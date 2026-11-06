"""Validate raw demonstrations and export image-backed LeRobot v3 datasets.

Every label is observed state[t+1], not an SDK command. The final raw frame has
no next-state target and is omitted. Splits are by episode; normalization stats
are computed from the training episodes only. Nothing is uploaded to the Hub.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from e14.collect_data import ACTION_NAMES, write_json


def image_path(episode: Path, relative: str) -> Path:
    path = (episode / relative).resolve()
    if not path.is_relative_to(episode.resolve()) or not path.is_file():
        raise ValueError(f"Missing or escaping image path: {relative}")
    return path


def load_raw_episode(path: str | Path, *, include_unknown: bool = False) -> dict:
    """Validate all timestamps/files before any exported dataset is created."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("format") != "piper-raw-v1" or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported raw format: {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete/failed recording cannot be exported: {path}")
    outcome = manifest.get("task_outcome")
    if outcome != "success" and not (include_unknown and outcome == "unknown"):
        raise ValueError(f"Task outcome {outcome!r} excluded: {path}")
    expected = {"state_names": ACTION_NAMES, "joint_unit": "degree",
                "gripper_unit": "meter_full_opening", "action_mode": "absolute_position",
                "label_semantics": "next_state"}
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError(f"State ordering/units/label semantics differ: {path}")
    fps = float(manifest["control_hz"])
    ratio = float(manifest["max_period_error_ratio"])
    if not math.isfinite(fps) or fps <= 0 or fps != int(fps) or not 0 < ratio < 1:
        raise ValueError("Export requires integer positive control_hz and timing ratio in (0,1)")
    if not isinstance(manifest.get("task"), str) or not manifest["task"].strip():
        raise ValueError("Missing task")
    cameras = manifest["cameras"]
    if not cameras:
        raise ValueError("No cameras")
    rows = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
    if len(rows) < 2 or len(rows) != manifest["frames"] or len(rows) != manifest["expected_frames"]:
        raise ValueError(f"Frame count mismatch: {path}")
    previous = None
    for index, row in enumerate(rows):
        state = np.asarray(row["state"], dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all() or row["frame_index"] != index:
            raise ValueError(f"Invalid state/index at frame {index}")
        if set(row["images"]) != set(cameras) or set(row["image_timestamps"]) != set(cameras):
            raise ValueError("Missing camera observation")
        captured = float(row["captured_monotonic"])
        feedback = float(row["feedback_monotonic"])
        collected = float(row["collected_monotonic"])
        stamps = [captured, feedback, collected, *map(float, row["image_timestamps"].values())]
        if not all(math.isfinite(t) for t in stamps) or min(stamps) < 0:
            raise ValueError("Invalid timestamp")
        if not 0 <= collected - captured <= float(manifest["feedback_max_age_ms"]) / 1000:
            raise ValueError("Stale/future captured observation")
        if not 0 <= collected - feedback <= float(manifest["feedback_max_age_ms"]) / 1000:
            raise ValueError("Stale/future feedback")
        synchronized = [feedback, *map(float, row["image_timestamps"].values())]
        if (max(synchronized) - min(synchronized)) * 1000 > float(manifest["max_sync_skew_ms"]):
            raise ValueError("Image/state synchronization failed")
        if previous is not None:
            dt = captured - previous["captured_monotonic"]
            if dt <= 0 or abs(dt - 1 / fps) > ratio / fps:
                raise ValueError(f"Missing/nonuniform frame at {index}")
            if feedback <= previous["feedback_monotonic"]:
                raise ValueError("Repeated robot feedback")
        for name, camera in cameras.items():
            stamp = float(row["image_timestamps"][name])
            if not 0 <= collected - stamp <= float(camera["max_age_ms"]) / 1000:
                raise ValueError(f"Stale/future image: {name}")
            if previous is not None and stamp <= previous["image_timestamps"][name]:
                raise ValueError(f"Repeated camera frame: {name}")
            source = image_path(path, row["images"][name])
            with Image.open(source) as image:
                image.load()
                if image.mode != "RGB" or image.size != (camera["width"], camera["height"]):
                    raise ValueError(f"Image format/dimensions mismatch: {source}")
        previous = row
    return {"path": path, "manifest": manifest, "rows": rows}


def build_samples(episode: dict):
    rows, root, task = episode["rows"], episode["path"], episode["manifest"]["task"]
    for index, row in enumerate(rows[:-1]):
        frame = {"observation.state": np.asarray(row["state"], dtype=np.float32),
                 "action": np.asarray(rows[index + 1]["state"], dtype=np.float32), "task": task}
        for name, relative in row["images"].items():
            with Image.open(image_path(root, relative)) as image:
                frame[f"observation.images.{name}"] = np.asarray(image).copy()
        yield frame


def encoded_episode_buffer(episode, features, episode_index):
    """Official save_episode(episode_data=...) buffer using original image paths.

    load_raw_episode already validated decoded dimensions, bytes and timing.
    The official writer embeds file bytes; no PNG round trip is necessary.
    """
    rows=episode['rows']; count=len(rows)-1; m=episode['manifest']
    buffer={name: [] for name in features}
    buffer.update(size=count, task=[m['task']]*count, episode_index=episode_index,
                  frame_index=list(range(count)), timestamp=[i/m['control_hz'] for i in range(count)])
    buffer['observation.state']=[np.asarray(r['state'],np.float32) for r in rows[:-1]]
    buffer['action']=[np.asarray(r['state'],np.float32) for r in rows[1:]]
    for camera in m['cameras']:
        buffer[f'observation.images.{camera}']=[str(image_path(episode['path'],r['images'][camera])) for r in rows[:-1]]
    return buffer


def episode_split(count: int, validation_fraction: float = .2, seed: int = 42) -> dict:
    if count < 2:
        raise ValueError("At least two accepted episodes are required for disjoint train/validation")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    order = np.random.default_rng(seed).permutation(count).tolist()
    validation_count = max(1, min(count - 1, round(count * validation_fraction)))
    return {"train": sorted(order[validation_count:]), "validation": sorted(order[:validation_count])}


def preserve_encoded_images(destination, episodes):
    """Keep original JPEG bytes in image-backed parquet, not decoded PNG copies.

    LeRobot's writer first emits its standard representation. Rewrite only
    image payloads, in small batches, retaining indices, schema and all labels.
    """
    rows=[(episode['path'],row['images']) for episode in episodes for row in episode['rows'][:-1]]
    if not any(Path(relative).suffix.lower() in ('.jpg','.jpeg') for _,images in rows for relative in images.values()):
        return False
    import pyarrow as pa
    import pyarrow.parquet as pq
    files=sorted((destination/'data').rglob('*.parquet'))
    if not files:raise ValueError('Missing exported parquet files')
    seen=set()
    for path in files:
        source=pq.ParquetFile(path)
        temporary=path.with_suffix('.parquet.tmp')
        with pq.ParquetWriter(temporary,source.schema_arrow,compression='zstd') as writer:
            for batch in source.iter_batches(batch_size=32):
                table=pa.Table.from_batches([batch])
                indexes=table.column('index').to_pylist()
                if (len(set(indexes))!=len(indexes) or
                        any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(rows) or i in seen for i in indexes)):
                    raise ValueError('Unexpected exported row index')
                seen.update(indexes)
                for name in episodes[0]['manifest']['cameras']:
                    key=f'observation.images.{name}'
                    field=table.schema.field(key)
                    if not pa.types.is_struct(field.type) or set(field.type.names)!={'bytes','path'}:
                        raise ValueError('Unsupported encoded image storage type')
                    payloads=[]
                    for index in indexes:
                        root,images=rows[index]
                        payloads.append({'bytes':image_path(root,images[name]).read_bytes(),'path':None})
                    table=table.set_column(table.schema.get_field_index(key),field,pa.array(payloads,type=field.type))
                writer.write_table(table)
        source.close()
        temporary.replace(path)
    if seen!=set(range(len(rows))):raise ValueError('Exported row mapping is incomplete')
    return True


def export_dataset(raw_root: str | Path, output: str | Path, repo_id: str, *,
                   validation_fraction: float = .2, seed: int = 42, include_unknown: bool = False,
                   direct_encoded_images: bool = False) -> dict:
    raw_root, output = Path(raw_root), Path(output)
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new path: {output}")
    paths = sorted(p for p in raw_root.glob("episode_*") if p.is_dir())
    accepted, excluded = [], []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "complete" or manifest.get("task_outcome") == "failure":
            excluded.append({"source": str(path), "reason": "failed/incomplete recording or task"})
            continue
        if manifest.get("task_outcome") != "success" and not include_unknown:
            excluded.append({"source": str(path), "reason": "unknown task outcome"})
            continue
        accepted.append(load_raw_episode(path, include_unknown=include_unknown))
    split = episode_split(len(accepted), validation_fraction, seed)
    first = accepted[0]["manifest"]
    schema = {"schema_version": 1, "protocol_version": "piper-v1", "robot_type": "piper",
              "action_names": ACTION_NAMES, "joint_unit": "degree", "gripper_unit": "meter_full_opening",
              "action_mode": "absolute_position", "label_semantics": "next_state",
              "control_hz": first["control_hz"],
              "cameras": {name: {k: value[k] for k in ("width", "height")}
                          for name, value in first["cameras"].items()},
              "mock": any(e["manifest"].get("mock", False) for e in accepted)}
    for episode in accepted:
        m = episode["manifest"]
        dims = {name: {k: c[k] for k in ("width", "height")} for name, c in m["cameras"].items()}
        if m["control_hz"] != schema["control_hz"] or dims != schema["cameras"]:
            raise ValueError("Episodes have inconsistent frequency or cameras")
    features = {"observation.state": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
                "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES}}
    features.update({f"observation.images.{name}": {"dtype": "image", "shape": (c["height"], c["width"], 3),
                                                       "names": ["height", "width", "channels"]}
                     for name, c in schema["cameras"].items()})
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.utils import serialize_dict
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    destination = temporary / "dataset"
    dataset = None
    try:
        dataset = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=int(schema["control_hz"]),
                                        robot_type="piper", features=features, use_videos=False)
        train_stats = []
        mapping = []
        for index, episode in enumerate(accepted):
            if direct_encoded_images:
                dataset.save_episode(episode_data=encoded_episode_buffer(episode,dataset.meta.features,index))
            else:
                for frame in build_samples(episode):
                    dataset.add_frame(frame)
                dataset.save_episode()
            print(f"Exported episode {index+1}/{len(accepted)}",flush=True)
            mapping.append({"episode_index": index, "raw_source": str(episode["path"].resolve()),
                            "frames": len(episode["rows"]) - 1,
                            "task_outcome": episode["manifest"]["task_outcome"]})
            if index in split["train"]:
                rows = episode["rows"]
                stats_input = {"observation.state": np.asarray([r["state"] for r in rows[:-1]], np.float32),
                               "action": np.asarray([r["state"] for r in rows[1:]], np.float32)}
                for name in schema["cameras"]:
                    stats_input[f"observation.images.{name}"] = [str(image_path(episode["path"], r["images"][name]))
                                                                for r in rows[:-1]]
                train_stats.append(compute_episode_stats(stats_input, features))
        dataset.finalize()
        original_image_bytes=True if direct_encoded_images else preserve_encoded_images(destination,accepted)
        write_json(destination / "split.json", split)
        write_json(destination / "train_stats.json", serialize_dict(aggregate_stats(train_stats)))
        write_json(destination / "deployment_schema.json", schema)
        report = {"schema_version": 1, "status": "complete", "repo_id": repo_id,
                  "label_semantics": "next_state", "final_raw_frame_omitted": True,
                  "seed": seed, "split": split, "episodes": mapping, "excluded": excluded,
                  "mock": schema["mock"], "images_embedded_in_parquet": True,
                  "original_encoded_image_bytes_preserved": original_image_bytes,
                  "direct_encoded_images": direct_encoded_images,
                  "normalization_stats": "train_stats.json (training split only)"}
        write_json(destination / "export_manifest.json", report)
        destination.rename(output)
        return report
    finally:
        if dataset is not None:
            dataset.finalize()
        shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/piper_pickplace")
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direct-encoded-images", action="store_true", help="Use official episode_data buffer to embed original image bytes without PNG conversion")
    parser.add_argument("--include-unknown", action="store_true",
                        help="Explicitly admit complete recordings with unannotated task outcome")
    args = parser.parse_args(argv)
    report = export_dataset(args.raw_root, args.output, args.repo_id,
                            validation_fraction=args.validation_fraction, seed=args.seed,
                            include_unknown=args.include_unknown,direct_encoded_images=args.direct_encoded_images)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
