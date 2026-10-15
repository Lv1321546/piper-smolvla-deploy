"""Read-only timestamped PiPER demonstrations. Never enable or command motors.

Actions are not invented here: export_dataset labels observation[t] with the next
observed state, explicitly marked next_state rather than recorded commands.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ACTION_NAMES = [f"joint{i}" for i in range(1, 7)] + ["gripper_width"]


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def collection_settings(config: dict) -> dict:
    robot, collection = config["robot"], config.get("collection", {})
    settings = {
        "control_hz": float(robot["control_hz"]),
        "feedback_max_age_ms": float(robot["feedback_max_age_ms"]),
        "max_sync_skew_ms": float(collection.get("max_sync_skew_ms", 100)),
        "max_period_error_ratio": float(collection.get("max_period_error_ratio", .5)),
    }
    if not all(math.isfinite(v) and v > 0 for v in settings.values()):
        raise ValueError("Collection frequency and timing tolerances must be positive finite numbers")
    if settings["max_period_error_ratio"] >= 1:
        raise ValueError("max_period_error_ratio must be < 1; missing ticks cannot be hidden")
    if not config.get("cameras"):
        raise ValueError("At least one RGB camera is required")
    for name, camera in config["cameras"].items():
        if not name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Invalid camera name: {name!r}")
        if any(not isinstance(camera.get(d), int) or camera[d] <= 0 for d in ("width", "height")):
            raise ValueError(f"Camera {name} needs positive integer width and height")
    if robot.get("joint_unit", "degree") != "degree" or robot.get("gripper_unit", "meter_full_opening") != "meter_full_opening":
        raise ValueError("Required units: joint degree and gripper full opening meter")
    if list(robot.get("state_action_order", ACTION_NAMES)) != ACTION_NAMES:
        raise ValueError("Unsupported state ordering")
    return settings


def validate_sample(observation: dict, camera_frame: dict, config: dict, settings: dict, now: float) -> dict:
    state = np.asarray(observation["state"], dtype=np.float64)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError("State must be seven finite numbers")
    captured, feedback = (float(observation[k]) for k in ("captured_monotonic", "feedback_monotonic"))
    images = camera_frame["images"]
    image_times = {name: float(t) for name, t in camera_frame["image_timestamps"].items()}
    if set(images) != set(config["cameras"]) or set(image_times) != set(images):
        raise ValueError("Camera keys differ from config")
    times = [captured, feedback, *image_times.values()]
    if not all(math.isfinite(t) and 0 <= now - t < 86400 for t in times):
        raise ValueError("Invalid, future or stale monotonic timestamp")
    if (now - feedback) * 1000 > settings["feedback_max_age_ms"]:
        raise ValueError("Stale robot feedback")
    synchronized = [feedback, *image_times.values()]
    skew = (max(synchronized) - min(synchronized)) * 1000
    if skew > settings["max_sync_skew_ms"]:
        raise ValueError(f"Image/state skew {skew:.1f} ms exceeds tolerance")
    for name, image in images.items():
        camera = config["cameras"][name]
        expected = (camera["height"], camera["width"], 3)
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != expected:
            raise ValueError(f"Camera {name}: expected uint8 RGB {expected}")
        if (now - image_times[name]) * 1000 > float(camera.get("max_age_ms", settings["feedback_max_age_ms"])):
            raise ValueError(f"Stale image from {name}")
    return {"state": state.tolist(), "captured_monotonic": captured, "feedback_monotonic": feedback,
            "image_timestamps": image_times, "collected_monotonic": now, "sync_skew_ms": skew}


def record_episode(robot, cameras, output: str | Path, task: str, config: dict, duration: float,
                   *, mock: bool = False, outcome: str = "unknown", clock: Callable = time.monotonic,
                   sleep: Callable = time.sleep) -> dict:
    """Fail on missed ticks rather than omit frames. Completion is not task success."""
    settings = collection_settings(config)
    if not task.strip() or not math.isfinite(duration) or duration <= 0:
        raise ValueError("A task and positive duration are required")
    if outcome not in {"unknown", "success", "failure"}:
        raise ValueError("Invalid task outcome")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    period = 1 / settings["control_hz"]
    count = max(2, int(math.floor(duration * settings["control_hz"] + 1e-9)) + 1)
    started = clock()
    manifest = {
        "schema_version": 1, "format": "piper-raw-v1", "status": "recording",
        "task": task, "task_outcome": outcome, "mock": mock, "frames": 0,
        "expected_frames": count, "requested_duration_s": duration,
        "created_unix_s": time.time(), "started_monotonic": started,
        "state_names": ACTION_NAMES, "joint_unit": "degree", "gripper_unit": "meter_full_opening",
        "action_mode": "absolute_position", "label_semantics": "next_state",
        "label_note": "No commands recorded; export pairs observation[t] with observed state[t+1].",
        "cameras": {name: {"width": c["width"], "height": c["height"], "color": "RGB",
                             "max_age_ms": c.get("max_age_ms", settings["feedback_max_age_ms"])}
                    for name, c in config["cameras"].items()}, **settings,
    }
    write_json(output / "manifest.json", manifest)
    previous = None
    previous_feedback = None
    previous_images = None
    try:
        with (output / "frames.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(count):
                deadline = started + index * period
                remaining = deadline - clock()
                if remaining > 0:
                    sleep(remaining)
                if clock() - deadline > period * settings["max_period_error_ratio"]:
                    raise RuntimeError(f"Missed collection deadline at frame {index}; no ticks skipped")
                observation, camera_frame = robot.get_observation(), cameras.read()
                sample = validate_sample(observation, camera_frame, config, settings, clock())
                if previous is not None:
                    dt = sample["captured_monotonic"] - previous
                    if dt <= 0 or abs(dt - period) > period * settings["max_period_error_ratio"]:
                        raise RuntimeError(f"Nonuniform sampling at frame {index}: {dt:.6f} s")
                    if sample["feedback_monotonic"] <= previous_feedback:
                        raise RuntimeError(f"Repeated robot feedback at frame {index}")
                    if any(sample["image_timestamps"][name] <= previous_images[name]
                           for name in sample["image_timestamps"]):
                        raise RuntimeError(f"Repeated camera frame at frame {index}")
                previous = sample["captured_monotonic"]
                previous_feedback = sample["feedback_monotonic"]
                previous_images = sample["image_timestamps"]
                paths = {}
                for name, image in camera_frame["images"].items():
                    relative = Path("images") / name / f"{index:06d}.png"
                    (output / relative).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(image).save(output / relative, compress_level=1)
                    paths[name] = relative.as_posix()
                sample.update(frame_index=index, images=paths)
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
                stream.flush()
                manifest["frames"] = index + 1
        manifest.update(status="complete", finished_monotonic=clock())
        write_json(output / "manifest.json", manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", task_outcome="unknown", error=f"{type(exc).__name__}: {exc}",
                        finished_monotonic=clock())
        write_json(output / "manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Local raw root; copy to NAS after collection")
    parser.add_argument("--task", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--duration", type=float, default=2)
    parser.add_argument("--outcome", choices=["unknown", "success", "failure"], default="unknown",
                        help="Explicit operator task outcome, independent of recording completeness")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = json.loads(args.config.read_text())
    collection_settings(config)
    from e14.piper_robot import PiperRobot
    from e14.cameras import CameraReader
    robot, cameras = PiperRobot(config, mock=args.mock), CameraReader(config["cameras"], mock=args.mock)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        robot.connect()
        cameras.connect()
        existing = [int(p.name.split("_")[1]) for p in args.output.glob("episode_[0-9]*") if p.is_dir()]
        offset = max(existing, default=-1) + 1
        for index in range(args.episodes):
            destination = args.output / f"episode_{offset + index:06d}"
            print(f"Recording {destination} ({args.duration}s), read-only; outcome={args.outcome}", flush=True)
            manifest = record_episode(robot, cameras, destination, args.task, config, args.duration,
                                      mock=args.mock, outcome=args.outcome)
            print(f"Saved {manifest['frames']} raw frames; next-state labels derived at export", flush=True)
    finally:
        cameras.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
