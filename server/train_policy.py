"""Fine-tune a local SmolVLA checkpoint on exported PiPER episodes.

The default only validates inputs and prints a plan. --execute invokes LeRobot's
official training pipeline, evaluates held-out episodes and writes a deployment
manifest. Nothing is downloaded or published. Numerical validation is not a
claim of real-robot task success.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import subprocess
import importlib.metadata
from contextlib import contextmanager
from pathlib import Path

from server.policy_runner import (PolicyRunner, processor_file_hashes, sha256_file,
                                  validate_schema, validate_stats)
from shared.protocol import ACTION_NAMES


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_split(split, total_episodes):
    groups = []
    for name in ("train", "validation"):
        values = split.get(name)
        if (not isinstance(values, list) or not values
                or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < total_episodes for x in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"split.json needs nonempty unique {name} episode indices")
        groups.append(set(values))
    if groups[0] & groups[1]:
        raise ValueError("Training and validation episodes overlap")
    if groups[0] | groups[1] != set(range(total_episodes)):
        raise ValueError("split.json must account for every exported episode")
    return split


def make_plan(dataset_root, base_model, vlm_path, output_dir, steps=5000, batch_size=8, *,
              seed=42, lr=1e-4, warmup_steps=None, decay_lr=2.5e-6,
              save_freq=2000, eval_freq=2000, log_freq=50, stop_after=None,
              resume=None, use_amp=False, augment_images=False):
    root, base, vlm, output = [Path(p).expanduser().resolve() for p in
                               (dataset_root, base_model, vlm_path, output_dir)]
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if not root.is_dir() or not (root / "meta/info.json").is_file():
        raise ValueError("No exported LeRobot dataset; collect and export real demonstrations first")
    info = read_json(root / "meta/info.json")
    schema = validate_schema(read_json(root / "deployment_schema.json"))
    split = validate_split(read_json(root / "split.json"), info["total_episodes"])
    provenance = read_json(root / "export_manifest.json")
    if provenance.get("status") != "complete" or provenance.get("split") != split:
        raise ValueError("Dataset export must be complete and match split.json")
    stats = read_json(root / "train_stats.json")
    validate_stats(stats)
    if info.get("robot_type") != "piper" or not math.isclose(float(info.get("fps", 0)), schema["control_hz"]):
        raise ValueError("LeRobot robot_type/fps differ from deployment schema")
    for name in ("observation.state", "action"):
        feature = info.get("features", {}).get(name, {})
        if feature.get("shape") != [7] or feature.get("names") != list(ACTION_NAMES):
            raise ValueError(f"Invalid PiPER feature names or dimensions: {name}")
    for name, camera in schema["cameras"].items():
        feature = info.get("features", {}).get(f"observation.images.{name}", {})
        if feature.get("shape") != [camera["height"], camera["width"], 3]:
            raise ValueError(f"Camera dimensions differ from schema: {name}")
    if not (base / "config.json").is_file() or not (base / "model.safetensors").is_file():
        raise ValueError("base_model must contain local config.json and model.safetensors")
    if not vlm.is_dir() or not (vlm / "config.json").is_file():
        raise ValueError("vlm_path must contain a local VLM/tokenizer bundle")
    base_config = read_json(base / "config.json")
    if base_config.get("type") != "smolvla":
        raise ValueError("Expected a SmolVLA base checkpoint")
    if min(base_config.get("max_state_dim", 32), base_config.get("max_action_dim", 32)) < 7:
        raise ValueError("Base padded projection dimensions cannot support seven PiPER values")
    stop_after = steps if stop_after is None else stop_after
    warmup_steps = min(1000, max(1, steps // 10)) if warmup_steps is None else warmup_steps
    if not 0 < stop_after <= steps or not 0 < warmup_steps < steps:
        raise ValueError("Require 0 < stop_after <= steps and 0 < warmup_steps < steps")
    if not (math.isfinite(lr) and math.isfinite(decay_lr) and 0 < decay_lr <= lr):
        raise ValueError("Require finite 0 < decay_lr <= lr")
    if min(save_freq, eval_freq, log_freq) <= 0:
        raise ValueError("save/eval/log frequencies must be positive")
    resume_path = resolve_checkpoint(resume) if resume is not None else None
    if output.exists() and resume_path is None:
        raise ValueError("output_dir already exists; choose a new run directory")
    if resume_path is not None:
        previous = read_json(output / "experiment_plan.json")
        expected = dict(dataset_root=str(root), base_model=str(base), vlm_path=str(vlm),
                        steps=steps, batch_size=batch_size, seed=seed, lr=lr,
                        warmup_steps=warmup_steps, decay_lr=decay_lr, use_amp=use_amp,
                        augment_images=augment_images, stats_sha256=sha256_file(root / "train_stats.json"))
        expected.update(schema=schema, episode_split=split)
        for key, value in expected.items():
            if previous.get(key) != value:
                raise ValueError(f"Resume invariant changed: {key}")
        checkpoint_step = read_json(resume_path / "training_state/training_step.json")["step"]
        if not checkpoint_step < stop_after:
            raise ValueError("stop_after must be greater than resumed checkpoint step")
        if not resume_path.is_relative_to(output / "checkpoints"):
            raise ValueError("Resume checkpoint must belong to output_dir")
    return {"status": "plan_only", "dataset_root": str(root), "base_model": str(base),
            "vlm_path": str(vlm), "output_dir": str(output), "steps": steps, "batch_size": batch_size,
            "seed": seed, "lr": lr, "warmup_steps": warmup_steps, "decay_lr": decay_lr,
            "save_freq": save_freq, "eval_freq": eval_freq, "log_freq": log_freq,
            "stop_after": stop_after, "resume_checkpoint": str(resume_path) if resume_path else None,
            "use_amp": use_amp, "augment_images": augment_images,
            "camera_order": [name for name in ("front", "astra") if name in schema["cameras"]]
                            + sorted(set(schema["cameras"]) - {"front", "astra"}),
            "schema": schema, "episode_split": split, "stats_path": str(root / "train_stats.json"),
            "mock_dataset": bool(schema.get("mock", False) or provenance.get("mock", False)),
            "stats_sha256": sha256_file(root / "train_stats.json"),
            "base_model_sha256": sha256_file(base / "model.safetensors"),
            "normalization_source": "training episodes only",
            "base_projection_dims": {"state": base_config.get("max_state_dim", 32),
                                     "action": base_config.get("max_action_dim", 32)},
            "note": "Six-to-seven adaptation keeps padded projection sizes; PiPER fine-tuning is still required."}



def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_file() and path.name == "train_config.json":
        path = path.parent
    if path.name == "pretrained_model":
        path = path.parent
    for relative in ("pretrained_model/train_config.json", "pretrained_model/model.safetensors",
                     "training_state/training_step.json"):
        if not (path / relative).is_file():
            raise ValueError(f"Incomplete resume checkpoint: {path / relative}")
    return path

def build_train_config(plan, device="cuda", num_workers=4):
    """Public LeRobot 0.6.2 dataclasses; no shell/CLI string interpolation."""
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    policy = SmolVLAConfig.from_pretrained(plan["base_model"], local_files_only=True)
    policy.pretrained_path = Path(plan["base_model"])
    policy.vlm_model_name, policy.load_vlm_weights = plan["vlm_path"], False
    policy.device, policy.push_to_hub = device, False
    policy.empty_cameras = 0
    policy.adapt_to_pi_aloha = policy.use_delta_joint_actions_aloha = False
    policy.input_features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,))}
    policy.input_features.update({f"observation.images.{name}": PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, spec["height"], spec["width"]))
        for name, spec in plan["schema"]["cameras"].items()})
    policy.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy.input_features = {"observation.state": policy.input_features["observation.state"],
                             **{f"observation.images.{name}": policy.input_features[f"observation.images.{name}"]
                                for name in plan["camera_order"]}}
    policy.scheduler_decay_steps = plan["steps"]
    policy.scheduler_warmup_steps = plan["warmup_steps"]
    policy.scheduler_decay_lr = plan["decay_lr"]
    policy.optimizer_lr = plan["lr"]
    policy.optimizer_betas, policy.optimizer_eps = (0.9, 0.95), 1e-8
    policy.optimizer_weight_decay, policy.optimizer_grad_clip_norm = 1e-10, 10
    policy.freeze_vision_encoder = policy.train_expert_only = policy.train_state_proj = True
    policy.use_amp = plan["use_amp"]
    policy.chunk_size, policy.n_action_steps, policy.n_obs_steps = 50, 50, 1
    dataset = DatasetConfig(repo_id="local/piper", root=plan["dataset_root"],
                            episodes=plan["episode_split"]["train"], use_imagenet_stats=False,
                            video_backend="pyav", eval_split=0)
    cfg = TrainPipelineConfig(dataset=dataset, policy=policy, output_dir=Path(plan["output_dir"]),
                              steps=plan["stop_after"], batch_size=plan["batch_size"], num_workers=num_workers,
                              seed=plan["seed"], cudnn_deterministic=True,
                              env_eval_freq=0, eval_steps=0, save_freq=plan["save_freq"],
                              log_freq=plan["log_freq"])
    cfg.dataset.image_transforms.enable = plan["augment_images"]
    cfg.accelerator.mixed_precision = "fp16" if plan["use_amp"] else "no"
    cfg.accelerator.gradient_accumulation.steps = 1
    if plan.get("resume_checkpoint"):
        checkpoint_cfg = TrainPipelineConfig.from_pretrained(
            str(Path(plan["resume_checkpoint"]) / "pretrained_model"), local_files_only=True)
        cfg.optimizer, cfg.scheduler = checkpoint_cfg.optimizer, checkpoint_cfg.scheduler
        cfg.resume, cfg.checkpoint_path = True, Path(plan["resume_checkpoint"])
        cfg.policy.pretrained_path = Path(plan["resume_checkpoint"]) / "pretrained_model"
    return cfg


@contextmanager
def official_training_adapters(train_module, plan):
    """Keep the official optimizer/checkpoint loop; adapt episode split/processors.

    LeRobot's built-in eval_split re-splits episodes; our exporter already wrote
    the definitive split. Loading SO100's saved processors would also restore
    foreign dimensions/camera keys. Both changes are scoped to this training call.
    """
    import torch
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies import make_pre_post_processors

    original_datasets = train_module.make_train_eval_datasets
    original_processors = train_module.make_pre_post_processors
    stats = read_json(plan["stats_path"])
    tensor_stats = {key: {name: torch.tensor(value, dtype=torch.float32)
                          for name, value in entries.items()} for key, entries in stats.items()}

    def datasets(cfg):
        result = []
        for group in ("train", "validation"):
            local_cfg = copy.deepcopy(cfg)
            local_cfg.dataset.episodes = list(plan["episode_split"][group])
            local_cfg.dataset.eval_split = 0
            ds = make_dataset(local_cfg)
            # Never let the metadata's whole-dataset aggregate reach normalization.
            ds.meta.stats = {key: {name: torch.tensor(value, dtype=torch.float32)
                                   for name, value in entries.items()} for key, entries in stats.items()}
            result.append(ds)
        return tuple(result)

    def processors(policy_cfg, pretrained_path=None, pretrained_revision=None, **kwargs):
        return make_pre_post_processors(policy_cfg, pretrained_path=None,
                                        dataset_stats=tensor_stats)

    train_module.make_train_eval_datasets = datasets
    train_module.make_pre_post_processors = processors
    try:
        yield
    finally:
        train_module.make_train_eval_datasets = original_datasets
        train_module.make_pre_post_processors = original_processors


def validate_base_projections(plan):
    from safetensors import safe_open
    expected = {"model.state_proj.weight": (1, plan["base_projection_dims"]["state"]),
                "model.action_in_proj.weight": (1, plan["base_projection_dims"]["action"]),
                "model.action_out_proj.weight": (0, plan["base_projection_dims"]["action"])}
    with safe_open(str(Path(plan["base_model"]) / "model.safetensors"), framework="pt") as weights:
        for name, (axis, dimension) in expected.items():
            if name not in weights.keys() or weights.get_slice(name).get_shape()[axis] != dimension:
                raise ValueError(f"Base checkpoint has incompatible padded projection: {name}")


def evaluate_checkpoint(checkpoint, plan, samples=20, device="cuda"):
    from server.evaluate_training import evaluate_checkpoint as evaluate
    anchors = max(1, math.ceil(samples / len(plan["episode_split"]["validation"])))
    return evaluate(checkpoint, plan, anchors_per_episode=anchors, device=device,
                    seed=plan.get("seed", 42),
                    output_dir=Path(plan["output_dir"]) / "evaluations" / Path(checkpoint).resolve().parent.name)


def export_manifest(checkpoint, plan, validation):
    checkpoint = Path(checkpoint)
    if plan.get("mock_dataset"):
        raise ValueError("Mock demonstrations cannot produce a physical PiPER deployment manifest")
    if not validation.get("finite_predictions") or validation.get("samples", 0) <= 0:
        raise ValueError("Cannot export a deployment manifest without held-out validation")
    stats_path = checkpoint / "deployment_stats.json"
    stats_path.write_text(json.dumps(read_json(plan["stats_path"]), indent=2), encoding="utf-8")
    manifest = {**plan["schema"], "training_completed": True,
                "model_id": f"piper-smolvla-{Path(plan['output_dir']).name}",
                "episode_split": plan["episode_split"], "validation": validation,
                "stats_path": stats_path.name, "stats_sha256": sha256_file(stats_path),
                "model_sha256": sha256_file(checkpoint / "model.safetensors"),
                "config_sha256": sha256_file(checkpoint / "config.json"),
                "processor_sha256": processor_file_hashes(checkpoint),
                "normalization_source": "training episodes only", "training_steps": plan["steps"]}
    target = checkpoint / "deployment_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


def execute_training(plan, device="cuda", num_workers=4, validation_samples=20, *,
                     skip_final_evaluation=False):
    if plan.get("mock_dataset"):
        raise ValueError("Mock data is only for pipeline tests; collect real demonstrations before training")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import lerobot.scripts.lerobot_train as official
    from server.training_runtime import training_audit

    validate_base_projections(plan)
    cfg = build_train_config(plan, device, num_workers)
    original_argv = sys.argv
    try:
        # Public config.validate resolves resume from this real checkpoint config.
        sys.argv = [original_argv[0]]
        if plan.get("resume_checkpoint"):
            sys.argv += ["--config_path=" + str(Path(plan["resume_checkpoint"]) / "pretrained_model/train_config.json")]
        with official_training_adapters(official, plan), training_audit(official, plan):
            official.train(cfg)
    finally:
        sys.argv = original_argv
    checkpoint = Path(plan["output_dir"]) / "checkpoints/last/pretrained_model"
    if not checkpoint.is_dir():
        raise RuntimeError("Official trainer returned without its final checkpoint")
    result = {"checkpoint": str(checkpoint.resolve()), "trained_through_step": plan["stop_after"],
              "scheduler_decay_steps": plan["steps"],
              "plan_path": str(Path(plan["output_dir"]) / "experiment_plan.json")}
    if not skip_final_evaluation:
        validation = evaluate_checkpoint(checkpoint, plan, validation_samples, device)
        result["validation"] = validation
        if plan["stop_after"] == plan["steps"]:
            result["manifest"] = str(export_manifest(checkpoint, plan, validation))
    (Path(plan["output_dir"]) / (f"segment_result_{plan['stop_after']:06d}.json")).write_text(
        json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--decay-lr", type=float, default=2.5e-6)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--eval-freq", type=int, default=2000, help="Recorded evaluation cadence; independent checkpoint evaluator runs after training")
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--stop-after", type=int, help="Stop at this total update while preserving --steps scheduler horizon")
    parser.add_argument("--resume", help="Checkpoint step directory, last link, or pretrained_model directory")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--augment-images", action="store_true")
    parser.add_argument("--skip-final-evaluation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true", help="Actually train; default is plan-only")
    args = parser.parse_args(argv)
    plan = make_plan(args.dataset_root, args.base_model, args.vlm_path, args.output_dir,
                     args.steps, args.batch_size, seed=args.seed, lr=args.lr,
                     warmup_steps=args.warmup_steps, decay_lr=args.decay_lr,
                     save_freq=args.save_freq, eval_freq=args.eval_freq, log_freq=args.log_freq,
                     stop_after=args.stop_after, resume=args.resume, use_amp=args.use_amp,
                     augment_images=args.augment_images)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        print(json.dumps(execute_training(plan, args.device, args.num_workers,
                                          args.validation_samples,
                                          skip_final_evaluation=args.skip_final_evaluation), indent=2))
    else:
        print("Plan only: no weights loaded and no training started. Add --execute to train.")


if __name__ == "__main__":
    main()
