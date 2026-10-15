"""Offline PiPER SmolVLA inference; imports do not load GPU libraries.

The six-dimensional SO100 base checkpoint is only usable in the separate public
example. This service never slices a PiPER observation to fit a foreign robot.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import time
from pathlib import Path
from shared.protocol import ACTION_NAMES, PROTOCOL_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_file(checkpoint: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Processor paths must be relative files inside the checkpoint")
    root = checkpoint.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or external checkpoint processor file: {relative}")
    return path


def processor_file_hashes(checkpoint: Path) -> dict:
    """Hash processor configs and every declared state/artifact file, without GPU imports."""
    names = {"policy_preprocessor.json", "policy_postprocessor.json"}
    for config_name in tuple(names):
        config_path = checkpoint_file(checkpoint, config_name)
        pipeline = json.loads(config_path.read_text())
        if not isinstance(pipeline.get("steps"), list) or not pipeline["steps"]:
            raise ValueError("Processor config must contain steps")
        for step in pipeline["steps"]:
            state_file = step.get("state_file")
            if state_file:
                checkpoint_file(checkpoint, state_file)
                names.add(state_file)
            artifacts = step.get("artifacts", {})
            if not isinstance(artifacts, dict):
                raise ValueError("Invalid processor artifacts")
            for relative in artifacts.values():
                if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("External processor artifact path")
                artifact = (checkpoint / relative).resolve()
                if not artifact.is_relative_to(checkpoint.resolve()) or not artifact.exists():
                    raise ValueError("External or missing processor artifact")
                if artifact.is_file():
                    names.add(relative)
                else:
                    names.update(str(file.relative_to(checkpoint.resolve())) for file in artifact.rglob("*") if file.is_file())
    return {name: sha256_file(checkpoint_file(checkpoint, name)) for name in sorted(names)}


def validate_processor_manifest(checkpoint: Path, manifest: dict) -> None:
    expected = manifest.get("processor_sha256")
    actual = processor_file_hashes(checkpoint)
    if not isinstance(expected, dict) or expected != actual:
        raise ValueError("Processor files differ from deployment manifest")


def validate_schema(schema: dict, config: dict | None = None) -> dict:
    required = {
        "schema_version": 1, "protocol_version": PROTOCOL_VERSION,
        "robot_type": "piper", "action_names": list(ACTION_NAMES),
        "joint_unit": "degree", "gripper_unit": "meter_full_opening",
        "action_mode": "absolute_position", "label_semantics": "next_state",
    }
    for key, value in required.items():
        if schema.get(key) != value:
            raise ValueError(f"Incompatible deployment schema: {key}")
    hz = schema.get("control_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("Invalid schema control_hz")
    cameras = schema.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Schema must declare real camera names and resolutions")
    for camera in cameras.values():
        if not isinstance(camera, dict) or any(
            isinstance(camera.get(k), bool) or not isinstance(camera.get(k), int) or camera[k] <= 0
            for k in ("width", "height")
        ):
            raise ValueError("Invalid schema camera resolution")
    if config:
        robot = config.get("robot", {})
        for key in ("joint_unit", "gripper_unit", "action_mode"):
            if robot.get(key, required[key]) != required[key]:
                raise ValueError(f"Configured robot {key} differs from trained schema")
        if list(robot.get("state_action_order", ACTION_NAMES)) != list(ACTION_NAMES):
            raise ValueError("Configured action order differs from trained schema")
        if float(robot.get("control_hz", hz)) != float(hz):
            raise ValueError("Configured control_hz differs from trained schema")
        if set(config.get("server", {}).get("camera_map", {})) != set(cameras):
            raise ValueError("Configured camera names differ from trained schema")
        for name, camera in cameras.items():
            configured = config.get("cameras", {}).get(name, {})
            if any(configured.get(key) != camera[key] for key in ("width", "height")):
                raise ValueError(f"Configured camera resolution differs from trained schema: {name}")
    return schema


def validate_stats(stats: dict) -> None:
    for feature in ("observation.state", "action"):
        values = stats.get(feature, {})
        for key in ("mean", "std"):
            value = values.get(key)
            if not isinstance(value, list) or len(value) != len(ACTION_NAMES):
                raise ValueError(f"Missing seven-dimensional {feature}.{key} statistics")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
                raise ValueError(f"Invalid statistics: {feature}.{key}")
        if any(x < 0 for x in values["std"]):
            raise ValueError(f"Negative standard deviation: {feature}")


class PolicyRunner:
    def __init__(self, config: dict, mock: bool = False):
        self.config = config
        self.options = config.get("server", {})
        self.mock = mock or self.options.get("backend") == "mock"
        self.loaded = False
        self.policy = self.preprocess = self.postprocess = None
        self.schema = None
        self._metadata = {
            "protocol_version": PROTOCOL_VERSION, "backend": "mock" if self.mock else "smolvla",
            "execution_ready": False, "robot_type": "piper", "action_names": list(ACTION_NAMES),
            "joint_unit": "degree", "gripper_unit": "meter_full_opening",
            "action_mode": "absolute_position", "control_hz": config.get("robot", {}).get("control_hz", 20),
            "model_id": "mock-hold-current" if self.mock else str(self.options.get("model_path", "")),
        }

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)

    def load(self) -> "PolicyRunner":
        if self.loaded:
            return self
        if self.mock:
            self.loaded = True
            return self
        model_path = Path(self.options.get("model_path") or "").expanduser().resolve()
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise ValueError("server.model_path must be a local PiPER checkpoint directory")
        raw_config = json.loads((model_path / "config.json").read_text())
        for group, key in (("input_features", "observation.state"), ("output_features", "action")):
            if raw_config.get(group, {}).get(key, {}).get("shape") != [len(ACTION_NAMES)]:
                raise ValueError("Model is not a seven-dimensional PiPER policy; use the public example for SO100")
        manifest_path = Path(self.options.get("manifest_path") or model_path / "deployment_manifest.json")
        if not manifest_path.is_file():
            raise ValueError("Missing deployment manifest; changing a config dimension is insufficient")
        manifest = json.loads(manifest_path.read_text())
        self.schema = validate_schema(manifest, self.config)
        if manifest.get("mock", False):
            raise ValueError("Mock demonstrations are not a physical PiPER checkpoint")
        if sha256_file(model_path / "config.json") != manifest.get("config_sha256"):
            raise ValueError("Model configuration differs from deployment manifest")
        validate_processor_manifest(model_path, manifest)
        if manifest.get("training_completed") is not True or manifest.get("validation", {}).get("finite_predictions") is not True:
            raise ValueError("Manifest lacks completed training and held-out validation")
        split = manifest.get("episode_split", {})
        training, validation = split.get("train"), split.get("validation")
        if (not isinstance(training, list) or not training or not isinstance(validation, list) or not validation
                or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in training + validation)
                or len(training) != len(set(training)) or len(validation) != len(set(validation))
                or set(training) & set(validation)):
            raise ValueError("Manifest must record disjoint training and validation episodes")
        weights = model_path / "model.safetensors"
        if not weights.is_file() or sha256_file(weights) != manifest.get("model_sha256"):
            raise ValueError("Model weights differ from deployment manifest")
        stats_path = (model_path / manifest.get("stats_path", "deployment_stats.json")).resolve()
        if stats_path.parent != model_path.resolve():
            raise ValueError("Statistics must reside inside the checkpoint directory")
        if not stats_path.is_file() or sha256_file(stats_path) != manifest.get("stats_sha256"):
            raise ValueError("Normalization statistics differ from deployment manifest")
        stats = json.loads(stats_path.read_text())
        validate_stats(stats)
        vlm_path = Path(self.options.get("vlm_path") or raw_config.get("vlm_model_name", ""))
        if not vlm_path.is_dir():
            raise ValueError("server.vlm_path must reference local SmolVLM/tokenizer files")
        expected_features = {f"observation.images.{name}" for name in self.schema["cameras"]}
        camera_map = self.options.get("camera_map", {})
        if set(camera_map.values()) != expected_features:
            raise ValueError("Camera feature mapping differs from training")
        actual_features = {key for key, value in raw_config.get("input_features", {}).items()
                           if value.get("type", "").upper() == "VISUAL"}
        if actual_features != expected_features:
            raise ValueError("Checkpoint camera features differ from deployment schema")
        if any(raw_config.get(key, False) for key in (
            "use_relative_actions", "adapt_to_pi_aloha", "use_delta_joint_actions_aloha"
        )):
            raise ValueError("Only absolute PiPER joint targets are supported")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import numpy as np
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        device = self.options.get("device", "cuda:0")
        cfg = SmolVLAConfig.from_pretrained(str(model_path), local_files_only=True)
        cfg.device, cfg.vlm_model_name, cfg.load_vlm_weights = device, str(vlm_path), False
        if cfg.max_state_dim < 7 or cfg.max_action_dim < 7:
            raise ValueError("Padded projection dimensions cannot accommodate PiPER")
        self.policy = SmolVLAPolicy.from_pretrained(str(model_path), config=cfg, strict=True)
        self.policy.eval()
        tensor_stats = {f: {k: np.asarray(v) for k, v in entries.items()} for f, entries in stats.items()}
        self.preprocess, self.postprocess = make_pre_post_processors(
            cfg, str(model_path), preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_path)},
                "device_processor": {"device": device}, "normalizer_processor": {"stats": tensor_stats},
            }, postprocessor_overrides={
                "device_processor": {"device": "cpu"}, "unnormalizer_processor": {"stats": tensor_stats},
            },
        )
        self._metadata.update({"execution_ready": True, "control_hz": float(self.schema["control_hz"]),
                               "model_id": manifest.get("model_id", str(model_path))})
        self.loaded = True
        return self

    def predict(self, observation: dict, task: str) -> tuple[list[list[float]], float]:
        if not self.loaded:
            raise RuntimeError("Policy has not been loaded")
        start = time.perf_counter()
        state = observation.get("state")
        if not isinstance(state, list) or len(state) != len(ACTION_NAMES) or any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in state
        ):
            raise ValueError("PiPER requires seven finite state values")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("A task instruction is required")
        if self.mock:
            steps = int(self.options.get("mock_chunk_steps", 10))
            if not 1 <= steps <= int(self.options.get("max_chunk_steps", 100)):
                raise ValueError("Invalid mock_chunk_steps")
            return [list(map(float, state)) for _ in range(steps)], (time.perf_counter() - start) * 1000
        import numpy as np
        import torch
        images = observation.get("images", {})
        if set(images) != set(self.schema["cameras"]):
            raise ValueError("Observed camera names differ from trained camera names")
        raw = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": task}
        for name, target in self.options["camera_map"].items():
            image, expected = images[name], self.schema["cameras"][name]
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (expected["height"], expected["width"], 3):
                raise ValueError(f"Unexpected RGB image dimensions or dtype for {name}")
            raw[target] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        device = next(self.policy.parameters()).device
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            self.policy.reset()
            normalized = self.policy.predict_action_chunk(self.preprocess(raw))
            actions = self.postprocess(normalized).detach().cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != 7 or not torch.isfinite(actions).all():
            raise RuntimeError("Policy produced invalid PiPER actions")
        if not 0 < actions.shape[1] <= int(self.options.get("max_chunk_steps", 100)):
            raise RuntimeError("Policy chunk size exceeds configured maximum")
        return actions[0].tolist(), (time.perf_counter() - start) * 1000
