import copy
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from server.inference_server import create_app
from server.policy_runner import (PolicyRunner, processor_file_hashes, sha256_file,
                                  validate_processor_manifest, validate_schema, validate_stats)
from server.train_policy import build_train_config, execute_training, make_plan, validate_split
from shared.protocol import ACTION_NAMES, PROTOCOL_VERSION, encode_images, validate_response


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    cfg = json.loads((ROOT / "config.mock.json").read_text())
    cfg["server"]["port"] = 0
    return cfg


def schema():
    return {"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "robot_type": "piper",
            "action_names": list(ACTION_NAMES), "joint_unit": "degree",
            "gripper_unit": "meter_full_opening", "action_mode": "absolute_position",
            "label_semantics": "next_state", "control_hz": 20,
            "cameras": {"front": {"width": 320, "height": 240}}}


def statistics():
    return {name: {"mean": [0] * 7, "std": [1] * 7}
            for name in ("observation.state", "action")}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.runner = PolicyRunner(self.config, mock=True).load()
        self.server = create_app(self.runner, self.config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_address[1])
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, route, body=None):
        request = urllib.request.Request(self.url + route, data=None if body is None else body,
                                         headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=3) as response:
            return json.loads(response.read())

    def payload(self):
        return {"protocol_version": PROTOCOL_VERSION, "robot_type": "piper", "request_id": "test-1",
                "state": [0, 20, -30, 0, 0, 0, 0.04], "task": "Pick up the cube", "control_hz": 20,
                "images": encode_images({"front": np.zeros((240, 320, 3), dtype=np.uint8)})}

    def test_health_and_end_to_end_chunk(self):
        health = self.request("/health")
        self.assertEqual(health["status"], "ready")
        self.assertEqual(health["backend"], "mock")
        self.assertFalse(health["execution_ready"])
        payload = self.payload()
        result = self.request("/predict", json.dumps(payload).encode())
        validate_response(result, request_id="test-1", expected_hz=20)
        self.assertEqual(result["actions"], [payload["state"]] * 10)

    def test_wrong_frequency_is_rejected(self):
        payload = self.payload()
        payload["control_hz"] = 30
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request("/predict", json.dumps(payload).encode())
        self.assertEqual(raised.exception.code, 400)

    def test_invalid_image_and_nan_are_rejected(self):
        for value in (b'{"state":NaN}', b'[]'):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.request("/predict", value)
            self.assertEqual(raised.exception.code, 400)
        payload = self.payload()
        payload["images"]["front"] = "invalid-base64"
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.request("/predict", json.dumps(payload).encode())
        self.assertEqual(raised.exception.code, 400)

    def test_single_flight_returns_busy(self):
        entered, release = threading.Event(), threading.Event()
        original = self.runner.predict

        def slow(*args):
            entered.set()
            release.wait(timeout=3)
            return original(*args)

        self.runner.predict = slow
        response = []
        body = json.dumps(self.payload()).encode()
        worker = threading.Thread(target=lambda: response.append(self.request("/predict", body)))
        worker.start()
        self.assertTrue(entered.wait(timeout=2))
        try:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.request("/predict", body)
            self.assertEqual(raised.exception.code, 429)
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertEqual(len(response), 1)

    def test_network_bind_requires_opt_in(self):
        cfg = configuration()
        cfg["server"]["bind"] = "0.0.0.0"
        with self.assertRaisesRegex(ValueError, "Non-loopback"):
            create_app(self.runner, cfg)


class CheckpointTests(unittest.TestCase):
    def test_processor_manifest_covers_state_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            state = checkpoint / "normalizer.safetensors"
            state.write_bytes(b"test-state")
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (checkpoint / name).write_text(json.dumps({"steps": [{"state_file": state.name}]}))
            manifest = {"processor_sha256": processor_file_hashes(checkpoint)}
            self.assertEqual(len(manifest["processor_sha256"]), 3)
            validate_processor_manifest(checkpoint, manifest)
            state.write_bytes(b"changed-state")
            with self.assertRaisesRegex(ValueError, "Processor files differ"):
                validate_processor_manifest(checkpoint, manifest)
            state.write_bytes(b"test-state")
            pipeline = checkpoint / "policy_preprocessor.json"
            pipeline.write_text(pipeline.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Processor files differ"):
                validate_processor_manifest(checkpoint, manifest)

    def test_processor_external_state_path_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (checkpoint / name).write_text(json.dumps({"steps": [{"state_file": "../external"}]}))
            with self.assertRaisesRegex(ValueError, "relative files inside"):
                processor_file_hashes(checkpoint)

    def test_schema_rejects_wrong_order_units_and_resolution(self):
        validate_schema(schema(), configuration())
        for key, value in (("action_names", list(reversed(ACTION_NAMES))), ("joint_unit", "radian")):
            invalid = schema()
            invalid[key] = value
            with self.assertRaises(ValueError):
                validate_schema(invalid)
        config = configuration()
        config["cameras"]["front"]["width"] = 640
        with self.assertRaisesRegex(ValueError, "resolution"):
            validate_schema(schema(), config)

    def test_stats_need_seven_finite_dimensions(self):
        validate_stats(statistics())
        invalid = statistics()
        invalid["action"]["std"] = [1] * 6
        with self.assertRaises(ValueError):
            validate_stats(invalid)
        invalid = statistics()
        invalid["observation.state"]["mean"][0] = float("nan")
        with self.assertRaises(ValueError):
            validate_stats(invalid)

    def test_so100_base_cannot_be_execution_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text(json.dumps({
                "input_features": {"observation.state": {"shape": [6]}},
                "output_features": {"action": {"shape": [6]}}}))
            cfg = configuration()
            cfg["server"].update(backend="smolvla", model_path=str(model))
            with self.assertRaisesRegex(ValueError, "seven-dimensional"):
                PolicyRunner(cfg).load()

    def test_edited_dimensions_without_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            (model / "config.json").write_text(json.dumps({
                "input_features": {"observation.state": {"shape": [7]}},
                "output_features": {"action": {"shape": [7]}}}))
            cfg = configuration()
            cfg["server"].update(backend="smolvla", model_path=str(model))
            with self.assertRaisesRegex(ValueError, "manifest"):
                PolicyRunner(cfg).load()

    def test_manifest_hash_detects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder)
            config_path = model / "config.json"
            config_path.write_text(json.dumps({"input_features": {"observation.state": {"shape": [7]}},
                                               "output_features": {"action": {"shape": [7]}}}))
            manifest = {**schema(), "training_completed": True, "validation": {"finite_predictions": True},
                        "config_sha256": sha256_file(config_path),
                        "episode_split": {"train": [0], "validation": [1]}}
            (model / "deployment_manifest.json").write_text(json.dumps(manifest))
            config_path.write_text(config_path.read_text() + "\n")
            cfg = configuration()
            cfg["server"].update(backend="smolvla", model_path=str(model))
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                PolicyRunner(cfg).load()


class TrainingPlanTests(unittest.TestCase):
    def fixture(self, folder):
        root, base, vlm = [Path(folder) / name for name in ("dataset", "base", "vlm")]
        (root / "meta").mkdir(parents=True)
        base.mkdir()
        vlm.mkdir()
        features = {name: {"shape": [7], "names": list(ACTION_NAMES)}
                    for name in ("observation.state", "action")}
        features["observation.images.front"] = {"dtype": "image", "shape": [240, 320, 3]}
        (root / "meta/info.json").write_text(json.dumps({"robot_type": "piper", "fps": 20,
                                                         "total_episodes": 3, "features": features}))
        split = {"train": [0, 1], "validation": [2]}
        for name, value in (("split.json", split), ("deployment_schema.json", schema()),
                            ("train_stats.json", statistics()),
                            ("export_manifest.json", {"status": "complete", "split": split, "mock": False})):
            (root / name).write_text(json.dumps(value))
        (base / "config.json").write_text(json.dumps({"type": "smolvla", "max_state_dim": 32,
                                                       "max_action_dim": 32}))
        (base / "model.safetensors").write_bytes(b"fixture-only-not-real-weights")
        (vlm / "config.json").write_text("{}")
        return root, base, vlm, Path(folder) / "new_run"

    def test_split_rejects_leakage(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split({"train": [0, 1], "validation": [1]}, 2)

    def test_plan_is_offline_and_keeps_base_projection_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = make_plan(*self.fixture(folder), steps=50)
            self.assertEqual(plan["status"], "plan_only")
            self.assertEqual(plan["episode_split"]["validation"], [2])
            self.assertEqual(plan["base_projection_dims"], {"state": 32, "action": 32})
            self.assertFalse(Path(plan["output_dir"]).exists())

    def test_mock_data_cannot_start_real_training(self):
        with self.assertRaisesRegex(ValueError, "Mock data"):
            execute_training({"mock_dataset": True})

    def test_official_config_api_resolves_piper_features(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = make_plan(*self.fixture(folder), steps=50)
            cfg = build_train_config(plan, device="cpu", num_workers=0)
            self.assertEqual(cfg.policy.input_features["observation.state"].shape, (7,))
            self.assertEqual(cfg.policy.output_features["action"].shape, (7,))
            self.assertEqual(set(cfg.policy.image_features), {"observation.images.front"})
            self.assertEqual(cfg.policy.max_action_dim, 32)
            self.assertEqual(cfg.dataset.episodes, [0, 1])
            self.assertFalse(cfg.dataset.use_imagenet_stats)
            self.assertEqual(cfg.policy.vlm_model_name, plan["vlm_path"])


if __name__ == "__main__":
    unittest.main()
