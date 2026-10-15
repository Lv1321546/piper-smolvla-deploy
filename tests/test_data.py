import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from e14.collect_data import record_episode
from e14.export_dataset import build_samples, episode_split, export_dataset, load_raw_episode

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Robot:
    def __init__(self, clock, offset=0):
        self.clock, self.offset, self.index = clock, offset, 0

    def get_observation(self):
        state = [self.offset + self.index, 10., -30., 0., 30., 0., .04]
        self.index += 1
        return {"state": state, "captured_monotonic": self.clock(), "feedback_monotonic": self.clock()}


class Camera:
    def __init__(self, clock):
        self.clock = clock

    def read(self):
        return {"images": {"front": np.full((12, 16, 3), 75, np.uint8)},
                "image_timestamps": {"front": self.clock()}}


class DataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "config.mock.json").read_text())
        self.config["cameras"]["front"].update(width=16, height=12)

    def tearDown(self):
        self.temp.cleanup()

    def record(self, index=0, offset=0, **kwargs):
        clock = Clock()
        path = self.root / "raw" / f"episode_{index:06d}"
        path.parent.mkdir(exist_ok=True)
        record_episode(Robot(clock, offset), Camera(clock), path, "pick cube", self.config, .2,
                       mock=True, outcome="success", clock=clock, sleep=clock.sleep, **kwargs)
        return path

    def test_next_state_labels_drop_final_frame(self):
        path = self.record()
        episode = load_raw_episode(path)
        samples = list(build_samples(episode))
        self.assertEqual(len(samples), len(episode["rows"]) - 1)
        self.assertEqual(samples[0]["observation.state"][0], 0)
        self.assertEqual(samples[0]["action"][0], 1)
        self.assertEqual(samples[-1]["action"][0], len(episode["rows"]) - 1)

    def test_failed_capture_marked_not_exportable(self):
        clock = Clock()
        class SlowRobot(Robot):
            def get_observation(self):
                result = super().get_observation()
                clock.now += .1
                return result
        path = self.root / "failed"
        with self.assertRaises((ValueError, RuntimeError)):
            record_episode(SlowRobot(clock), Camera(clock), path, "pick cube", self.config, .2,
                           mock=True, outcome="success", clock=clock, sleep=clock.sleep)
        self.assertEqual(json.loads((path / "manifest.json").read_text())["status"], "failed")
        with self.assertRaisesRegex(ValueError, "Incomplete/failed"):
            load_raw_episode(path)

    def test_export_rejects_gap_skew_and_path_escape(self):
        path = self.record()
        original = [json.loads(line) for line in (path / "frames.jsonl").read_text().splitlines()]
        for edit in (lambda rows: rows[1].update(captured_monotonic=100.2),
                     lambda rows: rows[1]["image_timestamps"].update(front=99.0),
                     lambda rows: rows[0]["images"].update(front="../outside.png")):
            rows = copy.deepcopy(original)
            edit(rows)
            (path / "frames.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                load_raw_episode(path)

    def test_disjoint_reproducible_episode_split(self):
        split = episode_split(10, .2, 42)
        self.assertEqual(split, episode_split(10, .2, 42))
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(sorted(split["train"] + split["validation"]), list(range(10)))
        with self.assertRaises(ValueError):
            episode_split(1)

    def test_actual_lerobot_export_load_and_train_only_stats(self):
        # A real v0.6.2 writer/reader round trip, using synthetic hardware only.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        for index, offset in enumerate((0, 50, 100)):
            self.record(index, offset)
        output = self.root / "dataset"
        report = export_dataset(self.root / "raw", output, "local/piper_test")
        dataset = LeRobotDataset("local/piper_test", root=output)
        self.assertEqual(len(dataset), 12)
        row = dataset[0]
        self.assertEqual(tuple(row["observation.images.front"].shape), (3, 12, 16))
        np.testing.assert_allclose(row["action"].numpy(), [1, 10, -30, 0, 30, 0, .04], atol=1e-6)
        train_ids = report["split"]["train"]
        expected = np.mean([50 * i + frame for i in train_ids for frame in range(4)])
        stats = json.loads((output / "train_stats.json").read_text())
        self.assertAlmostEqual(stats["observation.state"]["mean"][0], expected, places=5)
        schema = json.loads((output / "deployment_schema.json").read_text())
        self.assertTrue(schema["mock"])
        self.assertEqual(schema["cameras"], {"front": {"width": 16, "height": 12}})
        self.assertEqual(schema["label_semantics"], "next_state")
        self.assertTrue(report["final_raw_frame_omitted"])
        with self.assertRaises(FileExistsError):
            export_dataset(self.root / "raw", output, "local/piper_test")


if __name__ == "__main__":
    unittest.main()
