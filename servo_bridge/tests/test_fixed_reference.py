"""Fixed target must stay independent of the arm's current measured state."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from servo_bridge.core import load_config
from servo_bridge.start_reference import load_fixed_reference
from servo_bridge.hold_to_home import load_home_reference


class FixedReferenceTests(unittest.TestCase):
    def test_startup_and_buttons_load_identical_persistent_joint_target(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        reference = dict(schema_version=1,joint_unit='degree',
                         joint_angles=[2.681,80.357,-64.467,-3.659,62.954,2.46])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'home.json'
            path.write_text(json.dumps(reference))
            before = path.read_bytes()
            self.assertEqual(load_fixed_reference(path,cfg),reference)
            np.testing.assert_allclose(load_home_reference(path),np.radians(reference['joint_angles']))
            self.assertEqual(path.read_bytes(),before)

    def test_unusable_fixed_target_is_rejected_before_startup(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        reference = dict(schema_version=1,joint_unit='degree',
                         joint_angles=[2.681,80.357,-64.467,-3.659,70.381,2.46])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'home.json';path.write_text(json.dumps(reference))
            with self.assertRaises(ValueError):
                load_fixed_reference(path,cfg)


if __name__=='__main__':unittest.main()
