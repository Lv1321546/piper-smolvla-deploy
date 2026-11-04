"""Keyboard reset cancellation holds the actual jaw, not the old open goal."""
from pathlib import Path
import time
import unittest

import numpy as np

from servo_bridge.core import Feedback, NAMES, load_config
from servo_bridge.sdk_worker import Worker
from servo_bridge.tests.test_geometry_hold import Writer


class ResetGripperHoldTests(unittest.TestCase):
    def test_cancel_opening_holds_fresh_measured_jaw(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        writer = Writer()
        now = 100.
        worker = Worker(cfg, writer, clock=lambda:now)
        q = np.radians(cfg['sim_initial_joints_deg'])
        feedback = Feedback(q, now, .04, True, True)
        packet = dict(version=1, monotonic=now, servo_status=0, status_monotonic=now,
                      deadman=True, joint_names=list(NAMES), positions=q.tolist(),
                      velocities=[0.]*6, horizon_s=.02, gripper={'width_m':.07})
        worker.packet(packet, feedback)
        self.assertEqual(writer.grippers[-1][0], 70000)
        now += .02
        feedback.received = now
        feedback.width = .05
        worker.packet(dict(version=1, monotonic=now, servo_status=0, status_monotonic=now,
                           deadman=False, hold_gripper=True), feedback)
        self.assertEqual(writer.grippers[-1][0], 50000)
        self.assertFalse(worker.moved)


if __name__ == '__main__':
    unittest.main()
