"""Normal-speed hold return and latched keyboard reset without hardware."""
import math
from pathlib import Path
import unittest

import numpy as np

from servo_bridge.core import load_config
from servo_bridge.hold_to_home import HoldToHomePolicy
from servo_bridge.tests.test_hold_to_home import Clock, pad


class KeyboardResetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        self.clock = Clock()
        self.home = np.radians(self.cfg['sim_initial_joints_deg'])
        self.q = self.home+np.radians([15, -30, -25, 10, -15, 20])
        self.width = .04
        self.policy = HoldToHomePolicy(self.cfg, home_q=self.home, clock=self.clock)
        self.step()

    def step(self, sample=None, quiet=True):
        self.clock.advance(.02)
        return self.policy.step(sample or pad(), self.policy.manual.kin.forward(self.q)[:3, :3], .02,
            measured_q=self.q, measured_width=self.width, home_may_start=quiet)

    def test_default_hold_return_uses_normal_joint_speed(self):
        self.assertEqual(self.policy.speed_deg_s, self.cfg['joint_speed_deg_s'])
        self.assertEqual(self.policy.acceleration_deg_s2, self.cfg['joint_acceleration_deg_s2'])
        maximum = 0
        for _ in range(150):
            velocity = self.step(pad(a=True))
            maximum = max(maximum, float(np.max(abs(velocity))))
        self.assertGreater(math.degrees(maximum), 3.)
        self.assertLessEqual(math.degrees(maximum), self.cfg['joint_speed_deg_s']+1e-8)
        np.testing.assert_array_equal(self.step(), 0)
        self.assertFalse(self.policy.active)

    def test_one_click_returns_then_opens_to_maximum_and_finishes(self):
        self.policy.request_reset()
        maximum = 0
        opened = False
        for _ in range(800):
            velocity = self.step()
            maximum = max(maximum, float(np.max(abs(velocity))))
            self.q += velocity*.02
            target = self.policy.reset_gripper_target
            if target is not None:
                opened = True
                self.assertLessEqual(max(abs(self.q-self.home)), self.policy.tolerance)
                self.assertEqual(target, .07)
                self.width += min(.002, target-self.width)
            if self.policy.reset_state == 'complete':
                break
        else:
            self.fail(str(self.policy.diagnostics))
        self.assertTrue(opened)
        self.assertGreater(math.degrees(maximum), 3.)
        self.assertFalse(self.policy.active)
        self.assertFalse(self.policy.reset_busy)
        self.assertAlmostEqual(self.width, .07, places=3)
        for _ in range(10):
            np.testing.assert_array_equal(self.step(), 0)
            self.assertFalse(self.policy.active)

    def test_zero_candidate_required_and_duplicate_request_does_not_restart(self):
        self.policy.request_reset()
        stamp = self.policy.home_requested_at
        self.assertFalse(self.policy.request_reset())
        for _ in range(20):
            np.testing.assert_array_equal(self.step(quiet=False), 0)
            self.assertFalse(self.policy.active)
        self.assertEqual(stamp, self.policy.home_requested_at)
        self.assertGreater(np.linalg.norm(self.step()), 0)
        self.assertTrue(self.policy.active)

    def test_cancel_stops_and_does_not_resume_automatically(self):
        self.policy.request_reset()
        for _ in range(10):
            self.step()
        self.assertTrue(self.policy.active)
        self.policy.cancel_reset()
        for _ in range(30):
            np.testing.assert_array_equal(self.step(), 0)
            self.assertIsNone(self.policy.reset_gripper_target)
        self.assertEqual(self.policy.reset_state, 'cancelled')

    def test_manual_input_or_disconnect_cancels_auto_reset(self):
        for sample in (pad(lb=True), pad(a=True), pad(axes={'left_y': .3}),
                       pad(axes={'left_trigger': .3}), pad(connected=False)):
            with self.subTest(sample=sample):
                self.setUp()
                self.policy.request_reset()
                self.step()
                np.testing.assert_array_equal(self.step(sample), 0)
                self.assertFalse(self.policy.reset_busy)
                self.assertFalse(self.policy.active)
                self.assertIsNone(self.policy.reset_gripper_target)

    def test_opening_requires_arm_hold_and_times_out_with_no_feedback_progress(self):
        self.q = self.home.copy()
        self.policy.request_reset()
        self.step()
        self.assertEqual(self.policy.reset_state, 'opening')
        self.assertFalse(self.policy.active)
        for _ in range(10):
            self.step(quiet=False)
            self.assertIsNone(self.policy.reset_gripper_target)
        for _ in range(300):
            self.step()
        self.assertEqual(self.policy.reset_state, 'failed')
        self.assertFalse(self.policy.active)
        self.assertIsNone(self.policy.reset_gripper_target)


if __name__ == '__main__':
    unittest.main()
