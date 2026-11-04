"""Startup movement contracts with fake SDK and passive feedback; no hardware."""
import unittest
from pathlib import Path

import numpy as np

from servo_bridge.core import load_config
from servo_bridge.joint_start_pose import DEFAULT_POSE, move_to_pose, validate_pose


class FakeArm:
    def __init__(self):
        self.calls = []

    def ConnectPort(self, **kwargs):
        self.calls.append(('connect', kwargs))

    def MotionCtrl_2(self, *args):
        self.calls.append(('mode', args))

    def JointCtrl(self, *args):
        self.calls.append(('joint', args))

    def DisconnectPort(self):
        self.calls.append(('disconnect',))


class StartupPoseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / 'config.joint.example.json')
        self.target, _, _, _ = validate_pose(DEFAULT_POSE, self.cfg)
        self.current = self.target + np.array([10, 0, 0, 0, 0, 0])
        self.arm = FakeArm()
        self.now = 0.

    def sleep(self, dt):
        self.now += dt

    def factory(self, *args, **kwargs):
        return self.arm

    def snap(self, q):
        return {'state': list(q) + [.04]}

    def test_default_pose_has_margin_and_low_condition(self):
        _, margins, condition, _ = validate_pose(DEFAULT_POSE, self.cfg)
        self.assertGreater(min(margins), 12)
        self.assertLess(condition, 15)

    def test_rejects_invalid_targets(self):
        for text in ('0 0', 'nan 90 -35 -17 51 -25', '150 90 -35 -17 51 -25',
                     '0 0 0 0 0 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_pose(text, self.cfg)

    def test_already_at_pose_restores_execution_speed_after_measured_hold(self):
        self.assertFalse(move_to_pose(self.target, self.cfg,
            lambda: self.snap(self.target), self.factory, clock=lambda:self.now, sleep=self.sleep))
        self.assertEqual([c[1][2] for c in self.arm.calls if c[0]=='mode'], [70])
        joints = [c[1] for c in self.arm.calls if c[0]=='joint']
        self.assertEqual(joints, [tuple(round(v*1000) for v in self.target)])
        joint_index = next(i for i,c in enumerate(self.arm.calls) if c[0]=='joint')
        mode_index = next(i for i,c in enumerate(self.arm.calls) if c[0]=='mode')
        self.assertLess(joint_index, mode_index)
        self.assertEqual(self.arm.calls[-1], ('disconnect',))

    def test_changed_pose_cannot_switch_to_normal_speed(self):
        samples = iter([self.target, self.current, self.current])
        with self.assertRaisesRegex(RuntimeError, '发生变化'):
            move_to_pose(self.target, self.cfg, lambda:self.snap(next(samples)),
                         self.factory, clock=lambda:self.now, sleep=self.sleep)
        self.assertFalse(any(c[0]=='mode' for c in self.arm.calls))

    def test_low_speed_move_then_restore_only_after_three_good_samples(self):
        samples = iter([self.current, self.current, self.target,
                        self.target, self.target])
        result = move_to_pose(self.target, self.cfg, lambda: self.snap(next(samples)),
                              self.factory, clock=lambda: self.now, sleep=self.sleep)
        self.assertTrue(result)
        self.assertEqual([c[1][2] for c in self.arm.calls if c[0] == 'mode'], [5, 70])
        self.assertEqual([c[1] for c in self.arm.calls if c[0] == 'joint'],
                         [tuple(round(v * 1000) for v in self.target)])
        self.assertEqual(self.arm.calls[-1], ('disconnect',))

    def test_timeout_holds_current_and_does_not_start_fast_mode(self):
        with self.assertRaisesRegex(RuntimeError, '超时'):
            move_to_pose(self.target, self.cfg, lambda: self.snap(self.current),
                         self.factory, clock=lambda: self.now, sleep=self.sleep, timeout=.3)
        joints = [c[1] for c in self.arm.calls if c[0] == 'joint']
        self.assertEqual(joints[-1], tuple(round(v * 1000) for v in self.current))
        self.assertEqual([c[1][2] for c in self.arm.calls if c[0] == 'mode'], [5])
        self.assertEqual(self.arm.calls[-1], ('disconnect',))

    def test_keyboard_interrupt_requests_measured_hold(self):
        def interrupted(dt):
            self.now += dt
            if self.now > .2:
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            move_to_pose(self.target, self.cfg, lambda: self.snap(self.current),
                         self.factory, clock=lambda: self.now, sleep=interrupted)
        self.assertEqual([c[1] for c in self.arm.calls if c[0] == 'joint'][-1],
                         tuple(round(v * 1000) for v in self.current))

    def test_out_of_range_feedback_never_opens_sdk(self):
        bad = self.target.copy()
        bad[0] = 160
        with self.assertRaisesRegex(RuntimeError, '反馈'):
            move_to_pose(self.target, self.cfg, lambda: self.snap(bad), self.factory)
        self.assertEqual(self.arm.calls, [])


if __name__ == '__main__':
    unittest.main()
