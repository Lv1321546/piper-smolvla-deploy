"""Live XYZ profile: LB deadman, faster fine mode, and bounded rotation."""
from pathlib import Path
import unittest

import numpy as np

from servo_bridge.core import load_config
from servo_bridge.twist_policy import TwistPolicy


class CartesianFineTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / 'config.servo.example.json')
        self.dt = self.cfg['publish_period_s']

    def test_rb_alone_cannot_move_and_lb_rb_moves_at_half_speed(self):
        policy = TwistPolicy(self.cfg)
        pad = dict(connected=True, instance=1, axes={}, buttons={}, hat=[0, 0])
        policy.step(pad, np.eye(3), self.dt)
        pad['buttons'] = {'rb': True}
        pad['axes'] = {'left_y': -1.}
        for _ in range(20):
            np.testing.assert_array_equal(policy.step(pad, np.eye(3), self.dt), 0)
        self.assertFalse(policy.active)
        pad['axes'] = {}
        policy.step(pad, np.eye(3), self.dt)
        pad['buttons']['lb'] = True
        pad['axes'] = {'left_y': -1.}
        for _ in range(100):
            result = policy.step(pad, np.eye(3), self.dt)
        self.assertAlmostEqual(result[0], .03)
        self.assertTrue(policy.active)
        pad['buttons']['rb'] = False
        for _ in range(100):
            result = policy.step(pad, np.eye(3), self.dt)
        self.assertAlmostEqual(result[0], .06)
        pad['buttons']['lb'] = False
        np.testing.assert_array_equal(policy.step(pad, np.eye(3), self.dt), 0)

    def test_fine_rotation_and_diagonal_translation_respect_fine_caps(self):
        policy = TwistPolicy(self.cfg)
        pad = dict(connected=True, instance=1, axes={}, buttons={}, hat=[0, 0])
        policy.step(pad, np.eye(3), self.dt)
        pad['buttons'] = dict(lb=True, rb=True)
        pad['axes'] = dict(left_x=1., left_y=-1., right_y=-1., right_x=1.)
        pad['hat'] = [1, 1]
        for _ in range(100):
            result = policy.step(pad, np.eye(3), self.dt, reference_may_advance=False)
        self.assertAlmostEqual(np.linalg.norm(result[:3]), .03)
        self.assertAlmostEqual(np.linalg.norm(result[3:]), .15)

    def test_rejects_invalid_fine_scale_and_keeps_legacy_default(self):
        cfg = {**self.cfg}
        del cfg['fine_speed_scale']
        self.assertEqual(TwistPolicy(cfg).fine_scale, .25)
        for value in (True, 0, -1, 1.1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TwistPolicy({**cfg, 'fine_speed_scale': value})
