"""Joint isolation, rearming, speed and stop contracts; no ROS/CAN required."""
from copy import deepcopy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from servo_bridge.core import load_config
from servo_bridge.joint_policy import JointPolicy


PROFILE = Path(__file__).resolve().parents[1] / 'config.joint.example.json'


def pad(**buttons):
    return dict(connected=True, instance=1, axes={}, hat=[0, 0], buttons=buttons)


class JointPolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROFILE)
        self.dt = self.cfg['publish_period_s']

    def armed(self, mapping=None):
        p = JointPolicy(self.cfg, mapping)
        p.step(pad(), np.eye(3), self.dt)
        return p

    def test_each_input_moves_only_its_joint_and_reverse(self):
        inputs = [('left_x', 1), ('left_y', -1), ('right_y', -1),
                  ('hat_x', 1), ('hat_y', 1), ('right_x', 1)]
        for index, (key, sign) in enumerate(inputs):
            for direction in (-1, 1):
                with self.subTest(joint=index + 1, direction=direction):
                    policy = self.armed()
                    sample = pad(lb=True)
                    if key.startswith('hat'):
                        sample['hat'][0 if key == 'hat_x' else 1] = sign * direction
                    else:
                        sample['axes'][key] = sign * direction
                    for _ in range(60):
                        result = policy.step(sample, np.eye(3), self.dt)
                    expected = np.zeros(6)
                    expected[index] = direction * math.radians(self.cfg['joint_speed_deg_s'])
                    np.testing.assert_allclose(result, expected)
                    self.assertIsNone(policy.orientation)

    def test_no_movement_until_neutral_with_lb_released(self):
        policy = JointPolicy(self.cfg)
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt), 0)
        policy.step(pad(a=True), np.eye(3), self.dt)
        self.assertFalse(policy.neutral_seen)
        sample2 = pad()
        sample2['axes']['right_trigger'] = 1
        policy.step(sample2, np.eye(3), self.dt)
        self.assertFalse(policy.neutral_seen)
        policy.step(pad(), np.eye(3), self.dt)
        self.assertGreater(policy.step(sample, np.eye(3), self.dt)[0], 0)

    def test_release_immediately_zeroes_and_requires_neutral(self):
        policy = self.armed()
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        policy.step(sample, np.eye(3), self.dt)
        sample['buttons']['lb'] = False
        np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt), 0)
        sample['buttons']['lb'] = True
        np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt), 0)
        self.assertFalse(policy.active)

    def test_fine_speed_and_acceleration_bound(self):
        policy = self.armed()
        sample = pad(lb=True, rb=True)
        sample['axes']['right_x'] = 1
        previous = np.zeros(6)
        for _ in range(60):
            velocity = policy.step(sample, np.eye(3), self.dt)
            self.assertLessEqual(float(np.max(abs(velocity - previous))),
                                 math.radians(self.cfg['joint_acceleration_deg_s2']) * self.dt + 1e-10)
            previous = velocity
        self.assertAlmostEqual(velocity[5], math.radians(self.cfg['joint_fine_speed_deg_s']))

    def test_rb_requires_lb_and_switches_speed_while_moving(self):
        policy = self.armed()
        sample = pad(rb=True)
        sample['axes']['left_x'] = 1
        for _ in range(30):
            np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt), 0)
        self.assertFalse(policy.active)
        policy.step(pad(), np.eye(3), self.dt)
        sample['buttons']['lb'] = True
        for _ in range(60):
            result = policy.step(sample, np.eye(3), self.dt)
        self.assertAlmostEqual(result[0], math.radians(8))
        sample['buttons']['rb'] = False
        for _ in range(60):
            result = policy.step(sample, np.eye(3), self.dt)
        self.assertAlmostEqual(result[0], math.radians(24))
        sample['buttons']['rb'] = True
        for _ in range(60):
            result = policy.step(sample, np.eye(3), self.dt)
        self.assertAlmostEqual(result[0], math.radians(8))

    def test_invalid_fine_speed_is_rejected(self):
        for value in (True, float('nan'), 0, -1, self.cfg['joint_speed_deg_s'] + 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                JointPolicy({**self.cfg, 'joint_fine_speed_deg_s': value})

    def test_start_disconnect_and_device_change_latch_stop(self):
        for failure in ('start', 'disconnect', 'device'):
            with self.subTest(failure=failure):
                policy = self.armed()
                sample = pad(lb=True)
                sample['axes']['left_x'] = 1
                policy.step(sample, np.eye(3), self.dt)
                bad = deepcopy(sample)
                if failure == 'start':
                    bad['buttons']['start'] = True
                elif failure == 'disconnect':
                    bad['connected'] = False
                else:
                    bad['instance'] = 2
                np.testing.assert_array_equal(policy.step(bad, np.eye(3), self.dt), 0)
                policy.step(pad(), np.eye(3), self.dt)
                np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt), 0)
                self.assertTrue(policy.faulted)

    def test_input_direction_and_invalid_profile(self):
        policy = self.armed({'input_directions': {'left_x': -1}})
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        self.assertLess(policy.step(sample, np.eye(3), self.dt)[0], 0)
        for value in (True, float('nan'), -1, 1000):
            with self.subTest(value=value):
                cfg = {**self.cfg, 'joint_speed_deg_s': value}
                with self.assertRaises(ValueError):
                    JointPolicy(cfg)

    def test_invalid_axes_and_period_are_rejected(self):
        policy = self.armed()
        for value in (True, float('nan'), 1.1):
            sample = pad(lb=True)
            sample['axes']['left_trigger'] = value
            with self.assertRaises(ValueError):
                policy.step(sample, np.eye(3), self.dt)
        with self.assertRaises(ValueError):
            policy.step(pad(), np.eye(3), float('nan'))

    def test_live_joint_config_reverses_all_six_inputs_without_changing_gripper(self):
        root = PROFILE.parents[1]
        original = json.loads((root / 'gamepad.servo.example.json').read_text())['gamepad']
        joint = json.loads((root / 'gamepad.joint.example.json').read_text())['gamepad']
        self.assertEqual({k: v for k, v in joint.items() if k != 'input_directions'},
                         {k: v for k, v in original.items() if k != 'input_directions'})
        sample = pad(lb=True)
        sample['axes'] = dict(left_x=.2, left_y=-.3, right_y=-.4, right_x=.5,
                              left_trigger=.25, right_trigger=.75)
        sample['hat'] = [1, -1]
        before, after = self.armed(original), self.armed(joint)
        for _ in range(60):
            old = before.step(sample, np.eye(3), self.dt)
            new = after.step(sample, np.eye(3), self.dt)
        np.testing.assert_allclose(new, -old)
        self.assertEqual(joint['input_directions'],
                         {key: -value for key, value in original['input_directions'].items()})


if __name__ == '__main__':
    unittest.main()
