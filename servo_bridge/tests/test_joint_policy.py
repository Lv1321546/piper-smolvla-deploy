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

    def test_every_joint_blocks_outward_but_allows_inward_without_releasing_lb(self):
        inputs = [('left_x', 1), ('left_y', -1), ('right_y', -1),
                  ('hat_x', 1), ('hat_y', 1), ('right_x', 1)]
        for index, (key, sign) in enumerate(inputs):
            for side, outward in ((0, -1), (1, 1)):
                policy = self.armed()
                q = np.radians(self.cfg['sim_initial_joints_deg'])
                q[index] = policy.limits[index, side] - outward*.03
                sample = pad(lb=True)
                def command(direction):
                    if key.startswith('hat'):
                        sample['hat'][0 if key == 'hat_x' else 1] = sign*direction
                    else:
                        sample['axes'][key] = sign*direction
                with self.subTest(joint=index+1, side=side):
                    command(outward)
                    for _ in range(20):
                        v = policy.step(sample, np.eye(3), self.dt, measured_q=q)
                        np.testing.assert_array_equal(v, 0)
                    command(-outward)
                    v = policy.step(sample, np.eye(3), self.dt, measured_q=q)
                    self.assertLess(v[index]*outward, 0)
                    self.assertTrue(policy.active)

    def test_long_motion_brakes_before_margin_then_retreats(self):
        policy = self.armed()
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        q[0] = policy.limits[0, 1]-.3
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        previous = np.zeros(6)
        for _ in range(400):
            v = policy.step(sample, np.eye(3), self.dt, measured_q=q)
            self.assertFalse(policy.pause)
            self.assertLessEqual(max(abs(v-previous)), policy.acceleration*self.dt+1e-8)
            q += v*self.dt
            self.assertLessEqual(q[0], policy.limits[0, 1]-policy.margin+1e-8)
            previous = v
        self.assertLess(abs(v[0]), 1e-6)
        sample['axes']['left_x'] = -1
        self.assertLess(policy.step(sample, np.eye(3), self.dt, measured_q=q)[0], 0)

    def test_integrated_reference_near_limit_and_invalid_feedback(self):
        policy = self.armed()
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        reference = q.copy()
        reference[0] = policy.limits[0, 1]-.03
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        np.testing.assert_array_equal(policy.step(sample, np.eye(3), self.dt,
            measured_q=q, reference_q=reference), 0)
        q[0] = policy.limits[0, 1]+.001
        with self.assertRaises(ValueError):
            policy.step(sample, np.eye(3), self.dt, measured_q=q)

    def test_filtered_execution_with_feedback_lag_stays_inside_servo_margin(self):
        from servo_bridge.core import Feedback, NAMES
        from servo_bridge.execution import VelocityTarget
        policy = self.armed()
        execution = VelocityTarget(self.cfg)
        q = np.radians(self.cfg['sim_initial_joints_deg'])
        q[0] = policy.limits[0, 1]-.6
        sample = pad(lb=True)
        sample['axes']['left_x'] = 1
        near = None
        holds = 0
        for index in range(700):
            if index == 500:
                near = q[0]
                sample['axes']['left_x'] = -1
            velocity = policy.step(sample, np.eye(3), self.dt,
                measured_q=q, reference_q=execution.position)
            if policy.pause:
                holds += 1
                execution.reset()  # Runtime sends measured hold and clears the adapter.
            now = 100.+index*self.dt
            _, target, _ = execution.step(NAMES, velocity,
                feedback=Feedback(q.copy(), now, ready=True), now=now,
                stamp_age_s=0., horizon_s=self.dt)
            q += (target-q)*(1-np.exp(-self.dt/.08))
            self.assertLess(target[0], policy.limits[0, 1]-.05)
            self.assertLess(q[0], policy.limits[0, 1]-.05)
        self.assertLess(q[0], near-.2)
        self.assertLessEqual(holds, 3)

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
