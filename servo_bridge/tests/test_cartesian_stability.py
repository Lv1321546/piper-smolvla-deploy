"""Regressions for neutral twitch, Cartesian lift and startup sensor bounds."""
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from servo_bridge.cartesian_policy import CartesianPolicy
from servo_bridge.core import load_config
from servo_bridge.start_reference import reference_from_snapshot

PROFILE = Path(__file__).resolve().parents[1]/'config.cartesian.example.json'


class CartesianStabilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROFILE)
        self.dt = self.cfg['publish_period_s']
        self.q = np.radians(self.cfg['sim_initial_joints_deg'])
        self.pad = dict(connected=True, instance=1, axes={}, buttons={}, hat=[0, 0])

    def test_neutral_brakes_without_reallocating_or_recentering_joints(self):
        policy = CartesianPolicy(self.cfg)
        rotation = policy.kin.forward(self.q)[:3, :3]
        policy.step(self.pad, rotation, self.dt, measured_q=self.q)
        policy.previous = np.array([.08, -.12, .15, .03, -.04, .01])
        policy.orientation = Rotation.from_rotvec([.1, -.1, 0]).as_matrix()@rotation
        previous = policy.previous.copy()
        sample = {**self.pad, 'buttons': {'lb': True}}
        for _ in range(30):
            velocity = policy.step(sample, rotation, self.dt, measured_q=self.q)
            self.assertTrue(np.all(abs(velocity) <= abs(previous)+1e-12))
            self.assertTrue(np.all(velocity*previous >= 0))
            self.assertLessEqual(max(abs(velocity-previous)), policy.gate.acceleration*self.dt+1e-10)
            previous = velocity
        np.testing.assert_array_equal(velocity, np.zeros(6))
        for _ in range(10):
            np.testing.assert_array_equal(policy.step(sample, rotation, self.dt, measured_q=self.q), 0)

    def test_neutral_braking_respects_filter_coasting_near_joint_boundary(self):
        policy = CartesianPolicy(self.cfg)
        q = self.q.copy()
        q[4] = policy.kin.high[4]-.11
        rotation = policy.kin.forward(q)[:3, :3]
        policy.step(self.pad, rotation, self.dt, measured_q=q)
        policy.previous[4] = .2
        output = policy.step({**self.pad, 'buttons': {'lb': True}}, rotation, self.dt, measured_q=q)
        self.assertTrue(policy.pause)
        np.testing.assert_array_equal(output, 0)

    def test_flexible_heading_lifts_within_explicit_envelope_and_joint_bounds(self):
        cfg = {**self.cfg, 'cartesian_orientation_drift_rad': math.radians(30),
               'cartesian_orientation_weight': .08, 'execution_velocity_time_constant_s': .12}
        policy = CartesianPolicy(cfg)
        q = self.q.copy()
        initial = policy.kin.forward(q)
        policy.step(self.pad, initial[:3, :3], self.dt, measured_q=q)
        sample = {**self.pad, 'buttons': {'lb': True}, 'axes': {'right_y': -.6}}
        previous = np.zeros(6)
        for _ in range(400):
            velocity = policy.step(sample, policy.kin.forward(q)[:3, :3], self.dt, measured_q=q)
            self.assertFalse(policy.pause)
            self.assertLessEqual(max(abs(velocity-previous)), policy.gate.acceleration*self.dt+1e-8)
            q += velocity*self.dt
            self.assertTrue(np.all(q >= policy.kin.low))
            self.assertTrue(np.all(q <= policy.kin.high))
            pose = policy.kin.forward(q)
            drift = np.linalg.norm(Rotation.from_matrix(initial[:3, :3]@pose[:3, :3].T).as_rotvec())
            self.assertLessEqual(drift, math.radians(30)+.005)
            previous = velocity
        self.assertGreater(pose[2, 3]-initial[2, 3], .16)

    def test_start_reference_accepts_ended_teaching_and_preserves_sensor_values(self):
        snapshot = self.snapshot()
        reference = reference_from_snapshot(snapshot, self.cfg, self.collection(), now=100.)
        self.assertEqual(reference['joint_angles'], snapshot['state'][:6])
        self.assertEqual(reference['gripper_width_m'], .04)
        np.testing.assert_allclose(reference['base_to_link6'], CartesianPolicy(self.cfg).kin.forward(self.q))
        self.assertFalse(reference['tcp_calibrated'])

    def test_start_reference_rejects_real_out_of_model_feedback_and_staleness(self):
        snapshot = self.snapshot()
        snapshot['state'][4] = 69.932
        snapshot['state'][6] = .102
        with self.assertRaisesRegex(ValueError, 'J5=.*夹爪反馈'):
            reference_from_snapshot(snapshot, self.cfg, self.collection(), now=100.)
        snapshot = self.snapshot()
        with self.assertRaisesRegex(ValueError, '新鲜'):
            reference_from_snapshot(snapshot, self.cfg, self.collection(), now=101.)

    def snapshot(self):
        return dict(state=self.cfg['sim_initial_joints_deg']+[.04], feedback_monotonic=99.99,
                    arm_status=0, err_code=0, motor_faults=[0]*6, gripper_faults=0,
                    ctrl_mode=2, teach_status=2, enabled=[True]*6, mode_feed=1)

    def collection(self):
        return {'robot': {'gripper_limits_m': [0, .07]}}


if __name__ == '__main__':
    unittest.main()
