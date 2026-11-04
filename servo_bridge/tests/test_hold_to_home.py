"""Hold-to-home activation, release and bounded return without ROS or CAN."""
import math
from pathlib import Path
import unittest

import numpy as np

from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import load_config
from servo_bridge.hold_to_home import HoldToHomePolicy


PROFILE = Path(__file__).resolve().parents[1]/'config.cartesian.example.json'


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def pad(*, axes=None, hat=None, connected=True, instance=1, **buttons):
    return dict(connected=connected, instance=instance, axes=axes or {},
                buttons=buttons, hat=[0, 0] if hat is None else hat)


class HoldToHomeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(PROFILE)
        self.cfg.update(home_hold_s=.6, home_speed_deg_s=3.,
                        home_acceleration_deg_s2=15., home_tolerance_deg=.3)
        self.dt = self.cfg['publish_period_s']
        self.home = np.radians(self.cfg['sim_initial_joints_deg'])
        self.q = self.home + np.radians([8., -8., 10., -5., -8., 6.])
        self.kin = PiperKinematics(self.cfg['joint_limits_deg'])
        self.clock = Clock()
        self.policy = HoldToHomePolicy(self.cfg, home_q=self.home, clock=self.clock)

    def step(self, sample, q=None, *, may_start=True, reference_q=None):
        q = self.q if q is None else q
        self.clock.advance(self.dt)
        return self.policy.step(sample, self.kin.forward(q)[:3, :3], self.dt,
                                measured_q=q, reference_q=reference_q,
                                home_may_start=may_start)

    def hold_for(self, seconds, sample=None, q=None, *, may_start=True):
        sample = pad(a=True) if sample is None else sample
        velocity = np.zeros(6)
        for _ in range(math.ceil(seconds/self.dt)):
            velocity = self.step(sample, q, may_start=may_start)
        return velocity

    def activate_home(self, sample=None):
        self.step(pad())
        velocity = self.hold_for(.8, sample)
        self.assertTrue(self.policy.home_active, self.policy.diagnostics)
        self.assertTrue(self.policy.active)
        self.assertFalse(self.policy.gripper_allowed)
        self.assertGreater(np.linalg.norm(velocity), 0)
        return velocity

    def test_startup_held_a_needs_release_before_a_fresh_press(self):
        for _ in range(60):
            np.testing.assert_array_equal(self.step(pad(a=True)), 0)
            self.assertFalse(self.policy.active)
            self.assertFalse(self.policy.home_active)
            self.assertFalse(self.policy.gripper_allowed)
        self.step(pad())
        self.hold_for(.8)
        self.assertTrue(self.policy.home_active)

    def test_short_press_stays_still_and_cannot_close_gripper(self):
        self.step(pad())
        for _ in range(20):
            np.testing.assert_array_equal(self.step(pad(a=True)), 0)
            self.assertFalse(self.policy.active)
            self.assertFalse(self.policy.gripper_allowed)
        np.testing.assert_array_equal(self.step(pad()), 0)
        self.assertFalse(self.policy.home_active)
        for _ in range(40):
            np.testing.assert_array_equal(self.step(pad()), 0)

    def test_a_alone_long_hold_returns_and_lb_rb_combinations_are_supported(self):
        for buttons in ({'a': True}, {'a': True, 'lb': True},
                        {'a': True, 'lb': True, 'rb': True}):
            with self.subTest(buttons=buttons):
                self.clock = Clock()
                self.policy = HoldToHomePolicy(self.cfg, home_q=self.home, clock=self.clock)
                velocity = self.activate_home(pad(**buttons))
                self.assertTrue(np.all(velocity*(self.home-self.q) >= -1e-12))
                self.assertLessEqual(max(abs(velocity)), math.radians(3)+1e-9)

    def test_manual_motion_is_held_immediately_when_a_is_pressed(self):
        self.step(pad())
        moving = pad(lb=True, axes={'left_y': -.5})
        velocity = self.hold_for(.2, moving)
        self.assertTrue(self.policy.active)
        self.assertGreater(np.linalg.norm(velocity), 0)
        np.testing.assert_array_equal(self.step(pad(a=True, lb=True)), 0)
        self.assertFalse(self.policy.active)
        self.assertFalse(self.policy.gripper_allowed)

    def test_return_release_stops_and_held_lb_cannot_resume_manual_motion(self):
        self.activate_home()
        np.testing.assert_array_equal(self.step(pad(lb=True, axes={'left_y': -.5})), 0)
        self.assertFalse(self.policy.active)
        self.assertFalse(self.policy.home_active)
        for _ in range(50):
            np.testing.assert_array_equal(self.step(pad(lb=True, axes={'left_y': -.5})), 0)
            self.assertFalse(self.policy.active)
        self.step(pad())
        velocity = self.hold_for(.15, pad(lb=True, axes={'left_y': -.5}))
        self.assertTrue(self.policy.active)
        self.assertFalse(self.policy.home_active)
        self.assertGreater(np.linalg.norm(velocity), 0)
        self.assertTrue(self.policy.gripper_allowed)

    def test_return_can_restart_only_after_a_release_and_new_long_hold(self):
        self.activate_home()
        np.testing.assert_array_equal(self.step(pad()), 0)
        for _ in range(20):
            np.testing.assert_array_equal(self.step(pad(a=True)), 0)
            self.assertFalse(self.policy.active)
        self.hold_for(.4)
        self.assertTrue(self.policy.home_active)

    def test_conflicting_inputs_block_until_a_is_released(self):
        conflicts = [pad(a=True, axes={'left_x': .2}),
                     pad(a=True, hat=[1, 0]),
                     pad(a=True, axes={'left_trigger': .2}),
                     pad(a=True, axes={'right_trigger': .2}),
                     pad(a=True, b=True)]
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                self.clock = Clock()
                self.policy = HoldToHomePolicy(self.cfg, home_q=self.home, clock=self.clock)
                self.step(pad())
                self.step(pad(a=True))
                np.testing.assert_array_equal(self.step(conflict), 0)
                self.assertFalse(self.policy.gripper_allowed)
                for _ in range(60):
                    np.testing.assert_array_equal(self.step(pad(a=True)), 0)
                    self.assertFalse(self.policy.active)
                self.step(pad())
                self.hold_for(.8)
                self.assertTrue(self.policy.home_active)

    def test_conflicting_input_during_return_stops_without_gripper_side_effect(self):
        self.activate_home()
        np.testing.assert_array_equal(self.step(pad(a=True, axes={'left_trigger': .5})), 0)
        self.assertFalse(self.policy.active)
        self.assertFalse(self.policy.home_active)
        self.assertFalse(self.policy.gripper_allowed)
        for _ in range(60):
            np.testing.assert_array_equal(self.step(pad(a=True)), 0)

    def test_zero_candidate_handshake_prevents_home_arming_early(self):
        self.step(pad())
        self.hold_for(1., may_start=False)
        self.assertFalse(self.policy.active)
        self.assertFalse(self.policy.home_active)
        self.assertFalse(self.policy.gripper_allowed)
        velocity = self.hold_for(.1, may_start=True)
        self.assertTrue(self.policy.home_active)
        self.assertGreater(np.linalg.norm(velocity), 0)

    def test_active_return_does_not_require_another_zero_candidate(self):
        self.activate_home()
        for _ in range(50):
            velocity = self.step(pad(a=True), may_start=False)
            self.assertTrue(self.policy.active)
            self.assertTrue(self.policy.home_active)
            self.assertGreater(np.linalg.norm(velocity), 0)
            self.assertFalse(self.policy.gripper_allowed)

    def test_reset_motion_keeps_long_press_time(self):
        self.step(pad())
        self.step(pad(a=True))
        requested_at = self.policy.home_requested_at
        self.assertIsNotNone(requested_at)
        for _ in range(40):
            self.policy.reset_motion()
            self.step(pad(a=True), may_start=False)
            self.assertEqual(self.policy.home_requested_at, requested_at)
        self.hold_for(.1)
        self.assertTrue(self.policy.home_active)

    def test_disconnect_start_and_device_change_latch_return_closed(self):
        stops = [pad(a=True, connected=False), pad(a=True, start=True),
                 pad(a=True, instance=2)]
        for stop in stops:
            with self.subTest(stop=stop):
                self.clock = Clock()
                self.policy = HoldToHomePolicy(self.cfg, home_q=self.home, clock=self.clock)
                self.activate_home()
                np.testing.assert_array_equal(self.step(stop), 0)
                self.assertFalse(self.policy.active)
                self.assertTrue(self.policy.faulted)
                self.step(pad())
                for _ in range(60):
                    np.testing.assert_array_equal(self.step(pad(a=True)), 0)
                    self.assertFalse(self.policy.active)

    def test_return_progress_speed_acceleration_and_completion_hold(self):
        q = self.q.copy()
        initial_error = max(abs(self.home-q))
        self.step(pad(), q)
        previous = np.zeros(6)
        was_active = False
        moved = False
        for _ in range(600):
            velocity = self.step(pad(a=True), q)
            self.assertTrue(np.isfinite(velocity).all())
            self.assertLessEqual(max(abs(velocity)), math.radians(3)+1e-9)
            if self.policy.home_active:
                self.assertFalse(self.policy.gripper_allowed)
                if was_active:
                    self.assertLessEqual(max(abs(velocity-previous)),
                                         math.radians(15)*self.dt+1e-8)
                moved = moved or bool(np.any(velocity))
            q += velocity*self.dt
            self.assertTrue(np.all(q >= self.kin.low))
            self.assertTrue(np.all(q <= self.kin.high))
            previous = velocity
            was_active = self.policy.home_active
        self.assertTrue(moved)
        self.assertLess(max(abs(self.home-q)), initial_error*.1)
        self.assertLessEqual(max(abs(self.home-q)), math.radians(.3)+1e-8)
        for _ in range(30):
            np.testing.assert_array_equal(self.step(pad(a=True), q), 0)
            self.assertFalse(self.policy.home_active)
            self.assertFalse(self.policy.gripper_allowed)
        # Sensor noise and a small residual do not restart a held completion.
        displaced = q+np.radians([1., 0., 0., 0., 0., 0.])
        for _ in range(30):
            np.testing.assert_array_equal(self.step(pad(a=True), displaced), 0)
            self.assertFalse(self.policy.home_active)

    def test_home_reference_requires_finite_six_joints_inside_soft_limits(self):
        near_limit = self.home.copy()
        near_limit[4] = self.kin.high[4]-.05
        invalid = [None, np.zeros(5), np.full(6, np.nan), near_limit]
        for home in invalid:
            with self.subTest(home=home), self.assertRaises(ValueError):
                HoldToHomePolicy(self.cfg, home_q=home, clock=self.clock)

    def test_invalid_hold_speed_acceleration_and_tolerance_are_rejected(self):
        for key, value in [('home_hold_s', 0), ('home_hold_s', float('nan')),
                           ('home_speed_deg_s', -1), ('home_speed_deg_s', 100),
                           ('home_acceleration_deg_s2', 0), ('home_tolerance_deg', 0)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                HoldToHomePolicy({**self.cfg, key: value}, home_q=self.home, clock=self.clock)


if __name__ == '__main__':
    unittest.main()
