"""Protocol/safety regressions using synthetic feedback and a fake SDK only."""
import json
import math
from pathlib import Path
import socket
import struct
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from servo_bridge.core import CommandGuard, Feedback, NAMES, load_config, ordered_values, sdk_units
from servo_bridge.twist_policy import TwistPolicy
from servo_bridge.sdk_worker import SDKJointWriter, Worker, validate_arm, ACK
from servo_bridge.sources import PassiveCan


class Clock:
    def __init__(self):
        self.t = 100.
    def __call__(self):
        return self.t


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.clock = Clock()
        self.guard = CommandGuard(self.cfg, clock=self.clock)
        self.q = np.radians([0., 70., -80., 0., 30., 0.])
        self.feedback = Feedback(self.q.copy(), 100., .04, True, True, 'fake')

    def call(self, **kwargs):
        args = dict(names=list(NAMES), positions=self.q, velocities=np.zeros(6), feedback=self.feedback,
                    stamp_age_s=.01, horizon_s=.05)
        args.update(kwargs)
        return self.guard.validate(**args)

    def test_joint_order_and_units(self):
        names = list(reversed(NAMES))
        raw = self.call(names=names, positions=list(reversed(self.q)), velocities=[0.] * 6)
        self.assertEqual(raw, [0, 70000, -80000, 0, 30000, 0])
        self.assertEqual(sdk_units([math.pi / 2] * 6), [90000] * 6)

    def test_duplicate_unknown_and_malformed_joints(self):
        for names in (['joint1'] * 6, list(NAMES[:-1]), list(NAMES[:-1]) + ['joint7']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.call(names=names)
        for value in ([True] * 6, ['0'] * 6, [float('nan')] * 6, [float('inf')] * 6):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.call(positions=value)

    def test_stale_future_feedback_and_command(self):
        for age in (-.001, .151, float('nan')):
            with self.subTest(age=age), self.assertRaises(ValueError):
                self.call(stamp_age_s=age)
        for stamp in (99., 101.):
            self.feedback.received = stamp
            with self.assertRaises(ValueError):
                self.call()

    def test_position_velocity_acceleration_and_lead(self):
        outside = self.q.copy(); outside[1] = -.001
        with self.assertRaisesRegex(ValueError, 'position'):
            self.call(positions=outside)
        too_far = self.q.copy(); too_far[0] += math.radians(1.01)
        with self.assertRaisesRegex(ValueError, 'tracking'):
            self.call(positions=too_far)
        with self.assertRaisesRegex(ValueError, 'velocity'):
            self.call(velocities=[.151] * 6)
        with self.assertRaisesRegex(ValueError, 'acceleration'):
            self.call(velocities=[.026] * 6)
        self.assertIsNone(self.guard.last_q)
        self.call()
        self.clock.t += .05; self.feedback.received = self.clock.t
        jumped = self.q.copy(); jumped[0] += .01
        with self.assertRaisesRegex(ValueError, 'derived'):
            self.call(positions=jumped)
        np.testing.assert_allclose(self.guard.last_q, self.q)

    def test_quantization_cannot_cross_limit(self):
        self.guard.limits[0, 1] = math.radians(.0006)
        q = self.q.copy(); q[0] = math.radians(.00059)
        with self.assertRaisesRegex(ValueError, 'position'):
            self.call(positions=q)

    def test_reject_without_advancing_previous_reference(self):
        self.call()
        self.clock.t += .02
        with self.assertRaisesRegex(ValueError, 'too fast'):
            self.call()
        self.assertEqual(self.guard.last_received, 100.)

    def test_quantized_velocity_cannot_exceed_bound(self):
        self.call()
        self.clock.t += .05; self.feedback.received = self.clock.t
        q = self.q.copy(); q[0] = .15 * .05
        with self.assertRaisesRegex(ValueError, 'quantization'):
            self.call(positions=q)


class TwistTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.policy = TwistPolicy(self.cfg)
        self.pad = dict(connected=True, instance=1, axes={}, buttons={'lb': False}, hat=[0, 0])

    def tick(self, **kwargs):
        return self.policy.step(self.pad, np.eye(3), .05, **kwargs)

    def test_neutral_deadman_and_release(self):
        self.pad['buttons']['lb'] = True; self.pad['axes']['left_y'] = -1.
        np.testing.assert_array_equal(self.tick(), np.zeros(6))
        self.pad['buttons']['lb'] = False; self.pad['axes'] = {}; self.tick()
        self.pad['buttons']['lb'] = True; self.pad['axes']['left_y'] = -1.
        self.assertGreater(self.tick()[0], 0)
        self.pad['buttons']['lb'] = False
        np.testing.assert_array_equal(self.tick(), np.zeros(6))
        self.assertFalse(self.policy.active)

    def test_bounded_correction_retains_heading(self):
        self.tick(); self.pad['buttons']['lb'] = True; self.tick()
        measured = Rotation.from_euler('z', .3).as_matrix()
        result = self.policy.step(self.pad, measured, .05)
        self.assertAlmostEqual(result[5], -.02)
        np.testing.assert_allclose(self.policy.orientation, np.eye(3))

    def test_rotation_reference_cannot_wind_up_when_blocked(self):
        self.tick(); self.pad['buttons']['lb'] = True; self.pad['axes']['right_x'] = 1.
        for _ in range(1000):
            result = self.tick()
        self.assertLessEqual(np.linalg.norm(Rotation.from_matrix(self.policy.orientation).as_rotvec()), .05 + 1e-10)
        self.assertLessEqual(np.linalg.norm(result[3:]), self.cfg['angular_speed_rad_s'] + 1e-10)
        ref = self.policy.orientation.copy()
        for _ in range(10):
            self.tick(reference_may_advance=False)
        np.testing.assert_allclose(self.policy.orientation, ref)

    def test_historical_cartesian_mapping(self):
        self.tick(); self.pad['buttons']['lb'] = True
        self.pad['axes'] = dict(left_x=1., left_y=-1., right_x=1., right_y=-1.)
        self.pad['hat'] = [1, 1]
        result = self.tick()
        self.assertTrue(np.all(np.sign(result) == [1, -1, 1, 1, -1, 1]))

    def test_diagonal_speed_is_bounded_and_bad_mapping_rejected(self):
        self.tick(); self.pad['buttons']['lb'] = True
        self.pad['axes'] = dict(left_x=1., left_y=-1., right_y=-1.)
        for _ in range(10):
            result = self.tick()
        self.assertLessEqual(np.linalg.norm(result[:3]), self.cfg['linear_speed_m_s'] + 1e-12)
        with self.assertRaises(ValueError):
            TwistPolicy(self.cfg, {'input_directions': {'left_x': 2}})


class WorkerTests(unittest.TestCase):
    class FakeSDK:
        calls = []
        def __init__(self, channel):
            self.calls.append(('construct', channel))
        def ConnectPort(self, **kwargs):
            self.calls.append(('connect', kwargs))
        def JointCtrl(self, *values):
            self.calls.append(('joint', values))
        def DisconnectPort(self):
            self.calls.append(('disconnect',))

    def setUp(self):
        self.FakeSDK.calls = []
        self.cfg = load_config(); self.clock = Clock()
        self.q = np.radians([0., 70., -80., 0., 30., 0.])
        self.feedback = Feedback(self.q, 100., .04, True, True, 'fake')
        self.writer = SDKJointWriter('vcan-test', self.FakeSDK)
        self.worker = Worker(self.cfg, self.writer, clock=self.clock)
        self.packet = dict(version=1, monotonic=100., joint_names=list(NAMES), positions=self.q.tolist(),
                           velocities=[0.] * 6, deadman=True, servo_status=0, status_monotonic=100., horizon_s=.05)

    def test_sdk_has_no_queries_enable_mode_or_gripper(self):
        self.worker.packet(self.packet, self.feedback)
        self.writer.close()
        self.assertEqual([c[0] for c in self.FakeSDK.calls], ['construct', 'connect', 'joint', 'disconnect'])
        self.assertEqual(self.FakeSDK.calls[1][1], dict(piper_init=False, start_thread=False))

    def test_absent_unready_stale_and_servo_halt_cannot_write(self):
        for key, value in [('servo_status', 2), ('monotonic', 99.), ('monotonic', 101.), ('status_monotonic', 99.)]:
            p = {**self.packet, key: value}
            with self.assertRaises(ValueError):
                self.worker.packet(p, self.feedback)
        self.feedback.ready = False
        with self.assertRaises(ValueError):
            self.worker.packet(self.packet, self.feedback)
        self.assertNotIn('joint', [c[0] for c in self.FakeSDK.calls])

    def test_deadman_release_holds_once_after_motion(self):
        self.worker.packet(self.packet, self.feedback)
        self.clock.t += .05; self.feedback.received = self.clock.t
        p = {**self.packet, 'monotonic': self.clock.t, 'deadman': False}
        self.worker.packet(p, self.feedback)
        self.assertFalse(self.worker.moved)
        self.worker.hold(self.feedback)
        self.assertEqual(sum(c[0] == 'joint' for c in self.FakeSDK.calls), 2)

    def test_stale_feedback_cannot_become_stop_command(self):
        self.worker.packet(self.packet, self.feedback)
        self.clock.t += 1.
        with self.assertRaises(RuntimeError):
            self.worker.hold(self.feedback)
        self.assertEqual(sum(c[0] == 'joint' for c in self.FakeSDK.calls), 1)

    def test_explicit_relay_stop_holds_once_and_exits_without_trajectory(self):
        self.worker.packet(self.packet, self.feedback)
        self.clock.t += .05; self.feedback.received = self.clock.t
        stop = dict(version=1, monotonic=self.clock.t, stop_requested=True,
                    reason='Joint acceleration limit exceeded')
        with self.assertRaisesRegex(RuntimeError, 'Relay requested stop'):
            self.worker.packet(stop, self.feedback)
        self.assertFalse(self.worker.moved)
        self.assertEqual(sum(c[0] == 'joint' for c in self.FakeSDK.calls), 2)

    def test_explicit_stop_without_motion_sends_no_hold_target(self):
        with self.assertRaisesRegex(RuntimeError, 'Relay requested stop'):
            self.worker.packet(dict(version=1, monotonic=self.clock.t,
                                    stop_requested=True), self.feedback)
        self.assertNotIn('joint', [c[0] for c in self.FakeSDK.calls])

    def test_example_and_expired_arming_remain_closed(self):
        example = json.loads((Path(__file__).parents[1] / 'arm.example.json').read_text())
        with self.assertRaises(ValueError):
            validate_arm(example, 'can0', now=100.)
        armed = {k: True for k in ('operator_present', 'model_and_control_point_checked', 'measured_joint_hold_checked', 'only_one_can_writer')}
        armed.update(expires_unix_s=110., can_interface='can0', acknowledgement=ACK)
        validate_arm(armed, 'can0', now=100.)
        armed['measured_joint_hold_checked'] = False
        with self.assertRaises(ValueError):
            validate_arm(armed, 'can0', now=100.)
        validate_arm(armed, 'can0', now=100., commission_hold_only=True)
        with self.assertRaises(ValueError):
            validate_arm(armed, 'can0', now=111.)


class PassiveReceiveTests(unittest.TestCase):
    def test_passive_can_receives_without_any_send_or_timestamp_renewal(self):
        frames = []
        for i, can_id in enumerate((0x2a5, 0x2a6, 0x2a7)):
            raw = struct.pack('>ii', *([1000, 2000] if i == 0 else [-3000, 4000]))
            frames.append(struct.pack('=IB3x8s', can_id, 8, raw))
        for can_id in range(0x261, 0x267):
            frames.append(struct.pack('=IB3x8s', can_id, 8, bytes([0, 0, 0, 0, 0, 0x40, 0, 0])))
        frames.append(struct.pack('=IB3x8s', 0x2a8, 8, struct.pack('>i', 40000) + bytes([0, 0, 0x40, 0])))
        frames.append(struct.pack('=IB3x8s', 0x2a1, 8, bytes([1, 0, 1, 0, 0, 0, 0, 0])))
        class Socket:
            def setsockopt(self, *args): pass
            def bind(self, *args): pass
            def settimeout(self, *args): pass
            def close(self): pass
            def send(self, *args): raise AssertionError('Passive source must never send')
            def sendto(self, *args): raise AssertionError('Passive source must never send')
            def recv(self, size):
                if frames:
                    return frames.pop(0)
                time.sleep(.005)
                raise socket.timeout()
        with patch('servo_bridge.sources.socket.socket', return_value=Socket()):
            source = PassiveCan('fake-can')
            try:
                until = time.monotonic() + 1.
                while frames and time.monotonic() < until:
                    time.sleep(.005)
                first = source.sample(time.monotonic(), .05)[0]
                second = source.sample(time.monotonic() + 10., .05)[0]
                self.assertEqual(first.received, second.received)
                self.assertTrue(first.ready)
                np.testing.assert_allclose(np.degrees(first.q), [1, 2, -3, 4, -3, 4])
            finally:
                source.close()


if __name__ == '__main__':
    unittest.main()
