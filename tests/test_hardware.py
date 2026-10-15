"""Contract/failure tests with synthetic CAN frames and camera captures only."""
import copy
import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from e14.cameras import CameraReader
from e14.piper_robot import (ACTION_NAMES, FeedbackStore, JOINT_FRAMES,
                            LeRobotPiperAdapter, LeRobotPiperAdapterConfig, MockBackend, PiperRobot,
                            REQUIRED_FRAMES, _SDKBackend, ensure_can)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "config.mock.json").read_text())


class Clock:
    def __init__(self):
        self.value = 100.

    def __call__(self):
        return self.value

    def advance(self, duration=.1):
        self.value += duration


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = MockBackend(self.clock)
        self.backend.refresh()

    def test_raw_degree_and_metre_conversion(self):
        self.backend.state = [12.345, -.314, -99.123, -1.234, 5.678, 99.999, .056789]
        self.backend.enabled = True
        self.backend.gripper_enabled = True
        self.backend.refresh()
        snap = self.backend.store.snapshot()
        np.testing.assert_allclose(snap['state'], self.backend.state, atol=1e-9)
        self.assertEqual(snap['enabled'], [True] * 6)
        self.assertTrue(snap['gripper_enabled'])
        self.assertEqual(snap['motor_faults'], [0] * 6)

    def test_all_frame_ids_required(self):
        self.assertEqual(set(self.backend.store.frames), set(REQUIRED_FRAMES))
        for missing in REQUIRED_FRAMES:
            saved = self.backend.store.frames.pop(missing)
            with self.assertRaisesRegex(RuntimeError, 'Missing feedback'):
                self.backend.store.snapshot()
            self.backend.store.frames[missing] = saved

    def test_one_pair_freezing_does_not_gain_other_frames_timestamp(self):
        store = self.backend.store
        initial = store.snapshot()['frame_timestamps']['0x2a5']
        self.clock.advance(.4)
        store.ingest(0x2A6, b'\0' * 8)
        snap = store.snapshot()
        self.assertEqual(snap['frame_timestamps']['0x2a5'], initial)
        self.assertEqual(snap['feedback_monotonic'], initial)
        self.assertGreater(snap['frame_timestamps']['0x2a6'], initial)

    def test_motor_and_gripper_fault_bit_meanings(self):
        store = self.backend.store
        # Motor bit7 is a stall fault, bit6 is enable. Gripper bit7 is homed.
        store.ingest(0x261, bytes([0, 0, 0, 0, 0, 0xC0, 0, 0]))
        store.ingest(0x2A8, bytes([0, 0, 0, 0, 0, 0, 0xC0, 0]))
        snap = store.snapshot()
        self.assertTrue(snap['enabled'][0])
        self.assertEqual(snap['motor_faults'][0], 0x80)
        self.assertTrue(snap['gripper_enabled'])
        self.assertEqual(snap['gripper_faults'], 0)
        store.ingest(0x2A8, bytes([0, 0, 0, 0, 0, 0, 0x41, 0]))
        self.assertEqual(store.snapshot()['gripper_faults'], 1)

    def test_unrelated_or_short_frame_ignored_and_invalid_time_rejected(self):
        store = FeedbackStore(self.clock)
        store.ingest(0x123, b'\0' * 8)
        store.ingest(0x2A5, b'\0' * 7)
        self.assertFalse(store.frames)
        for stamp in (float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                store.ingest(0x2A5, b'\0' * 8, received_at=stamp)


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = MockBackend(self.clock)
        self.robot = PiperRobot(config(), mock=True, backend=self.backend, clock=self.clock)
        self.robot.connect()

    def tearDown(self):
        self.robot.disconnect()

    def prepare(self):
        # Let a mocked mode command produce a new receipt timestamp as CAN does.
        original = self.backend.mode_control
        def mode(speed):
            self.clock.advance(.01)
            original(speed)
        self.backend.mode_control = mode
        self.robot.prepare_control()

    def test_connect_observe_disconnect_are_read_only(self):
        observation = self.robot.get_observation()
        self.assertEqual(len(observation['state']), 7)
        self.assertEqual(observation['feedback_monotonic'], self.clock())
        self.robot.disconnect()
        self.assertEqual(self.backend.commands, [])
        with self.assertRaisesRegex(RuntimeError, 'not connected'):
            self.robot.get_observation()

    def test_frozen_feedback_rejected_on_read_and_motion(self):
        self.prepare()
        self.backend.freeze_feedback()
        self.clock.advance(.4)
        with self.assertRaisesRegex(RuntimeError, 'Stale feedback'):
            self.robot.get_observation()
        before = list(self.backend.commands)
        with self.assertRaisesRegex(RuntimeError, 'Stale feedback'):
            self.robot.send_action(self.backend.state)
        self.assertEqual(self.backend.commands, before)

    def test_stale_single_pair_rejected_even_with_fresh_other_pairs(self):
        self.backend.frozen = True
        self.clock.advance(.4)
        for frame_id in REQUIRED_FRAMES[1:]:
            data = self.backend.store.frames[frame_id][0]
            self.backend.store.ingest(frame_id, data)
        with self.assertRaisesRegex(RuntimeError, '0x2a5'):
            self.robot.get_observation()

    def test_future_feedback_rejected(self):
        self.backend.frozen = True
        data = self.backend.store.frames[0x2A5][0]
        self.backend.store.ingest(0x2A5, data, received_at=self.clock() + .1)
        with self.assertRaisesRegex(RuntimeError, 'Stale feedback'):
            self.robot.get_observation()

    def test_explicit_prepare_sends_mode_enable_but_no_joint_target(self):
        with self.assertRaisesRegex(RuntimeError, 'prepare_control'):
            self.robot.send_action(self.backend.state)
        self.prepare()
        self.assertEqual([x[0] for x in self.backend.commands], ['mode', 'enable', 'gripper'])
        self.assertTrue(self.robot._prepared)

    def test_teaching_mode_is_not_reset(self):
        self.backend.mode = 2
        with self.assertRaisesRegex(RuntimeError, 'physical button'):
            self.robot.prepare_control()
        self.assertFalse(self.backend.commands)
        self.assertFalse(self.robot._prepared)

    def test_small_feedback_boundary_tolerance_but_strict_target(self):
        self.backend.state[1] = -.314
        self.prepare()
        goal = list(self.backend.state)
        goal[1] = 0
        self.robot.send_action(goal)
        self.assertEqual(self.backend.commands[-2][1][1], 0)
        goal[1] = -.1
        with self.assertRaises(ValueError):
            self.robot.send_action(goal)

    def test_action_conversion_and_flat_lerobot_adapter(self):
        self.prepare()
        goal = self.backend.state.copy()
        goal[0], goal[6] = .123, .041234
        sent = self.robot.send_action(dict(zip(ACTION_NAMES, goal)))
        self.assertEqual(sent, dict(zip(ACTION_NAMES, goal)))
        self.assertIn(('joint', [123, 0, -30000, 0, 30000, 0]), self.backend.commands)
        self.assertIn(('gripper', 41234, 500), self.backend.commands)
        adapter = LeRobotPiperAdapter(config(), mock=True)
        try:
            adapter.connect()
            self.assertEqual(set(adapter.get_observation()), set(ACTION_NAMES))
            self.assertEqual(set(adapter.observation_features), set(ACTION_NAMES))
        finally:
            adapter.disconnect()

    def test_target_shape_nan_range_and_delta_rejected_without_send(self):
        self.prepare()
        base = self.backend.state.copy()
        values = [base[:-1], [float('nan')] * 7, [True] * 7,
                  {**dict(zip(ACTION_NAMES, base)), 'extra': 1}]
        for axis, value in ((0, 151), (0, 2), (1, -1), (6, .08), (6, .043)):
            target = base.copy()
            target[axis] = value
            values.append(target)
        before = list(self.backend.commands)
        for target in values:
            with self.assertRaises(ValueError):
                self.robot.send_action(target)
        self.assertEqual(self.backend.commands, before)

    def test_changed_commands_cannot_bypass_cycle_limit(self):
        self.prepare()
        goal = self.backend.state.copy()
        self.robot.send_action(goal)
        goal[0] += .2
        with self.assertRaisesRegex(RuntimeError, 'faster'):
            self.robot.send_action(goal)
        self.clock.advance(.05)
        self.robot.send_action(goal)

    def test_disabled_gripper_may_not_change(self):
        self.robot.settings['enable_gripper'] = False
        self.prepare()
        goal = self.backend.state.copy()
        goal[6] += .001
        with self.assertRaisesRegex(RuntimeError, 'enable_gripper=false'):
            self.robot.send_action(goal)

    def test_partial_send_failure_attempts_stop_and_latches_unprepared(self):
        self.prepare()
        def failed_gripper(*_):
            raise OSError('simulated CAN send failure')
        self.backend.gripper = failed_gripper
        with self.assertRaisesRegex(OSError, 'simulated CAN'):
            self.robot.send_action(self.backend.state)
        self.assertEqual(self.backend.commands[-1][0], 'emergency_stop')
        self.assertFalse(self.robot._prepared)

    def test_emergency_stop_does_not_resume_or_disable(self):
        self.prepare()
        self.robot.stop()
        self.assertFalse(self.robot._prepared)
        self.assertEqual(self.backend.commands[-1], ('emergency_stop',))
        with self.assertRaisesRegex(RuntimeError, 'PiPER error'):
            self.robot.prepare_control()
        self.assertFalse(self.robot._prepared)
        self.assertNotIn('reset', [x[0] for x in self.backend.commands])

    def test_hold_requires_fresh_feedback(self):
        self.prepare()
        self.robot.settings['stop_strategy'] = 'hold'
        self.backend.freeze_feedback()
        self.clock.advance(.4)
        before = list(self.backend.commands)
        with self.assertRaisesRegex(RuntimeError, 'Stale feedback'):
            self.robot.stop()
        self.assertEqual(self.backend.commands, before)
        self.assertFalse(self.robot._prepared)

    def test_hold_corrects_tolerated_feedback_boundary_only(self):
        self.backend.state[1] = -.314
        self.prepare()
        self.robot.settings['stop_strategy'] = 'hold'
        self.robot.stop()
        self.assertEqual(self.backend.commands[-2][0], 'joint')
        self.assertEqual(self.backend.commands[-2][1][1], 0)
        self.assertFalse(self.robot._prepared)

    def test_hold_boundary_correction_obeys_step_limit(self):
        self.backend.state[1] = -.314
        self.prepare()
        self.robot.settings['stop_strategy'] = 'hold'
        self.robot.settings['max_step_delta'][1] = .1
        before = list(self.backend.commands)
        with self.assertRaisesRegex(RuntimeError, 'Boundary correction'):
            self.robot.stop()
        self.assertEqual(before, self.backend.commands)

    def test_official_lerobot_config_factory_and_dataset_features(self):
        try:
            from lerobot.robots.config import RobotConfig
            from lerobot.robots.utils import make_robot_from_config
            from lerobot.utils.feature_utils import hw_to_dataset_features
        except ImportError:
            self.skipTest('LeRobot is not installed in this Python')
        config_class = RobotConfig.get_choice_class('piper_smolvla')
        self.assertIs(config_class, LeRobotPiperAdapterConfig)
        adapter = make_robot_from_config(config_class(deployment=config(), mock=True, id='test-piper'))
        self.assertIsInstance(adapter, LeRobotPiperAdapter)
        self.assertEqual(adapter.id, 'test-piper')
        try:
            adapter.connect()
            self.assertEqual(set(adapter.get_observation()), set(ACTION_NAMES))
            features = hw_to_dataset_features(adapter.observation_features, 'observation')
            self.assertEqual(features['observation.state']['shape'], (7,))
            self.assertEqual(features['observation.state']['names'], list(ACTION_NAMES))
        finally:
            adapter.disconnect()

    def test_failed_prepare_does_not_leave_prepared_flag(self):
        self.prepare()
        self.backend.state[0] = 200
        with self.assertRaisesRegex(RuntimeError, 'feedback'):
            self.robot.prepare_control()
        self.assertFalse(self.robot._prepared)

    def test_disconnect_failure_still_clears_connection(self):
        def fail():
            raise OSError('simulated close error')
        self.backend.disconnect = fail
        with self.assertRaises(OSError):
            self.robot.disconnect()
        self.assertFalse(self.robot.is_connected)
        self.assertFalse(self.robot._prepared)

    def test_failed_connect_preserves_original_error_and_clears_flags(self):
        settings = config()
        settings['robot']['connect_timeout_s'] = .01
        backend = MockBackend(self.clock)
        backend.frozen = True
        def fail_close():
            raise OSError('secondary cleanup failure')
        backend.disconnect = fail_close
        robot = PiperRobot(settings, mock=True, backend=backend, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'Feedback wait timed out'):
            robot.connect()
        self.assertFalse(robot.is_connected)

    def test_real_control_cannot_start_with_unverified_stop(self):
        settings = config()
        settings['robot']['stop_strategy_verified'] = False
        robot = PiperRobot(settings, backend=self.backend, clock=self.clock)
        try:
            robot.connect()
            self.assertEqual(len(robot.get_observation()['state']), 7)
            with self.assertRaisesRegex(RuntimeError, 'stop_strategy_verified'):
                robot.prepare_control()
            self.assertEqual(self.backend.commands, [])
        finally:
            robot.disconnect()

    def test_bad_configuration_rejected(self):
        for key, value in [('speed_percent', True), ('gripper_effort', 5001),
                           ('stop_strategy', 'reset'), ('control_hz', 0),
                           ('max_step_delta', [0] * 7), ('joint_unit', 'radian')]:
            candidate = config()
            candidate['robot'][key] = value
            with self.assertRaises(ValueError):
                PiperRobot(candidate, mock=True)


class SDKContractTests(unittest.TestCase):
    def test_callback_is_installed_before_connect_and_only_real_frames_refresh(self):
        clock = Clock()
        class Bus:
            CAN_STATUS = types.SimpleNamespace(SEND_MESSAGE_SUCCESS=1)
            result = 1
            def SendCanMessage(self, *_):
                return self.result
        class Interface:
            def __init__(self, name, **kwargs):
                self.name, self.kwargs = name, kwargs
                self.bus = Bus()
                self.callback = self.ParseCANFrame  # SDK 0.6.2 constructor behaviour.
            def ParseCANFrame(self, frame):
                clock.advance(.01)  # SDK parsing must not become a new arrival time.
            def GetCanBus(self):
                return self.bus
            def ConnectPort(self, **kwargs):
                self.connect_options = kwargs
            def DisconnectPort(self):
                self.closed = True
        with patch.dict(sys.modules, {'piper_sdk': types.SimpleNamespace(C_PiperInterface_V2=Interface)}):
            backend = _SDKBackend('fake-can', clock)
        backend.connect()
        self.assertEqual(backend.arm.connect_options, {'piper_init': False})
        self.assertFalse(backend.arm.kwargs['start_sdk_joint_limit'])
        frame = types.SimpleNamespace(arbitration_id=0x2A5, data=b'\0' * 8,
                                      is_error_frame=False, is_remote_frame=False, is_extended_id=False)
        original_stamp = clock()
        backend.arm.callback(frame)
        self.assertEqual(backend.store.frames[0x2A5][1], original_stamp)
        for flag in ('is_error_frame', 'is_remote_frame', 'is_extended_id'):
            invalid = copy.copy(frame)
            setattr(invalid, flag, True)
            backend.arm.callback(invalid)
            self.assertEqual(backend.store.frames[0x2A5][1], original_stamp)
        short = copy.copy(frame)
        short.data = b'\0' * 7
        backend.arm.callback(short)
        self.assertEqual(backend.store.frames[0x2A5][1], original_stamp)
        backend.refresh()
        self.assertEqual(backend.store.frames[0x2A5][1], original_stamp)
        backend.arm.bus.result = -1
        with self.assertRaisesRegex(RuntimeError, 'CAN send failed'):
            backend.arm.bus.SendCanMessage(0, b'\0' * 8)
        backend.disconnect()
        self.assertTrue(backend.arm.closed)

    def test_socketcan_healthy_no_mutation_and_busoff_requires_explicit_restart(self):
        healthy = {'flags': ['UP'], 'linkinfo': {'info_kind': 'can', 'info_data':
                   {'state': 'ERROR-ACTIVE', 'bittiming': {'bitrate': 1000000}}}}
        result = types.SimpleNamespace(stdout=json.dumps([healthy]))
        with patch('e14.piper_robot.subprocess.run', return_value=result) as run:
            ensure_can('can0')
            self.assertEqual(run.call_count, 1)
        healthy['linkinfo']['info_data']['state'] = 'BUS-OFF'
        with patch('e14.piper_robot.subprocess.run', return_value=types.SimpleNamespace(stdout=json.dumps([healthy]))) as run:
            with self.assertRaisesRegex(RuntimeError, 'BUS-OFF'):
                ensure_can('can0', initialize=True)
            self.assertEqual(run.call_count, 1)


class CameraTests(unittest.TestCase):
    def test_mock_named_rgb_frames_and_timestamps(self):
        clock = Clock()
        reader = CameraReader(config()['cameras'], mock=True, clock=clock)
        with reader:
            sample = reader.read()
            image = sample['images']['front']
            self.assertEqual(image.shape, (240, 320, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(sample['image_timestamps'], {'front': clock()})
            image[:] = 0
            self.assertTrue(reader.read()['images']['front'].any())
        self.assertFalse(reader.is_connected)

    def test_invalid_config_and_duplicate_device_rejected(self):
        camera = {'device': 0, 'width': 4, 'height': 3, 'fps': 30, 'max_age_ms': 10}
        with self.assertRaises(ValueError): CameraReader({})
        with self.assertRaises(ValueError): CameraReader({'front': camera, 'wrist': camera})
        for key, value in [('device', None), ('width', True), ('fps', 0), ('max_age_ms', float('nan'))]:
            with self.assertRaises(ValueError): CameraReader({'front': {**camera, key: value}})

    def test_real_capture_contract_rgb_freshness_and_bounded_close(self):
        clock = Clock()
        unblock = threading.Event()
        class Capture:
            released = False
            count = 0
            def isOpened(self): return True
            def set(self, *_): return True
            def get(self, *_): return 30
            def read(self):
                self.count += 1
                if self.count == 1:
                    return True, np.tile(np.array([1, 2, 3], dtype=np.uint8), (3, 4, 1))
                unblock.wait(1)
                return False, None
            def release(self): self.released = True
        capture = Capture()
        reader = CameraReader({'front': {'device': 'fake', 'width': 4, 'height': 3,
                                        'fps': 30, 'max_age_ms': 50, 'join_timeout_s': .01}},
                              clock=clock, capture_factory=lambda *_: capture)
        try:
            reader.connect()
            sample = reader.read()
            self.assertEqual(sample['images']['front'][0, 0].tolist(), [3, 2, 1])
            clock.advance(.1)
            with self.assertRaisesRegex(TimeoutError, 'stale'):
                reader.read()
            started = time.monotonic()
            reader.close()
            self.assertLess(time.monotonic() - started, .2)
            self.assertFalse(capture.released)  # Never release a blocked cap from another thread.
            with self.assertRaisesRegex(RuntimeError, 'still blocked'):
                reader.connect()
        finally:
            unblock.set()
            for thread in reader._threads:
                thread.join(1)
            reader.close()
        self.assertTrue(capture.released)

    def test_capture_open_failure_releases(self):
        capture = types.SimpleNamespace(isOpened=lambda: False, release=lambda: None)
        reader = CameraReader({'front': {'device': 'fake', 'width': 4, 'height': 3,
                                        'fps': 30, 'max_age_ms': 50}},
                              capture_factory=lambda *_: capture)
        with self.assertRaisesRegex(RuntimeError, 'Could not open'):
            reader.connect()
        self.assertFalse(reader.is_connected)

    def test_camera_geometry_failure_is_not_silently_resized(self):
        released = []
        capture = types.SimpleNamespace(isOpened=lambda: True, set=lambda *_: True,
                    get=lambda *_: 30, read=lambda: (True, np.zeros((2, 2, 3), np.uint8)),
                    release=lambda: released.append(True))
        reader = CameraReader({'front': {'device': 'fake', 'width': 4, 'height': 3,
                                        'fps': 30, 'max_age_ms': 50}},
                              capture_factory=lambda *_: capture)
        with self.assertRaisesRegex(RuntimeError, 'Expected uint8 BGR'):
            reader.connect()
        self.assertFalse(reader.is_connected)
        self.assertTrue(released)


if __name__ == '__main__':
    unittest.main()
