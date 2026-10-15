"""Interactive teaching tests use synthetic CAN/cameras, never physical devices."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from e14.cameras import CameraReader
from e14.collect_data import record_episode
from e14.export_dataset import load_raw_episode, build_samples
from e14.piper_robot import MockBackend, PiperRobot
from e14.teaching_ui import Dashboard, GripperControl, TeachingSession

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.t = 100.
    def __call__(self):
        return self.t
    def sleep(self, seconds):
        self.t += seconds


class FakeDashboard:
    def __init__(self):
        self.pending = []
        self.state = None
        self.scheduled = []
    def drain(self):
        if self.scheduled:
            return self.scheduled.pop(0)
        commands, self.pending = self.pending, []
        return commands
    def update(self, state, images):
        self.state = state


class TeachingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.clock = Clock()
        self.config = json.loads((ROOT / 'config.mock.json').read_text())
        self.config['cameras']['front'].update(width=16, height=12)
        # Explicit manual gripper works even with policy execution disabled.
        self.config['robot'].update(enable_gripper=False, stop_strategy_verified=False)
        self.backend = MockBackend(self.clock)
        self.backend.mode = 2
        self.robot = PiperRobot(self.config, mock=True, backend=self.backend, clock=self.clock)
        self.robot.connect()
        original_mode = self.backend.mode_control
        def mode(speed):
            self.clock.sleep(.001)
            original_mode(speed)
        self.backend.mode_control = mode
        self.cameras = CameraReader(self.config['cameras'], mock=True, clock=self.clock)
        self.cameras.connect()
        self.args = SimpleNamespace(open_width=.06, close_width=0., gripper_speed=.03,
                                    duration=.5, task='Lift block and drop onto mouse pad',
                                    home_file=self.root/'home.json', output=self.root/'raw',
                                    mock=True, episodes=2)
        self.dashboard = FakeDashboard()
        self.session = TeachingSession(self.args, self.robot, self.cameras, self.dashboard,
                                       clock=self.clock, sleep=self.clock.sleep)

    def tearDown(self):
        self.cameras.close()
        self.robot.disconnect()
        self.temp.cleanup()

    def test_manual_gripper_never_commands_arm_or_clears_errors(self):
        event = self.robot.command_gripper(.05)
        self.assertEqual(event['commanded_width_m'], .05)
        self.assertEqual(self.backend.commands, [('gripper', 50000, 500)])
        self.assertFalse(self.robot._prepared)
        for value in (-.001, .071, float('nan'), True):
            with self.assertRaises(ValueError):
                self.robot.command_gripper(value)
        before = list(self.backend.commands)
        self.backend.frozen = True
        self.clock.sleep(.4)
        with self.assertRaisesRegex(RuntimeError, 'Stale'):
            self.robot.command_gripper(.05)
        self.assertEqual(before, self.backend.commands)

    def test_gripper_fault_or_wrong_mode_never_sends(self):
        self.backend.mode = 0
        with self.assertRaises(RuntimeError):
            self.robot.command_gripper(.05)
        self.backend.mode = 2
        self.backend.arm_status = 1
        with self.assertRaises(RuntimeError):
            self.robot.command_gripper(.05)
        self.assertEqual(self.backend.commands, [])

    def test_ramp_is_bounded_and_changes_direction_from_actual_width(self):
        control = self.session.gripper
        control.tick()
        self.assertAlmostEqual(self.backend.state[6], .0415)
        before = len(self.backend.commands)
        control.tick()
        self.assertEqual(len(self.backend.commands), before)
        self.clock.sleep(1)  # Delay must not produce a large target jump.
        control.tick()
        self.assertAlmostEqual(self.backend.state[6], .043)
        control.request(False)
        control.tick()
        self.assertAlmostEqual(self.backend.state[6], .0415)
        self.assertTrue(all(c[0] == 'gripper' for c in self.backend.commands))

    def test_start_waits_for_measured_open_not_command_target(self):
        self.session.handle('record')
        self.assertTrue(self.session.start_requested)
        self.session.poll()
        self.assertFalse(self.session.gripper.is_open(self.session.latest))
        for _ in range(45):
            self.clock.sleep(.05)
            self.session.poll()
        self.assertTrue(self.session.gripper.is_open(self.session.latest))
        self.assertFalse(self.args.output.exists())

    def test_record_stops_labels_audit_and_next_state_are_exportable(self):
        self.robot.command_gripper(.06)
        self.session.publish()
        self.dashboard.scheduled = [['close'], [], ['open'], ['success']]
        self.session.record()
        path = self.args.output/'episode_000000'
        manifest = json.loads((path/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'complete')
        self.assertEqual(manifest['task_outcome'], 'success')
        self.assertEqual(manifest['stop_reason'], 'operator')
        self.assertEqual(manifest['expected_frames'], 3)
        episode = load_raw_episode(path)
        samples = list(build_samples(episode))
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(float(samples[0]['action'][6]), episode['rows'][1]['state'][6])
        events = [json.loads(s) for s in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['event'] == 'gripper_command' for e in events))
        self.assertEqual(self.session.completed, 1)
        self.assertTrue(all(c[0] == 'gripper' for c in self.backend.commands))

    def test_home_pose_is_saved_without_motion_and_return_blocked_in_recording(self):
        self.session.handle('save_home')
        data = json.loads(self.args.home_file.read_text())
        self.assertEqual(data['joint_angles'], self.backend.state[:6])
        self.assertTrue((self.root/'home-front.png').exists())
        self.assertEqual(self.backend.commands, [])
        self.session.status = 'recording'
        self.session.handle('home')
        self.assertEqual(self.backend.commands, [])
        self.assertIsNone(self.session.home_target)

    def test_home_return_preserves_pending_review_and_opens_only_on_arrival(self):
        self.session.handle('save_home')
        self.backend.state[0] = 5.
        self.backend.state[6] = .02
        self.backend.refresh()
        self.session.status = 'review'
        self.session.handle('home')
        for _ in range(30):
            if self.session.home_target is None:
                break
            self.clock.sleep(.05)
            self.session._home_tick()
        self.assertIsNone(self.session.home_target)
        self.assertEqual(self.session.status, 'review')
        self.assertLessEqual(abs(self.backend.state[0]), .5)
        self.assertEqual(self.backend.state[6], .02)
        self.assertEqual(self.session.gripper.target, .06)
        self.assertTrue(any(c[0] == 'joint' for c in self.backend.commands))

    def test_quit_sends_no_extra_gripper_command_and_never_auto_releases(self):
        self.session.gripper.request(False)
        before = list(self.backend.commands)
        self.dashboard.pending = ['quit']
        self.assertFalse(self.session.poll())
        self.assertEqual(self.backend.commands, before)

    def test_feedback_boundary_offsets_never_become_out_of_range_commands(self):
        self.robot.settings['gripper_limits_m'] = [0, .1]
        for measured in (.1006, .1021, -.0006):
            with self.subTest(measured=measured):
                self.backend.state[6] = measured
                self.backend.refresh()
                self.session.gripper.request(True)
                event = self.session.gripper.events[-1]
                self.assertAlmostEqual(event['measured_width_m'], measured)
                self.assertAlmostEqual(self.robot.get_feedback()['state'][6], measured)
                self.session.gripper.tick()
                sent = self.backend.commands[-1][1] / 1_000_000
                self.assertGreaterEqual(sent, 0)
                self.assertLessEqual(sent, .1)
                self.assertLessEqual(abs(sent - event['ramp_start_m']), .002)
        # Direct targets remain strict; no firmware range/zero is rewritten.
        with self.assertRaises(ValueError):
            self.robot.command_gripper(.1006)
        self.assertTrue(all(c[0] == 'gripper' for c in self.backend.commands))

    def test_outlier_suspends_old_ramp_but_keeps_preview_and_home_capture(self):
        self.backend.state[6] = .08  # Outside mock's 70 mm limit + 3 mm.
        self.backend.refresh()
        before = list(self.backend.commands)
        self.dashboard.pending = ['open', 'save_home']
        self.session.poll()
        self.assertIsNone(self.session.gripper.target)
        self.assertIn('3 mm', self.session.gripper.error)
        self.assertEqual(self.backend.commands, before)
        self.assertTrue(self.args.home_file.exists())
        self.assertEqual(self.dashboard.state['gripper_width_mm'], 80)
        self.assertTrue(self.dashboard.state['warnings'])
        self.assertIn('已记录', self.dashboard.state['message'])

    def test_startup_outlier_does_not_close_dashboard(self):
        self.backend.state[6] = .08
        self.backend.refresh()
        session = TeachingSession(self.args, self.robot, self.cameras, self.dashboard,
                                  clock=self.clock, sleep=self.clock.sleep)
        session.poll()
        self.assertIsNotNone(session.gripper.error)
        self.assertIsNone(session.gripper.target)
        self.assertEqual(self.dashboard.state['status'], 'preparing')
        self.assertFalse(self.backend.commands)

    def test_ignored_gripper_commands_cancel_pending_record_without_mode_switch(self):
        def ignore(width, effort):
            self.backend.commands.append(('gripper', width, effort))
            self.backend.refresh()  # Fresh frames, but no acknowledgement/motion.
        self.backend.gripper = ignore
        self.session.handle('record')
        for _ in range(55):
            self.session.poll()
            self.clock.sleep(.05)
        self.assertFalse(self.session.start_requested)
        self.assertIsNone(self.session.gripper.target)
        self.assertIn('未响应', self.session.gripper.error)
        before = list(self.backend.commands)
        self.session.poll()
        self.assertEqual(before, self.backend.commands)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(self.dashboard.state['ctrl_mode'], 2)
        self.assertTrue(all(c[0] == 'gripper' for c in self.backend.commands))
        # Recovery is an explicit new request; no automatic mode/reset command.
        self.backend.gripper = lambda w, e: MockBackend.gripper(self.backend, w, e)
        self.dashboard.pending = ['open']
        self.session.poll()
        self.assertIsNone(self.session.gripper.error)

    def test_contact_during_close_is_not_required_to_reach_zero(self):
        def contact(width, effort):
            MockBackend.gripper(self.backend, max(width, 30000), effort)
        self.backend.gripper = contact
        self.session.gripper.request(False)
        for _ in range(100):
            self.session.poll()
            self.clock.sleep(.05)
        self.assertIsNone(self.session.gripper.error)
        self.assertEqual(self.backend.state[6], .03)

    def test_feedback_failure_keeps_camera_and_allows_explicit_retry(self):
        self.backend.frozen = True
        self.clock.sleep(.4)
        self.session.poll()
        self.assertIsNone(self.session.latest)
        self.assertTrue(self.session.device_errors)
        self.assertTrue(self.dashboard.state['image_age_ms'])
        self.assertIsNone(self.session.gripper.target)
        self.assertFalse(self.backend.commands)
        self.backend.frozen = False
        self.dashboard.pending = ['open']
        self.session.poll()
        self.assertFalse(self.session.device_errors)
        self.assertIsNone(self.session.gripper.error)

    def test_control_failure_marks_episode_failed_and_keeps_session_available(self):
        self.robot.command_gripper(.06)
        self.session.publish()
        def broken(width, effort):
            raise RuntimeError('Synthetic CAN send failure')
        self.backend.gripper = broken
        self.dashboard.scheduled = [['close']]
        self.session.record()
        path = self.args.output/'episode_000000'
        manifest = json.loads((path/'manifest.json').read_text())
        self.assertEqual(manifest['status'], 'failed')
        self.assertEqual(manifest['task_outcome'], 'unknown')
        self.assertEqual(self.session.status, 'preparing')
        self.assertEqual(self.session.completed, 0)
        self.assertFalse(self.session.quit)
        self.session.poll()
        self.assertIn('本条录制失败', self.dashboard.state['message'])
        with self.assertRaises(ValueError):
            load_raw_episode(path)

    def test_home_capture_retains_raw_joint_boundary_and_does_not_use_jaw_limits(self):
        self.backend.state[1] = -.314
        self.backend.state[6] = .1006
        self.backend.refresh()
        self.session.handle('save_home')
        data = json.loads(self.args.home_file.read_text())
        self.assertEqual(data['joint_angles'][1], -.314)
        self.assertEqual(data['joints_outside_target_limits'], [2])
        self.assertFalse(self.backend.commands)
        # Saving an observation never silently turns it into a legal motion goal.
        with self.assertRaises(ValueError):
            self.session.handle('home')

    def test_home_block_reason_is_visible_without_hiding_command_message(self):
        self.session.handle('save_home')
        self.robot.mock = False  # Synthetic backend; exercise real-control gates.
        self.dashboard.pending = ['home']
        self.session.poll()
        self.assertIn('退出', self.dashboard.state['message'])
        self.assertTrue(any('停止策略' in w for w in self.dashboard.state['warnings']))
        self.backend.mode = 1
        self.dashboard.pending = ['home']
        self.session.poll()
        self.assertIn('stop_strategy_verified=false', self.dashboard.state['message'])
        self.assertFalse(any(c[0] in {'mode', 'enable', 'joint'} for c in self.backend.commands))

    def test_home_cancel_latches_off_targets_even_if_stop_fails(self):
        self.session.handle('save_home')
        self.backend.state[0] = 5.
        self.backend.refresh()
        self.session.handle('home')
        def fail_stop():
            raise RuntimeError('Synthetic stop send failure; confirm physical stop')
        self.robot.stop = fail_stop
        before = list(self.backend.commands)
        self.dashboard.pending = ['finish']
        self.session.poll()
        self.assertIsNone(self.session.home_target)
        self.assertEqual(self.session.status, 'preparing')
        self.assertEqual(self.backend.commands, before)
        self.assertIn('confirm physical stop', self.dashboard.state['message'])
        self.clock.sleep(.05)
        self.session.poll()
        self.assertEqual(self.backend.commands, before)


class DashboardTests(unittest.TestCase):
    def test_preview_intents_and_cross_origin_protection(self):
        with Dashboard(0) as dashboard:
            dashboard.update({'status':'preparing','status_label':'准备中','message':'test'},
                             {'front':np.zeros((12,16,3),np.uint8)})
            with urlopen(dashboard.url, timeout=2) as r:
                page = r.read().decode()
            self.assertIn('O 张开', page)
            with urlopen(dashboard.url+'/state', timeout=2) as r:
                self.assertEqual(json.load(r)['status'], 'preparing')
            def post(token=None, origin=None):
                headers={'Content-Type':'application/json'}
                if token: headers['X-Teaching-Token']=token
                if origin: headers['Origin']=origin
                return urlopen(Request(dashboard.url+'/command',data=b'{"command":"close"}',headers=headers),timeout=2)
            for token,origin in [(None,None),(dashboard.token,'http://untrusted.example')]:
                with self.assertRaises(HTTPError) as error:
                    post(token,origin)
                self.assertEqual(error.exception.code,403)
            with post(dashboard.token,dashboard.url) as r:
                self.assertEqual(r.status,202)
            self.assertEqual(dashboard.drain(),['close'])
            with urlopen(dashboard.url+'/stream',timeout=2) as r:
                self.assertEqual(r.readline(),b'--frame\r\n')
                self.assertIn(b'image/jpeg',r.readline())
                size=int(r.readline().split(b':')[1]);r.readline()
                image=Image.open(io.BytesIO(r.read(size)))
                self.assertEqual(image.size,(16,12))


if __name__ == '__main__':
    unittest.main()
