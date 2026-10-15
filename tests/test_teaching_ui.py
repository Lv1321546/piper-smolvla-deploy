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
