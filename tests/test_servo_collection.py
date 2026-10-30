"""No CAN, SDK, gamepad, ROS process, or camera hardware is used here."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from e14.cameras import CameraReader
from e14.piper_robot import PiperRobot,MockBackend
from e14.servo_collect import ServoSession,MockServo,ServoProcess,run_servo_collection
from e14.export_dataset import load_raw_episode,build_samples
from tests.test_teaching_ui import Clock,FakeDashboard

ROOT=Path(__file__).resolve().parents[1]
class ServoCollectionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.clock=Clock()
  cfg=json.loads((ROOT/'config.mock.json').read_text());cfg['cameras']['front'].update(width=16,height=12)
  self.backend=MockBackend(self.clock);self.backend.mode=1;self.backend.enabled=True
  self.robot=PiperRobot(cfg,mock=True,backend=self.backend,clock=self.clock);self.robot.connect()
  self.cam=CameraReader(cfg['cameras'],mock=True,clock=self.clock);self.cam.connect()
  self.args=SimpleNamespace(open_width=.06,close_width=0.,gripper_speed=.03,duration=.5,task='push block',home_file=self.root/'home.json',output=self.root/'raw',mock=True,execute=False,episodes=2,run_seconds=2.,preview_port=0)
  self.ui=FakeDashboard();self.bridge=MockServo(self.robot,self.clock)
  self.session=ServoSession(self.args,self.robot,self.cam,self.ui,self.bridge,clock=self.clock,sleep=self.clock.sleep)
 def tearDown(self):
  self.bridge.close();self.cam.close();self.robot.disconnect();self.tmp.cleanup()
 def test_episode_exports_actual_next_state_and_images(self):
  self.session.poll();self.session.handle('record');self.session.record()
  path=self.session.review_path;self.assertIsNotNone(path)
  manifest=json.loads((path/'manifest.json').read_text());self.assertEqual(manifest['status'],'complete')
  self.assertEqual(manifest['operator_session']['mode'],'moveit_servo_gamepad')
  rows=[json.loads(s) for s in (path/'frames.jsonl').read_text().splitlines()]
  self.assertGreater(rows[-1]['state'][0],rows[0]['state'][0])
  self.assertTrue((path/rows[0]['images']['front']).exists())
  self.session.handle('success');self.assertEqual(json.loads((path/'manifest.json').read_text())['task_outcome'],'success')
  frames=list(build_samples(load_raw_episode(path)))
  self.assertEqual(len(frames),len(rows)-1)
  import numpy as np
  np.testing.assert_allclose(frames[0]['action'],rows[1]['state'])
  self.assertEqual(self.backend.commands,[])
 def test_gripper_and_home_cannot_invoke_legacy_motion(self):
  for c in ('open','close','home'):self.session.handle(c)
  self.assertEqual(self.backend.commands,[]);self.assertIsNone(self.session.gripper.target)
 def test_runtime_failure_marks_episode_failed_and_stops(self):
  self.session.poll();original=self.bridge.poll;n=0
  def fail():
   nonlocal n
   n+=1
   if n==4:raise RuntimeError('injected Servo loss')
   return original()
  self.bridge.poll=fail;self.session.record()
  manifest=json.loads(next(self.args.output.glob('episode_*/manifest.json')).read_text())
  self.assertEqual(manifest['status'],'failed');self.assertEqual(manifest['task_outcome'],'unknown')
  self.assertTrue(self.bridge.stopped);self.assertTrue(self.session.quit)
 def test_camera_failure_stops_control(self):
  self.cam.read=lambda:(_ for _ in ()).throw(RuntimeError('camera stale'))
  self.session.poll();self.assertTrue(self.bridge.stopped);self.assertTrue(self.session.quit)
 def test_button_edges_and_held_startup_key(self):
  def buttons(x):return [{'event':'input','gamepad':{'buttons':{'x':x}},'monotonic':self.clock()}]
  self.bridge.poll=lambda:buttons(True)
  self.session.poll();self.assertFalse(self.session.start_requested)
  self.bridge.poll=lambda:buttons(False);self.session.poll()
  self.bridge.poll=lambda:buttons(True);self.session.poll();self.assertTrue(self.session.start_requested)
  self.session.status='recording';self.session.poll();self.assertFalse(self.session.stop_recording)
  self.bridge.poll=lambda:buttons(False);self.session.poll()
  self.bridge.poll=lambda:buttons(True);self.session.poll();self.assertTrue(self.session.stop_recording)
 def test_mock_rejects_real_execution_before_construction(self):
  self.args.config=ROOT/'config.mock.json';self.args.interactive=False;self.args.execute=True
  with patch('e14.servo_collect.PiperRobot') as robot:
   with self.assertRaises(ValueError):run_servo_collection(self.args)
   robot.assert_not_called()
 def test_telemetry_reader_retains_partial_json(self):
  args=SimpleNamespace(output=self.root,execute=False)
  b=ServoProcess(args,self.robot.config);b.path.mkdir(parents=True);b.trace.parent.mkdir()
  b.trace.write_text('{"event":"input","monotonic":')
  self.assertEqual(b.poll(),[])
  import time
  with b.trace.open('a') as f:f.write(str(time.monotonic())+'}\n')
  self.assertEqual(len(b.poll()),1);b.close()

if __name__=='__main__':unittest.main()
