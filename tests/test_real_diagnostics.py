"""Offline diagnostics contracts; no CAN device is opened."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from e14.diagnose_can import capture
from tests import test_gamepad as fixtures


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.GamepadTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_passive_capture_records_stopped_feedback_and_external_controls(self):
        clock = self.fixture.clock
        backend = self.fixture.backend
        backend.arm_status = 1
        backend.refresh()
        frames = [SimpleNamespace(arbitration_id=can_id, data=data, timestamp=1.,
                                  is_error_frame=False, is_remote_frame=False, is_extended_id=False)
                  for can_id, (data, _) in backend.store.frames.items()]
        frames.append(SimpleNamespace(arbitration_id=0x150, data=b'\x01'+b'\0'*7, timestamp=1.,
                                      is_error_frame=False, is_remote_frame=False, is_extended_id=False))
        # This fake bus has no send method: capture must only call recv.
        class Bus:
            def recv(self, timeout):
                clock.sleep(.01)
                return frames.pop(0) if frames else None
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'log.jsonl'
            result = capture(Bus(), path, duration=.3, clock=clock)
            events = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(result['final_feedback']['arm_status'], 1)
            self.assertEqual(result['actuator_commands_sent'], 0)
            self.assertEqual(result['observed_control_counts'], {'0x150': 1})
            self.assertEqual(events[-1]['event'], 'summary')
            with self.assertRaises(FileExistsError):
                capture(Bus(), path, duration=.3, clock=clock)

    def test_capture_preserves_failure_and_summary(self):
        class Bus:
            def recv(self, timeout):
                raise OSError('read failed')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'log.jsonl'
            with self.assertRaisesRegex(OSError, 'read failed'):
                capture(Bus(), path)
            summary = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(summary['status'], 'failed')
            self.assertIn('Missing feedback frames', summary['final_feedback']['error'])

    def test_no_camera_preview_cannot_record_or_save_fake_reference(self):
        session, dashboard, pad, args = self.fixture.make_session()
        args.no_camera = True
        session.poll()
        for command in ['record', 'save_home']:
            with self.assertRaisesRegex(RuntimeError, '摄像头'):
                session.handle(command)
        self.assertFalse(args.output.exists())
        self.assertTrue(dashboard.state['camera_disabled'])


if __name__ == '__main__':
    unittest.main()
