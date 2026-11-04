"""Session intent freshness, deduplication and atomic publication."""
import json
from pathlib import Path
import tempfile
import unittest

from servo_bridge.ui_commands import ControlMailbox


class MailboxTests(unittest.TestCase):
    def test_fresh_commands_are_delivered_once_and_in_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ui.json'
            sender = ControlMailbox(path, clock=lambda: 100.)
            receiver = ControlMailbox(path, clock=lambda: 100.1)
            self.assertIsNone(receiver.receive())
            self.assertEqual(sender.send('reset'), 1)
            self.assertEqual(receiver.receive()['command'], 'reset')
            self.assertIsNone(receiver.receive())
            sender.send('cancel_reset')
            self.assertEqual(receiver.receive()['seq'], 2)
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_expired_or_unknown_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ui.json'
            sender = ControlMailbox(path, clock=lambda: 100.)
            receiver = ControlMailbox(path, clock=lambda: 101.)
            sender.send('reset')
            with self.assertRaises(ValueError):
                receiver.receive()
            self.assertIsNone(receiver.receive())
            with self.assertRaises(ValueError):
                sender.send('enable_arm')
            path.write_text(json.dumps(dict(version=1, seq=2, monotonic=101., command='enable_arm')))
            with self.assertRaises(ValueError):
                receiver.receive()


if __name__ == '__main__':
    unittest.main()
