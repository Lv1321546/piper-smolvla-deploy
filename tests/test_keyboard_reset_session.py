"""Dashboard reset intent stays outside recording and is cancellable."""
import unittest
from unittest.mock import Mock

from e14.servo_collect import PAGE
from tests import test_servo_collection as fixtures


class KeyboardResetSessionTests(unittest.TestCase):
    setUp = fixtures.ServoCollectionTests.setUp
    tearDown = fixtures.ServoCollectionTests.tearDown

    def enable_commands(self):
        self.args.execute = True
        self.args.mock = False
        self.bridge.send_control = Mock(side_effect=[1, 2, 3])
        self.bridge.latest = {'ui_command_seq': 0, 'planner': {'reset_state': 'idle'}}

    def test_h_maps_to_home_and_has_visible_cancel_controls(self):
        self.assertIn('KeyH:\'home\'', PAGE)
        self.assertIn('data-c="home">H 返回并张爪', PAGE)
        self.assertIn('空格 取消复位', PAGE)
        self.assertIn('s.reset_active', PAGE)

    def test_click_home_queues_one_reset_and_blocks_recording_until_done(self):
        self.enable_commands()
        self.session.handle('home')
        self.bridge.send_control.assert_called_once_with('reset')
        self.assertTrue(self.session.reset_in_progress())
        self.session.handle('home')
        self.session.handle('record')
        self.assertEqual(self.bridge.send_control.call_count, 1)
        self.assertFalse(self.session.start_requested)
        self.bridge.latest = {'ui_command_seq': 1, 'planner': {'reset_state': 'complete'}}
        self.assertFalse(self.session.reset_in_progress())
        self.session.handle('record')
        self.assertTrue(self.session.start_requested)

    def test_home_is_rejected_during_recording(self):
        self.enable_commands()
        self.session.status = 'recording'
        self.session.handle('home')
        self.bridge.send_control.assert_not_called()

    def test_space_cancels_reset_without_starting_recording(self):
        self.enable_commands()
        self.session.handle('home')
        self.session.handle('finish')
        self.assertEqual(self.bridge.send_control.call_args.args, ('cancel_reset',))
        self.assertTrue(self.session.reset_in_progress())
        self.bridge.latest = {'ui_command_seq': 2, 'planner': {'reset_state': 'cancelled'}}
        self.assertFalse(self.session.reset_in_progress())


if __name__ == '__main__':
    unittest.main()
