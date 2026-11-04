"""Real geometric halts may authorize hold only, never a new motion target."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from servo_bridge.core import Feedback, NAMES, load_config, sdk_units
from servo_bridge.sdk_worker import Worker


class Writer:
    def __init__(self):
        self.joints = []
        self.grippers = []

    def joint(self, raw):
        self.joints.append(raw)

    def gripper(self, *args):
        self.grippers.append(args)


class GeometryHoldTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1] / 'config.cartesian.example.json')
        self.now = 100.
        self.writer = Writer()
        self.worker = Worker(self.cfg, self.writer, clock=lambda: self.now)
        self.q = np.radians(self.cfg['sim_initial_joints_deg'])

    def feedback(self):
        return Feedback(self.q.copy(), self.now, .04, True, True)

    def motion(self):
        return dict(version=1, monotonic=self.now, servo_status=0, status_monotonic=self.now,
                    deadman=True, joint_names=list(NAMES), positions=self.q.tolist(),
                    velocities=[0.]*6, horizon_s=self.cfg['publish_period_s'])

    def hold(self):
        return dict(version=1, monotonic=self.now, servo_status=5, status_monotonic=self.now,
                    deadman=False, geometry_hold=True)

    def test_geometry_pause_holds_then_real_clear_status_can_resume(self):
        self.worker.packet(self.motion(), self.feedback())
        self.now += .02
        self.assertFalse(self.worker.packet(self.hold(), self.feedback()))
        self.assertEqual(self.writer.joints[-1], sdk_units(self.q))
        self.assertFalse(self.worker.moved)
        self.now += .02
        self.assertTrue(self.worker.packet(self.motion(), self.feedback()))

    def test_geometry_hold_cannot_transmit_targets_or_gripper(self):
        for field, value in [('positions', [100.]*6), ('velocities', [1.]*6),
                             ('joint_names', list(NAMES)), ('gripper', {'width_m': .07})]:
            packet = self.hold()
            packet[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.worker.packet(packet, self.feedback())
        self.assertEqual(self.writer.joints, [])
        self.assertEqual(self.writer.grippers, [])

    def test_collision_stale_status_and_unready_feedback_still_fail(self):
        for status in (3, 4, None):
            packet = self.hold()
            packet['servo_status'] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.worker.packet(packet, self.feedback())
        packet = self.hold()
        packet['status_monotonic'] -= 1
        with self.assertRaises(ValueError):
            self.worker.packet(packet, self.feedback())
        feedback = self.feedback()
        feedback.ready = False
        with self.assertRaises(ValueError):
            self.worker.packet(self.hold(), feedback)

    def test_motion_under_geometry_halt_still_rejected(self):
        for status in (2, 5):
            packet = self.motion()
            packet.update(servo_status=status, geometry_hold=True)
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.worker.packet(packet, self.feedback())

    def test_singularity_hold_is_target_free_and_can_resume_only_after_clear(self):
        self.worker.packet(self.motion(), self.feedback())
        self.now += .02
        packet = self.hold()
        packet['servo_status'] = 2
        self.assertFalse(self.worker.packet(packet, self.feedback()))
        self.assertFalse(self.worker.moved)
        self.assertEqual(self.writer.joints[-1], sdk_units(self.q))
        self.now += .02
        packet = self.motion()
        packet['servo_status'] = 6
        self.assertTrue(self.worker.packet(packet, self.feedback()))

    def test_worker_audits_only_explicit_gripper_transmissions(self):
        events = []
        self.worker.audit = events.append
        self.worker.packet(self.motion(), self.feedback())
        self.assertEqual(events, [])
        self.now += .02
        packet = self.motion()
        packet['gripper'] = dict(width_m=.04, effort=500)
        self.worker.packet(packet, self.feedback())
        self.assertEqual([event['event'] for event in events],
                         ['gripper_transmit_requested', 'gripper_transmitted'])
        self.assertEqual(events[-1]['width_sdk'], 40000)
        self.assertEqual(self.writer.grippers, [(40000, 500)])
        self.now += .02
        self.worker.packet(self.hold(), self.feedback())
        self.assertEqual(len(events), 2)  # Exit/hold never operates the gripper.


if __name__ == '__main__':
    unittest.main()
