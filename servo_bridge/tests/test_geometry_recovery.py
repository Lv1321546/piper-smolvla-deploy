"""Geometry status -> runtime hold -> independent worker, without ROS or CAN."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from servo_bridge.core import Feedback, NAMES, load_config
from servo_bridge.execution import VelocityTarget
from servo_bridge.geometry_recovery import GeometryRecovery
from servo_bridge.joint_policy import JointPolicy
from servo_bridge.sdk_worker import Worker
from servo_bridge.tests.test_geometry_hold import Writer


class Harness(GeometryRecovery):
    def event(self, kind, **fields):
        self.events.append(dict(event=kind, **fields))

    def latch_output(self, reason):
        self.output_latched = True


class GeometryRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.joint.example.json')
        self.now = 100.
        self.q = np.radians(self.cfg['sim_initial_joints_deg'])
        self.writer = Writer()
        self.worker = Worker(self.cfg, self.writer, clock=lambda: self.now)
        self.node = Harness()
        self.node.cfg = self.cfg
        self.node.args = SimpleNamespace(output_socket='fake')
        self.node.source = object()
        self.node.feedback = Feedback(self.q.copy(), self.now, .04, True, True)
        self.node.policy = JointPolicy(self.cfg)
        self.node.execution = VelocityTarget(self.cfg)
        self.node.output_latched = False
        self.node.events = []
        self.packets = []
        def send(data, address):
            packet = json.loads(data)
            self.packets.append(packet)
            self.worker.packet(packet, self.node.feedback)
        self.node.socket = SimpleNamespace(sendto=send)

    def test_hold_for_both_statuses_keeps_session_open_and_does_not_send_targets(self):
        for status in (2, 5):
            self.now += .02
            self.node.feedback.received = self.now
            with patch('servo_bridge.geometry_recovery.time.monotonic', return_value=self.now):
                self.node.status(SimpleNamespace(data=status))
            self.assertFalse(self.node.output_latched)
            self.assertTrue(self.node.geometry_paused)
            self.assertTrue(self.node.events[-2]['recoverable'])
            self.assertEqual(self.packets[-1]['servo_status'], status)
            self.assertNotIn('positions', self.packets[-1])
        self.assertEqual(self.writer.joints, [])
        self.assertEqual(self.writer.grippers, [])

    def test_real_clear_status_allows_guarded_motion_again(self):
        with patch('servo_bridge.geometry_recovery.time.monotonic', return_value=self.now):
            self.node.status(SimpleNamespace(data=5))
        self.now += .02
        self.node.feedback.received = self.now
        with patch('servo_bridge.geometry_recovery.time.monotonic', return_value=self.now):
            self.node.status(SimpleNamespace(data=0))
        self.assertTrue(self.worker.packet(dict(version=1, monotonic=self.now,
            servo_status=self.node.latest_status, status_monotonic=self.node.latest_status_received,
            deadman=True, joint_names=list(NAMES), positions=self.q.tolist(),
            velocities=[0.]*6, horizon_s=.02), self.node.feedback))

    def test_collision_and_stale_status_still_latch(self):
        for status in (3, 4):
            self.node.output_latched = False
            with patch('servo_bridge.geometry_recovery.time.monotonic', return_value=self.now):
                self.node.status(SimpleNamespace(data=status))
            self.assertTrue(self.node.output_latched)
            self.assertFalse(self.node.events[-1]['recoverable'])
        self.node.output_latched = False
        self.node.latest_status = 5
        self.node.latest_status_received = 0.
        with patch('servo_bridge.geometry_recovery.time.monotonic', return_value=self.now):
            self.node.geometry_hold('test stale')
        self.assertTrue(self.node.output_latched)


if __name__ == '__main__':
    unittest.main()
