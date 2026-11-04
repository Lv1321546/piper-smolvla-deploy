"""Exercise the real joint trajectory callback with fake SDK transport, no ROS node.

Run with the ROS environment sourced; no ROS initialization or CAN is used.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from servo_bridge.candidate_cadence import CandidateCadence
from servo_bridge.core import CommandGuard, Feedback, NAMES, load_config
from servo_bridge.execution import VelocityTarget
from servo_bridge.joint_runtime import Runtime
from servo_bridge.sdk_worker import Worker


class JointTrajectoryCadenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(Path(__file__).resolve().parents[1]/'config.joint.example.json')
        self.now = 1.
        self.q = np.radians(self.cfg['sim_initial_joints_deg'])
        self.events, self.writes = [], []
        writer = NS(joint=lambda raw: self.writes.append(raw))
        self.worker = Worker(self.cfg, writer, clock=lambda: self.now)
        self.node = NS(cfg=self.cfg, output_count=0, invalid_count=0,
            latest_status=0, latest_status_received=1., latest_input=1.,
            policy=NS(active=True, pause=False), execution=VelocityTarget(self.cfg),
            guard=CommandGuard(self.cfg), cadence=CandidateCadence(.02),
            source=object(), args=NS(execution_mode='integrated-velocity', output_socket='fake'),
            gripper_cmd=None, relay_active=False, output_latched=False, seq=1,
            get_clock=lambda: NS(now=lambda: NS(nanoseconds=int(self.now*1e9))))
        self.node.event = lambda kind, **fields: self.events.append(dict(event=kind, **fields))
        def latch(reason):
            self.node.output_latched = True
            self.node.event('relay_latched_closed', reason=reason)
        self.node.latch_output = latch
        def send(data, address):
            self.worker.packet(json.loads(data), self.node.feedback)
        self.node.socket = NS(sendto=send)

    def callback(self, now, *, velocity=.04, feedback_age=0.):
        self.now = now
        self.node.latest_input = self.node.latest_status_received = now
        self.node.feedback = Feedback(self.q.copy(), now-feedback_age, ready=True)
        msg = NS(header=NS(stamp=NS(sec=0, nanosec=0), frame_id='base_link'),
                 joint_names=list(NAMES), points=[NS(positions=self.q.tolist(),
                 velocities=[velocity, 0., 0., 0., 0., 0.],
                 time_from_start=NS(sec=0, nanosec=20_000_000))])
        with patch('servo_bridge.joint_runtime.time.monotonic', return_value=now):
            Runtime.trajectory(self.node, msg)
        if self.node.execution.position is not None:
            self.q = self.node.execution.position.copy()

    def test_recorded_30ms_then_9ms_burst_coalesces_without_output_or_integration(self):
        for now in (1., 1.020141, 1.040420, 1.070929):
            self.callback(now)
        q_before = self.node.execution.position.copy()
        self.callback(1.080187)  # Actual failing callback interval: ~9.3 ms.
        self.assertEqual(len(self.writes), 4)
        np.testing.assert_array_equal(self.node.execution.position, q_before)
        self.assertEqual(self.node.execution.last_time, 1.070929)
        self.assertEqual(self.events[-1]['event'], 'trajectory_coalesced')
        self.callback(1.100187)
        self.assertEqual(len(self.writes), 5)
        self.assertFalse(self.node.output_latched)
        self.assertEqual(self.node.invalid_count, 0)

    def test_real_timeout_still_latches(self):
        self.callback(1.)
        self.callback(1.2)
        self.assertTrue(self.node.output_latched)
        self.assertEqual(len(self.writes), 1)
        self.assertIn('interval=0.200000s', self.events[-1]['rejection'])

    def test_bad_candidate_or_stale_feedback_not_hidden_in_a_burst(self):
        for kwargs in (dict(velocity=10.), dict(feedback_age=.2)):
            with self.subTest(kwargs=kwargs):
                self.setUp()
                self.callback(1.)
                self.callback(1.009, **kwargs)
                self.assertTrue(self.node.output_latched)
                self.assertEqual(len(self.writes), 1)

    def test_invalid_jaw_feedback_is_audited_before_rejection(self):
        from rclpy.time import Time
        self.node.get_clock = lambda: NS(now=lambda: Time(seconds=1.))
        self.node.jaw_outside_model = 0
        snapshot = dict(gripper_feedback_raw_hex='0001906400004000',
                        frame_timestamps={'0x2a8': 1.}, gripper_enabled=True, gripper_faults=0)
        with patch('servo_bridge.joint_runtime.time.monotonic', return_value=1.), \
                self.assertRaisesRegex(ValueError, 'outside this 70 mm model'):
            Runtime.feedback_only(self.node, Feedback(self.q, 1., .1025, True, True), snapshot)
        self.assertEqual(self.events[-1]['event'], 'gripper_feedback_outside_model')
        self.assertEqual(self.events[-1]['gripper_feedback_raw_hex'], snapshot['gripper_feedback_raw_hex'])
        self.assertIsNone(self.events[-1]['previous_gripper_command'])
        self.assertEqual(self.writes, [])


if __name__ == '__main__':
    unittest.main()
