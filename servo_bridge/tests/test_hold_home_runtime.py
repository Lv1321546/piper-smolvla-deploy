"""Released/pending home input cannot forward an older Servo motion candidate."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from servo_bridge.cartesian_runtime import Runtime
except ImportError as exc:
    raise unittest.SkipTest('Requires ROS Humble message packages') from exc

from servo_bridge.candidate_cadence import CandidateCadence
from servo_bridge.core import CommandGuard, Feedback, NAMES, load_config
from servo_bridge.execution import VelocityTarget
from servo_bridge.sources import SimSource


class HoldHomeRuntimeTests(unittest.TestCase):
    def setUp(self):
        cfg = load_config(Path(__file__).resolve().parents[1]/'config.cartesian.example.json')
        source = SimSource(cfg)
        self.packets, self.events = [], []
        now = time.monotonic()
        self.node = SimpleNamespace(
            cfg=cfg, source=source, feedback=Feedback(source.q.copy(), now, .04, True, True),
            policy=SimpleNamespace(active=False, pause=False), latest_status=0,
            latest_status_received=now, latest_input=now, quiet_candidate_received=None,
            execution=VelocityTarget(cfg), guard=CommandGuard(cfg),
            cadence=CandidateCadence(cfg['publish_period_s']),
            socket=SimpleNamespace(sendto=lambda data, address:self.packets.append(json.loads(data))),
            args=SimpleNamespace(execution_mode='integrated-velocity', output_socket='fake'),
            output_latched=False, relay_active=False, geometry_paused=False,
            output_count=0, invalid_count=0, seq=1, gripper_cmd=None,
            event=lambda kind, **data:self.events.append(dict(event=kind, **data)),
            get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=time.time_ns())))
        self.node.latch_output = lambda reason:self.fail('Unexpected latch: '+reason)
        self.msg = JointTrajectory()
        self.msg.joint_names = list(NAMES)
        point = JointTrajectoryPoint()
        point.positions = source.q.tolist()
        point.velocities = [.04, 0., 0., 0., 0., 0.]
        point.time_from_start.nanosec = int(cfg['publish_period_s']*1e9)
        self.msg.points = [point]

    def test_stale_moving_candidate_after_release_only_holds_current_position(self):
        self.node.execution.position = self.node.source.q+.001
        self.node.execution.filtered_velocity[:] = .04
        Runtime.trajectory(self.node, self.msg)
        self.assertEqual(self.node.invalid_count, 0)
        self.assertEqual(len(self.packets), 1)
        self.assertFalse(self.packets[0]['deadman'])
        self.assertNotIn('gripper', self.packets[0])
        np.testing.assert_allclose(self.node.source.target, self.node.feedback.q, atol=1e-5)
        np.testing.assert_array_equal(self.node.execution.filtered_velocity, 0)
        self.assertIsNone(self.node.quiet_candidate_received)

    def test_fresh_zero_candidate_acknowledges_safe_home_transition(self):
        self.msg.points[0].velocities = [0.]*6
        Runtime.trajectory(self.node, self.msg)
        self.assertIsNotNone(self.node.quiet_candidate_received)
        self.assertFalse(self.packets[0]['deadman'])

    def test_nonzero_candidate_invalidates_earlier_quiet_acknowledgement(self):
        self.node.quiet_candidate_received = time.monotonic()
        Runtime.trajectory(self.node, self.msg)
        self.assertIsNone(self.node.quiet_candidate_received)

    def test_expired_input_cannot_acknowledge_home_transition(self):
        self.msg.points[0].velocities = [0.]*6
        self.node.latest_input -= 1
        Runtime.trajectory(self.node, self.msg)
        self.assertIsNone(self.node.quiet_candidate_received)
        self.assertEqual(self.packets, [])
        self.assertEqual(self.node.invalid_count, 1)


if __name__ == '__main__':
    unittest.main()
