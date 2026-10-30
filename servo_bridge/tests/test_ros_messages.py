"""ROS message contracts without creating nodes, DDS sockets or hardware."""
import time
import json
from types import SimpleNamespace
import unittest

try:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Int8
    from servo_bridge.runtime import Runtime
except ImportError as exc:
    raise unittest.SkipTest('Source ROS Humble setup.bash for ROS-message tests') from exc

from servo_bridge.core import CommandGuard, Feedback, NAMES, load_config
from servo_bridge.sources import SimSource
from servo_bridge.execution import VelocityTarget


class MessageTests(unittest.TestCase):
    def setup(self, status=0):
        cfg = load_config()
        source = SimSource(cfg)
        feedback = Feedback(source.q.copy(), time.monotonic(), .04, True, True, 'fake')
        self.events = []
        self.node = SimpleNamespace(
            output_count=0, latest_status=status, latest_input=time.monotonic(),
            latest_status_received=time.monotonic(), cfg=cfg,
            feedback=feedback, guard=CommandGuard(cfg), source=source,
            socket=None, output_latched=False, seq=1, invalid_count=0,
            relay_active=False, args=SimpleNamespace(execution_mode="servo-position"),
            execution=VelocityTarget(cfg), policy=SimpleNamespace(active=False),
            event=lambda kind, **data: self.events.append({'event': kind, **data}),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=time.time_ns())))
        msg = JointTrajectory()
        msg.header.frame_id = 'base_link'
        msg.joint_names = list(NAMES)
        point = JointTrajectoryPoint()
        point.positions = source.q.tolist()
        point.velocities = [0.] * 6
        point.time_from_start.nanosec = 50000000
        msg.points = [point]
        return msg

    def test_live_guard_rejection_latches_and_requests_worker_stop_once(self):
        msg = self.setup()
        packets = []
        self.node.socket = SimpleNamespace(sendto=lambda data, address: packets.append(json.loads(data)))
        self.node.args = SimpleNamespace(output_socket='fake-local-socket', execution_mode='servo-position')
        self.node.policy = SimpleNamespace(active=True)
        self.node.latch_output = lambda reason: Runtime.latch_output(self.node, reason)
        msg.points[0].velocities = [.15] * 6
        Runtime.trajectory(self.node, msg)
        self.assertTrue(self.node.output_latched)
        self.assertEqual(len(packets), 1)
        self.assertIs(packets[0]['stop_requested'], True)
        self.assertNotIn('positions', packets[0])
        Runtime.trajectory(self.node, msg)
        self.assertEqual(len(packets), 1)

    def test_integrated_mode_smooths_raw_velocity_jump_before_final_guard(self):
        msg = self.setup()
        self.node.args.execution_mode = 'integrated-velocity'
        self.node.policy.active = True
        # A legal speed demand with an abrupt change: old ordering rejected
        # it before the execution layer could enforce acceleration.
        msg.points[0].velocities = [.06, 0., 0., 0., 0., 0.]
        Runtime.trajectory(self.node, msg)
        self.assertTrue(self.events[-1]['accepted'])
        velocity = self.events[-1]['execution_velocities'][0]
        self.assertLessEqual(velocity, .02 + 1e-10)
        self.assertGreater(velocity, 0.)

    def test_zero_stamp_means_immediate_but_requires_current_lease(self):
        msg = self.setup()
        Runtime.trajectory(self.node, msg)
        self.assertTrue(self.events[-1]['accepted'])
        self.node.latest_input -= 1.
        Runtime.trajectory(self.node, msg)
        self.assertFalse(self.events[-1]['accepted'])

    def test_servo_halt_never_reaches_simulated_or_sdk_sink(self):
        msg = self.setup(status=2)
        Runtime.trajectory(self.node, msg)
        self.assertFalse(self.events[-1]['accepted'])
        self.assertIsNone(self.node.guard.last_q)

    def test_stale_status_never_reaches_output(self):
        msg = self.setup()
        self.node.latest_status_received -= 1.
        Runtime.trajectory(self.node, msg)
        self.assertFalse(self.events[-1]['accepted'])

    def test_halt_status_requests_only_a_simulated_measured_hold(self):
        self.setup()
        self.node.source.target += .01
        Runtime.status(self.node, Int8(data=4))
        self.assertTrue((self.node.source.target == self.node.feedback.q).all())
        self.assertEqual(self.node.latest_status, 4)

    def test_multiple_points_cannot_silently_ignore_horizon(self):
        msg = self.setup()
        msg.points.append(msg.points[0])
        Runtime.trajectory(self.node, msg)
        self.assertFalse(self.events[-1]['accepted'])


if __name__ == '__main__':
    unittest.main()
