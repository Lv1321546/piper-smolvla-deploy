"""Run the real Humble Servo binary with an audit or synthetic output sink.

    /usr/bin/python3 -m servo_bridge.runtime --source sim --duration 18

No SDK is imported by this process. CAN feedback is passive. Real output
requires a separately started, manually armed sdk_worker, never auto-spawned.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive

from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import CommandGuard, NAMES, STATUS, load_config
from servo_bridge.sources import LogSource, SimSource, PassiveCan, synthetic_pad
from servo_bridge.twist_policy import TwistPolicy


class Runtime(Node):
    def __init__(self, args, cfg, source, file):
        super().__init__('piper_servo_audit')
        self.args, self.cfg, self.source, self.file = args, cfg, source, file
        self.kin = PiperKinematics(cfg['joint_limits_deg'])
        mapping = source.session['gamepad'] if args.source == 'log' else {}
        self.pad = None
        if args.input == 'gamepad':
            from e14.gamepad import PygamePad
            mapping = json.loads(Path(args.gamepad_config).read_text()).get('gamepad', {})
            self.pad = PygamePad(mapping)
        self.policy = TwistPolicy(cfg, mapping)
        self.guard = CommandGuard(cfg)
        self.joints_pub = self.create_publisher(JointState, '/piper_servo/joint_states', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/piper_servo/twist', 10)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        self.create_subscription(JointTrajectory, '/piper_servo/candidate_trajectory', self.trajectory, 1)
        self.create_subscription(Int8, '/piper_servo/status', self.status, 10)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.feedback = None
        self.latest_status = None
        self.latest_status_received = None
        self.latest_input = None
        self.seq = 0
        self.started = None
        self.output_count = 0
        self.invalid_count = 0
        self.jaw_outside_model = 0
        self.joint_history = []
        self.socket = None
        self.output_latched = False
        self.relay_active = False
        if args.output_socket:
            if args.source != 'can' or args.input != 'gamepad':
                raise ValueError('SDK relay is only allowed with passive CAN feedback + live gamepad; no log/synthetic input')
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.hardware_output = 'external manually armed worker requested; not confirmed' if self.socket else False
        self.event('session', source=args.source, input=args.input, hardware_output=self.hardware_output,
                   sdk_socket_requested=bool(args.output_socket), config=cfg,
                   note='Candidate trajectory is not hardware execution or training data')

    def event(self, kind, **kwargs):
        self.file.write(json.dumps({'event': kind, 'monotonic': time.monotonic(), **kwargs}, allow_nan=False) + '\n')

    def status(self, msg):
        self.latest_status = int(msg.data)
        self.latest_status_received = time.monotonic()
        self.event('servo_status', value=int(msg.data), label=STATUS.get(msg.data, 'UNKNOWN'))
        if self.latest_status in (2, 4, 5):
            if isinstance(self.source, SimSource) and self.feedback is not None:
                self.source.target = self.feedback.q.copy()
            if self.socket:
                self.latch_output(STATUS.get(self.latest_status, 'Servo halt'))

    def latch_output(self, reason):
        """A rejected live command ends this test; never silently skip/recover."""
        if self.output_latched:
            return
        self.output_latched = True
        self.event('relay_latched_closed', reason=reason)
        print(f'SDK relay STOPPED: {reason}. Output latched closed; inspect logs before restarting.', flush=True)
        try:
            self.socket.sendto(json.dumps({'version': 1, 'monotonic': time.monotonic(),
                                          'stop_requested': True, 'reason': reason}).encode(), self.args.output_socket)
        except OSError as exc:
            self.event('worker_stop_notification_failed', detail=str(exc))

    def trajectory(self, msg):
        before = time.perf_counter()
        self.output_count += 1
        reason, raw = None, None
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        # Humble Servo deliberately uses stamp=0 ("begin immediately"). It is
        # not a Unix-epoch-stale command. Require a current input lease and a
        # fresh measured state instead; do not invent an output creation time.
        age = ((self.get_clock().now().nanoseconds - stamp) / 1e9 if stamp else
               time.monotonic() - self.latest_input if self.latest_input is not None else math.inf)
        point = msg.points[0] if len(msg.points) == 1 else None
        horizon = point.time_from_start.sec + point.time_from_start.nanosec / 1e9 if point else 0.
        q = list(point.positions) if point else []
        v = list(point.velocities) if point else []
        try:
            if self.socket and self.output_latched:
                raise ValueError('Hardware relay is latched closed; restart only after supervised recovery')
            if self.latest_status not in (0, 1, 6):
                raise ValueError('Servo halt/collision/joint-bound or status not yet available')
            if self.latest_status_received is None or time.monotonic() - self.latest_status_received > self.cfg['command_max_age_s']:
                raise ValueError('Servo status feedback is stale')
            if point is None:
                raise ValueError('Expected one Servo trajectory point')
            if msg.header.frame_id not in ('', 'base_link'):
                raise ValueError('Unexpected trajectory frame')
            raw = self.guard.validate(list(msg.joint_names), q, v, feedback=self.feedback,
                                      stamp_age_s=age, horizon_s=horizon)
            if isinstance(self.source, SimSource):
                self.source.target = self.guard.last_q.copy()
            if self.socket:
                # The worker independently checks real CAN, freshness, expiry,
                # health, names, limits and the ongoing deadman lease.
                data = {'version': 1, 'monotonic': time.monotonic(), 'joint_names': list(NAMES),
                        'positions': self.guard.last_q.tolist(), 'velocities': self.guard.last_velocity.tolist(),
                        'deadman': bool(self.policy.active), 'servo_status': self.latest_status,
                        'status_monotonic': self.latest_status_received,
                        'horizon_s': horizon}
                self.socket.sendto(json.dumps(data).encode(), self.args.output_socket)
                self.relay_active = bool(self.policy.active)
        except (ValueError, OSError) as exc:
            reason = str(exc)
            self.invalid_count += 1
            if self.socket and (self.policy.active or self.relay_active):
                self.latch_output(reason)
        self.event('trajectory', input_seq=self.seq, status=self.latest_status,
                   names=list(msg.joint_names), positions=q, velocities=v,
                   stamp_age_s=age, horizon_s=horizon, accepted=reason is None,
                   age_basis='header_stamp' if stamp else 'latest_input_lease; output_stamp_zero_begin_immediately',
                   rejection=reason, sdk_units_audit=raw,
                   processing_ms=(time.perf_counter() - before) * 1000,
                   input_to_callback_ms=(time.monotonic() - self.latest_input) * 1000 if self.latest_input else None)

    def feedback_only(self, feedback):
        self.feedback = feedback
        joint = JointState()
        age = time.monotonic() - feedback.received
        joint.header.stamp = (self.get_clock().now() - rclpy.duration.Duration(nanoseconds=int(age * 1e9))).to_msg()
        joint.header.frame_id = 'base_link'
        joint.name = list(NAMES) + ['joint7', 'joint8']
        # The pinned gripper model spans 70 mm, hardware can report ~100 mm.
        # Do not silently remap this discrepancy. Affected input is rejected.
        if not 0 <= feedback.width <= .07:
            self.jaw_outside_model += 1
            raise ValueError(f'Jaw feedback {feedback.width:g} m is outside this 70 mm model; calibrate/replace model')
        joint.position = feedback.q.tolist() + [feedback.width / 2, -feedback.width / 2]
        joint.velocity = [0.] * 8
        self.joints_pub.publish(joint)
        self.joint_history.append((joint.header.stamp.sec * 10**9 + joint.header.stamp.nanosec, feedback.q.copy()))
        self.joint_history = self.joint_history[-100:]

    def tick(self, dt, elapsed):
        started = time.perf_counter()
        feedback, pad, original = self.source.sample(time.monotonic(), dt)
        if time.monotonic() - feedback.received > self.cfg['feedback_max_age_s']:
            raise RuntimeError('Measured feedback is stale; no new twist is published')
        self.feedback_only(feedback)
        if self.pad:
            pad = asdict(self.pad.poll())
            if not 0 <= time.monotonic() - pad['timestamp'] <= self.cfg['command_max_age_s']:
                raise RuntimeError('Gamepad input is stale')
        elif pad is None:
            pad = synthetic_pad(elapsed, self.args.scenario) if self.args.input == 'synthetic' else {
                'connected': True, 'instance': 0, 'buttons': {'lb': False}, 'axes': {}, 'hat': [0, 0]}
        pose, jac = self.kin.forward(feedback.q, jacobian=True)
        was_active = self.policy.active
        twist = self.policy.step(pad, pose[:3, :3], dt,
                                 reference_may_advance=self.latest_status in (0, 1, 6))
        if self.socket and was_active and not self.policy.active and not self.output_latched:
            # Release must not depend on a decelerating Servo candidate passing
            # the trajectory guard. The worker owns the commissioned hold.
            try:
                self.socket.sendto(json.dumps({'version': 1, 'monotonic': time.monotonic(),
                                              'servo_status': self.latest_status,
                                              'status_monotonic': self.latest_status_received,
                                              'deadman': False}).encode(), self.args.output_socket)
                self.relay_active = False
                self.guard = CommandGuard(self.cfg)
                self.event('relay_deadman_released')
            except OSError as exc:
                self.latch_output(str(exc))
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = map(float, twist[:3])
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = map(float, twist[3:])
        input_published = not (self.args.scenario == 'timeout' and elapsed >= 3.)
        if input_published:
            self.twist_pub.publish(msg)
            self.latest_input = time.monotonic()
        self.seq += 1
        tf_position_error = tf_angle_error = None
        try:
            tf = self.tf.lookup_transform('base_link', 'link6', rclpy.time.Time())
            translation, rot = tf.transform.translation, tf.transform.rotation
            # TF is asynchronous. Compare it with the joint state carrying the
            # same ROS timestamp, never with a later cycle's measurement.
            tf_ns = tf.header.stamp.sec * 10**9 + tf.header.stamp.nanosec
            same_stamp = [q for ns, q in self.joint_history if ns == tf_ns]
            if same_stamp:
                tf_pose = self.kin.forward(same_stamp[-1])
                tf_position_error = np.linalg.norm(np.array([translation.x, translation.y, translation.z]) - tf_pose[:3, 3])
                tf_angle_error = np.linalg.norm(Rotation.from_matrix(
                    Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]).inv().as_matrix() @ tf_pose[:3, :3]).as_rotvec())
        except Exception:
            pass
        error_angle = np.linalg.norm(Rotation.from_matrix(self.policy.orientation @ pose[:3, :3].T).as_rotvec()) if self.policy.orientation is not None else None
        self.event('input', seq=self.seq, elapsed_s=elapsed, feedback_source=feedback.source,
                   measured_q_rad=feedback.q.tolist(), measured_pose=pose.tolist(), jaw_width_m=feedback.width,
                   feedback_age_s=time.monotonic() - feedback.received, twist=twist.tolist(),
                   orientation_goal=self.policy.orientation.tolist() if self.policy.orientation is not None else None,
                   orientation_error_rad=float(error_angle) if error_angle is not None else None,
                   jacobian_condition=float(np.linalg.cond(jac)), status=self.latest_status,
                   active=self.policy.active, original_monotonic=original.get('monotonic') if original else None,
                   original_ik_accepted=original.get('planner', {}).get('accepted') if original else None,
                   input_published=input_published,
                   tf_latest_position_gap_m=float(tf_position_error) if tf_position_error is not None else None,
                   tf_same_stamp_angle_error_rad=float(tf_angle_error) if tf_angle_error is not None else None,
                   processing_ms=(time.perf_counter() - started) * 1000)

    def scene(self):
        # Simulated table only; not a calibrated real table. Keep self collision
        # active in every run, even where the physical scene is unavailable.
        if self.args.source != 'sim':
            return
        scene = PlanningScene()
        scene.is_diff = True
        obj = CollisionObject()
        obj.header.frame_id = 'base_link'
        obj.id = 'synthetic_table'
        obj.operation = CollisionObject.ADD
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [2., 2., .05]
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.orientation.w = 1.
        pose.position.z = self.cfg['sim_table_top_z_m'] - .025
        obj.primitives, obj.primitive_poses = [box], [pose]
        scene.world.collision_objects = [obj]
        self.scene_pub.publish(scene)
        if self.args.scenario == 'collision':
            obstacle = CollisionObject()
            obstacle.header.frame_id = 'base_link'
            obstacle.id = 'deliberate_test_obstacle'
            obstacle.operation = CollisionObject.ADD
            shape = SolidPrimitive()
            shape.type = SolidPrimitive.BOX
            shape.dimensions = [.10, .10, .10]
            at = Pose()
            at.orientation.w = 1.
            xyz = self.kin.forward(self.source.q)[:3, 3]
            at.position.x, at.position.y, at.position.z = map(float, xyz)
            obstacle.primitives, obstacle.primitive_poses = [shape], [at]
            scene.world.collision_objects = [obstacle]
            self.scene_pub.publish(scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model-dir', default='models/piper_servo')
    parser.add_argument('--source', choices=('log', 'sim', 'can'), default='sim')
    parser.add_argument('--log')
    parser.add_argument('--input', choices=('recorded', 'synthetic', 'gamepad', 'none'))
    parser.add_argument('--gamepad-config', default='gamepad.local.teleop.json')
    parser.add_argument('--scenario', choices=('xyz', 'x', 'singularity', 'collision', 'timeout'), default='xyz')
    parser.add_argument('--duration', type=float, default=18.)
    parser.add_argument('--max-cycles', type=int)
    parser.add_argument('--output', default='outputs/servo_bridge/run')
    parser.add_argument('--output-socket', help='Explicit future relay to a separately manually armed SDK worker')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('duration must be positive and finite')
    args.input = args.input or {'log': 'recorded', 'sim': 'synthetic', 'can': 'none'}[args.source]
    if args.source == 'can' and args.input in ('recorded', 'synthetic'):
        parser.error('CAN feedback cannot use synthetic/recorded input')
    if args.source == 'log' and (not args.log or args.input != 'recorded'):
        parser.error('--source log requires --log and recorded input')
    if args.output_socket and (args.source != 'can' or args.input != 'gamepad'):
        parser.error('SDK relay requires passive CAN feedback + live gamepad; no log/synthetic source')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    params = Path(args.model_dir).resolve() / 'params.yaml'
    if not params.is_file():
        parser.error('Generate model parameters with servo_bridge.prepare first')
    os.environ.setdefault('ROS_DOMAIN_ID', '83')
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    os.environ['ROS_LOG_DIR'] = str(output / 'ros_logs')
    # Prevent two instances of this integration from mixing simulated/log
    # feedback with a live bridge in the same ROS domain.
    domain_lock = open(f"/tmp/piper-servo-domain-{os.environ['ROS_DOMAIN_ID']}.lock", 'a')
    try:
        fcntl.flock(domain_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another Servo bridge already owns this ROS domain; stop it or use a different ROS_DOMAIN_ID')
    source = LogSource(args.log) if args.source == 'log' else SimSource(cfg) if args.source == 'sim' else PassiveCan(cfg['can_interface'])
    processes, handles = [], []
    from contextlib import nullcontext
    event_file = (output / 'events.jsonl').open('w')
    rclpy.init()
    node = None
    try:
        # Keep the log open through interrupt handling and child cleanup.
        with nullcontext(event_file) as file:
            node = Runtime(args, cfg, source, file)
            for executable, name, extra in (
                ('/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher', 'robot_state_publisher', ['-r', 'joint_states:=/piper_servo/joint_states']),
                ('/opt/ros/humble/lib/moveit_servo/servo_node_main', 'servo_node', [])):
                handle = (output / f'{name}.log').open('w')
                handles.append(handle)
                processes.append(subprocess.Popen([executable, '--ros-args', '-r', f'__node:={name}',
                                                     '--params-file', str(params), *extra], stdout=handle,
                                                    stderr=subprocess.STDOUT, start_new_session=True))
            client = node.create_client(Trigger, '/servo_node/start_servo')
            boot = time.monotonic()
            initial = None
            while time.monotonic() - boot < 15.:
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('A ROS process exited at startup; inspect its log')
                try:
                    if initial is None or args.source == 'can':
                        if isinstance(source, LogSource):
                            saved = source.index
                            initial = source.sample(time.monotonic(), cfg['publish_period_s'])[0]
                            source.index = saved
                        else:
                            initial = source.sample(time.monotonic(), cfg['publish_period_s'])[0]
                    initial.received = time.monotonic() if args.source != 'can' else initial.received
                    node.feedback_only(initial)
                except RuntimeError:
                    pass
                rclpy.spin_once(node, timeout_sec=.02)
                if client.service_is_ready() and time.monotonic() - boot > 3.:
                    break
            if not client.service_is_ready():
                raise RuntimeError('Servo start service unavailable (15s)')
            for topic in ('/piper_servo/joint_states', '/piper_servo/twist',
                          '/piper_servo/status', '/piper_servo/candidate_trajectory'):
                if node.count_publishers(topic) != 1:
                    raise RuntimeError(f'{topic}: expected exactly one publisher; refusing mixed sources')
            future = client.call_async(Trigger.Request())
            while not future.done() and time.monotonic() - boot < 20.:
                node.feedback_only(initial)
                rclpy.spin_once(node, timeout_sec=.01)
            if not future.done() or not future.result().success:
                raise RuntimeError('Servo did not confirm startup')
            node.scene()
            started = next_tick = time.monotonic()
            print(f'MoveIt Servo started. Source={args.source}; hardware output={node.hardware_output}. Logs: {output}', flush=True)
            hz_ticks = os.sysconf('SC_CLK_TCK')
            def cpu_seconds(process):
                fields = Path(f'/proc/{process.pid}/stat').read_text().split()
                return (int(fields[13]) + int(fields[14])) / hz_ticks
            cpu_before = [cpu_seconds(p) for p in processes]
            while rclpy.ok():
                now = time.monotonic()
                if args.source == 'log':
                    done = source.index >= len(source.rows)
                else:
                    done = now - started >= args.duration
                if done or args.max_cycles and node.seq >= args.max_cycles:
                    break
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError('ROS process exited during run')
                if now >= next_tick:
                    # Logs are replayed at the recorded ~20 Hz schedule; sensor
                    # readings remain exogenous, never modified by candidate output.
                    dt = cfg['publish_period_s']
                    node.tick(dt, now - started)
                    if isinstance(source, LogSource) and source.index < len(source.rows):
                        next_tick = started + source.rows[source.index]['monotonic'] - source.rows[0]['monotonic']
                    else:
                        next_tick += dt
                    if now - next_tick > dt:
                        node.event('scheduler_overrun', lag_s=now - next_tick)
                        next_tick = now + dt
                rclpy.spin_once(node, timeout_sec=.002)
            node.event('finished', cycles=node.seq, trajectories=node.output_count,
                       rejected=node.invalid_count, hardware_output=node.hardware_output,
                       child_cpu_s={name: cpu_seconds(p) - before for name, p, before in zip(
                           ('robot_state_publisher', 'servo_node'), processes, cpu_before)})
            print(f'Finished: {node.seq} inputs; {node.output_count} candidates; {node.invalid_count} rejected. Hardware output={node.hardware_output}.', flush=True)
    except KeyboardInterrupt:
        if node:
            node.event('interrupted', hardware_output=node.hardware_output)
    finally:
        # No actuator cleanup commands. These are only local ROS processes.
        if node:
            if node.pad:
                node.pad.close()
            if node.socket:
                node.socket.close()
            node.destroy_node()
        source.close()
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=3.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3.)
        for handle in handles:
            handle.close()
        rclpy.try_shutdown()
        event_file.close()
        domain_lock.close()


if __name__ == '__main__':
    main()
