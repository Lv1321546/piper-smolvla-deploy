"""Bounded position following for absolute PiPER actions; no IK or ROS required.

The model's 20 Hz future targets are pursued with velocity/acceleration limits.
The independent SDK guard checks the actual quantized joint command as well.
"""
import math
import time

import numpy as np

from shared.protocol import finite_number, vector7
from servo_bridge.core import Feedback, NAMES


class PositionFollower:
    def __init__(self, config):
        self.config = config
        robot, options = config['robot'], config['inference']
        self.period = 1 / finite_number(robot['control_hz'], 'control_hz', positive=True)
        self.bounds = np.array(robot['joint_limits_deg'] + [robot['gripper_limits_m']], float)
        if (self.bounds.shape != (7, 2) or not np.isfinite(self.bounds).all()
                or np.any(self.bounds[:, 0] >= self.bounds[:, 1])):
            raise ValueError('Seven ordered physical position bounds required')
        self.vmax = np.array(vector7(options['max_velocity'], 'inference.max_velocity'))
        self.amax = np.array(vector7(options['max_acceleration'], 'inference.max_acceleration'))
        self.lead = np.array(vector7(options['max_model_delta'], 'inference.max_model_delta'))
        self.tolerance = np.array(vector7(options['boundary_tolerance'], 'inference.boundary_tolerance'))
        if np.any(self.vmax <= 0) or np.any(self.amax <= 0) or np.any(self.lead <= 0) or np.any(self.tolerance < 0):
            raise ValueError('Positive velocity/acceleration/model delta and nonnegative tolerances required')
        self.position = self.velocity = None

    def checked_target(self, action, measured):
        action, measured = np.array(vector7(action, 'model action')), np.array(vector7(measured))
        if np.any(action < self.bounds[:, 0] - self.tolerance) or np.any(action > self.bounds[:, 1] + self.tolerance):
            raise ValueError('Model target exceeds physical limits plus boundary tolerance')
        target = np.clip(action, self.bounds[:, 0], self.bounds[:, 1])
        if np.any(abs(target - measured) > self.lead):
            raise ValueError('Model target is too far from current feedback')
        return target

    def reset(self, measured):
        q = np.array(vector7(measured))
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            raise ValueError('Initial measured state is outside command limits')
        self.position, self.velocity = q, np.zeros(7)

    def step(self, action, measured, *, dt=None):
        dt = self.period if dt is None else finite_number(dt, 'follower dt', positive=True)
        if not .5*self.period <= dt <= 2*self.period:
            raise ValueError('Follower cadence is too fast or stale')
        dt = min(dt, self.period)
        target = self.checked_target(action, measured)
        if self.position is None:
            self.reset(measured)
        error = target - self.position
        # Reserve margin for SDK quantization and the independent writer clock.
        acceleration = .8*self.amax
        # Exact discrete stopping distance includes this tick and every later
        # deceleration tick: dt*(k*v - a*dt*k*(k-1)/2). A continuous sqrt(2*a*d)
        # bound can overrun a fully closed jaw by one final control interval.
        decrement = acceleration*dt
        k = np.maximum(1., np.ceil((np.sqrt(1 + 8*abs(error)/(acceleration*dt*dt))-1)/2))
        braking = (abs(error)/dt + decrement*k*(k-1)/2)/k
        desired = np.sign(error) * np.minimum(.9*self.vmax, np.minimum(abs(error)/dt, braking))
        velocity = self.velocity + np.clip(desired-self.velocity,
                                           -acceleration*dt, acceleration*dt)
        position = self.position + velocity*dt
        # Remove floating-point roundoff only, never a physical limit violation.
        for side in (0, 1):
            close = np.isclose(position, self.bounds[:, side], atol=1e-12, rtol=0.)
            position[close] = self.bounds[close, side]
        if np.any(position < self.bounds[:, 0]) or np.any(position > self.bounds[:, 1]):
            raise ValueError('Bounded follower would cross a physical limit')
        self.position, self.velocity = position, velocity
        return position.copy(), velocity.copy(), target


def guard_config(config):
    follower = PositionFollower(config)
    robot, options = config['robot'], config['inference']
    return dict(can_interface=robot['can_interface'], publish_period_s=follower.period,
                feedback_max_age_s=robot['feedback_max_age_ms']/1000,
                command_max_age_s=2*follower.period,
                maximum_tracking_error_deg=finite_number(options['max_tracking_error_deg'], positive=True),
                maximum_joint_step_deg=float(max(follower.vmax[:6])*follower.period + .002),
                maximum_joint_velocity_rad_s=math.radians(float(max(follower.vmax[:6]))) + .001,
                maximum_joint_acceleration_rad_s2=math.radians(float(max(follower.amax[:6]))) + .001,
                joint_limits_deg=robot['joint_limits_deg'], gripper_limits_m=robot['gripper_limits_m'],
                gripper_effort=robot['gripper_effort'])


def select_position_mode(writer, snapshot, config):
    """Replace any previous goal before selecting the commissioned MOVE_J speed.

    Called only by explicit real execution, under the shared writer lock.
    Selecting a known speed matters even if another tool already selected CAN.
    """
    speed = config['robot']['speed_percent']
    if isinstance(speed, bool) or not isinstance(speed, int) or not 1 <= speed <= 100:
        raise ValueError('robot.speed_percent must be an integer 1..100')
    cfg = guard_config(config)
    from servo_bridge.prepare_control import inspect_snapshot
    if not inspect_snapshot(snapshot, cfg, now=time.monotonic()):
        raise RuntimeError('CAN/MOVE_J must be ready before selecting policy output mode')
    writer.joint([round(x*1000) for x in snapshot['state'][:6]])
    writer.sdk.MotionCtrl_2(0x01, 0x01, speed, 0x00)


class PolicyOutput:
    """Reuse the audited raw SDK writer/guard/hold, with a seven-axis limiter."""
    def __init__(self, config, writer, *, clock=time.monotonic):
        from servo_bridge.sdk_worker import Worker
        self.config, self.writer, self.clock = config, writer, clock
        self.cfg = guard_config(config)
        self.holder = Worker(self.cfg, writer, clock=clock)
        self.last_width = self.last_time = None

    def send(self, position, velocity, feedback, snapshot):
        position, velocity = np.array(vector7(position)), np.array(vector7(velocity))
        now = self.clock()
        if not feedback.ready or snapshot.get('gripper_faults') or not snapshot.get('gripper_enabled'):
            raise RuntimeError('Policy execution needs fresh enabled CAN/MOVE_J and gripper feedback')
        raw_gripper = self.holder._gripper_command({'gripper': {'width_m': float(position[6])}})
        width = raw_gripper[0] / 1e6
        previous = feedback.width if self.last_width is None else self.last_width
        dt = self.cfg['publish_period_s'] if self.last_time is None else min(now-self.last_time, self.cfg['publish_period_s'])
        if abs(width - previous) > self.config['inference']['max_velocity'][6]*dt + 2e-6:
            raise ValueError('Gripper SDK target exceeds velocity limit')
        if abs(width - feedback.width) > self.config['inference']['max_model_delta'][6]:
            raise ValueError('Gripper tracking difference exceeds limit')
        raw = self.holder.guard.validate(list(NAMES), np.radians(position[:6]), np.radians(velocity[:6]),
                    feedback=feedback, stamp_age_s=0., horizon_s=self.cfg['publish_period_s'], now=now)
        self.holder.moved = True  # A partially failed write also needs a hold attempt.
        self.writer.joint(raw)
        self.writer.gripper(*raw_gripper)
        self.last_width, self.last_time = width, now

    def hold(self, feedback):
        # Joint position hold only. Leave the existing jaw command intact;
        # never open, disable, reset or switch to the damped-falling stop mode.
        self.holder.hold(feedback)

    def resume(self, follower, measured):
        """Restart joints from measured hold, retaining the jaw's last target.

        A real jaw can settle with a small encoder offset. Keep its command
        reference so resuming cannot jump the jaw goal back to that reading.
        """
        if self.holder.moved:
            raise RuntimeError('Resume requires a successful position hold first')
        reference = vector7(measured)
        if self.last_width is not None:
            reference[6] = self.last_width
        follower.reset(reference)
        self.last_time = None


class SimulatedArm:
    """Synthetic test plant/writer; not a robot-dynamics or task-success test."""
    def __init__(self, config):
        self.config = config
        self.state = np.array(config['inference'].get('mock_initial_state', [0.,80.,-60.,0.,40.,0.,.04]), float)
        self.commands = []

    def sample(self, now, dt):
        feedback = Feedback(np.radians(self.state[:6]), now, float(self.state[6]), True, True, 'simulated_policy')
        snapshot = dict(state=self.state.tolist(), feedback_monotonic=now, ctrl_mode=1, mode_feed=1,
                        arm_status=0, err_code=0, motor_faults=[0]*6, gripper_faults=0,
                        gripper_enabled=True, enabled=[True]*6)
        return feedback, None, snapshot

    def joint(self, raw):
        self.commands.append(('joint', list(raw)))
        self.state[:6] = np.array(raw)/1000.

    def gripper(self, raw, effort):
        self.commands.append(('gripper', raw))
        self.state[6] = raw/1e6

    def close(self):
        pass
