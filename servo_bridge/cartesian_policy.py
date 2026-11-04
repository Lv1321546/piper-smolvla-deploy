"""Bounded, position-priority Cartesian velocity control (no ROS or SDK).

Solve a six-variable damped least-squares task with joint velocity,
acceleration, braking-distance and reference-lead bounds. Orientation is soft
inside an explicit envelope; it is never allowed to drift without a bound.
The downstream Servo still owns collision checking and trajectory smoothing.
"""
import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import vector
from servo_bridge.joint_policy import JointPolicy


class CartesianPolicy:
    def __init__(self, cfg, mapping=None):
        self.cfg, self.mapping = cfg, mapping or {}
        self.gate = JointPolicy(cfg, mapping)
        self.kin = PiperKinematics(cfg['joint_limits_deg'])
        self.mode = cfg.get('cartesian_orientation_mode', 'position')
        if self.mode not in ('position', 'hold'):
            raise ValueError('cartesian_orientation_mode must be position or hold')
        defaults = dict(cartesian_orientation_drift_rad=math.radians(15),
                        cartesian_hold_drift_rad=math.radians(2),
                        cartesian_joint_margin_rad=.10, cartesian_lookahead_s=.20,
                        cartesian_damping=.025, cartesian_orientation_weight=.15,
                        cartesian_velocity_smoothing_weight=.05)
        for key, default in defaults.items():
            value = cfg.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (float, int)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be positive and finite')
            setattr(self, key.removeprefix('cartesian_'), float(value))
        if self.joint_margin_rad <= .05 or np.any(self.kin.high - self.kin.low <= 2*self.joint_margin_rad):
            raise ValueError('Cartesian joint margin must exceed Servo margin and leave a usable range')
        self.fine_scale = cfg.get('fine_speed_scale', .5)
        if (isinstance(self.fine_scale, bool) or not isinstance(self.fine_scale, (int, float)) or
                not math.isfinite(self.fine_scale) or not 0 < self.fine_scale <= 1):
            raise ValueError('fine_speed_scale must be in (0, 1]')
        self.orientation = None
        self.previous = np.zeros(6)
        self.twist = np.zeros(6)
        self.diagnostics = {}
        self.pause = False

    @property
    def active(self):
        return self.gate.active

    @property
    def faulted(self):
        return self.gate.faulted

    def reset_motion(self):
        self.previous[:] = 0
        self.twist[:] = 0

    def step(self, pad, measured_rotation, dt, *, reference_may_advance=True,
             measured_q=None, reference_q=None):
        was_active = self.active
        self.gate.step(pad, measured_rotation, dt)  # Same validated LB/RB/Start handshake as joint demo.
        self.pause = False
        self.diagnostics = {'orientation_mode': self.mode, 'message': '松开 LB，输入回中后再按住 LB。'}
        if not self.active:
            self.reset_motion()
            self.orientation = None
            if self.faulted:
                self.diagnostics['message'] = 'Start/断连/设备变更已锁定输入；请退出并重新启动。'
            return np.zeros(6)
        q = vector(measured_q)
        reference = q if reference_q is None else vector(reference_q)
        if np.any(q < self.kin.low) or np.any(q > self.kin.high):
            raise ValueError('Measured Cartesian joints outside model bounds')
        pose, jac = self.kin.forward(q, jacobian=True)
        rotation = pose[:3, :3]
        if not was_active or self.orientation is None:
            self.orientation = rotation.copy()
        axes, buttons = pad.get('axes', {}), pad.get('buttons', {})
        signs = self.mapping.get('input_directions', {})
        lx, ly, rx, ry = [axes.get(k, 0.) * signs.get(k, 1)
                          for k in ('left_x', 'left_y', 'right_x', 'right_y')]
        hx, hy = np.asarray(pad.get('hat', (0, 0))) * [signs.get('hat_x', 1), signs.get('hat_y', 1)]
        fine = self.fine_scale if buttons.get('rb') else 1.
        request = np.r_[[-ly, -lx if self.mapping.get('invert_y', True) else lx, -ry],
                        [hx, -hy if self.mapping.get('invert_pitch', True) else hy, rx]]
        request[:3] *= self.cfg['linear_speed_m_s'] * fine
        request[3:] *= self.cfg['angular_speed_rad_s'] * fine
        for part, limit in ((slice(0, 3), self.cfg['linear_speed_m_s']*fine),
                            (slice(3, 6), self.cfg['angular_speed_rad_s']*fine)):
            request[part] *= min(1., limit / max(np.linalg.norm(request[part]), 1e-12))
        acceleration = np.r_[[self.cfg['linear_acceleration_m_s2']]*3,
                             [self.cfg['angular_acceleration_rad_s2']]*3]
        self.twist += np.clip(request-self.twist, -acceleration*dt, acceleration*dt)
        neutral_motion = np.linalg.norm(request) < 1e-10
        # Brake directly in joint velocity space. An IK solve at zero input
        # can redistribute the remaining speed between joints and twitch.
        if neutral_motion:
            velocity = self.previous + np.clip(-self.previous,
                -self.gate.acceleration*dt, self.gate.acceleration*dt)
            direction_state = np.where(velocity >= 0, np.maximum(q, reference), np.minimum(q, reference))
            coast = velocity*self.cfg.get('execution_velocity_time_constant_s', .15)
            stopping = direction_state + velocity*abs(velocity)/(2*self.gate.acceleration) + coast
            blocked = ((velocity > 0) & (stopping > self.kin.high-self.joint_margin_rad) |
                       (velocity < 0) & (stopping < self.kin.low+self.joint_margin_rad))
            if np.any(blocked):
                self.reset_motion()
                return self._pause('停止距离接近边界，保持当前位置。',
                                   [f'joint{i+1}' for i in np.flatnonzero(blocked)])
            self.previous = velocity.copy()
            self.twist[:] = 0
            if self.mode == 'position':
                self.orientation = rotation.copy()
            self.diagnostics.update(message='输入回中，平滑停止并保持当前位置。',
                requested_twist=request.tolist(), achieved_twist=(jac@velocity).tolist(),
                constrained=False, braking=bool(np.any(velocity)))
            return velocity
        envelope = self.orientation_drift_rad if self.mode == 'position' else self.hold_drift_rad
        if reference_may_advance and np.linalg.norm(request[3:]) > 1e-8:
            proposed = Rotation.from_rotvec(self.twist[3:]*dt).as_matrix() @ self.orientation
            error = Rotation.from_matrix(proposed @ rotation.T).as_rotvec()
            # Do not accumulate an unreachable orientation target.
            if np.linalg.norm(error) <= envelope:
                self.orientation = proposed
        error = Rotation.from_matrix(self.orientation @ rotation.T).as_rotvec()
        output = self.solve(q, reference, jac, self.twist, error, envelope, dt,
                            neutral=neutral_motion, angular_limit=self.cfg['angular_speed_rad_s']*fine,
                            joint_speed_limit=self.gate.fine_speed if buttons.get('rb') else self.gate.speed)
        if self.pause:
            self.reset_motion()
        else:
            self.previous = output.copy()
        return output

    def solve(self, q, reference, jac, twist, error, envelope, dt, *, neutral=False,
              angular_limit=None, joint_speed_limit=None):
        self.pause = False
        acceleration = self.gate.acceleration
        speed = self.gate.speed if joint_speed_limit is None else joint_speed_limit
        speed = max(speed, np.max(abs(self.previous))-acceleration*dt)
        lower_position = self.kin.low + self.joint_margin_rad
        upper_position = self.kin.high - self.joint_margin_rad
        distance_low = np.maximum(0., np.minimum(q, reference)-lower_position)
        distance_high = np.maximum(0., upper_position-np.maximum(q, reference))
        # Include the downstream velocity filter's coasting time in the soft
        # boundary. The independent hard position and collision guards remain.
        lookahead = max(self.lookahead_s,
                        2*self.cfg.get('execution_velocity_time_constant_s', .15)+2*dt)
        low = np.maximum.reduce([np.full(6, -speed),
                                 -np.sqrt(2*acceleration*distance_low),
                                 -distance_low/lookahead,
                                 self.previous-acceleration*dt])
        high = np.minimum.reduce([np.full(6, speed),
                                  np.sqrt(2*acceleration*distance_high),
                                  distance_high/lookahead,
                                  self.previous+acceleration*dt])
        limiting = [f'joint{i+1}' for i in range(6)
                    if min(distance_low[i], distance_high[i]) < speed*self.lookahead_s]
        if np.any(low > high + 1e-10):
            return self._pause('边界附近需要保持；可反向操作退回。', limiting)
        high = np.maximum(high, low)
        orientation_weight = 2. if self.mode == 'hold' else self.orientation_weight
        # The manual angular request is not multiplied into translation. A
        # higher heading weight is useful only when the operator asks to rotate.
        if np.linalg.norm(twist[3:]) > 1e-6:
            orientation_weight = max(orientation_weight, 1.)
        if neutral:
            desired = np.zeros(6)
        else:
            desired = twist.copy()
            if self.mode == 'hold' or np.linalg.norm(twist[3:]) > 1e-6:
                correction = error.copy()
                if np.linalg.norm(correction) <= self.cfg['orientation_hold_deadband_rad']:
                    correction[:] = 0
                desired[3:] += correction * self.cfg['orientation_hold_gain_s_inv']
        linear_scale = self.cfg['linear_speed_m_s']
        angular_scale = self.cfg['angular_speed_rad_s']
        task = np.vstack([jac[:3]*speed/linear_scale,
                          orientation_weight*jac[3:]*speed/angular_scale])
        goal = np.r_[desired[:3]/linear_scale, orientation_weight*desired[3:]/angular_scale]
        # Regularization keeps the inverse bounded even when J loses rank.
        # A small centering preference only acts during requested movement.
        center = (self.kin.low+self.kin.high)/2
        preference = np.clip((center-q)/(self.kin.high-self.kin.low), -.5, .5) * .2
        if neutral:
            preference[:] = 0
        matrix = np.vstack([task, self.damping*np.eye(6)])
        target = np.r_[goal, self.damping*preference]
        # Prefer a continuous joint solution when several joint combinations
        # produce the same requested Cartesian motion.
        continuity = self.velocity_smoothing_weight
        hessian = matrix.T @ matrix + continuity*np.eye(6)
        rhs = matrix.T @ target + continuity*(self.previous/speed)
        horizon = max(dt, self.lookahead_s)
        angular_prediction = jac[3:] * speed * horizon
        permitted = max(envelope, float(np.linalg.norm(error)))

        def orientation_constraint(u):
            return permitted - np.linalg.norm(error-angular_prediction@u)

        def orientation_gradient(u):
            residual = error-angular_prediction@u
            return residual @ angular_prediction / max(np.linalg.norm(residual), 1e-12)

        angular_map = jac[3:]*speed
        angular_limit = self.cfg['angular_speed_rad_s'] if angular_limit is None else angular_limit
        # Gear changes decelerate within the configured angular acceleration.
        angular_limit = max(angular_limit, np.linalg.norm(jac[3:]@self.previous)
                            - self.cfg['angular_acceleration_rad_s2']*dt)
        constraints = [dict(type='ineq', fun=orientation_constraint, jac=orientation_gradient),
                       dict(type='ineq', fun=lambda u: angular_limit-np.linalg.norm(angular_map@u),
                            jac=lambda u: -(angular_map@u)@angular_map /
                            max(np.linalg.norm(angular_map@u), 1e-12))]
        requested_norm = float(np.linalg.norm(twist[:3]))
        lateral_limit = .25*requested_norm + .0005
        if not neutral and requested_norm > .001:
            direction = twist[:3]/requested_norm
            lateral_map = (np.eye(3)-np.outer(direction, direction)) @ jac[:3] * speed
            lateral_limit = max(lateral_limit, np.linalg.norm(lateral_map@(self.previous/speed))
                                - self.cfg['linear_acceleration_m_s2']*dt)
            constraints.append(dict(type='ineq',
                fun=lambda u: lateral_limit-np.linalg.norm(lateral_map@u),
                jac=lambda u: -(lateral_map@u)@lateral_map /
                max(np.linalg.norm(lateral_map@u), 1e-12)))
        initial = np.clip(self.previous/speed, low/speed, high/speed)
        result = minimize(lambda u: .5*u@hessian@u-rhs@u, initial,
                          jac=lambda u: hessian@u-rhs, method='SLSQP',
                          bounds=list(zip(low/speed, high/speed)),
                          constraints=constraints,
                          options={'maxiter': 50, 'ftol': 1e-9})
        if (not result.success or not np.isfinite(result.x).all() or
                any(c['fun'](result.x) < -1e-6 for c in constraints)):
            return self._pause('当前方向受姿态/边界约束；可反向或调整朝向。', limiting)
        velocity = result.x*speed
        achieved = jac @ velocity
        progress = float(achieved[:3] @ twist[:3] / max(requested_norm**2, 1e-12))
        lateral = float(np.linalg.norm(achieved[:3]-progress*twist[:3]))
        # Do not silently substitute a large sideways movement for the user's
        # requested direction. A blocked task stays recoverable, not latched.
        if (not neutral and requested_norm > .001 and
                (progress < .03 or lateral > lateral_limit + 1e-6)):
            return self._pause('该方向暂不可达；保持位置，可反向退回。', limiting)
        constrained = bool(limiting or (requested_norm > .001 and progress < .8))
        self.diagnostics.update(message='接近边界/运动能力受限，已限速，可反向退回。' if constrained
                                else 'XYZ 位置优先控制中；LB+RB 精细档。',
                                requested_twist=twist.tolist(), achieved_twist=achieved.tolist(),
                                orientation_error_rad=float(np.linalg.norm(error)),
                                limiting_joints=limiting, constrained=constrained,
                                progress=progress, lateral_speed_m_s=lateral,
                                orientation_limit_rad=envelope,
                                joint_margin_rad=self.joint_margin_rad,
                                solver_iterations=int(getattr(result, 'nit', 0)))
        return velocity

    def _pause(self, message, limiting):
        self.pause = True
        self.diagnostics.update(message=message, constrained=True, limiting_joints=limiting,
                                achieved_twist=[0.]*6, geometry_pause=True)
        return np.zeros(6)
