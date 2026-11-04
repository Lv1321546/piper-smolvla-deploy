"""ROS-independent gamepad -> six joint velocities, without Cartesian IK."""
import math

import numpy as np

from servo_bridge.core import CommandGuard, vector


class JointPolicy:
    def __init__(self, cfg, mapping=None):
        self.cfg, self.mapping = cfg, mapping or {}
        self.signs = self.mapping.get('input_directions', {})
        names = {'left_x', 'left_y', 'right_x', 'right_y', 'hat_x', 'hat_y'}
        if (not isinstance(self.signs, dict) or set(self.signs) - names or
                any(isinstance(v, bool) or v not in (-1, 1) for v in self.signs.values())):
            raise ValueError('input_directions requires known axes with +1/-1 values')
        for key in ('joint_speed_deg_s', 'joint_acceleration_deg_s2'):
            value = cfg.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f'{key} must be finite and positive')
        self.speed = math.radians(cfg['joint_speed_deg_s'])
        fine_speed = cfg.get('joint_fine_speed_deg_s', cfg['joint_speed_deg_s'] * .25)
        if (isinstance(fine_speed, bool) or not isinstance(fine_speed, (int, float)) or
                not math.isfinite(fine_speed) or not 0 < fine_speed <= cfg['joint_speed_deg_s']):
            raise ValueError('joint_fine_speed_deg_s must be positive and no faster than normal speed')
        self.fine_speed = math.radians(fine_speed)
        self.acceleration = math.radians(cfg['joint_acceleration_deg_s2'])
        if self.speed > cfg['maximum_joint_velocity_rad_s']:
            raise ValueError('Joint input speed exceeds the execution velocity limit')
        if self.acceleration > cfg['maximum_joint_acceleration_rad_s2']:
            raise ValueError('Joint input acceleration exceeds the execution limit')
        self.active = self.neutral_seen = self.faulted = False
        self.instance = None
        self.orientation = None  # No orientation target or heading correction.
        self.velocity = np.zeros(6)
        self.limits = CommandGuard(cfg).limits
        self.margin = cfg.get('joint_soft_margin_rad', .10)
        if (isinstance(self.margin, bool) or not isinstance(self.margin, (int, float)) or
                not math.isfinite(self.margin) or self.margin <= .05 or
                np.any(np.diff(self.limits, axis=1) <= 2*self.margin)):
            raise ValueError('Joint soft margin must exceed Servo margin and leave a usable range')
        self.pause = False
        self.diagnostics = {}

    def step(self, pad, measured_rotation, dt, *, reference_may_advance=True,
             measured_q=None, reference_q=None):
        # measured_rotation/reference_may_advance are accepted for the shared
        # runtime interface; neither participates in joint control.
        if not math.isfinite(dt) or not 0 < dt <= 2 * self.cfg['publish_period_s']:
            raise ValueError('Invalid joint input period')
        self.pause = False
        self.diagnostics = {}
        axes, buttons = pad.get('axes', {}), pad.get('buttons', {})
        keys = ('left_x', 'left_y', 'right_x', 'right_y', 'left_trigger', 'right_trigger')
        values = vector([axes.get(k, 0.) for k in keys])
        hat = vector(pad.get('hat', (0, 0)), 2)
        if np.max(np.abs(values)) > 1 or np.max(np.abs(hat)) > 1:
            raise ValueError('Invalid normalized gamepad input')
        if (buttons.get('start') or
                (self.active and (not pad.get('connected') or pad.get('error') or
                                  pad.get('instance') != self.instance))):
            self.faulted = True
        if self.faulted or not pad.get('connected') or pad.get('error'):
            self.active = self.neutral_seen = False
            self.velocity[:] = 0
            return self.velocity.copy()
        if pad.get('instance') != self.instance:
            self.instance = pad.get('instance')
            self.active = self.neutral_seen = False
        neutral = (np.max(np.abs(values)) < .05 and np.max(np.abs(hat)) == 0
                   and not buttons.get('a') and not buttons.get('b'))
        if not buttons.get('lb'):
            self.active = False
            self.neutral_seen = neutral
            self.velocity[:] = 0
            return self.velocity.copy()
        if not self.active and not self.neutral_seen:
            return np.zeros(6)
        self.active = True
        lx, ly, rx, ry = [v * self.signs.get(k, 1) for k, v in zip(keys[:4], values[:4])]
        hx, hy = hat * [self.signs.get('hat_x', 1), self.signs.get('hat_y', 1)]
        controls = np.array([lx, -ly, -ry, hx, hy, rx])
        speed = self.fine_speed if buttons.get('rb') else self.speed
        demand = controls * speed
        previous = self.velocity.copy()
        self.velocity += np.clip(demand - self.velocity,
                                 -self.acceleration * dt, self.acceleration * dt)
        if measured_q is not None:
            q = vector(measured_q)
            reference = q if reference_q is None else vector(reference_q)
            if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                raise ValueError('Measured joints outside model bounds')
            # Include the integrated target's lead, braking distance, and the
            # execution filter's coasting time. Inside a margin, outward speed
            # is zero while inward speed remains available without rearming.
            low_distance = np.maximum(0., np.minimum(q, reference)-self.limits[:, 0]-self.margin)
            high_distance = np.maximum(0., self.limits[:, 1]-self.margin-np.maximum(q, reference))
            lookahead = max(.20, 2*self.cfg.get('execution_velocity_time_constant_s', .15)+2*dt)
            low = -np.minimum(np.sqrt(self.acceleration*low_distance), low_distance/lookahead)
            high = np.minimum(np.sqrt(self.acceleration*high_distance), high_distance/lookahead)
            bounded = np.clip(self.velocity, low, high)
            blocked = np.flatnonzero(abs(bounded-self.velocity) > 1e-9)
            if blocked.size:
                self.diagnostics = dict(limiting_joints=(blocked+1).tolist(),
                    message='接近关节限位：已限制靠近方向，反向可退回。')
            if np.max(abs(bounded-previous)) > self.acceleration*dt+1e-8:
                # An abrupt measured-state change needs a measured hold, not a
                # command that violates the downstream acceleration contract.
                self.pause = True
                bounded[:] = 0
            self.velocity = bounded
        return self.velocity.copy()
