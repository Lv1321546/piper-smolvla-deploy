"""ROS-independent gamepad -> six joint velocities, without Cartesian IK."""
import math

import numpy as np

from servo_bridge.core import vector


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

    def step(self, pad, measured_rotation, dt, *, reference_may_advance=True):
        # measured_rotation/reference_may_advance are accepted for the shared
        # runtime interface; neither participates in joint control.
        if not math.isfinite(dt) or not 0 < dt <= 2 * self.cfg['publish_period_s']:
            raise ValueError('Invalid joint input period')
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
        self.velocity += np.clip(demand - self.velocity,
                                 -self.acceleration * dt, self.acceleration * dt)
        return self.velocity.copy()
