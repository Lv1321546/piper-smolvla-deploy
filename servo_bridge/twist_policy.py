"""ROS-system-side heading policy; the SDK worker does not need SciPy."""
import numpy as np
from scipy.spatial.transform import Rotation
from servo_bridge.core import vector


class TwistPolicy:
    """Deadman + input ramp + bounded SO(3) heading correction, no position IK.

    The remembered orientation is corrected gradually within angular speed
    limits, rather than imposing an exact 6-D pose solve every control cycle.
    """
    def __init__(self, cfg, mapping=None):
        self.cfg, self.mapping = cfg, mapping or {}
        signs = self.mapping.get('input_directions', {})
        if (not isinstance(signs, dict) or set(signs) - {'left_x', 'left_y', 'right_x', 'right_y', 'hat_x', 'hat_y'} or
                any(isinstance(v, bool) or v not in (-1, 1) for v in signs.values())):
            raise ValueError('input_directions must contain known axes with +1 or -1 values')
        if any(not isinstance(self.mapping.get(key, True), bool) for key in ('invert_y', 'invert_pitch')):
            raise ValueError('invert_y and invert_pitch must be booleans')
        self.neutral_seen = self.active = False
        self.orientation = None
        self.velocity = np.zeros(6)
        self.instance = None

    def step(self, pad, measured_rotation, dt, *, reference_may_advance=True):
        axes = pad.get('axes', {})
        buttons = pad.get('buttons', {})
        values = np.array([axes.get(k, 0.) for k in ('left_x', 'left_y', 'right_x', 'right_y')], float)
        if not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
            raise ValueError('Invalid normalized gamepad axes')
        hat = vector(pad.get('hat', (0, 0)), 2)
        if np.max(np.abs(hat)) > 1:
            raise ValueError('Invalid hat input')
        if not 0 < dt <= 2 * self.cfg['publish_period_s']:
            raise ValueError('Invalid input period')
        neutral = np.max(np.abs(values)) < .05 and np.max(np.abs(hat)) == 0
        if pad.get('instance') != self.instance:
            self.instance = pad.get('instance')
            self.neutral_seen = self.active = False
        if not pad.get('connected', False) or pad.get('error') or buttons.get('start', False):
            self.neutral_seen = self.active = False
            self.velocity[:] = 0
            return np.zeros(6)
        if not buttons.get('lb', False):
            self.active = False
            self.neutral_seen = neutral
            self.orientation = None
            self.velocity[:] = 0
            return np.zeros(6)
        if not self.active and not self.neutral_seen:
            return np.zeros(6)
        if not self.active:
            self.orientation = np.asarray(measured_rotation, float).copy()
        self.active = True
        signs = self.mapping.get('input_directions', {})
        lx, ly, rx, ry = [v * signs.get(k, 1) for k, v in zip(
            ('left_x', 'left_y', 'right_x', 'right_y'), values)]
        hx, hy = hat * [signs.get('hat_x', 1), signs.get('hat_y', 1)]
        fine = .25 if buttons.get('rb') else 1.
        demand = np.r_[[-ly, -lx if self.mapping.get('invert_y', True) else lx, -ry],
                       [hx, -hy if self.mapping.get('invert_pitch', True) else hy, rx]]
        demand[:3] *= self.cfg['linear_speed_m_s'] * fine
        demand[:3] *= min(1., self.cfg['linear_speed_m_s'] * fine / max(np.linalg.norm(demand[:3]), 1e-12))
        demand[3:] *= self.cfg['angular_speed_rad_s'] * fine
        acceleration = np.r_[[self.cfg['linear_acceleration_m_s2']] * 3,
                             [self.cfg['angular_acceleration_rad_s2']] * 3]
        self.velocity += np.clip(demand - self.velocity, -acceleration * dt, acceleration * dt)
        next_orientation = Rotation.from_rotvec(self.velocity[3:] * dt).as_matrix() @ self.orientation
        old_error = np.linalg.norm(Rotation.from_matrix(self.orientation @ np.asarray(measured_rotation).T).as_rotvec())
        new_error = np.linalg.norm(Rotation.from_matrix(next_orientation @ np.asarray(measured_rotation).T).as_rotvec())
        if (reference_may_advance and
                (new_error <= self.cfg['orientation_reference_max_lead_rad'] or new_error < old_error)):
            self.orientation = next_orientation
        error = Rotation.from_matrix(self.orientation @ np.asarray(measured_rotation).T).as_rotvec()
        norm = np.linalg.norm(error)
        correction = np.zeros(3)
        deadband = self.cfg['orientation_hold_deadband_rad']
        if norm > deadband:
            magnitude = min(self.cfg['orientation_correction_max_rad_s'],
                            self.cfg['orientation_hold_gain_s_inv'] * (norm - deadband))
            correction = error / norm * magnitude
        twist = self.velocity.copy()
        twist[3:] += correction
        limit = self.cfg['angular_speed_rad_s']
        twist[3:] *= min(1., limit / max(np.linalg.norm(twist[3:]), 1e-12))
        return twist
