"""ROS-independent command contracts; no SDK import or CAN transmit path here."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

NAMES = tuple(f'joint{i}' for i in range(1, 7))
STATUS = {0: 'NO_WARNING', 1: 'DECELERATE_FOR_APPROACHING_SINGULARITY',
          2: 'HALT_FOR_SINGULARITY', 3: 'DECELERATE_FOR_COLLISION',
          4: 'HALT_FOR_COLLISION', 5: 'JOINT_BOUND',
          6: 'DECELERATE_FOR_LEAVING_SINGULARITY'}


def load_config(path=None):
    cfg = json.loads(Path(path or Path(__file__).with_name('config.json')).read_text())
    if cfg.get('schema') != 1 or cfg.get('hardware_output') is not False:
        raise ValueError('This profile must have schema=1 and hardware_output=false')
    for name in ('publish_period_s', 'feedback_max_age_s', 'command_max_age_s',
                 'maximum_tracking_error_deg', 'maximum_joint_step_deg',
                 'maximum_joint_velocity_rad_s', 'maximum_joint_acceleration_rad_s2',
                 'linear_speed_m_s', 'angular_speed_rad_s', 'linear_acceleration_m_s2',
                 'angular_acceleration_rad_s2', 'orientation_hold_gain_s_inv',
                 'orientation_hold_deadband_rad', 'orientation_correction_max_rad_s',
                 'orientation_reference_max_lead_rad',
                 'sim_time_constant_s'):
        value = cfg[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    bounds = np.asarray(cfg['joint_limits_deg'], float)
    if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError('Six finite ordered joint limit pairs required')
    if cfg['model']['planning_frame'] != 'base_link' or cfg['model']['control_frame'] != 'link6':
        raise ValueError('Only the audited base_link -> link6 model is supported')
    return cfg


def vector(value, size=6):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (size,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (float, int, np.floating, np.integer)) for v in raw):
        raise ValueError(f'{size} finite numbers required, not booleans or numeric strings')
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.isfinite(a).all():
        raise ValueError(f'{size} finite numbers required')
    return a


def ordered_values(names, values):
    """Never assume trajectory order. Reject duplicates, omissions and extras."""
    if len(names) != 6 or set(names) != set(NAMES):
        raise ValueError('Expected exactly joint1..joint6, once each')
    vals = vector(values)
    return np.array([vals[names.index(name)] for name in NAMES])


def sdk_units(radians):
    """SDK JointCtrl uses 0.001 degrees, not radians or encoder ticks."""
    return [round(math.degrees(v) * 1000) for v in vector(radians)]


@dataclass
class Feedback:
    q: np.ndarray
    received: float
    width: float = 0.04
    healthy: bool = True
    ready: bool = False
    source: str = 'unknown'


class CommandGuard:
    """Validate, do not clip. A rejected trajectory cannot update its reference."""
    def __init__(self, cfg, *, clock=time.monotonic):
        self.cfg, self.clock = cfg, clock
        self.last_q = self.last_received = None
        self.last_velocity = np.zeros(6)
        self.limits = np.radians(cfg['joint_limits_deg'])
        urdf = ET.parse(Path(__file__).resolve().parents[1] / 'third_party/gamepad_piper/piper.urdf').getroot()
        for i, name in enumerate(NAMES):
            limit = urdf.find(f"joint[@name='{name}']/limit")
            self.limits[i, 0] = max(self.limits[i, 0], float(limit.get('lower')))
            self.limits[i, 1] = min(self.limits[i, 1], float(limit.get('upper')))

    def validate(self, names, positions, velocities, *, feedback, stamp_age_s, horizon_s, now=None):
        now = self.clock() if now is None else now
        if (not math.isfinite(now) or not math.isfinite(stamp_age_s) or
                not 0 <= stamp_age_s <= self.cfg['command_max_age_s']):
            raise ValueError('Trajectory timestamp is stale, future or invalid')
        if not math.isfinite(horizon_s) or not 0 < horizon_s <= 2 * self.cfg['publish_period_s']:
            raise ValueError('Unexpected trajectory horizon')
        if (not isinstance(feedback, Feedback) or not feedback.healthy or
                not 0 <= now - feedback.received <= self.cfg['feedback_max_age_s']):
            raise ValueError('Missing, unhealthy, stale or future measured feedback')
        measured = vector(feedback.q)
        q = ordered_values(names, positions)
        velocity = ordered_values(names, velocities)
        # Validate the value that will really be quantized/transmitted as well.
        quantized = np.radians(np.array(sdk_units(q)) / 1000.)
        step = math.radians(self.cfg['maximum_joint_step_deg'])
        lead = math.radians(self.cfg['maximum_tracking_error_deg'])
        for target in (q, quantized):
            if np.any(target < self.limits[:, 0]) or np.any(target > self.limits[:, 1]):
                raise ValueError('Joint position limit exceeded')
            if np.max(np.abs(target - measured)) > lead + 1e-9:
                raise ValueError('Measured tracking-error limit exceeded')
            if self.last_q is not None and np.max(np.abs(target - self.last_q)) > step + 1e-9:
                raise ValueError('Previous-command step limit exceeded')
        if np.max(np.abs(velocity)) > self.cfg['maximum_joint_velocity_rad_s'] + 1e-8:
            raise ValueError('Joint velocity limit exceeded')
        dt = self.cfg['publish_period_s'] if self.last_received is None else now - self.last_received
        if dt < 0.5 * self.cfg['publish_period_s']:
            raise ValueError('Trajectory arrived too fast')
        # Incoming declared velocities and position-derived velocity must both
        # be plausible; compare acceleration at the nominal servo period so
        # scheduler jitter never creates a permissive safety window.
        dt_check = min(dt, self.cfg['publish_period_s'])
        dv_limit = self.cfg['maximum_joint_acceleration_rad_s2'] * dt_check
        if np.max(np.abs(velocity - self.last_velocity)) > dv_limit + 1e-8:
            raise ValueError('Joint acceleration limit exceeded')
        if self.last_q is not None:
            for target in (q, quantized):
                derived = (target - self.last_q) / max(min(dt, self.cfg['publish_period_s']), 1e-9)
                if np.max(np.abs(derived)) > self.cfg['maximum_joint_velocity_rad_s'] + 1e-5:
                    raise ValueError('Position-derived joint velocity limit exceeded (including SDK quantization)')
        # The next guard reference and simulated plant use the integer target
        # that would actually reach JointCtrl, not an unsent float reference.
        self.last_q, self.last_velocity, self.last_received = quantized.copy(), velocity.copy(), now
        return sdk_units(q)


