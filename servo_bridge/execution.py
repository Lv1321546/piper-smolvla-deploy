"""Bounded velocity-to-position adapter for the PiPER position SDK.

Humble Servo recomputes its one-step positions from measured state. A position
SDK is not a velocity controller: repeatedly forwarding that same tiny offset
can stall. This optional adapter integrates Servo velocities into a continuous
position reference. Fresh measured feedback still bounds reference lead via
CommandGuard; a stuck robot faults instead of winding up. No hardware imports.
"""
import math
import numpy as np
from servo_bridge.core import CommandGuard, NAMES, ordered_values

class VelocityTarget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.guard = CommandGuard(cfg)
        self.position = None
        self.velocity = np.zeros(6)
        self.filtered_velocity = np.zeros(6)
        self.last_time = None

    def reset(self):
        self.__init__(self.cfg)

    def validate_candidate(self, names, positions, velocities, feedback):
        q = ordered_values(names, positions)
        v = ordered_values(names, velocities)
        if np.any(q < self.guard.limits[:,0]) or np.any(q > self.guard.limits[:,1]):
            raise ValueError('Servo candidate joint position limit exceeded')
        if np.max(abs(q-feedback.q)) > math.radians(self.cfg['maximum_tracking_error_deg']):
            raise ValueError('Servo candidate measured tracking-error limit exceeded')
        if np.max(abs(v)) > self.cfg['maximum_joint_velocity_rad_s'] + 1e-8:
            raise ValueError('Servo candidate velocity limit exceeded')

    def step(self, names, velocities, *, feedback, now, stamp_age_s, horizon_s):
        desired = ordered_values(names, velocities)
        # Reject malformed/overspeed demand rather than hiding it by clipping.
        if np.max(np.abs(desired)) > self.cfg['maximum_joint_velocity_rad_s'] + 1e-8:
            raise ValueError('Servo velocity demand exceeds limit')
        period = self.cfg['publish_period_s']
        if self.last_time is not None and not .5*period <= now-self.last_time <= self.cfg['command_max_age_s']:
            raise ValueError('Execution cadence stale or too fast')
        dt = period if self.last_time is None else min(now-self.last_time, period)
        # Filter feedback/smoothing-induced candidate velocity ripple before
        # ramp limiting. Deadman release resets this memory; no buffered motion.
        tau = self.cfg.get('execution_velocity_time_constant_s', .15)
        if not isinstance(tau, (int,float)) or not math.isfinite(tau) or tau <= 0:
            raise ValueError('Execution velocity time constant must be positive')
        alpha = -math.expm1(-dt/tau)
        filtered = self.filtered_velocity + alpha*(desired-self.filtered_velocity)
        # Reserve timing margin for the independent worker receipt-time check.
        velocity = self.velocity + np.clip(filtered-self.velocity,
            -0.8*self.cfg['maximum_joint_acceleration_rad_s2']*dt,
             0.8*self.cfg['maximum_joint_acceleration_rad_s2']*dt)
        position = feedback.q.copy() if self.position is None else self.position
        target = position + velocity*dt
        raw = self.guard.validate(list(NAMES), target, velocity, feedback=feedback,
            stamp_age_s=stamp_age_s, horizon_s=horizon_s, now=now)
        # Commit only after all guards passed, retaining sub-quantization residue.
        self.position = target.copy()
        self.velocity = velocity.copy()
        self.filtered_velocity = filtered.copy()
        self.last_time = now
        return raw, self.guard.last_q.copy(), velocity.copy()
