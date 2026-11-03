"""Coalesce bursty ROS callbacks without integrating or transmitting twice.

This gate does not relax the execution/worker timestamp, acceleration or
feedback guards. A gap that is too long still reaches those guards and fails.
"""
import math


class CandidateCadence:
    def __init__(self, period):
        if isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period) or period <= 0:
            raise ValueError('Positive finite output period required')
        # Execution and worker reject intervals below 0.5*period. Leave extra
        # receipt-time margin while accepting normal ~period callbacks.
        self.minimum_interval = .75 * period
        self.last_output = None

    def should_defer(self, now):
        if not math.isfinite(now) or (self.last_output is not None and now < self.last_output):
            raise ValueError('Invalid or backwards output clock')
        return self.last_output is not None and now-self.last_output < self.minimum_interval

    def commit(self, now):
        if self.should_defer(now):
            raise ValueError('Output cadence committed before it was due')
        self.last_output = now
