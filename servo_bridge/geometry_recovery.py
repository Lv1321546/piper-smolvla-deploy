"""Recoverable geometric holds, shared by the three teleoperation runtimes.

Continue publishing user input to Servo while hardware holds measured position.
Only Servo's real, fresh clear/leaving status can permit hardware motion again.
No status reset service or substitute healthy status is used.
"""
import json
import time

from servo_bridge.core import CommandGuard, STATUS
from servo_bridge.sources import SimSource


GEOMETRY_STATUSES = (2, 5)


class GeometryRecovery:
    def status(self, msg):
        self.latest_status = int(msg.data)
        self.latest_status_received = time.monotonic()
        self.event('servo_status', value=self.latest_status,
                   label=STATUS.get(self.latest_status, 'UNKNOWN'),
                   recoverable=self.latest_status in GEOMETRY_STATUSES)
        if self.latest_status in GEOMETRY_STATUSES:
            self.geometry_hold('接近关节限位或奇异区域：已保持，请反向退离。')
        elif self.latest_status in (3, 4):
            if isinstance(self.source, SimSource) and self.feedback is not None:
                self.source.target = self.feedback.q.copy()
            if self.socket:
                self.latch_output(STATUS[self.latest_status])

    def geometry_hold(self, reason):
        now = time.monotonic()
        if (self.latest_status not in (0, 1, 2, 5, 6) or self.latest_status_received is None or
                not 0 <= now-self.latest_status_received <= self.cfg['command_max_age_s']):
            if self.socket:
                self.latch_output('Cannot verify fresh Servo status for geometry hold')
            return
        self.geometry_paused = True
        self.execution.reset()
        self.guard = CommandGuard(self.cfg)
        if hasattr(self.policy, 'reset_motion'):
            self.policy.reset_motion()
        else:
            self.policy.velocity[:] = 0
        if isinstance(self.source, SimSource) and self.feedback is not None:
            self.source.target = self.feedback.q.copy()
        if self.socket and not self.output_latched:
            try:
                self.socket.sendto(json.dumps(dict(version=1, monotonic=now,
                    deadman=False, geometry_hold=True, servo_status=self.latest_status,
                    status_monotonic=self.latest_status_received)).encode(), self.args.output_socket)
                self.relay_active = False
            except OSError as exc:
                self.latch_output(str(exc))
        self.event('geometry_hold', reason=reason, status=self.latest_status, recoverable=True)
