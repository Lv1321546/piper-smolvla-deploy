"""Feedback sources. The CAN source has only recv(), never SDK initialization."""
import json
from pathlib import Path
import socket
import threading
import time

import numpy as np
from e14.piper_robot import FeedbackStore
from servo_bridge.core import Feedback


def log_cycles(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    session = next(row for row in rows if row.get('event') == 'session')
    return session, [row for row in rows if row.get('event') == 'teleop_cycle']


class LogSource:
    def __init__(self, path):
        self.session, self.rows = log_cycles(path)
        self.index = 0

    def sample(self, now, dt):
        row = self.rows[self.index]
        snap = row['feedback_after']
        age = max(0., row['monotonic'] - snap['feedback_monotonic'])
        measured = row.get('planner', {}).get('measured_joints_deg') or snap['state'][:6]
        feedback = Feedback(np.radians(measured), now - age, snap['state'][6],
                            snap['arm_status'] == 0 and snap['err_code'] == 0 and not any(snap['motor_faults']),
                            False, 'recorded_real_feedback_rebased')
        self.index += 1
        return feedback, row['input'], row

    def close(self):
        pass


class SimSource:
    """Bounded first-order joint plant, explicitly not a real-arm dynamics test."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.q = np.radians(cfg['sim_initial_joints_deg'])
        self.target = self.q.copy()
        self.dq = np.zeros(6)
        self.width = cfg['sim_gripper_width_m']
        self.elapsed = 0.

    def sample(self, now, dt):
        desired = np.clip((self.target - self.q) / self.cfg['sim_time_constant_s'],
                          -self.cfg['maximum_joint_velocity_rad_s'], self.cfg['maximum_joint_velocity_rad_s'])
        self.dq += np.clip(desired - self.dq, -self.cfg['maximum_joint_acceleration_rad_s2'] * dt,
                           self.cfg['maximum_joint_acceleration_rad_s2'] * dt)
        step = self.dq * dt
        # Do not overshoot a position setpoint in this test plant.
        self.q += np.sign(self.target - self.q) * np.minimum(np.abs(step), np.abs(self.target - self.q))
        self.elapsed += dt
        return Feedback(self.q.copy(), now, self.width, True, True, 'synthetic_closed_loop'), None, None

    def close(self):
        pass


class PassiveCan:
    """Receive-only socket. Deliberately no send, SDK object, ip link or reset.

    Filters cover status, joints, jaw and driver enable/error feedback. recv
    timeout never renews timestamps. All required frame ages remain distinct.
    """
    def __init__(self, channel='can0'):
        self.store = FeedbackStore()
        self.stop = threading.Event()
        self.sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 0)
        self.sock.bind((channel,))
        self.sock.settimeout(.05)
        self.frames = 0
        self.error = None
        self.thread = threading.Thread(target=self._receive, daemon=True)
        self.thread.start()

    def _receive(self):
        import struct
        while not self.stop.is_set():
            try:
                frame = self.sock.recv(16)
                can_id, length, data = struct.unpack('=IB3x8s', frame)
                if can_id & (socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | socket.CAN_ERR_FLAG):
                    continue
                self.store.ingest(can_id, data[:length], received_at=time.monotonic())
                self.frames += 1
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop.is_set():
                    self.error = str(exc)
                return

    def sample(self, now, dt):
        if self.error:
            raise RuntimeError(self.error)
        snap = self.store.snapshot()
        healthy = snap['arm_status'] == 0 and snap['err_code'] == 0 and not any(snap['motor_faults'])
        ready = healthy and snap['ctrl_mode'] == 1 and snap['mode_feed'] == 1 and all(snap['enabled'])
        return Feedback(np.radians(snap['state'][:6]), snap['feedback_monotonic'],
                        snap['state'][6], healthy, ready, 'passive_real_can'), None, snap

    def close(self):
        self.stop.set()
        self.thread.join(timeout=.2)
        self.sock.close()


def synthetic_pad(elapsed, scenario='xyz'):
    pad = {'connected': True, 'instance': 0, 'buttons': {'lb': False}, 'axes': {}, 'hat': [0, 0]}
    # Neutral -> X -> pause -> Y -> pause -> Z -> pause -> yaw -> release.
    if elapsed < 1.:
        return pad
    pad['buttons']['lb'] = True
    if scenario == 'x':
        if elapsed < 5.:
            pad['axes']['left_y'] = -.5
    elif scenario == 'singularity':
        if elapsed < 5.:
            pad['axes']['left_y'] = -.5
    else:
        if 1. <= elapsed < 4.:
            pad['axes']['left_y'] = -.5
        elif 5. <= elapsed < 8.:
            pad['axes']['left_x'] = -.5
        elif 9. <= elapsed < 12.:
            pad['axes']['right_y'] = -.5
        elif 13. <= elapsed < 16.:
            pad['axes']['right_x'] = .4
        elif elapsed >= 17.:
            pad['buttons']['lb'] = False
    return pad
