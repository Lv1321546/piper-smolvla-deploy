"""Read-only CAN joint angles and firmware end pose; never imports the SDK."""
import argparse
import json
import math
import socket
import struct
import time

import numpy as np
from e14.piper_robot import FeedbackStore
from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--can-interface', default='can0')
    p.add_argument('--samples', type=int, default=10)
    p.add_argument('--interval', type=float, default=.2)
    args = p.parse_args()
    if args.samples < 1 or not math.isfinite(args.interval) or args.interval <= 0:
        p.error('samples and interval must be positive')
    store = FeedbackStore()
    kin = PiperKinematics(load_config()['joint_limits_deg'])
    pose_frames = {}
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    sock.bind((args.can_interface,))
    sock.settimeout(.05)
    count = 0
    next_output = time.monotonic() + .2
    deadline = time.monotonic() + 3.
    try:
        while count < args.samples:
            try:
                raw = sock.recv(16)
                can_id, length, data = struct.unpack('=IB3x8s', raw)
                now = time.monotonic()
                if can_id & (socket.CAN_EFF_FLAG | socket.CAN_RTR_FLAG | socket.CAN_ERR_FLAG) or length != 8:
                    continue
                store.ingest(can_id, data, received_at=now)
                if can_id in (0x2a2, 0x2a3, 0x2a4):
                    pose_frames[can_id] = (struct.unpack('>ii', data), now)
            except socket.timeout:
                now = time.monotonic()
            if now < next_output:
                continue
            try:
                snap = store.snapshot()
            except RuntimeError:
                if now >= deadline:
                    raise
                continue
            age = now-snap['feedback_monotonic']
            if age > .15:
                raise RuntimeError('Stale CAN feedback; no fresh state to display')
            q = snap['state'][:6]
            fk = kin.forward(np.radians(q))
            firmware = None
            if len(pose_frames) == 3 and now-min(v[1] for v in pose_frames.values()) <= .15:
                firmware = [x/1000 for i in (0x2a2, 0x2a3, 0x2a4) for x in pose_frames[i][0]]
            print(json.dumps(dict(joints_deg=q, fk_flange_xyz_mm=np.round(fk[:3,3]*1000,3).tolist(),
                firmware_xyz_mm=firmware[:3] if firmware else None,
                firmware_rpy_deg=firmware[3:] if firmware else None,
                gripper_width_mm=round(snap['state'][6]*1000,3),
                feedback_age_ms=round(age*1000,1), ctrl_mode=snap['ctrl_mode'],
                enabled=snap['enabled']), ensure_ascii=False), flush=True)
            count += 1
            next_output = now+args.interval
    finally:
        sock.close()


if __name__ == '__main__':
    main()
