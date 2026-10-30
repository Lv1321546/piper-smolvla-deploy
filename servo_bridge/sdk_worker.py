"""Opt-in, separately armed SDK writer for supervised commissioning only.

The ROS process uses system Python 3.10; this worker can run in the existing
LeRobot Python 3.12. It receives local Unix datagrams, independently receives
raw CAN feedback, and exposes JointCtrl only. Never auto-start this worker.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
from servo_bridge.core import CommandGuard, NAMES, load_config, sdk_units, vector
from servo_bridge.sources import PassiveCan

ACK = 'I authorize six-joint JointCtrl only; no enable, mode, reset, gripper or emergency-stop commands'


def validate_arm(data, channel, now=None, *, commission_hold_only=False):
    now = time.time() if now is None else now
    keys = ('operator_present', 'model_and_control_point_checked', 'only_one_can_writer')
    if not commission_hold_only:
        keys += ('measured_joint_hold_checked',)
    for key in keys:
        if data.get(key) is not True:
            raise ValueError(f'Output remains closed: {key} is not confirmed')
    expires = data.get('expires_unix_s', 0)
    if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not math.isfinite(expires) or not now < expires <= now + 1800:
        raise ValueError('Arming lease must expire within 30 minutes and be unexpired')
    if data.get('can_interface') != channel or data.get('acknowledgement') != ACK:
        raise ValueError('Arming acknowledgement / CAN interface mismatch')


class SDKJointWriter:
    """Explicit construction is the only SDK entry point in this new route."""
    def __init__(self, channel, factory=None):
        if factory is None:
            from piper_sdk import C_PiperInterface_V2
            factory = C_PiperInterface_V2
        self.sdk = factory(channel)
        # Important: the SDK default piper_init=True sends queries. Disable it;
        # the worker's independent passive receiver owns feedback collection.
        self.sdk.ConnectPort(piper_init=False, start_thread=False)

    def joint(self, raw):
        if len(raw) != 6 or any(isinstance(v, bool) or not isinstance(v, int) for v in raw):
            raise ValueError('Six integer SDK targets required')
        self.sdk.JointCtrl(*raw)

    def close(self):
        self.sdk.DisconnectPort()


class Worker:
    """Pure command/lease checks shared by the real worker and fake-SDK tests."""
    def __init__(self, cfg, writer, *, clock=time.monotonic):
        self.cfg, self.writer, self.clock = cfg, writer, clock
        self.guard = CommandGuard(cfg, clock=clock)
        self.last_packet = None
        self.moved = False

    def packet(self, packet, feedback):
        now = self.clock()
        if packet.get('version') != 1:
            raise ValueError('Unknown protocol version')
        timestamp = packet.get('monotonic')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError('Invalid command monotonic timestamp')
        if not 0 <= now - timestamp <= self.cfg['command_max_age_s']:
            raise ValueError('Command lease is stale or future')
        if self.last_packet is not None and timestamp <= self.last_packet:
            raise ValueError('Duplicate/out-of-order command')
        if packet.get('stop_requested') is True:
            self.hold(feedback)
            raise RuntimeError(f"Relay requested stop: {packet.get('reason', 'unspecified')}")
        if packet.get('servo_status') not in (0, 1, 6):
            raise ValueError('Servo warning/halt/collision state forbids transmission')
        status_stamp = packet.get('status_monotonic')
        if (isinstance(status_stamp, bool) or not isinstance(status_stamp, (int, float)) or
                not math.isfinite(status_stamp) or not 0 <= now - status_stamp <= self.cfg['command_max_age_s']):
            raise ValueError('Servo status is missing, stale or future')
        if not feedback.ready:
            raise ValueError('Requires already enabled, healthy CAN/MOVE_J; no mode or enable is sent')
        if packet.get('deadman') is not True:
            self.hold(feedback)
            self.last_packet = timestamp
            return False
        raw = self.guard.validate(packet['joint_names'], packet['positions'], packet['velocities'],
                                  feedback=feedback, stamp_age_s=now - timestamp,
                                  horizon_s=packet['horizon_s'], now=now)
        self.writer.joint(raw)
        self.last_packet = timestamp
        self.moved = True
        return True

    def hold(self, feedback):
        """One measured-position JointCtrl, only after this worker moved.

        There is no torque-disable/emergency-stop/reset fallback. If feedback
        is stale, unhealthy or outside limits, latch closed and require human
        intervention. Stop behaviour still needs physical commissioning.
        """
        if not self.moved:
            return
        now = self.clock()
        q = vector(feedback.q)
        quantized = np.radians(np.asarray(sdk_units(q)) / 1000.)
        bounds = self.guard.limits
        if (not feedback.ready or not feedback.healthy or
                not 0 <= now - feedback.received <= self.cfg['feedback_max_age_s'] or
                np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]) or
                np.any(quantized < bounds[:, 0]) or np.any(quantized > bounds[:, 1]) or
                self.guard.last_q is None or np.max(np.abs(q - self.guard.last_q)) > math.radians(self.cfg['maximum_tracking_error_deg'])):
            self.moved = False
            raise RuntimeError('Cannot verify measured-position hold; output latched closed; use commissioned hardware stop')
        self.writer.joint(sdk_units(q))
        self.moved = False
        self.guard.last_q = None
        self.guard.last_received = None
        self.guard.last_velocity[:] = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--arm-file', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--commission-hold', action='store_true', help='Supervised first test: send current measured joints once, observe 2s, then exit; no teleop socket')
    parser.add_argument('--socket', default='/tmp/piper-servo-sdk.sock')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Output remains closed without explicit --execute; SDK was not imported')
    cfg = load_config(args.config)
    arm_path = Path(args.arm_file)
    validate_arm(json.loads(arm_path.read_text()), cfg['can_interface'], commission_hold_only=args.commission_hold)
    sock_path = Path(args.socket)
    if sock_path.exists():
        parser.error('Socket path exists; verify no active writer before manual removal')
    lock = open(str(sock_path) + '.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = PassiveCan(cfg['can_interface'])
    sock, writer, worker = None, None, None
    try:
        # Passive readiness check before importing or constructing the SDK.
        deadline = time.monotonic() + 3.
        while True:
            try:
                feedback = source.sample(time.monotonic(), .05)[0]
                if feedback.ready and time.monotonic() - feedback.received <= cfg['feedback_max_age_s']:
                    break
            except RuntimeError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('No fresh healthy enabled CAN/MOVE_J feedback; output remains closed')
            time.sleep(.02)
        validate_arm(json.loads(arm_path.read_text()), cfg['can_interface'], commission_hold_only=args.commission_hold)
        writer = SDKJointWriter(cfg['can_interface'])
        if args.commission_hold:
            guard = CommandGuard(cfg)
            raw = guard.validate(list(NAMES), feedback.q, [0.] * 6, feedback=feedback,
                                 stamp_age_s=0., horizon_s=cfg['publish_period_s'])
            writer.joint(raw)
            initial = feedback.q.copy()
            end = time.monotonic() + 2.
            largest = 0.
            while time.monotonic() < end:
                latest = source.sample(time.monotonic(), .05)[0]
                if not latest.ready or time.monotonic() - latest.received > cfg['feedback_max_age_s']:
                    raise RuntimeError('Hold observation lost fresh ready feedback; operator intervention required')
                drift = float(np.degrees(np.max(np.abs(latest.q - initial))))
                largest = max(largest, drift)
                print(f'Hold observation: max joint change={drift:.3f} deg; no new goal sent', flush=True)
                time.sleep(.1)
            print(f'One measured-position JointCtrl sent; max observed change={largest:.3f} deg. Operator must confirm physical holding; this script does not mark commissioning complete.', flush=True)
            return
        worker = Worker(cfg, writer)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(str(sock_path))
        os.chmod(sock_path, 0o600)
        sock.settimeout(cfg['command_max_age_s'])
        print('SUPERVISED SDK OUTPUT ARMED. Six-joint JointCtrl only. Ctrl-C requests measured-position hold.', flush=True)
        while True:
            validate_arm(json.loads(arm_path.read_text()), cfg['can_interface'])
            try:
                raw = sock.recv(8192)
            except socket.timeout:
                if worker.moved:
                    raise RuntimeError('Input lease expired; stopping worker')
                continue
            feedback = source.sample(time.monotonic(), .05)[0]
            worker.packet(json.loads(raw), feedback)
    finally:
        try:
            if worker and worker.moved:
                worker.hold(source.sample(time.monotonic(), .05)[0])
        finally:
            try:
                if writer:
                    writer.close()
            finally:
                source.close()
                if sock:
                    sock.close()
                    sock_path.unlink(missing_ok=True)
                lock.close()


if __name__ == '__main__':
    main()
