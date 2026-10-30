"""Supervised J1 baseline only; no Cartesian IK, mode, enable or jaw control."""
import argparse
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from e14.gamepad import PygamePad
from servo_bridge.core import CommandGuard, NAMES, load_config
from servo_bridge.sdk_worker import SDKJointWriter, Worker, validate_arm
from servo_bridge.sources import PassiveCan


def advance(q, velocity, axis, initial, dt=.05):
    desired = -math.radians(2.) * axis
    dv = math.radians(4.) * dt
    speed = velocity + max(-dv, min(dv, desired - velocity))
    target = q.copy()
    target[0] += speed * dt
    if abs(target[0] - initial[0]) > math.radians(3.):
        raise StopIteration('3 degree J1 excursion reached')
    v = np.zeros(6); v[0] = speed
    return target, speed, v


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--arm-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.execute: p.error('Explicit --execute required')
    cfg = load_config()
    validate_arm(json.loads(args.arm_file.read_text()), 'can0')
    args.output.mkdir(parents=True, exist_ok=False)
    lock = open('/tmp/piper-servo-sdk.sock.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = pad = writer = worker = None
    reason = 'completed'; error = None; initial = None; initial_feedback = None
    commands = 0; started = None; neutral_seen = False
    q_target = None; speed = 0.; rows = []
    with (args.output / 'events.jsonl').open('w', buffering=1) as log:
        def event(kind, **data):
            row = {'event': kind, 'monotonic': time.monotonic(), **data}
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        try:
            source = PassiveCan('can0'); time.sleep(.4)
            feedback, _, snap = source.sample(time.monotonic(), .05)
            if not feedback.ready or time.monotonic() - feedback.received > .15:
                raise RuntimeError('Fresh enabled healthy CAN/MOVE_J required')
            initial_feedback = snap['state'][:6]
            pad = PygamePad(json.loads((ROOT/'gamepad.local.teleop.json').read_text())['gamepad'])
            state = pad.poll()
            if not state.connected or state.error:
                raise RuntimeError('Gamepad not ready: ' + state.error)
            writer = SDKJointWriter('can0')
            worker = Worker(cfg, writer)
            event('session', test_kind='J1-only live gamepad baseline', speed_limit_deg_s=2.,
                  acceleration_limit_deg_s2=4., excursion_limit_deg=3., motion_limit_s=5.,
                  model_uses='SDK six-joint angle limits; no Cartesian collision/IK claims',
                  initial_q_deg=initial_feedback, gamepad_guid=state.guid)
            print('READY: J1 only. Neutral first, hold LB + left stick sideways. +/-3 deg cap, <=5s. Other axes and jaw ignored.', flush=True)
            deadline = time.monotonic() + 300.
            while time.monotonic() < deadline:
                cycle = time.monotonic()
                validate_arm(json.loads(args.arm_file.read_text()), 'can0')
                feedback, _, snap = source.sample(cycle, .05)
                state = pad.poll()
                if not feedback.ready or not 0 <= time.monotonic()-feedback.received <= .15:
                    raise RuntimeError('Feedback not fresh/ready')
                if not state.connected or state.error or time.monotonic()-state.timestamp > .15:
                    raise RuntimeError('Gamepad unavailable/stale')
                if state.buttons.get('start'):
                    reason = 'Start requested measured hold'; break
                lb, axis = state.buttons.get('lb', False), state.axes['left_x']
                if not lb and all(abs(state.axes[k]) < .08 for k in ('left_x','left_y','right_x','right_y')):
                    neutral_seen = True
                event('feedback_input', q_deg=snap['state'][:6], feedback_age_s=time.monotonic()-feedback.received,
                      lb=lb, axis=axis, raw_pad=asdict(state))
                rows.append(feedback.q.copy())
                if started is None:
                    if not (neutral_seen and lb and abs(axis) > .15):
                        time.sleep(.05); continue
                    started = time.monotonic()
                    initial = feedback.q.copy(); q_target = initial.copy()
                    print('Live J1 input detected; bounded joint test starts.', flush=True)
                if not lb:
                    reason = 'LB released'; break
                if time.monotonic()-started >= 5.:
                    reason = '5 second motion budget reached'; break
                try:
                    target, speed, velocity = advance(q_target, speed, axis, initial)
                except StopIteration as exc:
                    reason = str(exc); break
                raw = worker.guard.validate(list(NAMES), target, velocity,
                    feedback=feedback, stamp_age_s=0., horizon_s=.05)
                event('joint_send_requested', units_001deg=raw, velocities=velocity.tolist())
                writer.joint(raw)
                worker.moved = True; commands += 1
                q_target = worker.guard.last_q.copy()
                event('sdk_joint_returned', units_001deg=raw,
                      note='SDK return is not motor execution acknowledgement')
                time.sleep(max(0., .05-(time.monotonic()-cycle)))
            else:
                reason = 'Waiting window expired'
        except KeyboardInterrupt:
            reason = 'Operator/software interrupted'
        except Exception as exc:
            reason = 'Guard/communication stop'; error = str(exc)
            event('fault', error=error)
        finally:
            try:
                if worker and worker.moved:
                    worker.hold(source.sample(time.monotonic(), .05)[0])
                    event('measured_hold_requested')
            except Exception as exc:
                error = (error or '') + '; hold: ' + str(exc)
            if source:
                # Observe actual final state after hold; no more goals.
                for _ in range(6):
                    try:
                        feedback, _, snap = source.sample(time.monotonic(), .05)
                        if time.monotonic()-feedback.received <= .15:
                            rows.append(feedback.q.copy()); event('post_stop_feedback', q_deg=snap['state'][:6])
                    except RuntimeError:
                        pass
                    time.sleep(.1)
            if writer: writer.close()
            if pad: pad.close()
            if source: source.close()
            lock.close()
            lease = json.loads(args.arm_file.read_text()); lease['expires_unix_s']=0
            args.arm_file.write_text(json.dumps(lease, indent=2))
            values = np.array(rows)
            result = {'reason':reason, 'error':error, 'joint_commands':commands,
                      'initial_q_deg':initial_feedback,
                      'final_q_deg':np.degrees(values[-1]).tolist() if len(values) else None,
                      'max_feedback_delta_deg':np.degrees(np.max(np.abs(values-values[0]),axis=0)).tolist() if len(values) else None,
                      'lease_revoked':True}
            (args.output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
            event('finished', **result); print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
