"""Recorded PiPER state replay. Default is offline inspection; no CAN/SDK.

This is a joint trajectory player, not a policy or MoveIt collision planner.
Recorded positions are interpolated exactly and uniformly retimed, not denoised.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from servo_bridge.core import CommandGuard, Feedback, NAMES, load_config, sdk_units, vector

ROOT = Path(__file__).resolve().parents[1]
STATE_NAMES = list(NAMES) + ['gripper_width']


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def load_episode(path):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    expected = dict(schema_version=1, status='complete',
                    state_names=STATE_NAMES, joint_unit='degree',
                    gripper_unit='meter_full_opening', action_mode='absolute_position')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Requires a complete piper-raw-v1 episode with the audited state order and units')
    kind = manifest.get('format')
    derived = kind == 'piper-replay-v1'
    if kind not in ('piper-raw-v1', 'piper-replay-v1'):
        raise ValueError('Unsupported episode/replay format')
    if derived and (manifest.get('purpose') != 'replay_only' or
                    manifest.get('target_semantics') != 'filtered_recorded_positions'):
        raise ValueError('Derived data must explicitly declare replay-only filtered targets')
    if manifest.get('mock') is not False:
        raise ValueError('Synthetic or unmarked data cannot be used for real-record replay')
    hz = positive(manifest.get('control_hz'), 'control_hz')
    rows = [json.loads(s) for s in (path / 'frames.jsonl').read_text().splitlines() if s.strip()]
    if len(rows) < 2 or any(type(manifest.get(k)) is not int or manifest[k] != len(rows)
                            for k in ('frames', 'expected_frames')):
        raise ValueError('Frame count mismatch or fewer than two frames')
    states, stamps, feedback = [], [], []
    for i, row in enumerate(rows):
        if type(row.get('frame_index')) is not int or row['frame_index'] != i:
            raise ValueError(f'Invalid frame index: {i}')
        states.append(vector(row['target_state' if derived else 'state'], 7))
        stamp_key = 'recorded_monotonic' if derived else 'captured_monotonic'
        stamps.append(positive(row.get(stamp_key), stamp_key))
        if not derived:
            feedback.append(positive(row.get('feedback_monotonic'), 'feedback_monotonic'))
    stamps, feedback = np.array(stamps), np.array(feedback)
    if np.any(np.diff(stamps) <= 0) or np.any(np.diff(feedback) <= 0):
        raise ValueError('Timestamps must be strictly increasing, including actual feedback')
    ratio = positive(manifest.get('max_period_error_ratio', .5), 'max_period_error_ratio')
    if ratio >= 1 or np.any(abs(np.diff(stamps) - 1 / hz) > ratio / hz):
        raise ValueError('Missing/nonuniform recording samples')
    if not derived:
        age = positive(manifest.get('feedback_max_age_ms'), 'feedback_max_age_ms') / 1000
        if np.any(stamps - feedback < 0) or np.any(stamps - feedback > age):
            raise ValueError('Stale or future recorded robot feedback')
    return stamps - stamps[0], np.array(states), manifest


class HermiteSpline:
    """Scalar C1, shape-preserving cubic Hermite curve; rest at both ends.

    Interior slopes use the weighted harmonic mean of same-sign secants.
    Adjacent opposite/zero slopes get zero tangent. No SciPy dependency.
    """
    def __init__(self, x, y):
        self.x = np.asarray(x, float)
        y = np.asarray(y, float)
        h = np.diff(self.x)
        delta = np.diff(y) / h
        slopes = np.zeros_like(y)
        for i in range(1, len(y)-1):
            left, right = delta[i-1], delta[i]
            if left*right > 0:
                w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
                slopes[i] = (w1+w2) / (w1/left+w2/right)
        self.coefficients = np.c_[y[:-1], slopes[:-1],
            (3*delta-2*slopes[:-1]-slopes[1:])/h,
            (slopes[:-1]+slopes[1:]-2*delta)/h**2]

    def __call__(self, t, nu=0):
        t = np.asarray(t)
        index = np.clip(np.searchsorted(self.x, t, side='right')-1, 0, len(self.x)-2)
        u = t-self.x[index]
        c = self.coefficients[index]
        if nu == 0:
            return c[...,0]+u*(c[...,1]+u*(c[...,2]+u*c[...,3]))
        if nu == 1:
            return c[...,1]+u*(2*c[...,2]+u*3*c[...,3])
        if nu == 2:
            return 2*c[...,2]+u*6*c[...,3]
        raise ValueError('Only position, velocity and acceleration supported')


def extrema(spline, derivative, duration):
    values = []
    for i, c in enumerate(spline.coefficients):
        h = spline.x[i+1]-spline.x[i]
        polynomial = np.polynomial.polynomial.polyder(c, m=derivative)
        critical = np.polynomial.polynomial.polyder(polynomial)
        roots = np.polynomial.polynomial.polyroots(critical) if len(critical)>1 else []
        points = [0., h]
        points += [float(np.real(r)) for r in roots if abs(np.imag(r)) < 1e-10 and 0 < np.real(r) < h]
        # Evaluate each segment separately so both sides of acceleration jumps
        # at C1 knots are included, rather than checking only the right side.
        values.extend(np.polynomial.polynomial.polyval(points, polynomial))
    return float(np.min(values)), float(np.max(values))

class ReplayPlan:
    def __init__(self, episode, cfg, *, speed=.5, max_velocity_deg_s=10.,
                 max_acceleration_deg_s2=30., replay_gripper=False):
        self.episode = Path(episode).resolve()
        self.times, self.states, self.manifest = load_episode(self.episode)
        self.cfg, self.replay_gripper = cfg, replay_gripper
        self.speed = positive(speed, 'speed')
        if self.speed > 1:
            raise ValueError('Replay speed must be <= 1; speedups are not supported')
        self.velocity = min(math.radians(positive(max_velocity_deg_s, 'max_velocity_deg_s')),
                            cfg['maximum_joint_velocity_rad_s'])
        self.acceleration = min(math.radians(positive(max_acceleration_deg_s2, 'max_acceleration_deg_s2')),
                                cfg['maximum_joint_acceleration_rad_s2'])
        self.bounds = CommandGuard(cfg).limits
        self.width_bounds = vector(cfg.get('gripper_limits_m', [0., .07]), 2)
        if self.width_bounds[0] < 0 or self.width_bounds[0] >= self.width_bounds[1]:
            raise ValueError('Invalid gripper limits')
        self.width_speed = positive(cfg.get('gripper_speed_m_s', .03), 'gripper_speed_m_s')
        self.effort = cfg.get('gripper_effort', 500)
        if type(self.effort) is not int or not 1 <= self.effort <= 5000:
            raise ValueError('Invalid gripper effort')
        self.recorded_duration = self.times[-1]
        values = np.c_[np.radians(self.states[:, :6]), self.states[:, 6]]
        self.splines = [HermiteSpline(self.times, values[:, i]) for i in range(7)]
        peak_v, peak_a = 0., 0.
        for i, spline in enumerate(self.splines[:6]):
            low, high = extrema(spline, 0, self.recorded_duration)
            # The rounded SDK target must also stay within the intersected limits.
            if (low < self.bounds[i, 0] or high > self.bounds[i, 1] or
                    math.radians(round(math.degrees(low)*1000)/1000) < self.bounds[i, 0] or
                    math.radians(round(math.degrees(high)*1000)/1000) > self.bounds[i, 1]):
                raise ValueError(f'Joint {i+1} recording/interpolation exceeds model limits')
            peak_v = max(peak_v, *map(abs, extrema(spline, 1, self.recorded_duration)))
            peak_a = max(peak_a, *map(abs, extrema(spline, 2, self.recorded_duration)))
        # Reserve 20% for scheduling/quantization; do not raise any profile caps.
        self.scale = max(1 / self.speed, peak_v / (.8*self.velocity),
                         math.sqrt(peak_a / (.8*self.acceleration)))
        if replay_gripper:
            low, high = extrema(self.splines[6], 0, self.recorded_duration)
            if low < self.width_bounds[0] or high > self.width_bounds[1]:
                raise ValueError('Recorded/interpolated gripper width exceeds configured limits')
            peak_width = max(map(abs, extrema(self.splines[6], 1, self.recorded_duration)))
            self.scale = max(self.scale, peak_width / (.8*self.width_speed))
        self.duration = self.recorded_duration * self.scale

    def sample(self, elapsed):
        t = min(self.recorded_duration, max(0., elapsed / self.scale))
        state = np.array([s(t) for s in self.splines])
        velocity = np.array([s(t, 1) / self.scale for s in self.splines[:6]])
        return state, velocity

    def summary(self):
        return dict(episode=str(self.episode), record_format=self.manifest['format'], processing=self.manifest.get('processing'), frames=len(self.states), task=self.manifest.get('task'),
                    recorded_duration_s=round(float(self.recorded_duration), 3),
                    requested_speed=self.speed, effective_speed=round(1/self.scale, 4),
                    playback_duration_s=round(float(self.duration), 3),
                    first_joints_deg=self.states[0, :6].tolist(),
                    last_joints_deg=self.states[-1, :6].tolist(),
                    gripper_replay=self.replay_gripper,
                    note='Automatic retiming includes velocity/acceleration limits; no collision planner. Derived input is filtered replay-only targets.' if self.manifest['format']=='piper-replay-v1' else 'Automatic retiming includes velocity/acceleration limits; no collision planner or denoising')


@dataclass
class Transition:
    initial: np.ndarray
    target: np.ndarray
    duration: float

    def sample(self, elapsed):
        u = min(1., max(0., elapsed / self.duration))
        h = 10*u**3 - 15*u**4 + 6*u**5
        dh = (30*u**2 - 60*u**3 + 30*u**4) / self.duration
        return self.initial + (self.target-self.initial)*h, (self.target-self.initial)[:6]*dh


def start_transition(initial, plan):
    initial = vector(initial, 7)
    target, _ = plan.sample(0.)
    if np.any(initial[:6] < plan.bounds[:, 0]) or np.any(initial[:6] > plan.bounds[:, 1]):
        raise ValueError('Current joints outside model bounds')
    delta = np.max(abs(target[:6] - initial[:6]))
    velocity = min(math.radians(5.), plan.velocity)
    duration = max(2., 1.875*delta/(.8*velocity), math.sqrt(5.774*delta/(.8*plan.acceleration)))
    if plan.replay_gripper:
        if not plan.width_bounds[0] <= initial[6] <= plan.width_bounds[1]:
            raise ValueError('Current gripper width outside configured limits')
        duration = max(duration, 1.875*abs(target[6]-initial[6])/(.8*plan.width_speed))
    else:
        target[6] = initial[6]
    return Transition(initial, target, duration)


def checked_feedback(source, cfg, now, *, gripper=False):
    f, _, snap = source.sample(now, cfg['publish_period_s'])
    if (not f.ready or not f.healthy or not 0 <= now-f.received <= cfg['feedback_max_age_s']
            or snap.get('gripper_faults', 0)):
        raise RuntimeError('Replay requires fresh, healthy, already-enabled CAN/MOVE_J feedback')
    vector(f.q)
    if not math.isfinite(f.width) or f.width < 0:
        raise RuntimeError('Invalid gripper feedback')
    if gripper and not snap.get('gripper_enabled'):
        raise RuntimeError('Gripper must already be enabled for gripper replay')
    return f


class ReplayOutput:
    """Direct SDK output with the existing guard/hold; no fabricated Servo status."""
    def __init__(self, plan, writer, *, clock=time.monotonic):
        from servo_bridge.sdk_worker import Worker
        self.plan, self.writer, self.clock = plan, writer, clock
        self.holder = Worker(plan.cfg, writer, clock=clock)
        self.previous_width = None
        self.previous_time = None

    def send(self, state, velocity, feedback):
        state, velocity = vector(state, 7), vector(velocity)
        now, plan = self.clock(), self.plan
        gripper = None
        if plan.replay_gripper:
            gripper = self.holder._gripper_command(dict(gripper=dict(width_m=float(state[6]), effort=plan.effort)))
            if self.previous_width is not None:
                dt = now-self.previous_time
                if abs(gripper[0]/1e6-self.previous_width) > plan.width_speed*min(dt, plan.cfg['publish_period_s'])+2e-6:
                    raise ValueError('Gripper replay step/speed exceeded')
        raw = self.holder.guard.validate(list(NAMES), state[:6], velocity, feedback=feedback,
                  stamp_age_s=0., horizon_s=plan.cfg['publish_period_s'], now=now)
        self.holder.moved = True  # Even a partially failed SDK write needs a hold attempt.
        self.writer.joint(raw)
        if gripper is not None:
            self.writer.gripper(*gripper)
            self.previous_width = gripper[0]/1e6
        self.previous_time = now

    def hold(self, feedback):
        self.holder.hold(feedback)


def play_segment(segment, source, output, plan, check_cancel, log, *, clock=time.monotonic, sleep=time.sleep):
    period = plan.cfg['publish_period_s']
    previous = output.holder.guard.last_received
    start = clock() if previous is None else max(clock(), previous + period)
    index = 0
    while True:
        due = start + index*period
        sleep(max(0., due-clock()))
        now = clock()
        if now-due > period*.5:
            raise RuntimeError('Replay scheduler missed a tick; refusing to catch up with a target jump')
        check_cancel()
        f = checked_feedback(source, plan.cfg, now, gripper=plan.replay_gripper)
        elapsed = min(index*period, segment.duration)
        state, velocity = segment.sample(elapsed)
        output.send(state, velocity, f)
        log(dict(event='target', monotonic=now, elapsed_s=elapsed,
                 target_joints_deg=np.degrees(state[:6]).tolist(), measured_joints_deg=np.degrees(f.q).tolist(),
                 gripper_target_m=float(state[6]) if plan.replay_gripper else None, measured_width_m=f.width))
        if elapsed >= segment.duration:
            return
        index += 1


def settle(target, source, output, plan, check_cancel, *, clock=time.monotonic, sleep=time.sleep):
    deadline, good = clock()+5., 0
    while clock() < deadline:
        sleep(plan.cfg['publish_period_s'])
        check_cancel()
        f = checked_feedback(source, plan.cfg, clock(), gripper=plan.replay_gripper)
        output.send(target, np.zeros(6), f)
        reached = np.max(abs(f.q-target[:6])) < math.radians(.5)
        if plan.replay_gripper:
            reached = reached and abs(f.width-target[6]) < .003
        good = good+1 if reached else 0
        if good >= 3:
            return
    raise RuntimeError('Final/start pose tracking did not settle; replay cancelled')


@contextmanager
def writer_lock(path=Path('/tmp/piper-servo-sdk.sock')):
    if path.exists():
        raise RuntimeError('SDK worker socket exists; exit the other controller first')
    with open(str(path)+'.lock', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def execute(plan, output_dir):
    from servo_bridge.sdk_worker import SDKJointWriter
    from servo_bridge.sources import PassiveCan

    with writer_lock():
        source, writer, control = PassiveCan(plan.cfg['can_interface']), None, None
        try:
            deadline = time.monotonic()+3.
            while True:
                try:
                    feedback = checked_feedback(source, plan.cfg, time.monotonic(), gripper=plan.replay_gripper)
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.02)
            transition = start_transition(np.r_[feedback.q, feedback.width], plan)
            print(f'Start transition {transition.duration:.1f}s; playback {plan.duration:.1f}s; gripper={plan.replay_gripper}', flush=True)
            output_dir.mkdir(parents=True, exist_ok=False)
            with (output_dir/'events.jsonl').open('x', buffering=1) as events:
                def log(row):
                    events.write(json.dumps(row, allow_nan=False)+'\n')
                log(dict(event='session', **plan.summary()))
                writer = SDKJointWriter(plan.cfg['can_interface'])
                control = ReplayOutput(plan, writer)
                try:
                    log(dict(event='phase', phase='start_transition'))
                    play_segment(transition, source, control, plan, lambda: None, log)
                    settle(transition.target, source, control, plan, lambda: None)
                    log(dict(event='phase', phase='recorded_playback'))
                    play_segment(plan, source, control, plan, lambda: None, log)
                    settle(plan.sample(plan.duration)[0], source, control, plan, lambda: None)
                    log(dict(event='complete'))
                except BaseException as exc:
                    log(dict(event='aborted', reason=str(exc), error=type(exc).__name__))
                    raise
                finally:
                    if control.holder.moved:
                        try:
                            control.hold(checked_feedback(source, plan.cfg, time.monotonic()))
                            log(dict(event='measured_hold_requested'))
                        except Exception as exc:
                            log(dict(event='hold_unavailable', reason=str(exc)))
                            print(f'Cannot verify position hold: {exc}; use on-site stop measures.', flush=True)
        finally:
            try:
                if writer:
                    writer.close()
            finally:
                source.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--profile', type=Path, default=ROOT/'servo_bridge/config.cartesian.example.json')
    p.add_argument('--speed', type=float, default=.5, help='Requested speed ratio (0,1]; limits may slow further')
    p.add_argument('--max-velocity-deg-s', type=float, default=10.)
    p.add_argument('--max-acceleration-deg-s2', type=float, default=30.)
    p.add_argument('--replay-gripper', action='store_true', help='Explicitly also reproduce recorded opening')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--arm-file', type=Path, help=argparse.SUPPRESS)  # Legacy option, ignored.
    p.add_argument('--output', type=Path, default=ROOT/'outputs/replay'/time.strftime('%Y%m%d_%H%M%S'))
    args = p.parse_args(argv)
    cfg = load_config(args.profile)
    plan = ReplayPlan(args.episode, cfg, speed=args.speed, max_velocity_deg_s=args.max_velocity_deg_s,
                      max_acceleration_deg_s2=args.max_acceleration_deg_s2, replay_gripper=args.replay_gripper)
    print(json.dumps(plan.summary(), indent=2, ensure_ascii=False))
    if not args.execute:
        print('INSPECTION ONLY: no CAN connection, SDK construction, motion or gripper command.')
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Signal received; stop replay and request measured hold')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        execute(plan, args.output)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
