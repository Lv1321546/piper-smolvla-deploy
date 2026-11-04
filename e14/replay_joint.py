"""Replay observed joint/hand positions on the original recording timeline.

Default is offline validation. Hardware output requires --execute, a commissioned
exclusive writer lock and fresh healthy CAN/MOVE_J feedback.
No mode/enable/reset command is sent. Default execution first returns to the
start pose, then replays smooth joints and the original linear gripper curve.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import time

import numpy as np

from e14.collect_data import ACTION_NAMES
from servo_bridge.core import CommandGuard, load_config, vector
from e14.replay_smoothing import prepare_trajectory


def load_episode(path, cfg):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    expected = dict(format='piper-raw-v1', status='complete', state_names=ACTION_NAMES,
                    joint_unit='degree', gripper_unit='meter_full_opening',
                    action_mode='absolute_position', label_semantics='next_state')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('录制格式、单位、关节顺序或完整状态不匹配')
    if manifest.get('operator_session', {}).get('mode') != 'moveit_servo_joint_gamepad':
        raise ValueError('此入口仅回放关节 demo 的完整录制')
    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) < 2 or len(rows) != manifest.get('frames'):
        raise ValueError('录制帧数不完整')
    states, stamps = [], []
    for index, row in enumerate(rows):
        if row.get('frame_index') != index:
            raise ValueError('录制帧索引不连续')
        states.append(vector(row['state'], 7))
        stamp = row['captured_monotonic']
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError('录制时间戳无效')
        stamps.append(stamp)
    states, stamps = np.asarray(states), np.asarray(stamps)
    stamps -= stamps[0]
    dt = np.diff(stamps)
    hz = manifest.get('control_hz')
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError('录制频率无效')
    if np.any(dt <= 0) or np.any(abs(dt - 1/hz) > .5/hz):
        raise ValueError('录制时间戳倒退、重复或存在丢帧')
    bounds = np.vstack([np.degrees(CommandGuard(cfg).limits), cfg['gripper_limits_m']])
    quantized = np.column_stack([np.round(states[:, :6]*1000)/1000,
                                np.round(states[:, 6]*1e6)/1e6])
    for values in (states, quantized):
        if np.any(values < bounds[:, 0]) or np.any(values > bounds[:, 1]):
            raise ValueError('录制位置超出当前关节/夹爪限位')
    velocity = np.abs(np.diff(states, axis=0) / dt[:, None])
    if np.any(velocity[:, :6] > math.degrees(cfg['maximum_joint_velocity_rad_s']) + 1e-6):
        raise ValueError('原速轨迹超过当前关节速度上限；拒绝偷偷降速')
    if np.any(velocity[:, 6] > cfg['gripper_speed_m_s'] + 1e-6):
        raise ValueError('原速夹爪轨迹超过当前夹爪速度上限')
    return manifest, stamps, states


def timeline(stamps, states, period):
    """Uniform linear interpolation plus exact endpoint; no extra knot bursts."""
    ticks, values, _ = prepare_trajectory(stamps, states, period, smoothing='linear')
    yield from zip(ticks, values)


def replay(stamps, states, send, *, period=.02, clock=time.monotonic,
           sleep=time.sleep, max_lateness=.05, prepared=None):
    if prepared is None:
        prepared = prepare_trajectory(stamps,states,period,smoothing='linear')
    ticks, values, summary = prepared
    started = clock()
    count, worst = 0, 0.
    for offset, target in zip(ticks, values):
        deadline = started + offset
        while clock() < deadline:
            sleep(max(0., deadline-clock()))
        late = clock()-deadline
        if late > max_lateness:
            raise RuntimeError('回放调度超时 %.3fs，停止；不会跳帧追赶或改变速度' % late)
        send(target)
        # A slow CAN write must also stop at the final frame.
        if clock()-deadline > max_lateness:
            raise RuntimeError('回放发送超时，停止')
        count += 1
        worst = max(worst, clock()-deadline)
    return dict(commands=count, recorded_duration_s=float(stamps[-1]),
                elapsed_s=clock()-started, max_lateness_s=worst, speed=1.0, **summary)


def alignment_plan(current, target, cfg):
    """Slow approach from any in-limit pose; path is not collision planned."""
    current, target = vector(current, 7), vector(target, 7)
    bounds = np.vstack([np.degrees(CommandGuard(cfg).limits), cfg['gripper_limits_m']])
    for value in (current, target):
        quantized = np.r_[np.round(value[:6]*1000)/1000, np.round(value[6]*1e6)/1e6]
        if (np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]) or
                np.any(quantized < bounds[:, 0]) or np.any(quantized > bounds[:, 1])):
            raise ValueError('首帧对齐位置超限')
    delta = abs(target-current)
    rates = np.r_[np.full(6, min(2., math.degrees(cfg['maximum_joint_velocity_rad_s']))),
                  min(.005, cfg['gripper_speed_m_s'])]
    # Cubic smoothstep peaks at 1.5 * average velocity and starts/ends at zero.
    acceleration = min(math.radians(10), cfg['maximum_joint_acceleration_rad_s2'])
    duration = max(.5, float(np.max(1.5*delta/rates)),
                   math.sqrt(6*math.radians(float(max(delta[:6])))/acceleration))
    offsets = np.unique(np.r_[np.arange(0, duration, cfg['publish_period_s']), duration])
    u = offsets/duration
    positions = current + (3*u*u-2*u*u*u)[:, None]*(target-current)
    return offsets, positions


def execute(path, manifest, stamps, states, config, cfg, arm_file, *, align_start=False, smoothing='smooth'):
    if manifest.get('mock') is not False:
        raise ValueError('模拟录制禁止实机回放')
    if config['robot']['can_interface'] != cfg['can_interface']:
        raise ValueError('CAN 配置不一致')
    # Prepare and validate the entire smooth path before importing/opening SDK.
    prepared = prepare_trajectory(stamps,states,cfg['publish_period_s'],cfg,smoothing=smoothing)
    print('回放轨迹：%s，固定 %.0f Hz，最大平滑偏差 %.3f°。' %
          (smoothing,prepared[2]['send_hz'],prepared[2]['joint_max_deviation_deg']),flush=True)
    from servo_bridge.sdk_worker import SDKJointWriter
    from e14.joint_collect import PassiveRobot
    bounds = np.vstack([np.degrees(CommandGuard(cfg).limits), cfg['gripper_limits_m']])
    robot, writer, moved = PassiveRobot(config), None, False
    max_error = cfg['maximum_tracking_error_deg']
    config['robot']['feedback_max_age_ms'] = min(config['robot']['feedback_max_age_ms'], cfg['feedback_max_age_s']*1000)

    def measured():
        snap = robot.get_feedback()
        value = vector(snap['state'], 7)
        if np.any(value < bounds[:, 0]) or np.any(value > bounds[:, 1]):
            raise RuntimeError('当前反馈越限，关闭输出')
        return value

    with open('/tmp/piper-servo-sdk.sock.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if Path('/tmp/piper-servo-sdk.sock').exists():
            raise RuntimeError('检测到已有控制 socket，请先退出采集 demo')
        try:
            robot.connect()
            current = measured()
            approach = alignment_plan(current, states[0], cfg) if align_start else None
            if not align_start and (np.max(abs(current[:6]-states[0, :6])) > 1.5 or abs(current[6]-states[0, 6]) > .005):
                raise RuntimeError('请先用关节 demo 对齐录制首帧，再退出 demo。首帧：%s；当前：%s' %
                                   (states[0].round(4).tolist(), current.round(4).tolist()))
            writer = SDKJointWriter(cfg['can_interface'])

            def send(target):
                nonlocal moved
                actual = measured()
                if np.max(abs(target[:6]-actual[:6])) > max_error or abs(target[6]-actual[6]) > .02:
                    raise RuntimeError('实际位置跟随误差过大，停止原速回放')
                moved = True  # Include a partially failed CAN send in hold cleanup.
                writer.joint([round(v*1000) for v in target[:6]])
                writer.gripper(round(target[6]*1e6), cfg['gripper_effort'])

            if approach is not None:
                print('低速对齐首帧：关节不超过 2°/s，夹爪不超过 5mm/s，预计 %.2f 秒；随后按录制原速回放。' % approach[0][-1], flush=True)
                replay(*approach, send, period=cfg['publish_period_s'])
                deadline, settled = time.monotonic()+3, 0
                while settled < 3:
                    actual = measured()
                    if np.max(abs(actual[:6]-states[0, :6])) > max_error or abs(actual[6]-states[0, 6]) > .02:
                        raise RuntimeError('首帧对齐跟随误差过大，取消回放')
                    arrived = np.max(abs(actual[:6]-states[0, :6])) <= .5 and abs(actual[6]-states[0, 6]) <= .002
                    settled = settled+1 if arrived else 0
                    if time.monotonic() > deadline:
                        raise RuntimeError('首帧对齐未到位，取消回放')
                    time.sleep(cfg['publish_period_s'])
                print('首帧对齐完成，开始原速回放。', flush=True)
            result = replay(stamps, states, send, period=cfg['publish_period_s'],prepared=prepared)
            # Leave the final target active on success; holding current now would
            # truncate the final movement before the robot has time to reach it.
            deadline = time.monotonic()+2
            while True:
                final = measured()
                if np.max(abs(final[:6]-states[-1, :6])) > max_error or abs(final[6]-states[-1, 6]) > .02:
                    raise RuntimeError('回放末端跟随误差过大')
                if np.max(abs(final[:6]-states[-1, :6])) <= .5 and abs(final[6]-states[-1, 6]) <= .002:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('回放末端未到位')
                time.sleep(.02)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except BaseException:
            if writer is not None and moved:
                try:
                    actual = measured()
                    writer.joint([round(v*1000) for v in actual[:6]])
                    writer.gripper(round(actual[6]*1e6), cfg['gripper_effort'])
                    print('回放已中断，已请求保持实测关节和夹爪位置。', flush=True)
                except Exception as exc:
                    print('无法确认位置保持，请使用现场停止措施：%s' % exc, flush=True)
            raise
        finally:
            if writer is not None:
                writer.close()
            robot.disconnect()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', nargs='?', type=Path, help='默认选择最新完整关节录制')
    parser.add_argument('--root', type=Path, default=Path('data/piper_raw_joint'))
    parser.add_argument('--config', type=Path, default=Path('config.local.json'))
    parser.add_argument('--profile', default=('servo_bridge/config.joint.local.json' if Path('servo_bridge/config.joint.local.json').is_file() else 'servo_bridge/config.joint.example.json'))
    parser.add_argument('--arm-file', help=argparse.SUPPRESS)  # Legacy, ignored.
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoothing', choices=('smooth','linear'), default='smooth',
                        help='默认平滑关节回放；linear 为线性插值对照，均保持原总时长')
    parser.add_argument('--align-start', dest='align_start', action='store_true', default=True,
                        help='先低速返回首帧，到位后原速回放（默认）')
    parser.add_argument('--no-align-start', dest='align_start', action='store_false',
                        help='关闭自动返回，仅允许从首帧附近回放')
    parser.add_argument('--mock', action='store_true', help='虚拟时钟验证，不接硬件')
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('回放收到停止信号')
    signal.signal(signal.SIGTERM, interrupted)
    if args.mock and args.execute:
        parser.error('--mock 与 --execute 不能同时使用')
    cfg = load_config(args.profile)
    try:
        candidates = []
        for path in sorted(args.root.glob('episode_*')):
            try:
                m, t, q = load_episode(path, cfg)
                candidates.append((path, m, t, q))
            except (ValueError, KeyError, OSError) as exc:
                print('不可回放 %s：%s' % (path, exc))
        if args.list:
            for p, m, t, _ in candidates:
                print('%s  %.3f 秒  %d 帧  %s' % (p, t[-1], m['frames'], m['task_outcome']))
            if not candidates:
                print('暂无完整关节录制，请先重新录制。')
            return
        if args.episode:
            m, t, q = load_episode(args.episode, cfg)
            path = args.episode
        elif candidates:
            path, m, t, q = candidates[-1]
        else:
            raise ValueError('暂无完整关节录制，请先运行 start_joint_demo.sh 录制')
        print('%s：%.3f 秒，%d 帧，原速 1.0×，首帧 %s' % (path, t[-1], len(t), q[0].round(4).tolist()))
        if args.execute:
            execute(path, m, t, q, json.loads(args.config.read_text()), cfg, args.arm_file,
                    align_start=args.align_start,smoothing=args.smoothing)
        elif args.mock:
            prepared = prepare_trajectory(t,q,cfg['publish_period_s'],cfg,smoothing=args.smoothing)
            now = [0.]
            result = replay(t, q, lambda _: None, clock=lambda: now[0],
                            sleep=lambda dt: now.__setitem__(0, now[0]+dt),prepared=prepared)
            print(json.dumps(result, indent=2))
        else:
            _,_,summary = prepare_trajectory(t,q,cfg['publish_period_s'],cfg,smoothing=args.smoothing)
            print(json.dumps(summary,ensure_ascii=False,indent=2))
            print('轨迹检查通过，未连接机械臂。实机回放请加 --execute。')
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, '%s\n' % exc)


if __name__ == '__main__':
    main()
