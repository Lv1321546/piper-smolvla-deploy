"""Explicit launcher-only CAN/MOVE_J takeover; never operates the gripper.

The collection facade and teleoperation worker remain receive-only / mode-free.
--check never constructs the SDK. Actual takeover requires --execute and the
explicit output flag, and is serialized with every other SDK writer.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np

from servo_bridge.core import CommandGuard, load_config
from servo_bridge.feedback_diagnostics import readiness_description


def inspect_snapshot(snapshot, cfg, *, now):
    stamp = snapshot.get('feedback_monotonic')
    if (not isinstance(stamp, (float, int)) or not math.isfinite(stamp) or
            not 0 <= now-stamp <= cfg['feedback_max_age_s']):
        raise RuntimeError('CAN 接管需要新鲜的完整反馈；反馈已过期或缺失')
    if (snapshot.get('arm_status') != 0 or snapshot.get('err_code') != 0 or
            len(snapshot.get('motor_faults', [])) != 6 or any(snapshot['motor_faults']) or
            snapshot.get('gripper_faults', 0) != 0):
        raise RuntimeError('存在机械臂/电机/夹爪故障；不自动清错或复位')
    if snapshot.get('ctrl_mode') not in (0, 1, 2):
        raise RuntimeError('仅支持待机、已结束示教或 CAN 模式；不接管其他控制源')
    if (snapshot.get('teach_status') in (1, 3) or
            snapshot.get('ctrl_mode') == 2 and snapshot.get('teach_status') not in (0, 2)):
        raise RuntimeError('示教记录/执行尚未结束，请先用实体按钮结束示教')
    if len(snapshot.get('enabled', [])) != 6:
        raise RuntimeError('电机使能反馈不完整')
    state = np.asarray(snapshot.get('state'), dtype=float)
    limits = CommandGuard(cfg).limits
    if state.shape != (7,) or not np.isfinite(state).all():
        raise RuntimeError('当前关节/夹爪反馈无效')
    quantized = np.r_[np.round(state[:6]*1000)/1000, np.round(state[6]*1e6)/1e6]
    for values in (state, quantized):
        if np.any(np.radians(values[:6]) < limits[:, 0]) or np.any(np.radians(values[:6]) > limits[:, 1]):
            raise RuntimeError('当前关节反馈超出模型/配置限位，不能接管')
        if not cfg['gripper_limits_m'][0] <= values[6] <= cfg['gripper_limits_m'][1]:
            raise RuntimeError('当前夹爪反馈超限，不能接管；不会自动开合或重置夹爪')
    return (snapshot['ctrl_mode'] == 1 and snapshot.get('mode_feed') == 1
            and all(snapshot['enabled']))


def prepare_control(cfg, read_feedback, factory, *, clock=time.monotonic,
                    sleep=time.sleep, timeout=5.):
    snapshot = read_feedback()
    if inspect_snapshot(snapshot, cfg, now=clock()):
        print('已处于健康、使能的 CAN/MOVE_J，无需接管。', flush=True)
        return False
    initial = np.asarray(snapshot['state'], float)
    print('启动 CAN 接管：先写当前关节保持目标，再以 5% 切换 CAN/MOVE_J；不操作夹爪。', flush=True)
    arm = factory(cfg['can_interface'], start_sdk_joint_limit=False, start_sdk_gripper_limit=False)
    mode_requested = False

    def checked():
        snap = read_feedback()
        ready = inspect_snapshot(snap, cfg, now=clock())
        if np.max(abs(np.asarray(snap['state'][:6])-initial[:6])) > 1.:
            raise RuntimeError('CAN 接管期间关节偏移超过 1°，取消启动')
        if abs(snap['state'][6]-initial[6]) > .005:
            raise RuntimeError('CAN 接管期间夹爪反馈变化超过 5 mm，取消启动')
        return snap, ready

    def hold(snap):
        arm.JointCtrl(*[round(v*1000) for v in snap['state'][:6]])

    try:
        arm.ConnectPort(piper_init=False, start_thread=False)
        deadline, settled, last_request = clock()+timeout, 0, -math.inf
        while clock() < deadline:
            snapshot, ready = checked()
            status_after_request = snapshot.get('frame_timestamps', {}).get('0x2a1', -math.inf) > last_request
            if ready and mode_requested and status_after_request:
                settled += 1
                if settled >= 3:
                    hold(snapshot)
                    arm.MotionCtrl_2(0x01, 0x01, 70, 0x00)
                    sleep(.05)
                    _, ready = checked()
                    if not ready:
                        raise RuntimeError('CAN 接管后模式/使能未保持，取消启动')
                    print('CAN/MOVE_J 和六关节使能已确认；随后执行原来的低速起始姿态移动。', flush=True)
                    return True
            else:
                settled = 0
            if clock()-last_request >= .1:
                # Replace the old goal before requesting CAN ownership. This
                # sequence still needs physical commissioning for the firmware
                # in use; the measured displacement guard is not an E-stop.
                hold(snapshot)
                mode_requested = True
                arm.MotionCtrl_2(0x01, 0x01, 5, 0x00)
                if not all(snapshot['enabled']):
                    arm.EnableArm(7)
                last_request = clock()
            sleep(.05)
        raise RuntimeError('CAN/MOVE_J 接管或使能确认超时，取消启动')
    except BaseException:
        if mode_requested:
            try:
                snapshot = read_feedback()
                if inspect_snapshot(snapshot, cfg, now=clock()):
                    hold(snapshot)
            except Exception as exc:
                print('无法确认接管后的保持，请使用现场停止措施：'+str(exc), flush=True)
        raise
    finally:
        arm.DisconnectPort()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm-file', help=argparse.SUPPRESS)  # Legacy, ignored.
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--preflight', action='store_true', help='只读检查是否可接管，允许尚未就绪')
    args = parser.parse_args(argv)
    cfg = load_config(args.profile)
    config = json.loads(Path(args.config).read_text())
    if config['robot']['can_interface'] != cfg['can_interface']:
        raise ValueError('CAN 接口配置不一致')
    if config['robot']['joint_limits_deg'] != cfg['joint_limits_deg']:
        raise ValueError('关节限位配置不一致')
    from servo_bridge.sources import PassiveCan
    source = PassiveCan(cfg['can_interface'])
    try:
        def read_feedback():
            return source.sample(time.monotonic(), cfg['publish_period_s'])[2]
        deadline = time.monotonic()+3.
        while True:
            try:
                snapshot = read_feedback()
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)
        ready = inspect_snapshot(snapshot, cfg, now=time.monotonic())
        print('当前关节角：'+str(snapshot['state'][:6]), flush=True)
        print('当前夹爪开口：%.1f mm' % (snapshot['state'][6]*1000), flush=True)
        if args.check or args.preflight:
            print(readiness_description(snapshot), flush=True)
            if not ready and args.check:
                raise RuntimeError('只读检查：正常启动会尝试 CAN 接管；本次未发送任何指令')
            print('只读预检通过；'+('已就绪。' if ready else '正常启动时将执行 CAN 接管。'), flush=True)
            return
        with open('/tmp/piper-servo-sdk.sock.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            from piper_sdk import C_PiperInterface_V2
            def checked_feedback():
                return read_feedback()
            prepare_control(cfg, checked_feedback, C_PiperInterface_V2)
    finally:
        source.close()


if __name__ == '__main__':
    main()
