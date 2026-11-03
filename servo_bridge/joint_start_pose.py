"""Validate and move to the joint demo's startup pose before the worker starts."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np

from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import CommandGuard, load_config

DEFAULT_POSE = '-5.6 90.2 -35.5 -17.4 51.2 -25.5'


def validate_pose(text, cfg):
    target = np.array([float(v) for v in text.split()], dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError('SAFE_POSE 必须是六个有限的关节角度（度）')
    bounds = np.degrees(CommandGuard(cfg).limits)
    margins = np.minimum(target - bounds[:, 0], bounds[:, 1] - target)
    if min(margins) < 5:
        raise ValueError('起始姿态必须在模型/配置限位内，且每个关节至少留 5° 余量')
    pose, jac = PiperKinematics(cfg['joint_limits_deg']).forward(np.radians(target), jacobian=True)
    condition = float(np.linalg.cond(jac))
    if not math.isfinite(condition) or condition > 25:
        raise ValueError('起始姿态条件数必须不超过 25，请调整 SAFE_POSE')
    return target, margins, condition, pose[:3, 3]


def move_to_pose(target, cfg, read_feedback, factory, *, clock=time.monotonic,
                 sleep=time.sleep, timeout=120.):
    """5% MOVE_J; fresh passive feedback; hold current position on failure.

    This is the same joint interpolation approach as the original launcher,
    not a collision-planned trajectory. No gripper/enable/reset commands.
    """
    bounds = np.degrees(CommandGuard(cfg).limits)

    def measured():
        snapshot = read_feedback()  # PassiveRobot also checks mode, health and age.
        q = np.asarray(snapshot['state'][:6], dtype=float)
        if (q.shape != (6,) or not np.isfinite(q).all() or
                np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1])):
            raise RuntimeError('当前关节反馈超出模型/配置范围，不能执行起始移动')
        return q

    # Defend direct calls as well as the command-line path.
    target, _, _, _ = validate_pose(' '.join(map(str, target)), cfg)
    current = measured()
    print('当前 %s，起始目标 %s' % (np.round(current, 2).tolist(), target.tolist()), flush=True)
    if np.max(abs(current - target)) <= 1.5:
        print('已在起始姿态附近，无需移动。', flush=True)
        return False
    arm = factory(cfg['can_interface'], start_sdk_joint_limit=False, start_sdk_gripper_limit=False)
    commanded = False
    try:
        arm.ConnectPort(piper_init=False, start_thread=False)
        arm.MotionCtrl_2(0x01, 0x01, 5, 0x00)
        sleep(.2)
        measured()  # Recheck after connection/mode command, before motion.
        commanded = True  # Even a partially failed write requires a hold attempt.
        arm.JointCtrl(*[round(v * 1000) for v in target])
        print('以 5% 速度移动到起始姿态，请保持运动路径空旷。Ctrl+C 请求当前位置保持。', flush=True)
        deadline, settled = clock() + timeout, 0
        while clock() < deadline:
            error = float(np.max(abs(measured() - target)))
            print('距起始姿态最大关节误差：%.2f°' % error, flush=True)
            settled = settled + 1 if error < 1.5 else 0
            if settled >= 3:
                arm.MotionCtrl_2(0x01, 0x01, 70, 0x00)
                print('已到达起始姿态，准备启动手柄关节控制。', flush=True)
                return True
            sleep(.2)
        raise RuntimeError('移动到起始姿态超时，取消启动采集')
    except BaseException:
        if commanded:
            try:
                hold = [round(v * 1000) for v in measured()]
                quantized = np.asarray(hold) / 1000.
                if np.any(quantized < bounds[:, 0]) or np.any(quantized > bounds[:, 1]):
                    raise RuntimeError('保持目标量化后超限')
                arm.JointCtrl(*hold)
                print('启动移动已中断，已请求保持当前实测位置。', flush=True)
            except Exception as exc:
                print('无法确认当前位置保持，请使用现场停止措施：%s' % exc, flush=True)
        raise
    finally:
        arm.DisconnectPort()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--safe-pose', default=DEFAULT_POSE)
    parser.add_argument('--arm-file', help=argparse.SUPPRESS)  # Legacy option, ignored.
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.profile)
    target, _, _, _ = validate_pose(args.safe_pose, cfg)
    if not args.execute:
        parser.error('起始移动需要 --execute')
    from e14.joint_collect import PassiveRobot
    from piper_sdk import C_PiperInterface_V2
    config = json.loads(Path(args.config).read_text())
    if config['robot']['can_interface'] != cfg['can_interface']:
        raise ValueError('CAN 接口配置不一致')
    # Serialize against the existing SDK worker and other startup moves.
    with open('/tmp/piper-servo-sdk.sock.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        robot = PassiveRobot(config)
        try:
            robot.connect()
            move_to_pose(target, cfg, robot.get_feedback, C_PiperInterface_V2)
        finally:
            robot.disconnect()


if __name__ == '__main__':
    main()
