"""Capture a validated startup reference from passive CAN joint feedback."""
import argparse
import datetime
import json
from pathlib import Path
import time

import numpy as np

from e14.gamepad_kinematics import PiperKinematics
from e14.piper_robot import ensure_can
from servo_bridge.core import CommandGuard, load_config, sdk_units, vector
from servo_bridge.prepare_control import inspect_snapshot


def reference_from_snapshot(snapshot, cfg, collection, *, now):
    stamp = snapshot.get('feedback_monotonic')
    if not isinstance(stamp, (int, float)) or not 0 <= now-stamp <= cfg['feedback_max_age_s']:
        raise ValueError('初始状态需要完整、新鲜的 CAN 反馈')
    state = np.asarray(snapshot['state'], dtype=float)
    limits = CommandGuard(cfg).limits
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('初始关节和夹爪反馈无效')
    errors = []
    for i, q in enumerate(np.radians(state[:6])):
        if not limits[i, 0] <= q <= limits[i, 1]:
            errors.append('J%d=%.3f° 超出模型范围 [%.3f, %.3f]°' %
                          (i+1, state[i], *np.degrees(limits[i])))
    jaw_low = max(cfg['gripper_limits_m'][0], collection['robot']['gripper_limits_m'][0])
    jaw_high = min(.07, cfg['gripper_limits_m'][1], collection['robot']['gripper_limits_m'][1])
    if not jaw_low <= state[6] <= jaw_high:
        errors.append('夹爪反馈 %.1f mm 超出当前模型范围 [%.1f, %.1f] mm' %
                      (state[6]*1000, jaw_low*1000, jaw_high*1000))
    if errors:
        raise ValueError('；'.join(errors)+'。请现场调整后重新读取初始状态。')
    inspect_snapshot(snapshot, cfg, now=now)
    quantized = np.radians(np.asarray(sdk_units(np.radians(state[:6])))/1000)
    if np.any(quantized < limits[:, 0]) or np.any(quantized > limits[:, 1]):
        raise ValueError('初始关节在 SDK 量化后超出模型范围')
    pose, jac = PiperKinematics(cfg['joint_limits_deg']).forward(np.radians(state[:6]), jacobian=True)
    return dict(schema_version=1, captured_at=datetime.datetime.now().astimezone().isoformat(),
                source='passive_can_joint_feedback', joint_unit='degree',
                joint_angles=state[:6].tolist(), gripper_width_m=float(state[6]),
                base_to_link6=pose.tolist(), pose_source='forward_kinematics_from_measured_joints',
                control_frame='link6', tcp_calibrated=False,
                feedback_monotonic=stamp, minimum_joint_margin_deg=float(np.min(np.minimum(
                    state[:6]-np.degrees(limits[:, 0]), np.degrees(limits[:, 1])-state[:6]))),
                jacobian_condition=float(np.linalg.cond(jac)))


def load_fixed_reference(path, cfg):
    reference = json.loads(Path(path).read_text())
    if reference.get('schema_version') != 1 or reference.get('joint_unit') != 'degree':
        raise ValueError('固定起点文件需要 schema_version=1 和 degree 单位')
    joints = vector(reference.get('joint_angles'))
    from servo_bridge.joint_start_pose import validate_pose
    validate_pose(' '.join(map(str, joints)), cfg)
    bounds = CommandGuard(cfg).limits
    margin = cfg.get('cartesian_joint_margin_rad', .1)
    if (np.any(np.radians(joints) < bounds[:, 0]+margin) or
            np.any(np.radians(joints) > bounds[:, 1]-margin)):
        raise ValueError('固定起点必须位于 A/H 返回的关节软限位内')
    return reference


def capture_reference(cfg, collection, *, timeout=3.):
    from servo_bridge.sources import PassiveCan
    ensure_can(cfg['can_interface'])
    source = PassiveCan(cfg['can_interface'])
    try:
        deadline = time.monotonic()+timeout
        while True:
            try:
                snapshot = source.store.snapshot()
                age = time.monotonic()-snapshot['feedback_monotonic']
                if not 0 <= age <= cfg['feedback_max_age_s']:
                    raise RuntimeError('CAN 反馈已过期')
                break
            except RuntimeError as exc:
                if time.monotonic() >= deadline:
                    raise RuntimeError('初始状态读取超时，请检查供电和 CAN：'+str(exc)) from exc
                time.sleep(.02)
        return reference_from_snapshot(snapshot, cfg, collection, now=time.monotonic())
    finally:
        source.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--fixed-home', type=Path)
    parser.add_argument('--print-joints', action='store_true')
    args = parser.parse_args()
    try:
        cfg = load_config(args.profile)
        collection = json.loads(Path(args.config).read_text())
        if cfg['can_interface'] != collection['robot']['can_interface']:
            raise ValueError('CAN 接口配置不一致')
        if cfg['joint_limits_deg'] != collection['robot']['joint_limits_deg']:
            raise ValueError('关节限位配置不一致')
        reference = (load_fixed_reference(args.fixed_home, cfg) if args.fixed_home
                     else capture_reference(cfg, collection))
        if args.print_joints:
            if not args.fixed_home:
                raise ValueError('--print-joints 需要 --fixed-home')
            print(' '.join(str(q) for q in reference['joint_angles']))
            return
        print(('固定起始关节（度）：' if args.fixed_home else '实测初始关节（度）：')+
              str(reference['joint_angles']), flush=True)
        print('夹爪 %.1f mm；模型 link6 位置（米）：%s' % (
            reference['gripper_width_m']*1000, [row[3] for row in reference['base_to_link6'][:3]]), flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix+'.tmp')
            temporary.write_text(json.dumps(reference, ensure_ascii=False, indent=2)+'\n')
            temporary.replace(args.output)
            print('初始状态已保存：'+str(args.output), flush=True)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, '初始状态检查失败：'+str(exc)+'\n')


if __name__ == '__main__':
    main()
