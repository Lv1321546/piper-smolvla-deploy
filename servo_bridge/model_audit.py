"""Check the recorded robot model offline; never imports the SDK interface."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import PyKDL as kdl
from scipy.spatial.transform import Rotation

from e14.gamepad_kinematics import PiperKinematics, URDF
from servo_bridge.core import NAMES, load_config
from servo_bridge.sources import log_cycles


def statistics(values):
    a = np.asarray(values, float)
    return dict(min=float(np.min(a)), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(np.max(a))) if a.size else None


def kdl_chain(urdf=URDF):
    root = ET.parse(urdf).getroot()
    chain = kdl.Chain()
    parent = 'base_link'
    for name in NAMES:
        joint = root.find(f"joint[@name='{name}']")
        if joint.get('type') != 'revolute' or joint.find('parent').get('link') != parent:
            raise ValueError('Unexpected joint chain/order')
        origin = joint.find('origin')
        xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
        axis = [float(v) for v in joint.find('axis').get('xyz').split()]
        frame = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))
        kj = kdl.Joint(name, frame.p, frame.M * kdl.Vector(*axis), kdl.Joint.RotAxis)
        parent = joint.find('child').get('link')
        chain.addSegment(kdl.Segment(parent, kj, frame))
    if parent != 'link6':
        raise ValueError('Control point is not the expected flange')
    return chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--sdk-fk', required=True, help='Installed piper_sdk/kinematics/piper_fk.py (pure math file only)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sdk_path = Path(args.sdk_fk)
    if sdk_path.name != 'piper_fk.py':
        parser.error('Only the inspected pure SDK kinematics file is allowed')
    spec = importlib.util.spec_from_file_location('offline_piper_fk_only', sdk_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session, rows = log_cycles(args.log)
    kin = PiperKinematics(session['robot']['joint_limits_deg'])
    chain = kdl_chain()
    fk = kdl.ChainFkSolverPos_recursive(chain)
    jac = kdl.ChainJntToJacSolver(chain)
    sdk = {i: module.C_PiperForwardKinematics(i) for i in (0, 1)}
    errors = {key: [] for key in ('kdl_position_m', 'kdl_orientation_rad', 'kdl_jacobian_max_abs',
                                 'sdk_offset0_position_m', 'sdk_offset1_position_m',
                                 'sdk_offset0_orientation_rad', 'sdk_offset1_orientation_rad')}
    cond = []
    limits_violated = 0
    active = []
    previous_gaps, tracking_gaps, tick_ms = [], [], []
    for row in rows:
        original = row.get('planner', {})
        q = np.radians(original.get('measured_joints_deg') or row['feedback_after']['state'][:6])
        pose, j = kin.forward(q, jacobian=True)
        joints = kdl.JntArray(6)
        for i, v in enumerate(q):
            joints[i] = v
        kf, kj = kdl.Frame(), kdl.Jacobian(6)
        fk.JntToCart(joints, kf)
        jac.JntToJac(joints, kj)
        kr = np.array([[kf.M[i, j] for j in range(3)] for i in range(3)])
        kp = np.array([kf.p[i] for i in range(3)])
        errors['kdl_position_m'].append(np.linalg.norm(kp - pose[:3, 3]))
        errors['kdl_orientation_rad'].append(np.linalg.norm(Rotation.from_matrix(kr @ pose[:3, :3].T).as_rotvec()))
        errors['kdl_jacobian_max_abs'].append(np.max(np.abs(np.array([[kj[i, c] for c in range(6)] for i in range(6)]) - j)))
        for offset, solver in sdk.items():
            # Use SDK DH matrices directly to avoid Euler singularity rounding
            # in CalFK's display conversion. No interface, socket or query.
            sdk_pose = np.eye(4)
            transform = solver._C_PiperForwardKinematics__LinkTransformtion
            for i, angle in enumerate(q):
                sdk_pose = sdk_pose @ np.array(transform(solver._alpha[i], solver._a[i], angle + solver._theta[i], solver._d[i])).reshape(4, 4)
            errors[f'sdk_offset{offset}_position_m'].append(np.linalg.norm(sdk_pose[:3, 3] / 1000. - pose[:3, 3]))
            errors[f'sdk_offset{offset}_orientation_rad'].append(np.linalg.norm(Rotation.from_matrix(sdk_pose[:3, :3] @ pose[:3, :3].T).as_rotvec()))
        if original:
            cond.append(float(np.linalg.cond(j)))
            active.append(original)
            target = original.get('target_joints_deg')
            if target is not None:
                target = np.asarray(target)
                bounds = np.asarray(session['robot']['joint_limits_deg'])
                limits_violated += int(np.any(target < bounds[:, 0]) or np.any(target > bounds[:, 1]))
                tracking_gaps.append(np.max(np.abs(target - np.degrees(q))))
                before = row.get('target_before_deg')
                if before is not None:
                    previous_gaps.append(np.max(np.abs(target - np.array(before))))
            tick_ms.append(row['control_elapsed_s'] * 1000.)
    result = {
        'log_sha256': hashlib.sha256(Path(args.log).read_bytes()).hexdigest(),
        'urdf_sha256': hashlib.sha256(Path(URDF).read_bytes()).hexdigest(),
        'sdk_fk_sha256': hashlib.sha256(sdk_path.read_bytes()).hexdigest(),
        'hardware_output': False, 'feedback': 'same recorded feedback; planner pre-command state used where present',
        'chain': list(NAMES), 'planning_frame': 'base_link', 'control_frame': 'link6',
        'control_point_calibrated_to_fingertips_or_camera': False,
        'records': len(rows), 'model_errors': {k: statistics(v) for k, v in errors.items()},
        'legacy': {'planning_cycles': len(active), 'failed': sum(not r.get('accepted') for r in active),
                   'jacobian_condition': statistics(cond), 'tracking_gap_deg': statistics(tracking_gaps),
                   'previous_target_step_deg': statistics(previous_gaps), 'joint_position_violations': limits_violated,
                   'recorded_control_tick_ms': statistics(tick_ms)},
        'interpretation': 'KDL/URDF/SDK comparison checks software conventions; it is not physical TCP, table, camera or joint-zero calibration.'}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
