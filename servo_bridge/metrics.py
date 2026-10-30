"""Summarize actual Servo output audits without treating replay as execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from e14.gamepad_kinematics import PiperKinematics
from servo_bridge.core import CommandGuard, NAMES, STATUS
from servo_bridge.model_audit import statistics


def summarize(path):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    session = next(r for r in rows if r['event'] == 'session')
    inputs = [r for r in rows if r['event'] == 'input']
    outputs = [r for r in rows if r['event'] == 'trajectory']
    finished = next((r for r in rows if r['event'] == 'finished'), {})
    cfg = session['config']
    kin = PiperKinematics(cfg['joint_limits_deg'])
    bounds = CommandGuard(cfg).limits
    samples = {r['seq']: r for r in inputs}
    errors_p, errors_r, leads, declared_v, acceleration = [], [], [], [], []
    predicted_signed_speed = []
    qs, ts, vs = [], [], []
    position_violations = 0
    halted_nonzero_outputs = 0
    original_failed_counter = Counter()
    for row in outputs:
        if row['names'] != list(NAMES) or len(row['positions']) != 6 or len(row['velocities']) != 6:
            continue
        sample = samples.get(row['input_seq'])
        q = np.array(row['positions']); v = np.array(row['velocities'])
        qs.append(q); vs.append(v); ts.append(row['monotonic'])
        position_violations += int(np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]))
        declared_v.append(np.max(np.abs(v)))
        if sample:
            actual = np.array(sample['measured_q_rad'])
            pose, requested = kin.forward(actual), np.array(sample['twist'])
            planned = kin.forward(q)
            dp = planned[:3, 3] - pose[:3, 3]
            dr = Rotation.from_matrix(planned[:3, :3] @ pose[:3, :3].T).as_rotvec()
            errors_p.append(np.linalg.norm(dp - requested[:3] * row['horizon_s']))
            errors_r.append(np.linalg.norm(dr - requested[3:] * row['horizon_s']))
            leads.append(np.max(np.abs(q - actual)))
            if np.linalg.norm(requested[:3]) > 1e-6:
                predicted_signed_speed.append(np.dot(dp, requested[:3] / np.linalg.norm(requested[:3])) / row['horizon_s'])
            if row['status'] in (2, 4):
                halted_nonzero_outputs += int(np.max(np.abs(q - actual)) > 1e-8 or np.max(np.abs(v)) > 1e-8)
            if sample['original_ik_accepted'] is False:
                original_failed_counter[STATUS.get(row['status'], str(row['status']))] += 1
    max_step = 0.
    position_derived_v = []
    if len(qs) > 1:
        dq = np.diff(np.array(qs), axis=0)
        dt = np.diff(np.array(ts))
        max_step = float(np.max(np.abs(dq)))
        valid = dt > 1e-6
        position_derived_v = np.max(np.abs(dq[valid] / dt[valid, None]), axis=1).tolist()
        acceleration = np.max(np.abs(np.diff(np.array(vs), axis=0)[valid] / dt[valid, None]), axis=1).tolist()
    phases = {}
    if session['source'] == 'sim':
        for name, start, end, axis in [('X', 1.2, 4., 0), ('Y', 5.2, 8., 1), ('Z', 9.2, 12., 2)]:
            chosen = [r for r in inputs if start <= r['elapsed_s'] <= end]
            if len(chosen) < 2:
                continue
            positions = np.array([r['measured_pose'] for r in chosen])[:, :3, 3]
            times = np.array([r['elapsed_s'] for r in chosen])
            requested = np.array([r['twist'] for r in chosen])[:, :3]
            requested_integral = np.sum(requested[:-1] * np.diff(times)[:, None], axis=0)
            displacement = positions[-1] - positions[0]
            phases[name] = {'measured_displacement_m': displacement.tolist(), 'input_integral_m': requested_integral.tolist(),
                            'position_tracking_residual_m': float(np.linalg.norm(displacement - requested_integral)),
                            'reverse_steps_above_1um': int(np.sum(np.diff(positions[:, axis]) < -1e-6)),
                            'cross_axis_drift_m': float(np.linalg.norm(np.delete(displacement, axis))),
                            'max_heading_error_rad': float(max(r['orientation_error_rad'] or 0 for r in chosen))}
    heading = [r['orientation_error_rad'] for r in inputs if r['orientation_error_rad'] is not None]
    hold_heading = [r['orientation_error_rad'] for r in inputs if r['orientation_error_rad'] is not None and r['elapsed_s'] < 13.]
    cpu = finished.get('child_cpu_s', {})
    return {
        'events_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'source': session['source'], 'hardware_output': session['hardware_output'], 'input_cycles': len(inputs), 'trajectory_count': len(outputs),
        'candidate_accept_count': sum(r['accepted'] for r in outputs),
        'candidate_rejections': dict(Counter(r['rejection'] for r in outputs if r['rejection'])),
        'output_status_counts': dict(Counter(STATUS.get(r['status'], str(r['status'])) for r in outputs)),
        'on_original_ik_failed_cycles_output_status_counts': dict(original_failed_counter),
        'on_original_ik_failed_input_cycles_latest_status_counts': dict(Counter(
            STATUS.get(r['status'], str(r['status'])) for r in inputs if r['original_ik_accepted'] is False)),
        'joint_position_violations': position_violations,
        'max_output_step_deg': float(np.degrees(max_step)),
        'max_candidate_lead_deg': float(np.degrees(max(leads))) if leads else None,
        'declared_joint_velocity_rad_s': statistics(declared_v),
        'position_difference_velocity_rad_s': statistics(position_derived_v),
        'declared_velocity_difference_acceleration_rad_s2': statistics(acceleration),
        'halt_status_nonzero_candidate_count': halted_nonzero_outputs,
        'one_period_candidate_translation_residual_m': statistics(errors_p),
        'one_period_candidate_rotation_residual_rad': statistics(errors_r),
        'instantaneous_candidate_progress_m_s': statistics(predicted_signed_speed),
        'reference_heading_error_rad': statistics(heading),
        'pre_rotation_heading_error_rad': statistics(hold_heading),
        'jacobian_condition': statistics([r['jacobian_condition'] for r in inputs]),
        'input_wrapper_processing_ms': statistics([r['processing_ms'] for r in inputs]),
        'output_guard_processing_ms': statistics([r['processing_ms'] for r in outputs]),
        'latest_input_to_output_callback_ms_includes_servo_phase': statistics([r['input_to_callback_ms'] for r in outputs if r['input_to_callback_ms'] is not None]),
        'child_cpu_s_including_collision_threads': cpu,
        'servo_cpu_ms_per_output_amortized_not_single_calculation': cpu.get('servo_node', 0) * 1000. / max(len(outputs), 1) if cpu else None,
        'fk_same_stamp_tf_position_error_m': statistics([r['tf_latest_position_gap_m'] for r in inputs if r['tf_latest_position_gap_m'] is not None]),
        'fk_same_stamp_tf_angle_error_rad': statistics([r['tf_same_stamp_angle_error_rad'] for r in inputs if r['tf_same_stamp_angle_error_rad'] is not None]),
        'simulated_xyz_phases': phases,
        'scheduler_overruns': sum(r['event'] == 'scheduler_overrun' for r in rows),
        'limitations': [
            'Log replay is open-loop with recorded real feedback: candidate output cannot change subsequent measurements.',
            'Synthetic first-order plant is not PiPER motor/firmware/contact dynamics.',
            'Input callback latency includes asynchronous Servo period and DDS scheduling; it is not IK calculation time.',
            'Butterworth smoothing does not prove a hard acceleration/jerk bound. The independent guard rejects violations.',
            'No physical motion, zero calibration, fingertip TCP, wrist-camera/table calibration or emergency stopping was validated.'
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {str(Path(p).parent): summarize(p) for p in args.events}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    for p, r in result.items():
        print(p, r['input_cycles'], r['output_status_counts'], r['candidate_rejections'])


if __name__ == '__main__':
    main()
