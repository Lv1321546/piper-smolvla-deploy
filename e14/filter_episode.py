"""Offline SG-style joint filtering into a replay-only trajectory; no CAN/SDK."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

from e14.gamepad_kinematics import PiperKinematics
from e14.replay_data import HermiteSpline, ReplayPlan, extrema, load_episode, positive, ROOT
from servo_bridge.core import load_config


def fingerprint(path):
    return {name: hashlib.sha256((path/name).read_bytes()).hexdigest()
            for name in ('manifest.json', 'frames.jsonl')}


def smoothstep(x):
    x = np.clip(x, 0., 1.)
    return x**3 * (10.-15.*x+6.*x*x)


def smooth_joints(times, states, *, window=7, order=2, strength=1., protect=()):
    times, states = np.asarray(times, float), np.asarray(states, float)
    if (times.ndim != 1 or states.shape != (len(times), 7) or len(times)<3
            or not np.isfinite(times).all() or not np.isfinite(states).all()
            or np.any(np.diff(times)<=0)):
        raise ValueError('Finite chronological timestamps and Nx7 states required')
    if type(window) is not int or window<3 or window%2!=1 or window>len(times):
        raise ValueError('window must be odd, >=3 and no longer than the recording')
    if type(order) is not int or not 1<=order<window:
        raise ValueError('Polynomial order must be an integer in [1,window)')
    strength = positive(strength,'strength')
    if strength>1:
        raise ValueError('strength must be <=1')
    half = window//2
    smoothed = np.empty_like(states[:, :6])
    # Fit at actual recorded times, not synthetic uniform timestamps. On an
    # exactly uniform grid this is the usual centered SG polynomial estimator.
    for i in range(len(times)):
        start = max(0, min(i-half, len(times)-window))
        span = slice(start, start+window)
        u = (times[span]-times[i]) / (times[start+window-1]-times[start])
        design = np.vander(u, order+1, increasing=True)
        smoothed[i] = np.linalg.lstsq(design, states[span, :6], rcond=None)[0][0]
    taper = half*float(np.median(np.diff(times)))
    weights = smoothstep(np.minimum(times-times[0], times[-1]-times)/taper)
    for start,end in protect:
        if (not math.isfinite(start) or not math.isfinite(end)
                or not 0<=start<end<=times[-1]-times[0]):
            raise ValueError('Protected intervals must be seconds in [0,recorded_duration] with start<end')
        relative = times-times[0]
        distance = np.maximum(start-relative, relative-end)
        weights = np.minimum(weights, smoothstep(distance/taper))
    result = states.copy()
    result[:, :6] += strength*weights[:, None]*(smoothed-states[:, :6])
    # Exact anchors, including JSON/float identity; jaw was never filtered.
    result[[0,-1], :6] = states[[0,-1], :6]
    return result


def target_manifest(source, manifest, parameters, hashes):
    return dict(schema_version=1, format='piper-replay-v1', purpose='replay_only',
                target_semantics='filtered_recorded_positions', status='complete',
                mock=False, state_names=manifest['state_names'], joint_unit='degree',
                gripper_unit='meter_full_opening', action_mode='absolute_position',
                control_hz=manifest['control_hz'], max_period_error_ratio=manifest['max_period_error_ratio'],
                frames=manifest['frames'], expected_frames=manifest['expected_frames'],
                task=manifest.get('task'), source_task_outcome=manifest.get('task_outcome'),
                task_outcome='unknown', created_unix_s=time.time(),
                processing=dict(method='local_polynomial_sg', **parameters),
                source=dict(path=str(source.resolve()), sha256=hashes),
                note='Hypothetical replay targets, not measured states or executed actions. No camera files copied; not a training dataset.')


def write_targets(path, times_absolute, states, manifest):
    (path/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (path/'frames.jsonl').open('w') as stream:
        for i,(stamp,state) in enumerate(zip(times_absolute,states)):
            stream.write(json.dumps(dict(frame_index=i,recorded_monotonic=float(stamp),target_state=state.tolist()),allow_nan=False)+'\n')


def plan_metrics(plan):
    return dict(playback_duration_s=float(plan.duration), effective_speed=1/plan.scale,
                interpolated_peak_velocity_deg_s=max(math.degrees(max(map(abs,extrema(s,1,plan.recorded_duration)))) for s in plan.splines[:6]),
                interpolated_peak_acceleration_deg_s2=max(math.degrees(max(map(abs,extrema(s,2,plan.recorded_duration)))) for s in plan.splines[:6]))


def compare(before, after, cfg):
    joint_max=[]
    # Both interpolants share knots. Check their difference continuously, not
    # merely the original sample deviations, using exact polynomial extrema.
    for a,b in zip(before.splines[:6],after.splines[:6]):
        delta=HermiteSpline.__new__(HermiteSpline)
        delta.x=a.x;delta.coefficients=b.coefficients-a.coefficients
        joint_max.append(math.degrees(max(map(abs,extrema(delta,0,before.recorded_duration)))))
    times=np.r_[np.concatenate([before.times[:-1]+fraction*np.diff(before.times) for fraction in (0.,.25,.5,.75)]),before.times[-1]]
    times=np.sort(times)
    kin=PiperKinematics(cfg['joint_limits_deg'])
    deviations=[]
    for t in times:
        a=np.array([s(t) for s in before.splines[:6]])
        b=np.array([s(t) for s in after.splines[:6]])
        old,new=kin.forward(a),kin.forward(b)
        mm=float(np.linalg.norm(old[:3,3]-new[:3,3])*1000)
        angle=math.degrees(math.acos(float(np.clip((np.trace(old[:3,:3].T@new[:3,:3])-1)/2,-1,1))))
        deviations.append((float(t),mm,angle))
    deviations=np.array(deviations)
    return dict(before=plan_metrics(before),after=plan_metrics(after),
                joint_max_deviation_deg=joint_max,
                joint_rms_deviation_deg=np.sqrt(np.mean((after.states[:,:6]-before.states[:,:6])**2,axis=0)).tolist(),
                flange_max_sampled_deviation_mm=float(np.max(deviations[:,1])),
                flange_orientation_max_sampled_deviation_deg=float(np.max(deviations[:,2])),
                fk_check_samples=len(times),
                endpoints_preserved=bool(np.array_equal(after.states[[0,-1]],before.states[[0,-1]])),
                gripper_unchanged=bool(np.array_equal(after.states[:,6],before.states[:,6])),
                note='FK deviations are sampled base_link->link6 model values, not calibrated fingertip/camera errors or collision checks'),deviations


def filter_episode(episode, output, cfg, *, window=7, order=2, strength=1., protect=(),
                   max_joint_deviation_deg=1., max_flange_deviation_mm=5., max_orientation_deviation_deg=2.,
                   speed=1., max_velocity_deg_s=10., max_acceleration_deg_s2=30.):
    source,output=Path(episode).resolve(),Path(output).resolve()
    if output==source or source in output.parents:
        raise ValueError('Output must be separate from the original episode, never inside it')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; choose a new directory')
    thresholds=[positive(v,n) for v,n in ((max_joint_deviation_deg,'max_joint_deviation_deg'),
                  (max_flange_deviation_mm,'max_flange_deviation_mm'),(max_orientation_deviation_deg,'max_orientation_deviation_deg'))]
    hashes=fingerprint(source)
    times,states,manifest=load_episode(source)
    if manifest['format']!='piper-raw-v1':
        raise ValueError('Filter the original real episode, not an already filtered trajectory')
    filtered=smooth_joints(times,states,window=window,order=order,strength=strength,protect=protect)
    parameters=dict(window=window,order=order,strength=strength,protected_intervals_s=list(protect),
                    endpoint_policy='exact anchors with smooth taper',gripper_policy='unchanged',
                    thresholds=dict(joint_deg=thresholds[0],flange_mm=thresholds[1],orientation_deg=thresholds[2]))
    derived=target_manifest(source,manifest,parameters,hashes)
    # ReplayPlan performs the same joint/model/SDK-rounding checks as playback.
    # Temporary targets are never exposed as completed output if checks fail.
    options=dict(speed=speed,max_velocity_deg_s=max_velocity_deg_s,max_acceleration_deg_s2=max_acceleration_deg_s2)
    before=ReplayPlan(source,cfg,**options)
    with tempfile.TemporaryDirectory(prefix='piper-filter-check-') as temp:
        candidate=Path(temp)
        first=json.loads((source/'frames.jsonl').read_text().splitlines()[0])['captured_monotonic']
        absolute=times+first
        write_targets(candidate,absolute,filtered,derived)
        after=ReplayPlan(candidate,cfg,**options)
        # An out-of-range measured opening must never be clipped. It may
        # prohibit optional jaw playback while joint-only playback remains valid.
        try:
            jaw_plan=ReplayPlan(candidate,cfg,replay_gripper=True,**options)
            jaw_check=dict(valid=True,playback_duration_s=float(jaw_plan.duration))
        except ValueError as exc:
            jaw_check=dict(valid=False,reason=str(exc))
        report,deviations=compare(before,after,cfg)
        report['optional_gripper_replay']=jaw_check
        observed=[max(report['joint_max_deviation_deg']),report['flange_max_sampled_deviation_mm'],report['flange_orientation_max_sampled_deviation_deg']]
        for name,value,limit in zip(('joint deviation (deg)','sampled flange deviation (mm)','sampled orientation deviation (deg)'),observed,thresholds):
            if value>limit+1e-9:
                raise ValueError(f'{name}: {value:.4f} exceeds {limit:.4f}; use a smaller window/strength; output not written')
        if fingerprint(source)!=hashes:
            raise RuntimeError('Source recording changed during filtering; output not written')
        report.update(source=str(source),output=str(output),parameters=parameters,
                      planning_comparison_options=options,source_sha256=hashes)
        (candidate/'filter_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with (candidate/'comparison.csv').open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['time_s']+[f'raw_joint{i}_deg' for i in range(1,7)]+[f'filtered_joint{i}_deg' for i in range(1,7)])
            for t,a,b in zip(times,states,filtered):writer.writerow([t,*a[:6],*b[:6]])
        with (candidate/'flange_deviation.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['time_s','flange_deviation_mm','orientation_deviation_deg']);writer.writerows(deviations)
        output.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive creation: cannot overwrite an existing run, even if another
        # process created it while validation was running.
        output.mkdir()
        try:
            for p in candidate.iterdir():
                with (output/p.name).open('xb') as f:f.write(p.read_bytes())
        except BaseException:
            # Retain evidence but ensure an incomplete copy cannot be replayed.
            (output/'manifest.json').write_text(json.dumps(dict(status='incomplete',format='piper-replay-v1')))
            raise
    return report


def interval(text):
    try:
        start,end=map(float,text.split(':'))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use START:END seconds, e.g. 12:15') from exc
    return start,end


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile',type=Path,default=ROOT/'servo_bridge/config.cartesian.example.json')
    p.add_argument('--window',type=int,default=7)
    p.add_argument('--order',type=int,default=2)
    p.add_argument('--strength',type=float,default=1.)
    p.add_argument('--protect',type=interval,action='append',default=[])
    p.add_argument('--max-joint-deviation-deg',type=float,default=1.)
    p.add_argument('--max-flange-deviation-mm',type=float,default=5.)
    p.add_argument('--max-orientation-deviation-deg',type=float,default=2.)
    p.add_argument('--speed',type=float,default=1.,help='For duration comparison only; does not start replay')
    args=p.parse_args(argv)
    report=filter_episode(args.episode,args.output,load_config(args.profile),window=args.window,order=args.order,
           strength=args.strength,protect=args.protect,max_joint_deviation_deg=args.max_joint_deviation_deg,
           max_flange_deviation_mm=args.max_flange_deviation_mm,max_orientation_deviation_deg=args.max_orientation_deviation_deg,speed=args.speed)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('Saved replay-only targets. No CAN/SDK/robot commands. Original recording unchanged.')


if __name__=='__main__':main()
