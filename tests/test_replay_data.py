"""Replay tests use temporary recordings, an ideal plant and fake SDK only."""
import copy
from contextlib import ExitStack
import time
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from e14.replay_data import (ReplayPlan, ReplayOutput, Transition, checked_feedback,
                             execute, load_episode, main, play_segment, settle, start_transition,
                             writer_lock)
from servo_bridge.core import Feedback, load_config

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now
    def sleep(self, dt): self.now += dt


class Plant:
    def __init__(self, clock, q):
        self.clock, self.q = clock, q.copy()
        self.width, self.commands, self.grippers = .06, [], []
        self.ready, self.age, self.fault, self.grip_enabled = True, 0., 0, True
        self.stuck = False
    def sample(self, now, dt):
        return Feedback(self.q.copy(), now-self.age, self.width, True, self.ready), None, dict(gripper_faults=self.fault, gripper_enabled=self.grip_enabled)
    def joint(self, raw):
        self.commands.append(raw)
        if not self.stuck: self.q = np.radians(np.array(raw)/1000)
    def gripper(self, raw, effort):
        self.grippers.append((raw, effort)); self.width = raw/1e6


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name)
        self.cfg = load_config(ROOT/'servo_bridge/config.cartesian.example.json')
        self.times = np.arange(21)*.05
        self.states = np.tile([0., 80., -60., -10., 40., 0., .06], (21,1))
        self.states[:,0] = np.sin(self.times*np.pi)*2
        self.states[:,6] = .06-self.times*.01
        self.write()
    def tearDown(self): self.tmp.cleanup()
    def write(self, rows=None, **overrides):
        manifest = dict(schema_version=1, format='piper-raw-v1', status='complete', mock=False,
                        state_names=[f'joint{i}' for i in range(1,7)]+['gripper_width'], joint_unit='degree',
                        gripper_unit='meter_full_opening', action_mode='absolute_position', frames=21,
                        expected_frames=21, control_hz=20, feedback_max_age_ms=300, max_period_error_ratio=.5, task='push')
        manifest.update(overrides)
        (self.path/'manifest.json').write_text(json.dumps(manifest))
        self.rows = rows if rows is not None else [dict(frame_index=i, captured_monotonic=100.+t,
                       feedback_monotonic=100.+t-.01, state=s.tolist()) for i,(t,s) in enumerate(zip(self.times,self.states))]
        (self.path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
    def plan(self, **kwargs): return ReplayPlan(self.path,self.cfg,**kwargs)
    def test_exact_knots_and_clamped_endpoint_velocity(self):
        p=self.plan()
        for t,s in zip(self.times,self.states):
            state,_=p.sample(t*p.scale)
            np.testing.assert_allclose(np.degrees(state[:6]), s[:6], atol=1e-9)
            self.assertAlmostEqual(state[6],s[6])
        np.testing.assert_allclose(p.sample(0)[1],0,atol=1e-10)
        np.testing.assert_allclose(p.sample(p.duration)[1],0,atol=1e-10)
    def test_retiming_respects_limits_and_preserves_record(self):
        p=self.plan(speed=1,max_velocity_deg_s=.5,max_acceleration_deg_s2=.2)
        self.assertGreater(p.scale,1)
        for t in np.linspace(0,p.duration,1000):
            self.assertLessEqual(np.max(abs(p.sample(t)[1])),p.velocity*.801)
            for s in p.splines[:6]:
                self.assertLessEqual(abs(s(t/p.scale,2)/p.scale**2),p.acceleration*.801)
    def test_bad_records_rejected(self):
        for field,value in [('mock',True),('mock',None),('status','failed'),('frames',22),('joint_unit','radian')]:
            with self.subTest(field=field,value=value):
                self.write(**{field:value})
                with self.assertRaises(ValueError): self.plan()
        self.write(); rows=copy.deepcopy(self.rows); rows[2]['state'][0]=True
        self.write(rows)
        with self.assertRaises(ValueError): self.plan()
    def test_bad_timestamps_and_missing_samples(self):
        for key in ('captured_monotonic','feedback_monotonic'):
            self.write(); rows=copy.deepcopy(self.rows);rows[2][key]=rows[1][key]
            self.write(rows)
            with self.assertRaises(ValueError):self.plan()
        self.write();rows=copy.deepcopy(self.rows);rows[4]['captured_monotonic']+=.04
        self.write(rows)
        with self.assertRaises(ValueError):self.plan()
    def test_limits_not_silently_clipped(self):
        self.states[5,0]=151;self.write()
        with self.assertRaises(ValueError):self.plan()
    def test_gripper_outliers_only_block_gripper_replay(self):
        self.states[:,6]=.09;self.write();self.plan()
        with self.assertRaises(ValueError):self.plan(replay_gripper=True)
    def test_invalid_options(self):
        for speed in (0,2,float('nan'),True):
            with self.assertRaises(ValueError):self.plan(speed=speed)
    def test_transition_starts_at_current_and_is_rest_to_rest(self):
        p=self.plan();q=p.sample(0)[0];q[0]+=math.radians(20)
        tr=start_transition(q,p)
        np.testing.assert_allclose(tr.sample(0)[0],q)
        np.testing.assert_allclose(tr.sample(tr.duration)[0],p.sample(0)[0])
        np.testing.assert_allclose(tr.sample(0)[1],0)
        np.testing.assert_allclose(tr.sample(tr.duration)[1],0,atol=1e-12)
        self.assertGreater(tr.duration,7.)
    def test_all_phases_guarded_and_gripper_opt_in(self):
        for gripper in (False,True):
            p=self.plan(replay_gripper=gripper);clock=Clock()
            q=p.sample(0)[0][:6];q[0]+=math.radians(3)
            plant=Plant(clock,q);out=ReplayOutput(p,plant,clock=clock)
            tr=start_transition(np.r_[q,plant.width],p);logs=[]
            play_segment(tr,plant,out,p,lambda:None,logs.append,clock=clock,sleep=clock.sleep)
            settle(tr.target,plant,out,p,lambda:None,clock=clock,sleep=clock.sleep)
            play_segment(p,plant,out,p,lambda:None,logs.append,clock=clock,sleep=clock.sleep)
            settle(p.sample(p.duration)[0],plant,out,p,lambda:None,clock=clock,sleep=clock.sleep)
            np.testing.assert_allclose(plant.q,p.sample(p.duration)[0][:6],atol=math.radians(.001))
            out.hold(plant.sample(clock(),.02)[0])
            self.assertFalse(out.holder.moved)
            self.assertEqual(bool(plant.grippers),gripper)
    def test_bad_feedback_and_gripper_readiness(self):
        plant=Plant(Clock(),np.radians(self.states[0,:6]))
        for attr,value in [('ready',False),('age',1.),('fault',1),('grip_enabled',False)]:
            old=getattr(plant,attr);setattr(plant,attr,value)
            with self.assertRaises(RuntimeError):checked_feedback(plant,self.cfg,100.,gripper=True)
            setattr(plant,attr,old)
    def test_tracking_failure_stops_stuck_plant(self):
        p=self.plan();clock=Clock();plant=Plant(clock,p.sample(0)[0][:6]);plant.stuck=True
        out=ReplayOutput(p,plant,clock=clock)
        tr=Transition(np.r_[plant.q,plant.width],np.r_[plant.q+np.radians([10,0,0,0,0,0]),plant.width],10.)
        with self.assertRaisesRegex(ValueError,'tracking'):
            play_segment(tr,plant,out,p,lambda:None,lambda r:None,clock=clock,sleep=clock.sleep)
    def test_lease_and_scheduler_fail_before_write(self):
        p=self.plan();clock=Clock();plant=Plant(clock,p.sample(0)[0][:6]);out=ReplayOutput(p,plant,clock=clock)
        def expired(): raise ValueError('expired')
        with self.assertRaisesRegex(ValueError,'expired'):
            play_segment(p,plant,out,p,expired,lambda r:None,clock=clock,sleep=clock.sleep)
        self.assertEqual(plant.commands,[])
        def delayed(dt): clock.sleep(dt+.02)
        with self.assertRaisesRegex(RuntimeError,'scheduler'):
            play_segment(p,plant,out,p,lambda:None,lambda r:None,clock=clock,sleep=delayed)
        self.assertEqual(plant.commands,[])
    def test_exclusive_lock_and_existing_socket(self):
        path=self.path/'writer.sock'
        with writer_lock(path):
            with self.assertRaises(BlockingIOError):
                with writer_lock(path): self.fail('second writer')
        path.touch()
        with self.assertRaises(RuntimeError):
            with writer_lock(path): self.fail('existing socket')
    def test_execute_owns_lock_closes_devices_and_holds_on_interrupt(self):
        import e14.replay_data as replay
        p=self.plan();clock=Clock();plant=Plant(clock,p.sample(0)[0][:6])
        plant.close_calls=0
        def close(): plant.close_calls+=1
        plant.close=close
        segment_count=0
        def segment(seg,source,output,plan,lease,log):
            nonlocal segment_count
            segment_count+=1
            if segment_count==2: raise KeyboardInterrupt('test interruption')
            return play_segment(seg,source,output,plan,lease,log,clock=clock,sleep=clock.sleep)
        with ExitStack() as stack:
            stack.enter_context(patch('servo_bridge.sources.PassiveCan',return_value=plant))
            stack.enter_context(patch('servo_bridge.sdk_worker.SDKJointWriter',return_value=plant))
            stack.enter_context(patch.object(replay,'writer_lock',lambda:writer_lock(self.path/'lock')))
            stack.enter_context(patch.object(replay.time,'monotonic',clock))
            stack.enter_context(patch.object(replay,'ReplayOutput',side_effect=lambda plan,writer:ReplayOutput(plan,writer,clock=clock)))
            stack.enter_context(patch.object(replay,'play_segment',side_effect=segment))
            stack.enter_context(patch.object(replay,'settle',side_effect=lambda *a:settle(*a,clock=clock,sleep=clock.sleep)))
            with self.assertRaises(KeyboardInterrupt):execute(p,self.path/'audit')
        self.assertEqual(plant.close_calls,2)
        rows=[json.loads(s) for s in (self.path/'audit/events.jsonl').read_text().splitlines()]
        self.assertIn('aborted',[r['event'] for r in rows])
        self.assertEqual(rows[-1]['event'],'measured_hold_requested')
        self.assertEqual(plant.grippers,[])

    def test_default_cli_never_opens_hardware(self):
        with patch('servo_bridge.sdk_worker.SDKJointWriter') as sdk,patch('servo_bridge.sources.PassiveCan') as can:
            main(['--episode',str(self.path)])
            sdk.assert_not_called();can.assert_not_called()


if __name__=='__main__':unittest.main()
