"""No hardware: filtered targets stay separate from measured training data."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from e14.filter_episode import filter_episode, fingerprint, smooth_joints
from e14.replay_data import ReplayPlan, ReplayOutput, load_episode, play_segment
from servo_bridge.core import load_config
from tests.test_replay_data import Clock, Plant

ROOT = Path(__file__).resolve().parents[1]


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root/'raw';self.source.mkdir()
        self.output = self.root/'filtered'
        self.cfg = load_config(ROOT/'servo_bridge/config.cartesian.example.json')
        self.times = np.arange(81)*.05
        self.states = np.tile([0.,80.,-60.,-10.,40.,0.,.06],(len(self.times),1))
        self.states[:,0] = 2*np.sin(self.times) + .1*np.where(np.arange(len(self.times))%2,1,-1)
        self.states[:,6] = .06-.002*self.times
        self.write()
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        manifest = dict(schema_version=1,format='piper-raw-v1',status='complete',mock=False,
             state_names=[f'joint{i}' for i in range(1,7)]+['gripper_width'],joint_unit='degree',
             gripper_unit='meter_full_opening',action_mode='absolute_position',label_semantics='next_state',
             frames=len(self.times),expected_frames=len(self.times),control_hz=20,
             max_period_error_ratio=.5,feedback_max_age_ms=300,task='push',task_outcome='success')
        (self.source/'manifest.json').write_text(json.dumps(manifest))
        rows=[dict(frame_index=i,captured_monotonic=100.+float(t),feedback_monotonic=99.99+float(t),state=s.tolist())
              for i,(t,s) in enumerate(zip(self.times,self.states))]
        (self.source/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def test_reduces_noise_and_preserves_anchors_jaw_and_size(self):
        filtered=smooth_joints(self.times,self.states)
        self.assertEqual(filtered.shape,self.states.shape)
        np.testing.assert_array_equal(filtered[:,6],self.states[:,6])
        np.testing.assert_array_equal(filtered[[0,-1]],self.states[[0,-1]])
        clean=2*np.sin(self.times)
        self.assertLess(np.mean((filtered[8:-8,0]-clean[8:-8])**2),
                        np.mean((self.states[8:-8,0]-clean[8:-8])**2)*.3)
    def test_quadratic_preserved_on_nonuniform_recorded_times(self):
        times=self.times.copy();times[1:-1]+=.001*np.sin(np.arange(1,len(times)-1))
        states=self.states.copy();states[:,0]=times**2+2*times+1
        result=smooth_joints(times,states)
        np.testing.assert_allclose(result[:,0],states[:,0],atol=1e-10)
    def test_protected_samples_unchanged_and_tapered(self):
        result=smooth_joints(self.times,self.states,protect=[(1.,2.)])
        mask=(self.times>=1)&(self.times<=2)
        np.testing.assert_array_equal(result[mask],self.states[mask])
        self.assertGreater(np.max(abs(result[~mask,:6]-self.states[~mask,:6])),0)
        np.testing.assert_array_equal(smooth_joints(self.times,self.states,protect=[(0.,4.)]),self.states)
    def test_invalid_parameters(self):
        for options in (dict(window=4),dict(window=True),dict(window=83),dict(order=7),
                        dict(order=True),dict(strength=0),dict(strength=2),dict(strength=float('nan')),
                        dict(protect=[(2,1)]),dict(protect=[(-1,1)]),dict(protect=[(1,float('inf'))])):
            with self.subTest(options=options),self.assertRaises(ValueError):smooth_joints(self.times,self.states,**options)
    def test_source_immutable_and_output_is_replay_only(self):
        hashes=fingerprint(self.source)
        report=filter_episode(self.source,self.output,self.cfg)
        self.assertEqual(hashes,fingerprint(self.source))
        times,states,manifest=load_episode(self.output)
        self.assertEqual(manifest['format'],'piper-replay-v1')
        self.assertEqual(manifest['purpose'],'replay_only')
        self.assertEqual(manifest['task_outcome'],'unknown')
        self.assertEqual(manifest['source_task_outcome'],'success')
        np.testing.assert_allclose(times,self.times,atol=1e-12)
        rows=[json.loads(s) for s in (self.output/'frames.jsonl').read_text().splitlines()]
        self.assertIn('target_state',rows[0]);self.assertNotIn('state',rows[0]);self.assertNotIn('feedback_monotonic',rows[0])
        self.assertTrue(report['gripper_unchanged']);self.assertTrue(report['endpoints_preserved'])
        self.assertEqual(len((self.output/'comparison.csv').read_text().splitlines()),len(times)+1)
        self.assertFalse((self.output/'images').exists())
        plan=ReplayPlan(self.output,self.cfg,speed=1,replay_gripper=True)
        self.assertGreater(plan.duration,0)
    def test_derived_targets_cannot_be_exported_as_observed_training_data(self):
        from e14.export_dataset import load_raw_episode
        filter_episode(self.source,self.output,self.cfg)
        with self.assertRaisesRegex(ValueError,'Unsupported raw format'):load_raw_episode(self.output)
    def test_missing_replay_only_marker_rejected(self):
        filter_episode(self.source,self.output,self.cfg)
        p=self.output/'manifest.json';data=json.loads(p.read_text());data.pop('purpose');p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'replay-only'):load_episode(self.output)
    def test_deviation_rejected_without_output(self):
        hashes=fingerprint(self.source)
        with self.assertRaisesRegex(ValueError,'deviation'):
            filter_episode(self.source,self.output,self.cfg,max_joint_deviation_deg=.000001)
        self.assertFalse(self.output.exists());self.assertEqual(hashes,fingerprint(self.source))
    def test_existing_and_nested_outputs_not_overwritten(self):
        self.output.mkdir();marker=self.output/'keep';marker.write_text('keep')
        with self.assertRaises(FileExistsError):filter_episode(self.source,self.output,self.cfg)
        self.assertEqual(marker.read_text(),'keep')
        for p in (self.source,self.source/'new'):
            with self.assertRaises(ValueError):filter_episode(self.source,p,self.cfg)
    def test_repeated_filter_rejected(self):
        filter_episode(self.source,self.output,self.cfg)
        with self.assertRaisesRegex(ValueError,'original real episode'):
            filter_episode(self.output,self.root/'twice',self.cfg)
    def test_joint_limits_never_silently_clipped(self):
        self.states[:,1]=0;self.states[20,1]=5;self.write()
        with self.assertRaisesRegex(ValueError,'model limits'):filter_episode(self.source,self.output,self.cfg)
        self.assertFalse(self.output.exists())
    def test_large_jaw_feedback_not_clipped_or_used_as_joint_only_blocker(self):
        self.states[:,6]=.102;self.write()
        report=filter_episode(self.source,self.output,self.cfg)
        self.assertFalse(report['optional_gripper_replay']['valid'])
        plan=ReplayPlan(self.output,self.cfg)
        np.testing.assert_array_equal(plan.states[:,6],self.states[:,6])
        with self.assertRaisesRegex(ValueError,'gripper width'):ReplayPlan(self.output,self.cfg,replay_gripper=True)
    def test_filtered_plan_passes_full_mock_output_guard(self):
        with patch('servo_bridge.sources.PassiveCan') as can,patch('servo_bridge.sdk_worker.SDKJointWriter') as sdk:
            filter_episode(self.source,self.output,self.cfg)
            can.assert_not_called();sdk.assert_not_called()
        plan=ReplayPlan(self.output,self.cfg,replay_gripper=True,speed=1)
        clock=Clock();plant=Plant(clock,plan.sample(0)[0][:6]);plant.width=plan.sample(0)[0][6]
        control=ReplayOutput(plan,plant,clock=clock)
        play_segment(plan,plant,control,plan,lambda:None,lambda row:None,clock=clock,sleep=clock.sleep)
        np.testing.assert_allclose(np.degrees(plant.q),plan.states[-1,:6],atol=.001)
        self.assertTrue(plant.grippers)


if __name__=='__main__':unittest.main()
