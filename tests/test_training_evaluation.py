"""Pure metric tests; no torch, GPU, dataset library or hardware needed."""
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from server.evaluate_training import anchors_for_episodes, physical_errors, aggregate_episodes, select_checkpoints, discover_inputs, actual_checkpoint_step, main

class EvaluationTests(unittest.TestCase):
    def test_anchors_are_fixed_and_do_not_cross_episodes(self):
        expected={5:[0,2,4],17:[5,6]}
        self.assertEqual(anchors_for_episodes([5]*5+[17]*2,3),expected)
        self.assertEqual(anchors_for_episodes([5]*5+[17]*2,3),expected)
    def test_horizons_ignore_padding_and_use_mm_only_for_gripper(self):
        target=np.zeros((50,7)); predicted=np.ones((50,7));state=np.zeros(7)
        valid=np.arange(50)<8
        metrics=physical_errors(predicted,target,state,valid)
        self.assertEqual(set(metrics),{'1','5'})
        self.assertEqual(metrics['1']['model'],[1]*6+[1000])
        self.assertEqual(metrics['1']['hold'],[0]*7)
    def test_physical_metrics_reject_padded_action_dimensions(self):
        with self.assertRaises(ValueError):
            physical_errors(np.zeros((50,32)),np.zeros((50,32)),np.zeros(7),np.ones(50,bool))
    def test_episodes_get_equal_weight_not_frame_weight(self):
        rows=[{'episode_index':1,'loss':1.,'errors':{'1':{'model':[1]*7,'hold':[2]*7}}}]*10
        rows+=[{'episode_index':2,'loss':3.,'errors':{'1':{'model':[3]*7,'hold':[4]*7}}}]
        report=aggregate_episodes(rows)
        self.assertEqual(report['loss_episode_mean'],2.)
        self.assertEqual(report['horizons']['1']['model_mae_per_dimension'],[2]*7)
        self.assertEqual(report['horizons']['1']['valid_anchors'],11)
    def test_nonfinite_checkpoint_is_never_selected(self):
        reports=[dict(checkpoint='a',finite_predictions=True,loss_episode_mean=.4,horizons={}),
                 dict(checkpoint='b',finite_predictions=True,loss_episode_mean=.2,horizons={}),
                 dict(checkpoint='c',finite_predictions=True,loss_episode_mean=float('nan'),horizons={})]
        selection=select_checkpoints(reports)
        self.assertEqual(selection['selected'],'b');self.assertEqual(selection['runner_up'],'a')
    def test_horizon_uses_exported_action_without_second_shift(self):
        target=np.tile(np.arange(1,51)[:,None],(1,7)).astype(float)
        valid=np.ones(50,bool)
        metrics=physical_errors(target.copy(),target,np.zeros(7),valid)
        self.assertEqual(metrics['1']['hold'][:6],[1]*6)
        self.assertEqual(metrics['50']['hold'][:6],[50]*6)
        self.assertEqual(metrics['50']['model'],[0]*7)

    def test_discovers_experiment_plan_and_numeric_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);plan=run/'experiment_plan.json';plan.write_text('{}')
            (run/'plan.json').write_text('{}')
            for step in ('000200','002000'):
                (run/'checkpoints'/step/'pretrained_model').mkdir(parents=True)
            (run/'checkpoints/last').symlink_to('002000',target_is_directory=True)
            found,paths=discover_inputs(run_dir=run)
            self.assertEqual(found,plan)
            self.assertEqual([p.parent.name for p in paths],['000200','002000'])
            # Explicit checkpoint works without --run-dir, too.
            self.assertEqual(discover_inputs(plan,checkpoints=[paths[0]])[1],[paths[0]])
    def test_cli_discovers_plan_and_selects_without_loading_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);(run/'experiment_plan.json').write_text('{}')
            checkpoint=run/'checkpoints/000200/pretrained_model';checkpoint.mkdir(parents=True)
            report=dict(checkpoint=str(checkpoint),finite_predictions=True,loss_episode_mean=.3,horizons={})
            with patch('server.evaluate_training.evaluate_checkpoint',return_value=report) as evaluate:
                main(['--run-dir',str(run),'--output',str(run/'reports')])
                self.assertEqual(evaluate.call_count,1)
            selection=json.loads((run/'reports/selection.json').read_text())
            self.assertEqual(selection['selected'],str(checkpoint))
    def test_actual_step_is_read_from_training_state_not_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'000200/pretrained_model';checkpoint.mkdir(parents=True)
            state=checkpoint.parent/'training_state';state.mkdir()
            (state/'training_step.json').write_text('{"step":200}')
            self.assertEqual(actual_checkpoint_step(checkpoint),200)

if __name__=='__main__':unittest.main()
