"""Offline plan/resume/strict-load regressions; no model construction or CUDA."""
import json
from pathlib import Path
import tempfile
import unittest

from server.train_policy import make_plan, resolve_checkpoint
from server.training_runtime import strict_pretrained, current_metrics


class TrainingConfigTests(unittest.TestCase):
    def fixture(self, root):
        from tests.test_server import TrainingPlanTests
        return TrainingPlanTests().fixture(root)

    def test_stop_point_does_not_shorten_scheduler(self):
        with tempfile.TemporaryDirectory() as d:
            plan=make_plan(*self.fixture(d),steps=20000,batch_size=16,stop_after=200,save_freq=200)
            self.assertEqual(plan['steps'],20000)
            self.assertEqual(plan['stop_after'],200)
            self.assertEqual(plan['warmup_steps'],1000)
            self.assertEqual(plan['camera_order'],['front'])

    def test_resume_keeps_invariants_and_requires_future_step(self):
        with tempfile.TemporaryDirectory() as d:
            paths=self.fixture(d)
            plan=make_plan(*paths,steps=20000,batch_size=16,stop_after=200)
            out=paths[3];cp=out/'checkpoints/000200'
            (cp/'pretrained_model').mkdir(parents=True)
            (cp/'training_state').mkdir()
            for name in ('train_config.json','model.safetensors'):
                (cp/'pretrained_model'/name).write_text('{}')
            (cp/'training_state/training_step.json').write_text(json.dumps({'step':200}))
            (out/'experiment_plan.json').write_text(json.dumps(plan))
            resumed=make_plan(*paths,steps=20000,batch_size=16,stop_after=300,resume=cp)
            self.assertEqual(resumed['resume_checkpoint'],str(cp.resolve()))
            with self.assertRaisesRegex(ValueError,'invariant changed: batch_size'):
                make_plan(*paths,steps=20000,batch_size=8,stop_after=300,resume=cp)
            with self.assertRaisesRegex(ValueError,'greater than'):
                make_plan(*paths,steps=20000,batch_size=16,stop_after=200,resume=cp)

    def test_invalid_short_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'stop_after'):
                make_plan(*self.fixture(d),steps=20000,stop_after=20001)

    def test_checkpoint_config_paths_resolve_to_step(self):
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/'cp'
            (cp/'pretrained_model').mkdir(parents=True)
            (cp/'training_state').mkdir()
            for name in ('train_config.json','model.safetensors'):
                (cp/'pretrained_model'/name).touch()
            (cp/'training_state/training_step.json').touch()
            self.assertEqual(resolve_checkpoint(cp/'pretrained_model/train_config.json'),cp.resolve())

    def test_audit_uses_current_metrics_not_logging_window_average(self):
        class Tracker:
            def to_dict(self,use_avg=True):
                return dict(loss=999 if use_avg else 2,grad_norm=3,lr=.0001,
                            step_s=4,samples_per_s=5)
        metrics=current_metrics(Tracker(),1)
        self.assertEqual(metrics['loss'],2)
        self.assertNotIn('step_s',metrics)

    def test_audit_rejects_nonfinite_or_missing_update_metric(self):
        class Tracker:
            def to_dict(self,use_avg=True):return dict(loss=float('nan'),grad_norm=3,lr=.0001)
        with self.assertRaisesRegex(RuntimeError,'loss'):
            current_metrics(Tracker(),10)

    def test_strict_loading_overrides_factory_default_and_restores_inheritance(self):
        class Base:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return kwargs
        class Policy(Base):pass
        with strict_pretrained(Policy):
            self.assertEqual(Policy.from_pretrained('local',strict=False),{'strict':True,'local_files_only':True})
        self.assertNotIn('from_pretrained',Policy.__dict__)
        self.assertEqual(Policy.from_pretrained('local'),{})


if __name__=='__main__':unittest.main()
