import json
from pathlib import Path
import tempfile
import unittest
from server.training_acceptance import check

class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        def dump(name,obj):
            p=self.p/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj))
        self.dump=dump
        rows=[dict(step=i,loss=.1,grad_norm=1.,lr=i*1e-7,elapsed_s=i*.5) for i in range(1,301)]
        (self.p/'training_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        dump('parameter_samples_before_000000.json',{'train':[0.],'frozen':[1.]})
        dump('parameter_samples_after_000300.json',{'train':[.5],'frozen':[1.]})
        dump('parameter_manifest.json',{'parameters':[{'name':'train','requires_grad':True},{'name':'frozen','requires_grad':False}]})
        dump('experiment_plan.json',{'steps':20000,'stop_after':300,'resume_checkpoint':'checkpoint200','lr':1e-4,'warmup_steps':1000})
        dump('checkpoints/last/training_state/training_step.json',{'step':300})
        for name in ('optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors','scheduler_state.json'):
            (self.p/'checkpoints/last/training_state'/name).touch()
    def tearDown(self):self.tmp.cleanup()
    def test_successful_resume_and_update(self):self.assertEqual(check(self.p)['status'],'passed')
    def test_frozen_parameter_change_blocks_baseline(self):
        self.dump('parameter_samples_after_000300.json',{'train':[.5],'frozen':[2.]})
        with self.assertRaisesRegex(ValueError,'frozen'):check(self.p)
    def test_missing_optimizer_steps_blocks_baseline(self):
        p=self.p/'training_metrics.jsonl';p.write_text('\n'.join(p.read_text().splitlines()[1:])+'\n')
        with self.assertRaisesRegex(ValueError,'numbering'):check(self.p)

if __name__=='__main__':unittest.main()
