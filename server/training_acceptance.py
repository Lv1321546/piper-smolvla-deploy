"""Check the real 200+100 step smoke artifacts before starting a fresh baseline."""
import argparse
import json
import math
from pathlib import Path
import statistics


def check(run):
    run=Path(run)
    rows=[json.loads(line) for line in (run/'training_metrics.jsonl').read_text().splitlines()]
    if [r['step'] for r in rows] != list(range(1,301)):
        raise ValueError('Expected uninterrupted step numbering 1..300, including resume')
    for row in rows:
        for key in ('loss','grad_norm','lr'):
            if key not in row or not math.isfinite(float(row[key])):
                raise ValueError(f'Missing/nonfinite {key} at step {row["step"]}')
    start=json.loads((run/'parameter_samples_before_000000.json').read_text())
    final=json.loads((run/'parameter_samples_after_000300.json').read_text())
    params=json.loads((run/'parameter_manifest.json').read_text())
    flags={p['name']:p['requires_grad'] for p in params['parameters']}
    if not any(flags[name] and start[name]!=final[name] for name in start):
        raise ValueError('No watched trainable state projection parameter changed')
    if not any(not flags[name] for name in start):
        raise ValueError('No frozen vision parameter was audited')
    if any(not flags[name] and start[name]!=final[name] for name in start):
        raise ValueError('Watched frozen vision parameters changed')
    plan=json.loads((run/'experiment_plan.json').read_text())
    if plan['steps']!=20000 or plan['stop_after']!=300 or not plan['resume_checkpoint']:
        raise ValueError('Smoke did not preserve 20k scheduler and resume to300')
    last=(run/'checkpoints/last').resolve()
    step=json.loads((last/'training_state/training_step.json').read_text())['step']
    if step!=300: raise ValueError('Last checkpoint is not step300')
    increment=rows[200]['lr']-rows[199]['lr']
    if not 0 < increment <= 1.5*plan['lr']/plan['warmup_steps']:
        raise ValueError('Learning-rate continuity failed at step200 ->201')
    for name in ('optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors','scheduler_state.json'):
        if not (last/'training_state'/name).is_file():
            raise ValueError('Missing resumable training state: '+name)
    tail=rows[-100:]
    elapsed=[tail[i]['elapsed_s']-tail[i-1]['elapsed_s'] for i in range(1,len(tail))]
    report={'status':'passed','step':step,'scheduler_steps':plan['steps'],
            'trainable_parameter_update_verified':True,'sampled_frozen_parameters_unchanged':True,
            'resume_numbering_verified':True,'last100_median_seconds_per_step':statistics.median(elapsed),
            'peak_allocated_gib':max(r.get('peak_allocated_gib',0) for r in rows),
            'checkpoint':str(last),'note':'Offline training acceptance, no physical task-success claim'}
    (run/'smoke_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();print(json.dumps(check(args.run_dir),indent=2))

if __name__=='__main__':main()
