"""Scoped auditing around the public LeRobot trainer; no alternative training loop."""
from contextlib import contextmanager
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def revision(directory):
    result = subprocess.run(['git', '-C', str(directory), 'rev-parse', 'HEAD'],
                            text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


@contextmanager
def strict_pretrained(policy_class):
    """The stock factory defaults to strict=False; reject every missing/unexpected tensor."""
    descriptor = policy_class.__dict__.get('from_pretrained')
    original = policy_class.from_pretrained
    def load(cls, *args, **kwargs):
        kwargs.update(strict=True, local_files_only=True)
        return original(*args, **kwargs)
    policy_class.from_pretrained = classmethod(load)
    try:
        yield
    finally:
        if descriptor is None:
            delattr(policy_class, 'from_pretrained')
        else:
            policy_class.from_pretrained = descriptor


def current_metrics(tracker, step):
    # The public trainer logs window averages every log_freq steps; this audit
    # is per update, so it must explicitly request each meter's current value.
    metrics = tracker.to_dict(use_avg=False)
    for key in ('loss', 'grad_norm', 'lr'):
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f'Missing/nonfinite training {key} at step {step}: {metrics.get(key)}')
    # step_s and samples_per_s are set by the caller *after* update_policy.
    # At this hook they describe the preceding update, so do not mislabel them.
    metrics.pop('step_s', None)
    metrics.pop('samples_per_s', None)
    return metrics


@contextmanager
def training_audit(train_module, plan):
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    output = Path(plan['output_dir'])
    original_make = train_module.make_policy
    original_update = train_module.update_policy
    step = 0
    if plan.get('resume_checkpoint'):
        step = json.loads((Path(plan['resume_checkpoint']) / 'training_state/training_step.json').read_text())['step']
    started = time.monotonic()
    handle = None
    watched = {}

    def snapshot(policy):
        parameters = dict(policy.named_parameters())
        return {name: parameters[name].detach().flatten()[:64].float().cpu().tolist() for name in watched}

    def make(*args, **kwargs):
        nonlocal handle, watched
        with strict_pretrained(SmolVLAPolicy):
            policy = original_make(*args, **kwargs)
        output.mkdir(parents=True, exist_ok=True)
        dump(output / 'experiment_plan.json', plan)
        versions = {}
        for package in ('torch','torchvision','lerobot','transformers','accelerate','safetensors'):
            versions[package] = importlib.metadata.version(package)
        import lerobot
        dump(output / f'environment_segment_{step:06d}.json', {
            'python': sys.version, 'packages': versions,
            'project_commit': revision(Path(__file__).resolve().parents[1]),
            'lerobot_commit': revision(Path(lerobot.__file__).resolve().parents[2]),
            'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
            'resume_sample_order': 'Official EpisodeAwareSampler derives seed/epoch/offset from saved step; same batch16 and single data-parallel worker required.',
            'strict_pretrained': True})
        freeze = subprocess.run([sys.executable,'-m','pip','freeze'],text=True,capture_output=True,check=True)
        (output / 'pip_freeze.txt').write_text(freeze.stdout)
        rows = [{'name': name, 'shape': list(p.shape), 'numel': p.numel(), 'requires_grad': p.requires_grad}
                for name,p in policy.named_parameters()]
        dump(output / 'parameter_manifest.json', {'parameters':rows,
             'trainable':sum(p['numel'] for p in rows if p['requires_grad']),
             'total':sum(p['numel'] for p in rows), 'strict_pretrained':True})
        train = next((p['name'] for p in rows if p['requires_grad'] and 'state_proj' in p['name']),None)
        frozen = next((p['name'] for p in rows if not p['requires_grad'] and 'vision_model' in p['name']),None)
        watched = {name:None for name in (train,frozen) if name}
        dump(output / f'parameter_samples_before_{step:06d}.json', snapshot(policy))
        handle = (output / 'training_metrics.jsonl').open('a',buffering=1)
        print('Strict pretrained loading verified; missing_keys=[]; unexpected_keys=[]',flush=True)
        return policy

    def update(*args, **kwargs):
        nonlocal step
        result = original_update(*args, **kwargs)
        step += 1
        metrics = current_metrics(result[0], step)
        record = {'step':step, 'elapsed_s':time.monotonic()-started, **metrics}
        if torch.cuda.is_available():
            record['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        if handle:
            handle.write(json.dumps(record, default=float, allow_nan=False)+'\n')
        if step == 1 or step == plan['stop_after']:
            policy = kwargs.get('policy',args[1] if len(args)>1 else None)
            accelerator = kwargs.get('accelerator')
            if accelerator:
                policy = accelerator.unwrap_model(policy)
            dump(output / f'parameter_samples_after_{step:06d}.json',snapshot(policy))
        return result

    train_module.make_policy, train_module.update_policy = make, update
    try:
        yield
    finally:
        train_module.make_policy, train_module.update_policy = original_make, original_update
        if handle:
            handle.close()
