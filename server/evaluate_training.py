"""Deterministic offline SmolVLA validation; never imports robot/CAN APIs.

Use the existing train-only statistics and episode split. Action at chunk index k
is the exported action[t+k] (already observed state[t+k+1]); no extra shift.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np

HORIZONS = (1, 5, 10, 25, 50)


def anchors_for_episodes(episode_ids, count=50):
    if count <= 0:
        raise ValueError('anchors_per_episode must be positive')
    groups = {}
    for index, episode in enumerate(episode_ids):
        groups.setdefault(int(episode), []).append(index)
    return {episode: [indices[i] for i in np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)]
            for episode, indices in sorted(groups.items())}


def physical_errors(prediction, target, state, valid, horizons=HORIZONS):
    """Only seven physical dimensions enter metrics; padded future rows ignored."""
    prediction, target, state = map(lambda x: np.asarray(x, dtype=float), (prediction, target, state))
    valid = np.asarray(valid, dtype=bool)
    if (prediction.ndim != 2 or target.shape != prediction.shape or prediction.shape[1] != 7
            or state.shape != (7,) or valid.shape != (len(target),)
            or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise ValueError('Expected finite [T,7] prediction/target, [7] state and [T] validity')
    result = {}
    scale = np.array([1.] * 6 + [1000.])
    for horizon in horizons:
        i = horizon - 1
        if i < len(target) and valid[i]:
            result[str(horizon)] = {
                'model': (np.abs(prediction[i] - target[i]) * scale).tolist(),
                'hold': (np.abs(state - target[i]) * scale).tolist(),
            }
    return result


def aggregate_episodes(records):
    """Each episode gets equal weight, regardless of frame/anchor count."""
    grouped = {}
    for row in records:
        grouped.setdefault(int(row['episode_index']), []).append(row)
    per_episode = {}
    for episode, rows in sorted(grouped.items()):
        metrics = {}
        for h in map(str, HORIZONS):
            valid = [r['errors'][h] for r in rows if h in r.get('errors', {})]
            if valid:
                metrics[h] = {'valid_anchors': len(valid), **{
                    kind+'_mae_per_dimension': np.mean([r[kind] for r in valid], axis=0).tolist()
                    for kind in ('model', 'hold')}}
        losses = [r['loss'] for r in rows if r.get('loss') is not None]
        per_episode[str(episode)] = {'anchors': sum(bool(r.get('errors')) for r in rows),
                                    'loss_samples': len(losses),
                                    'loss': float(np.mean(losses)) if losses else None, 'horizons': metrics}
    summary = {}
    for h in map(str, HORIZONS):
        values = [r['horizons'][h] for r in per_episode.values() if h in r['horizons']]
        if values:
            summary[h] = {'valid_episodes': len(values),
                          'valid_anchors': sum(r['valid_anchors'] for r in values), **{
                kind+'_mae_per_dimension': np.mean([r[kind+'_mae_per_dimension'] for r in values], axis=0).tolist()
                for kind in ('model', 'hold')}}
    losses = [r['loss'] for r in per_episode.values() if r['loss'] is not None]
    return {'loss_episode_mean': float(np.mean(losses)) if losses else None,
            'horizons': summary, 'per_episode': per_episode,
            'physical_units': ['degree']*6 + ['millimeter_full_opening']}


def select_checkpoints(reports):
    usable = [r for r in reports if r.get('finite_predictions') is True
              and isinstance(r.get('loss_episode_mean'), (int, float))
              and math.isfinite(r['loss_episode_mean'])]
    if not usable:
        raise ValueError('No finite evaluated checkpoint available for selection')
    ranked = sorted(usable, key=lambda r: (r['loss_episode_mean'], r['checkpoint']))
    return {'selection_metric': 'fixed-anchor masked flow-matching loss, episode equal weighting',
            'selected': ranked[0]['checkpoint'],
            'runner_up': ranked[1]['checkpoint'] if len(ranked) > 1 else None,
            'ranking': [{'checkpoint': r['checkpoint'], 'loss_episode_mean': r['loss_episode_mean'],
                         'horizons': r['horizons']} for r in ranked],
            'note': 'Offline model selection; physical task success has not been measured.'}


def evaluate_checkpoint(checkpoint, plan, anchors_per_episode=50, device='cuda', seed=42,
                        full_loss=False, output_dir=None):
    """Restore one model at a time; deterministic per-frame flow/inference noise.

    Explicit generators avoid changing training RNG. All loss calls use the
    actual SmolVLA forward mask and only seven unpadded action dimensions.
    """
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from torch.utils.data._utils.collate import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    checkpoint = Path(checkpoint).resolve()
    root = Path(plan['dataset_root'])
    cfg = SmolVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
    cfg.device, cfg.vlm_model_name, cfg.load_vlm_weights = device, str(plan['vlm_path']), False
    if cfg.output_features['action'].shape != (7,):
        raise ValueError('Validation requires a seven-dimensional PiPER checkpoint')
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg, strict=True).eval()
    tensor_device = next(policy.parameters()).device
    stats_path = Path(plan.get('stats_path', root/'train_stats.json'))
    stats = {f: {k: torch.tensor(v, dtype=torch.float32) for k, v in entries.items()}
             for f, entries in json.loads(stats_path.read_text()).items()}
    # New processors prevent restoring foreign SO100 dimensions/statistics.
    pre, post = make_pre_post_processors(cfg, pretrained_path=None, dataset_stats=stats)
    split = plan.get('episode_split') or json.loads((root/'split.json').read_text())
    fps = float(plan['schema']['control_hz'])
    dataset = LeRobotDataset('local/piper', root=root, episodes=split['validation'],
                            delta_timestamps={'action': [i/fps for i in range(cfg.chunk_size)]},
                            video_backend='pyav')
    episode_ids = np.asarray(dataset.hf_dataset.select_columns(['episode_index'])['episode_index']).reshape(-1)
    anchors = anchors_for_episodes(episode_ids, anchors_per_episode)
    if set(anchors) != set(split['validation']):
        raise ValueError('Validation dataset episodes differ from fixed split')
    anchor_set = {i for indices in anchors.values() for i in indices}
    indices = range(len(dataset)) if full_loss else sorted(anchor_set)
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir/'anchors.json').write_text(json.dumps(anchors, indent=2))
    records, latencies = [], []
    started = time.monotonic()
    errors_path = (output_dir/'predictions.jsonl').open('w') if output_dir else None
    try:
        for index in indices:
            row = dataset[int(index)]
            episode = int(row['episode_index'])
            target = row['action'].detach().cpu().numpy()
            valid = ~row['action_is_pad'].detach().cpu().numpy().astype(bool)
            if target.shape != (cfg.chunk_size, 7) or valid.shape != (cfg.chunk_size,) or not valid.any():
                raise ValueError('Unexpected chunk shape or all-padding validation sample')
            state = row['observation.state'].detach().cpu().numpy()
            batch = pre(default_collate([row]))
            batch['action_is_pad'] = torch.as_tensor(~valid, device=tensor_device).unsqueeze(0)
            generator = torch.Generator(device=tensor_device).manual_seed(seed + int(index)*2)
            shape = (1, cfg.chunk_size, cfg.max_action_dim)
            with torch.inference_mode():
                noise = torch.randn(shape, device=tensor_device, dtype=torch.float32, generator=generator)
                # Installed SmolVLA 0.6.2 uses Beta(1.5,1)*.999+.001.
                # Inverse CDF gives the same distribution with an explicit generator.
                flow_time = torch.rand((1,), device=tensor_device, generator=generator).pow(1/1.5)*.999+.001
                loss, _ = policy(batch, noise=noise, time=flow_time)
                loss = float(loss.item())
                if not math.isfinite(loss):
                    raise ValueError(f'Nonfinite validation loss at row {index}')
                record = {'dataset_index': int(index), 'episode_index': episode, 'loss': loss, 'errors': {}}
                if index in anchor_set:
                    prediction_batch = {k: v for k, v in batch.items() if k not in ('action', 'action_is_pad')}
                    policy.reset()
                    if tensor_device.type == 'cuda': torch.cuda.synchronize(tensor_device)
                    before = time.perf_counter()
                    generator.manual_seed(seed + int(index)*2 + 1)
                    noise = torch.randn(shape, device=tensor_device, dtype=torch.float32, generator=generator)
                    predicted = post(policy.predict_action_chunk(prediction_batch, noise=noise)).detach().cpu().numpy()[0]
                    if tensor_device.type == 'cuda': torch.cuda.synchronize(tensor_device)
                    latencies.append((time.perf_counter()-before)*1000)
                    record['errors'] = physical_errors(predicted, target, state, valid)
                    pairs = valid[1:] & valid[:-1]
                    step = np.abs(np.diff(predicted, axis=0))[pairs]
                    record['chunk_max_step_per_dimension'] = (step.max(axis=0)*np.array([1.]*6+[1000.])).tolist() if len(step) else [0.]*7
                    if errors_path:
                        errors_path.write(json.dumps({**record, 'target': target.tolist(),
                            'prediction': predicted.tolist(), 'valid': valid.tolist(), 'state': state.tolist()}, allow_nan=False)+'\n')
                        # One frame pair per episode, original observation not future data.
                        if index == anchors[episode][0]:
                            from PIL import Image
                            for camera in plan['schema']['cameras']:
                                image = row[f'observation.images.{camera}'].detach().cpu().numpy()
                                if image.shape[0] == 3: image = image.transpose(1, 2, 0)
                                if image.dtype != np.uint8: image = np.clip(image*255, 0, 255).astype(np.uint8)
                                Image.fromarray(image).save(output_dir/f'episode_{episode:06d}_{camera}.png')
            records.append(record)
        summary = aggregate_episodes(records)
        first = summary['horizons'].get('1', {}).get('model_mae_per_dimension')
        compatible_first = first[:6]+[first[6]/1000] if first else None
        report = {'checkpoint': str(checkpoint), 'finite_predictions': True,
                  'samples': len(anchor_set), 'loss_samples': len(records),
                  'full_loss': full_loss, 'anchors_per_episode': anchors_per_episode, 'seed': seed,
                  'loss_time_distribution': 'Beta(1.5,1)*0.999+0.001 (installed SmolVLA 0.6.2)',
                  'range_check': 'Not evaluated: validation schema does not carry commissioned robot limits',
                  'first_action_mae_per_dimension': compatible_first,
                  'units': ['degree']*6+['meter_full_opening'], **summary,
                  'inference_latency_ms': {k: float(np.percentile(latencies, p))
                                           for k,p in [('p50',50),('p95',95),('p99',99)]},
                  'elapsed_s': time.monotonic()-started,
                  'note': 'Offline numerical validation only; real-robot task success is not measured.'}
        if output_dir:
            (output_dir/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        return report
    finally:
        if errors_path: errors_path.close()
        del policy, dataset, pre, post
        gc.collect()
        if tensor_device.type == 'cuda': torch.cuda.empty_cache()


def discover_inputs(plan_path=None, run_dir=None, checkpoints=None):
    """Resolve saved plans and numeric checkpoints without importing GPU libraries."""
    run_dir = Path(run_dir) if run_dir else None
    if plan_path is None and run_dir:
        plan_path = next((run_dir/name for name in ('experiment_plan.json','plan.json','training_plan.json')
                          if (run_dir/name).is_file()), None)
    if plan_path is None or not Path(plan_path).is_file():
        raise ValueError('--plan or --run-dir with experiment_plan.json is required')
    if checkpoints:
        paths = [Path(p) for p in checkpoints]
    elif run_dir:
        paths = sorted(run_dir.glob('checkpoints/[0-9]*/pretrained_model'),
                       key=lambda p: int(p.parent.name))
    else:
        paths = []
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths or any(not p.is_dir() for p in paths):
        raise ValueError('No valid checkpoints supplied or found')
    return Path(plan_path), paths


def actual_checkpoint_step(checkpoint):
    path = Path(checkpoint).resolve().parent/'training_state/training_step.json'
    step = json.loads(path.read_text()).get('step')
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError('Checkpoint has no positive persisted training step')
    return step


def export_selected(selection, reports, plan):
    """Deployment manifests describe the actual chosen update, not the 20k budget."""
    from server.train_policy import export_manifest
    by_checkpoint = {str(Path(r['checkpoint']).resolve()): r for r in reports}
    artifacts = {}
    for role in ('selected','runner_up'):
        checkpoint = selection.get(role)
        if checkpoint is None:
            continue
        checkpoint = Path(checkpoint).resolve()
        step = actual_checkpoint_step(checkpoint)
        export_plan = dict(plan, steps=step)
        manifest_path = export_manifest(checkpoint, export_plan, by_checkpoint[str(checkpoint)])
        manifest = json.loads(manifest_path.read_text())
        manifest.update(training_steps=step, planned_training_steps=plan['steps'],
                        selection_role=role, selection_metric=selection['selection_metric'])
        manifest['model_id'] = manifest['model_id'] + f'-step{step:06d}'
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        artifacts[role] = str(manifest_path)
    return artifacts

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, help='Persisted training plan JSON')
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--checkpoint', type=Path, action='append')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--anchors-per-episode', type=int, default=50)
    p.add_argument('--full-loss', action='store_true', help='Compute loss on every validation frame')
    p.add_argument('--full-finalists', action='store_true', help='After anchor ranking, run full loss for top two')
    p.add_argument('--export-selected', action='store_true', help='Write deployment manifests for selected and runner-up')
    args = p.parse_args(argv)
    try:
        plan_path, checkpoints = discover_inputs(args.plan, args.run_dir, args.checkpoint)
    except ValueError as exc:
        p.error(str(exc))
    plan = json.loads(plan_path.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for checkpoint in checkpoints:
        name = checkpoint.parent.name
        if name == 'last': name = checkpoint.resolve().parent.name
        report = evaluate_checkpoint(checkpoint, plan, args.anchors_per_episode, args.device,
                                     args.seed, args.full_loss, args.output/name)
        reports.append(report)
        print(json.dumps({'checkpoint': str(checkpoint), 'loss_episode_mean':report['loss_episode_mean']}), flush=True)
    selection = select_checkpoints(reports)
    if args.full_finalists:
        selection['full_finalists'] = [evaluate_checkpoint(row['checkpoint'], plan, args.anchors_per_episode,
            args.device,args.seed,True,args.output/(Path(row['checkpoint']).parent.name+'_full'))
            for row in selection['ranking'][:2]]
    if args.export_selected:
        selection['deployment_manifests'] = export_selected(
            selection, reports + selection.get('full_finalists', []), plan)
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False))
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__': main()
