"""Send recorded observations through HTTP inference. Never connects to hardware."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from e14.inference_client import InferenceClient
from shared.config import load_config
from shared.deployment import check_model
from shared.recorded_observation import RecordedEpisode


def replay(config, episode, indices, output):
    dataset = RecordedEpisode(episode, config['cameras'], config['robot']['control_hz'])
    client = InferenceClient(config)
    metadata = check_model(client.health(), config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    with (output / 'predictions.jsonl').open('x', buffering=1) as stream:
        for index in indices:
            sample = dataset.sample(index)
            # Historical clocks are audit-only. Start a new local request clock;
            # never mistake a recorded timestamp for a live feedback timestamp.
            sample['captured_monotonic'] = time.monotonic()
            result = client.predict(sample, dataset.manifest['task'])
            check_model(result['response']['metadata'], config)
            if result['response']['metadata']['model_id'] != metadata['model_id']:
                raise ValueError('Model changed during recorded inference')
            actions = np.asarray(result['response']['actions'])
            delta = np.abs(actions[0] - sample['next_state'])
            row = dict(frame_index=index, recorded_monotonic=sample['recorded_monotonic'],
                       request_id=result['request_id'], state=sample['state'], next_state=sample['next_state'],
                       actions=actions.tolist(), first_action_abs_error=delta.tolist(),
                       inference_ms=result['response']['inference_ms'], round_trip_ms=result['round_trip_ms'])
            records.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    summary = dict(mode='recorded_observations_no_hardware', samples=len(records), metadata=metadata,
                   round_trip_p50_ms=statistics.median(r['round_trip_ms'] for r in records),
                   inference_p50_ms=statistics.median(r['inference_ms'] for r in records),
                   first_action_mae=np.mean([r['first_action_abs_error'] for r in records], axis=0).tolist(),
                   units=['degree']*6+['meter_full_opening'],
                   note='HTTP/deployment check only; sampled replay does not measure closed-loop task success.')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--episode', type=Path, required=True)
    parser.add_argument('--indices', default='0,50,100', help='Recorded frame indices, comma separated')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    indices = [int(x) for x in args.indices.split(',')]
    if not indices:
        parser.error('At least one replay index required')
    print(json.dumps(replay(load_config(args.config), args.episode, indices, args.output), indent=2))


if __name__ == '__main__':
    main()
