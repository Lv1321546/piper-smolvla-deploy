"""Read recorded RGB/state pairs for inference only. No robot or CAN imports."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from shared.protocol import ACTION_NAMES, vector7


class RecordedEpisode:
    def __init__(self, path, cameras, hz):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        expected = dict(format='piper-raw-v1', status='complete', state_names=list(ACTION_NAMES),
                        joint_unit='degree', gripper_unit='meter_full_opening',
                        action_mode='absolute_position', label_semantics='next_state', control_hz=hz)
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ValueError('Recording schema differs from inference configuration')
        actual = {name: {k: value[k] for k in ('width', 'height')}
                  for name, value in self.manifest['cameras'].items()}
        self.cameras = {name: {k: value[k] for k in ('width', 'height')}
                        for name, value in cameras.items()}
        if actual != self.cameras:
            raise ValueError('Recorded camera names/resolutions differ from deployment')
        self.rows = [json.loads(s) for s in (self.path / 'frames.jsonl').read_text().splitlines() if s.strip()]
        if len(self.rows) < 2 or len(self.rows) != self.manifest['frames']:
            raise ValueError('Incomplete recorded episode')
        if any(row['frame_index'] != i for i, row in enumerate(self.rows)):
            raise ValueError('Recorded frame indices are not contiguous')

    def sample(self, index):
        if not 0 <= index < len(self.rows) - 1:
            raise ValueError('Replay index must have a following frame for next-state comparison')
        row = self.rows[index]
        if set(row['images']) != set(self.cameras):
            raise ValueError('Missing recorded camera')
        images = {}
        for name, relative in row['images'].items():
            file = (self.path / relative).resolve()
            if not file.is_relative_to(self.path):
                raise ValueError('Recorded image path escapes episode directory')
            with Image.open(file) as image:
                image = np.asarray(image.convert('RGB')).copy()
            camera = self.cameras[name]
            if image.shape != (camera['height'], camera['width'], 3):
                raise ValueError('Recorded RGB dimensions differ from deployment')
            images[name] = image
        return {'state': vector7(row['state']), 'images': images,
                'recorded_monotonic': row['captured_monotonic'], 'frame_index': index,
                'next_state': vector7(self.rows[index + 1]['state'])}
