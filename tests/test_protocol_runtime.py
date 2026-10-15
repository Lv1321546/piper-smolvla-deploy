import copy
import json
from pathlib import Path
import unittest

import numpy as np

from shared.protocol import (ACTION_NAMES, decode_images, encode_images, validate_metadata,
                             validate_request, validate_response)
from e14.run_robot import select_action_chunk, validate_action

ROOT = Path(__file__).resolve().parents[1]


def metadata():
    return {'robot_type': 'piper', 'action_names': list(ACTION_NAMES), 'joint_unit': 'degree',
            'gripper_unit': 'meter_full_opening', 'action_mode': 'absolute_position',
            'control_hz': 20, 'model_id': 'test', 'backend': 'mock', 'execution_ready': False}


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.request = {'protocol_version': 'piper-v1', 'request_id': 'r1', 'robot_type': 'piper',
                        'task': 'move cube', 'state': [0, 1, -1, 0, 0, 0, .02], 'control_hz': 20,
                        'images': encode_images({'front': np.zeros((16, 24, 3), np.uint8)})}

    def test_images_and_schema_roundtrip(self):
        validate_request(self.request)
        self.assertEqual(decode_images(self.request['images'])['front'].shape, (16, 24, 3))

    def test_dimension_nan_and_corrupt_image(self):
        for state in ([0] * 6, [True] * 7, [float('nan')] * 7):
            value = {**self.request, 'state': state}
            with self.assertRaises(ValueError): validate_request(value)
        with self.assertRaises(ValueError): decode_images({'front': '!!!'})
        with self.assertRaises(ValueError): decode_images(self.request['images'], max_pixels=20)

    def test_mock_and_units_never_execute(self):
        with self.assertRaises(ValueError): validate_metadata(metadata(), 20, True)
        for key, value in [('action_names', list(ACTION_NAMES[:-1])), ('joint_unit', 'radian'),
                           ('gripper_unit', 'mm'), ('control_hz', 30)]:
            candidate = {**metadata(), key: value}
            with self.assertRaises(ValueError): validate_metadata(candidate, 20)

    def test_request_id_and_action_dimension(self):
        response = {'protocol_version': 'piper-v1', 'request_id': 'r1', 'metadata': metadata(),
                    'actions': [[0] * 7], 'inference_ms': .1}
        validate_response(response, 'r1', 20)
        with self.assertRaises(ValueError): validate_response(response, 'old', 20)
        with self.assertRaises(ValueError): validate_response({**response, 'actions': [[0] * 6]}, 'r1', 20)


class ControllerTest(unittest.TestCase):
    def test_time_alignment_expiry_and_horizon(self):
        result = {'captured_monotonic': 10., 'received_monotonic': 10.1,
                  'response': {'actions': [[float(i)] * 7 for i in range(20)]}}
        self.assertEqual(select_action_chunk(result, 10.16, 20, .5, 5)[0], 3)
        self.assertIsNone(select_action_chunk(result, 10.41, 20, .5, 5))
        with self.assertRaises(TimeoutError): select_action_chunk(result, 10.6, 20, .5, 5)

    def test_joint_and_gripper_commands_bounded(self):
        config = json.loads((ROOT / 'config.mock.json').read_text())
        state = [0, 10, -20, 0, 0, 0, .02]
        self.assertEqual(validate_action(state, state, config), state)
        for axis, target in ((0, 151), (1, -1), (2, 1), (6, .08), (0, 2), (6, .03)):
            action = state.copy()
            action[axis] = target
            with self.assertRaises(ValueError): validate_action(action, state, config)


if __name__ == '__main__':
    unittest.main()
