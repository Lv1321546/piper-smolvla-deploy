"""Versioned PiPER wire schema. No SDK imports and no motor commands."""
import base64
import io
import json
import math
from numbers import Real

ACTION_NAMES = ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper_width')
PROTOCOL_VERSION = 'piper-v1'
JOINT_UNIT = 'degree'
GRIPPER_UNIT = 'meter_full_opening'
ACTION_MODE = 'absolute_position'
MAX_CHUNK_STEPS = 100


def finite_number(value, name='value', positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return float(value)


def vector7(value, name='state'):
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(f'{name} must contain seven values in {ACTION_NAMES} order')
    return [finite_number(x, f'{name}[{i}]') for i, x in enumerate(value)]


def validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError('Request must be a JSON object')
    if payload.get('protocol_version') != PROTOCOL_VERSION or payload.get('robot_type') != 'piper':
        raise ValueError('Expected piper-v1 / piper observation')
    request_id = payload.get('request_id')
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValueError('Invalid request_id')
    task = payload.get('task')
    if not isinstance(task, str) or not task.strip() or len(task) > 4096:
        raise ValueError('task must be a nonempty string of at most 4096 characters')
    vector7(payload.get('state'))
    finite_number(payload.get('control_hz'), 'control_hz', positive=True)
    images = payload.get('images')
    if not isinstance(images, dict) or not 1 <= len(images) <= 4:
        raise ValueError('Expected one to four named images')
    for key, encoded in images.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError('Invalid camera name')
        if not isinstance(encoded, str) or not encoded:
            raise ValueError('Images must be nonempty base64 strings')
    return payload


def encode_images(images, quality=85):
    import numpy as np
    from PIL import Image
    result = {}
    for name, rgb in images.items():
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f'{name}: expected RGB uint8 HWC array')
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format='JPEG', quality=quality)
        result[name] = base64.b64encode(stream.getvalue()).decode('ascii')
    return result


def decode_images(images, max_pixels=2073600):
    import numpy as np
    from PIL import Image
    result = {}
    for name, encoded in images.items():
        try:
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format not in ('JPEG', 'PNG') or image.width * image.height > max_pixels:
                    raise ValueError('Image format or dimensions not allowed')
                result[name] = np.asarray(image.convert('RGB')).copy()
        except Exception as error:
            raise ValueError(f'Invalid image for {name}: {error}') from error
    return result


def validate_metadata(metadata, expected_hz=None, require_executable=False):
    if not isinstance(metadata, dict):
        raise ValueError('Missing model metadata')
    expected = {
        'robot_type': 'piper', 'action_names': list(ACTION_NAMES),
        'joint_unit': JOINT_UNIT, 'gripper_unit': GRIPPER_UNIT, 'action_mode': ACTION_MODE,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Incompatible model metadata: {key} must be {value!r}')
    hz = finite_number(metadata.get('control_hz'), 'model control_hz', positive=True)
    if expected_hz is not None and not math.isclose(hz, expected_hz, rel_tol=1e-6):
        raise ValueError('Model and controller frequencies differ')
    if not isinstance(metadata.get('model_id'), str) or not metadata['model_id']:
        raise ValueError('Missing model_id')
    if not isinstance(metadata.get('execution_ready'), bool):
        raise ValueError('execution_ready must be boolean')
    if require_executable and (metadata['execution_ready'] is not True or metadata.get('backend') != 'smolvla'):
        raise ValueError('Mock/base diagnostic policies cannot execute on physical PiPER')
    return metadata


def validate_response(response, request_id=None, expected_hz=None, require_executable=False):
    if not isinstance(response, dict) or response.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Response protocol mismatch')
    if request_id is not None and response.get('request_id') != request_id:
        raise ValueError('Stale or mismatched request_id')
    validate_metadata(response.get('metadata'), expected_hz, require_executable)
    actions = response.get('actions')
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_CHUNK_STEPS:
        raise ValueError('Expected a bounded nonempty action chunk')
    for i, action in enumerate(actions):
        vector7(action, f'actions[{i}]')
    latency = finite_number(response.get('inference_ms'), 'inference_ms')
    if latency < 0:
        raise ValueError('Negative inference latency')
    return response


def dumps(payload):
    return json.dumps(payload, allow_nan=False, separators=(',', ':')).encode('utf-8')


validate_observation = validate_request
validate_prediction = validate_response
