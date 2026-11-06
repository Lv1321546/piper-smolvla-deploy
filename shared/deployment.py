"""Contracts shared by live and recorded inference; hardware independent."""
from shared.protocol import PROTOCOL_VERSION, validate_metadata


def check_model(metadata, config, *, physical=False):
    validate_metadata(metadata, config['robot']['control_hz'], require_executable=physical)
    if metadata.get('protocol_version') != PROTOCOL_VERSION:
        raise ValueError('Deployment metadata protocol mismatch')
    expected = {name: {k: camera[k] for k in ('width', 'height')}
                for name, camera in config['cameras'].items()}
    if metadata.get('cameras') != expected or metadata.get('label_semantics') != 'next_state':
        raise ValueError('Model camera schema or action-label semantics differ from deployment')
    pinned = config.get('inference', {}).get('expected_model_id')
    if pinned and metadata['model_id'] != pinned:
        raise ValueError(f'Unexpected model_id: {metadata["model_id"]}; expected {pinned}')
    if type(metadata.get('action_chunk_steps')) is not int or metadata['action_chunk_steps'] < 1:
        raise ValueError('Model must declare its action chunk length')
    return metadata
