"""Load explicit configuration; relative storage paths follow the JSON file."""
import copy
import json
from pathlib import Path


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict):
        raise ValueError('Config must be a JSON object')
    config = copy.deepcopy(config)
    for section in ('robot', 'client', 'server', 'storage', 'collection', 'cameras'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Missing config section: {section}')
    config['_config_path'] = str(path)
    for key in ('raw_root', 'log_root'):
        value = config['storage'].get(key)
        if value:
            value = Path(value).expanduser()
            config['storage'][key] = str(value if value.is_absolute() else path.parent / value)
    return config
