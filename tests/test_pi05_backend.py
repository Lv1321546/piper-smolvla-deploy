import json
from pathlib import Path
import unittest
from shared.deployment import check_model
from shared.protocol import validate_metadata
from tests.test_protocol_runtime import metadata


class Pi05BackendTest(unittest.TestCase):
    def test_real_backend_contract_and_identity(self):
        cfg = json.loads((Path(__file__).parents[1] / 'config.pi05.json').read_text())
        m = dict(metadata(), backend='pi05', execution_ready=True,
                 protocol_version='piper-v1', label_semantics='next_state', action_chunk_steps=50,
                 model_id=cfg['inference']['expected_model_id'],
                 cameras={k: {x: v[x] for x in ('width','height')} for k,v in cfg['cameras'].items()})
        check_model(m, cfg, physical=True)
        with self.assertRaises(ValueError):
            check_model(dict(m, backend='smolvla'), cfg, physical=True)
        with self.assertRaises(ValueError):
            check_model(dict(m, model_id='another-checkpoint'), cfg, physical=True)
        for change in ({'execution_ready':False}, {'backend':'mock'}, {'joint_unit':'radian'}):
            with self.assertRaises(ValueError):
                validate_metadata(dict(m, **change), 20, require_executable=True)
