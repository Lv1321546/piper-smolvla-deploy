import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from e14.inference_client import InferenceClient
from e14.run_robot import run
from server.inference_server import create_app
from server.policy_runner import PolicyRunner

ROOT = Path(__file__).resolve().parents[1]


class MockEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = json.loads((ROOT / 'config.mock.json').read_text())
        self.config['storage']['log_root'] = self.temp.name
        self.config['server']['port'] = 0
        self.runner = PolicyRunner(self.config, mock=True).load()
        self.server = create_app(self.runner, self.config)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.config['client']['url'] = f'http://127.0.0.1:{self.server.server_address[1]}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_both_dry_run_and_mock_execution(self):
        for execute in (False, True):
            path = run(copy.deepcopy(self.config), mock=True, execute=execute, duration=.65, task='mock test')
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(records[-1]['success'])
            self.assertTrue(any(row['event'] == 'prediction' for row in records))
            commands = [row for row in records if row['event'] == 'observation' and row['sent']]
            self.assertEqual(bool(commands), execute)
            self.assertEqual(any(row['event'] == 'stop' for row in records), execute)

    def test_mock_metadata_rejected_for_hardware_execution(self):
        with self.assertRaises(ValueError):
            InferenceClient(self.config).health(require_executable=True)

    def test_normal_167ms_inference_does_not_exhaust_default_horizon(self):
        original = self.runner.predict
        def delayed(observation, task):
            time.sleep(.167)
            return original(observation, task)
        self.runner.predict = delayed
        path = run(copy.deepcopy(self.config), mock=True, execute=True, duration=1.2, task='latency test')
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(records[-1]['success'])
        predictions = [row for row in records if row['event'] == 'prediction']
        self.assertGreaterEqual(len(predictions), 3)
        self.assertTrue(all(row['round_trip_ms'] >= 160 for row in predictions))


if __name__ == '__main__':
    unittest.main()
