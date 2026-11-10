"""Replica scheduling, out-of-order results and fail-closed control integration."""
import concurrent.futures
import copy
import json
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from e14.deploy_policy import run
from e14.staggered_inference import StaggeredInference
from server.inference_server import create_app
from server.policy_runner import PolicyRunner


def configuration():
    return json.loads((Path(__file__).parents[1]/'config.inference.mock.json').read_text())


class SchedulerTests(unittest.TestCase):
    def test_tunnel_forwards_all_replica_ports_and_rejects_overflow(self):
        spec = importlib.util.spec_from_file_location('replica_tunnel',
                    Path(__file__).parents[1]/'scripts/inference_tunnel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch('sys.argv', ['tunnel', 'SuperSvr', '--local-port', '8095',
                               '--remote-port', '8085', '--replicas', '3']), \
             patch.object(module, 'tunnel_config', return_value='Host inference-target\n'), \
             patch.object(module.subprocess, 'call', return_value=0) as call:
            self.assertEqual(module.main(), 0)
        args = call.call_args.args[0]
        self.assertEqual(args.count('-L'), 3)
        for port in range(3):
            self.assertIn(f'127.0.0.1:{8095+port}:127.0.0.1:{8085+port}', args)
        with patch('sys.argv', ['tunnel', '--local-port', '65535', '--replicas', '2']), \
             self.assertRaises(SystemExit) as error:
            module.main()
        self.assertEqual(error.exception.code, 2)

    def test_staggering_bounds_pending_and_uses_latest_observation_without_queue(self):
        cfg = configuration()
        cfg['client'].update(url='http://127.0.0.1:8000', inference_interval_steps=4,
                             replica_urls=[f'http://127.0.0.1:{8000+i}' for i in range(3)])
        executor = Mock()
        executor.submit.side_effect = lambda *args: concurrent.futures.Future()
        with patch('e14.staggered_inference.concurrent.futures.ThreadPoolExecutor', return_value=executor):
            pool = StaggeredInference(cfg)
        self.addCleanup(pool.close)
        self.assertEqual(pool.submit({'frame':0}, 'task', 0.), 0)
        self.assertIsNone(pool.submit({'frame':1}, 'task', .1))
        self.assertEqual(pool.submit({'frame':2}, 'task', .2), 1)
        self.assertEqual(pool.submit({'frame':4}, 'task', .4), 2)
        self.assertIsNone(pool.submit({'frame':20}, 'task', 1.))
        self.assertEqual(executor.submit.call_count, 3)
        pool.pending[0].set_result('done')
        self.assertEqual(pool.completed()[0][0], 0)
        self.assertEqual(pool.submit({'frame':21}, 'task', 1.05), 0)
        self.assertEqual(executor.submit.call_args.args[2], {'frame':21})
        pool.pending[1].set_result('done')
        pool.completed()
        pool.defer(1, 2.)
        self.assertIsNone(pool.submit({'frame':30}, 'task', 1.5))
        self.assertEqual(pool.submit({'frame':40}, 'task', 2.), 1)

    def test_duplicate_or_misaligned_replica_urls_rejected(self):
        for urls in ([], ['http://127.0.0.1:8001'],
                     ['http://127.0.0.1:8000', 'http://127.0.0.1:8000/']):
            cfg = configuration()
            cfg['client'].update(url='http://127.0.0.1:8000', replica_urls=urls)
            with self.subTest(urls=urls), self.assertRaises(ValueError):
                StaggeredInference(cfg)


class MultiReplicaIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = configuration()
        self.cfg['storage']['log_root'] = self.temp.name
        self.runners, self.servers, self.threads = [], [], []
        urls = []
        for _ in range(3):
            cfg = copy.deepcopy(self.cfg)
            cfg['server']['port'] = 0
            runner = PolicyRunner(cfg, mock=True).load()
            server = create_app(runner, cfg)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.runners.append(runner); self.servers.append(server); self.threads.append(thread)
            urls.append(f'http://127.0.0.1:{server.server_address[1]}')
        self.cfg['client'].update(url=urls[0], replica_urls=urls, inference_interval_steps=4)

    def tearDown(self):
        for server in self.servers:
            server.shutdown(); server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)
        self.temp.cleanup()

    def rows(self):
        path = next(Path(self.temp.name).glob('policy-*'))
        return [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]

    def delay(self, index, seconds):
        original = self.runners[index].predict
        def delayed(*args):
            time.sleep(seconds)
            return original(*args)
        self.runners[index].predict = delayed

    def test_half_second_requests_stay_continuous_and_old_results_never_replace_new(self):
        for index, delay in enumerate((.59, .27, .42)):
            self.delay(index, delay)
        path = run(self.cfg, mock=True, execute=True, duration=2.7, no_preview=True)
        rows = self.rows(); end = rows[-1]
        self.assertTrue(end['success'])
        self.assertEqual(end['prediction_waits'], 0)
        self.assertEqual(end['prediction_resumes'], 1)
        self.assertGreater(end['sent_commands'], 20)
        self.assertTrue(any(r['event']=='prediction_discarded' and
                            r['reason']=='superseded by newer observation' for r in rows))
        active = [r for r in rows if r['event']=='cycle' and r['command'] is not None]
        sources = [r['action_source_monotonic'] for r in active]
        self.assertEqual(sources, sorted(sources))
        self.assertTrue(all(0 <= r['action_age_ms'] <= 1000 for r in active))
        self.assertEqual({r['replica_index'] for r in rows if r['event']=='prediction_requested'}, {0,1,2})
        # This test uses only the simulated writer, including the final hold.
        self.assertTrue(any(r['event']=='measured_joint_hold_requested' for r in rows))

    def test_model_hash_mismatch_between_replicas_rejected_at_startup(self):
        self.runners[1]._metadata['model_sha256'] = 'different-checkpoint'
        pool = StaggeredInference(self.cfg)
        self.addCleanup(pool.close)
        with self.assertRaisesRegex(ValueError, 'different model metadata'):
            pool.health()

    def test_ready_fresh_result_replaces_expired_plan_before_any_hold(self):
        engine = Mock()
        engine.health.return_value = self.runners[0].metadata
        engine.submit.return_value = None
        polls = 0
        def completed():
            nonlocal polls
            polls += 1
            if polls > 2:
                return []
            now = time.monotonic()
            captured = now-(.98 if polls==1 else .01)
            future = concurrent.futures.Future()
            response = dict(protocol_version='piper-v1', request_id=str(polls),
                            actions=[self.cfg['inference']['mock_initial_state']]*50,
                            metadata=self.runners[0].metadata, inference_ms=1.)
            future.set_result(dict(response=response, request_id=str(polls),
                                   captured_monotonic=captured, received_monotonic=now,
                                   request_started_monotonic=captured, round_trip_ms=(now-captured)*1000,
                                   replica_index=0, replica_url=self.cfg['client']['url']))
            return [(0,future)]
        engine.completed.side_effect = completed
        with patch('e14.deploy_policy.StaggeredInference', return_value=engine):
            run(self.cfg, mock=True, execute=True, duration=.3, no_preview=True)
        end = self.rows()[-1]
        self.assertTrue(end['success'])
        self.assertEqual(end['prediction_waits'], 0)
        self.assertEqual(end['prediction_resumes'], 1)


    def test_timeout_on_one_replica_does_not_pause_healthy_replicas(self):
        self.cfg['client'].update(timeout_ms=200, retry_interval_ms=50, inference_interval_steps=1)
        self.delay(0, .6)
        self.delay(1, .02)
        self.delay(2, .02)
        run(self.cfg, mock=True, execute=True, duration=1.2, no_preview=True)
        rows = self.rows(); end = rows[-1]
        self.assertTrue(end['success'])
        self.assertEqual(end['prediction_waits'], 0)
        self.assertGreater(end['prediction_retries'], 0)
        self.assertGreater(end['sent_commands'], 10)
        retries = [r for r in rows if r['event']=='prediction_retry']
        self.assertTrue(all(r['replica_index']==0 for r in retries))

    def test_unrecoverable_old_replica_error_still_stops_and_holds(self):
        def fail(*args):
            time.sleep(.6)
            raise RuntimeError('synthetic model failure on older request')
        self.runners[0].predict = fail
        self.delay(1, .1)
        self.delay(2, .1)
        with self.assertRaisesRegex(RuntimeError, 'HTTP 503'):
            run(self.cfg, mock=True, execute=True, duration=2., no_preview=True)
        rows = self.rows()
        self.assertFalse(rows[-1]['success'])
        self.assertGreater(rows[-1]['sent_commands'], 0)
        self.assertTrue(any(r['event']=='measured_joint_hold_requested' for r in rows))


if __name__ == '__main__':
    unittest.main()
