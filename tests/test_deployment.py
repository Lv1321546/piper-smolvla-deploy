"""Deployment failure paths with synthetic hardware and a loopback HTTP server."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from e14.deploy_policy import run
from e14.policy_control import PolicyOutput, PositionFollower, SimulatedArm, select_position_mode
from e14.replay_inference import replay
from server.inference_server import create_app
from server.policy_runner import PolicyRunner
from shared.deployment import check_model
from shared.protocol import ACTION_NAMES
from shared.recorded_observation import RecordedEpisode

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((ROOT/'config.inference.mock.json').read_text())


class FollowerTests(unittest.TestCase):
    def setUp(self):
        self.config = configuration()
        self.arm = SimulatedArm(self.config)
        self.now = 10.
        self.output = PolicyOutput(self.config, self.arm, clock=lambda: self.now)
        self.follower = PositionFollower(self.config)

    def test_joint_and_jaw_commands_continuous_at_sdk_precision(self):
        first = self.arm.state.copy()
        previous_position = first.copy()
        previous_velocity = np.zeros(7)
        moving = []
        for i in range(160):
            self.now += .05
            target = first + np.array(([5.,-5.,-5.,5.,-5.,5.,-.025] if i < 80
                                     else [-2.,2.,2.,-2.,2.,-2.,.015]))
            measured, _, snap = self.arm.sample(self.now, .05)
            position, velocity, _ = self.follower.step(target.tolist(), self.arm.state.tolist())
            self.assertTrue(np.all(abs(velocity) <= np.array(self.config['inference']['max_velocity'])+1e-9))
            self.assertTrue(np.all(abs(velocity-previous_velocity)/.05 <= np.array(self.config['inference']['max_acceleration'])+1e-8))
            self.assertTrue(np.all(abs(position-previous_position)/.05 <= np.array(self.config['inference']['max_velocity'])+1e-8))
            self.output.send(position.tolist(), velocity.tolist(), measured, snap)
            previous_position, previous_velocity = position, velocity
            moving.append(self.arm.state.copy())
        self.assertGreater(abs(moving[79][0]-first[0]), 3.)
        self.assertLess(moving[-1][0], moving[79][0])
        jaw_before = self.arm.state[6]
        self.output.hold(self.arm.sample(self.now, .05)[0])
        self.assertFalse(self.output.holder.moved)
        self.assertEqual(self.arm.state[6], jaw_before)
        self.assertEqual(self.arm.commands[-1][0], 'joint')

    def test_stuck_feedback_faults_instead_of_winding_up(self):
        state = self.arm.state.copy()
        for _ in range(50):
            self.now += .05
            target = state + np.array([10.,0.,0.,0.,0.,0.,0.])
            q, v, _ = self.follower.step(target.tolist(), state.tolist())
            feedback, _, snap = self.arm.sample(self.now, .05)
            try:
                self.output.send(q.tolist(), v.tolist(), feedback, snap)
            except ValueError as exc:
                self.assertIn('tracking-error', str(exc))
                break
            self.arm.state = state.copy()  # Simulate an actuator that does not move.
        else:
            self.fail('Stuck arm never faulted')

    def test_unsafe_model_target_and_cadence_rejected(self):
        state = self.arm.state.tolist()
        for axis, value in ((0, 151.), (6, -.002), (0, 30.), (1, float('nan'))):
            action = state.copy()
            action[axis] = value
            with self.assertRaises(ValueError):
                self.follower.step(action, state)
        for dt in (.001, .2):
            with self.assertRaisesRegex(ValueError, 'cadence'):
                self.follower.step(state, state, dt=dt)
        self.assertIsNone(self.follower.position)

    def test_only_small_boundary_noise_can_be_corrected(self):
        state = self.arm.state.tolist()
        action = state.copy()
        action[6] = -.0003
        self.assertEqual(self.follower.checked_target(action, state)[6], 0.)
        action[6] = -.003
        with self.assertRaises(ValueError):
            self.follower.checked_target(action, state)

    def test_jaw_reaches_both_endpoints_without_overshoot_or_acceleration_jump(self):
        state = self.arm.state.tolist()
        previous_velocity = np.zeros(7)
        for endpoint in (0., .07, 0.):
            target = state.copy()
            target[6] = endpoint
            for i in range(180):
                q, v, _ = self.follower.step(target,state)
                self.assertTrue(0. <= q[6] <= .07)
                self.assertTrue(np.all(abs(v-previous_velocity)/.05 <= np.array(self.config['inference']['max_acceleration'])+1e-8))
                state, previous_velocity = q.tolist(), v.copy()
            self.assertAlmostEqual(state[6],endpoint,places=10)

    def test_resume_restarts_joints_but_keeps_jaw_command_despite_encoder_offset(self):
        target = self.arm.state.copy()
        target[0] += 2.
        target[6] += .005
        for _ in range(6):
            self.now += .05
            fb, _, snap = self.arm.sample(self.now, .05)
            q, v, _ = self.follower.step(target.tolist(), self.arm.state.tolist())
            self.output.send(q.tolist(), v.tolist(), fb, snap)
        last_width = self.output.last_width
        self.arm.state[6] -= .0005  # A stable, real-style jaw encoder offset.
        fb, _, _ = self.arm.sample(self.now, .05)
        self.output.hold(fb)
        before = self.arm.state.copy()
        self.now += 2.
        self.output.resume(self.follower, before.tolist())
        q, v, _ = self.follower.step(target.tolist(), before.tolist())
        self.assertLessEqual(abs(v[0]), .8*self.config['inference']['max_acceleration'][0]*.05+1e-8)
        self.assertLessEqual(abs(q[6]-last_width), self.config['inference']['max_velocity'][6]*.05)
        fb, _, snap = self.arm.sample(self.now, .05)
        self.output.send(q.tolist(), v.tolist(), fb, snap)


class ContractTests(unittest.TestCase):
    def test_position_goal_precedes_mode_even_if_already_in_can(self):
        from types import SimpleNamespace
        config = configuration()
        calls = []
        writer = SimpleNamespace(joint=lambda q: calls.append(('joint', q)),
                    sdk=SimpleNamespace(MotionCtrl_2=lambda *args: calls.append(('mode', args))))
        _, _, snap = SimulatedArm(config).sample(time.monotonic(), .05)
        select_position_mode(writer, snap, config)
        self.assertEqual([x[0] for x in calls], ['joint','mode'])
        self.assertEqual(calls[1][1], (1,1,config['robot']['speed_percent'],0))

    def test_camera_names_resolution_and_model_pin(self):
        config = configuration()
        metadata = PolicyRunner(config, mock=True).load().metadata
        check_model(metadata, config)
        for key, value in (('cameras', {'front': {'width':64,'height':48}}),
                           ('model_id','another-model'), ('label_semantics','delta_action')):
            with self.assertRaises(ValueError):
                check_model({**metadata,key:value}, config)
        with self.assertRaises(ValueError):
            check_model(metadata, config, physical=True)

    def test_tunnel_drops_unrelated_forwarding_but_keeps_jump_and_key(self):
        spec = importlib.util.spec_from_file_location('inference_tunnel',ROOT/'scripts/inference_tunnel.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = {'hostname':['10.0.0.2'],'user':['lvzx'],'identityfile':['~/.ssh/id_key'],
                  'proxyjump':['desktop'],'remoteforward':['27891 localhost:7890']}
        hop = {'hostname':['10.0.0.1'],'user':['eureka'],'identityfile':['~/.ssh/id_key']}
        with patch.object(module, 'settings', side_effect=[target,hop]):
            config = module.tunnel_config('Inspur')
        self.assertNotIn('27891',config)
        self.assertIn('ProxyJump inference-jump',config)
        self.assertIn('~/.ssh/id_key',config)


class DeploymentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.config = configuration()
        self.config['storage']['log_root'] = self.temp.name
        self.config['server']['port'] = 0
        self.runner = PolicyRunner(self.config,mock=True).load()
        self.http = create_app(self.runner,self.config)
        self.thread = threading.Thread(target=self.http.serve_forever,daemon=True)
        self.thread.start()
        self.config['client']['url'] = f'http://127.0.0.1:{self.http.server_address[1]}'

    def tearDown(self):
        self.http.shutdown(); self.http.server_close(); self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_280ms_inference_overlaps_and_dry_run_never_opens_sdk(self):
        original = self.runner.predict
        def delayed(*args):
            time.sleep(.28)
            return original(*args)
        self.runner.predict = delayed
        with patch('servo_bridge.sdk_worker.SDKJointWriter',side_effect=AssertionError('real SDK opened')), \
             patch('servo_bridge.sources.PassiveCan',side_effect=AssertionError('physical CAN opened')):
            for execute in (False,True):
                path = run(copy.deepcopy(self.config),mock=True,execute=execute,duration=1.4,no_preview=True)
                summary = json.loads((path/'summary.json').read_text())
                self.assertTrue(summary['success'])
                self.assertGreaterEqual(summary['predictions'],3)
                self.assertEqual(summary['sent_commands']>0,execute)
                events = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
                self.assertEqual(any(x['event']=='measured_joint_hold_requested' for x in events),execute)

    def test_server_failure_cancels_execution_and_requests_measured_hold(self):
        original = self.runner.predict
        calls = 0
        def failing(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError('synthetic inference failure')
            return original(*args)
        self.runner.predict = failing
        with self.assertRaisesRegex(RuntimeError,'HTTP 503'):
            run(self.config,mock=True,execute=True,duration=.8,no_preview=True)
        path = next(Path(self.temp.name).glob('policy-*'))
        events = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(x['event']=='measured_joint_hold_requested' for x in events))
        self.assertFalse(events[-1]['success'])

    def test_exhausted_horizon_holds_discards_old_inflight_and_resumes_fresh(self):
        self.config['client']['execution_horizon'] = 2
        self.config['client']['inference_interval_steps'] = 1
        original = self.runner.predict
        calls = 0
        def delayed(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(.3)
            actions, elapsed = original(*args)
            for action in actions:
                action[0] += 2.
                action[6] += .005
            return actions, elapsed
        self.runner.predict = delayed
        path = run(self.config,mock=True,execute=True,duration=1.1,no_preview=True)
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(rows[-1]['success'])
        self.assertGreater(rows[-1]['prediction_waits'],0)
        self.assertTrue(any(x['event']=='measured_joint_hold_requested' for x in rows))
        discarded = [x['request_id'] for x in rows if x['event']=='prediction_discarded']
        self.assertTrue(discarded)
        self.assertTrue(any(x['event']=='prediction_discarded' and x['reason']=='source predates hold' for x in rows))
        resumed = [x for x in rows if x['event']=='prediction_resumed' and x['fresh_after'] is not None]
        self.assertTrue(resumed)
        self.assertTrue(all(x['captured_monotonic']>=x['fresh_after'] for x in resumed))
        self.assertFalse(set(discarded) & {x['request_id'] for x in resumed})
        paused = False
        resumes_checked = 0
        for row in rows:
            if row['event']=='prediction_wait_started' and not row.get('initial'):
                paused = True
            elif row['event']=='prediction_resumed':
                if paused:
                    paused = False
                    resumes_checked += 1
                    continue
            elif row['event']=='cycle':
                if paused:
                    self.assertIsNone(row['command'])
                    self.assertTrue(row['waiting_for_prediction'])
        self.assertGreater(resumes_checked,0)
        # Actual follower output resumes with bounded acceleration, not the old velocity.
        for resume in resumed:
            cycle = next(x for x in rows if x['event']=='cycle' and x['monotonic']>=resume['monotonic'])
            self.assertLessEqual(abs(cycle['velocity'][0]), .8*15*.05+1e-8)

    def test_initial_timeout_and_busy_responses_retry_then_start(self):
        self.config['client'].update(timeout_ms=100,max_action_age_ms=500,retry_interval_ms=50)
        original = self.runner.predict
        calls = 0
        def delayed(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                time.sleep(.4)
            return original(*args)
        self.runner.predict = delayed
        path = run(self.config,mock=True,execute=True,duration=1.,no_preview=True)
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(rows[-1]['success'])
        self.assertGreater(rows[-1]['sent_commands'],0)
        errors = [x for x in rows if x['event']=='prediction_retry']
        self.assertTrue(any('deadline' in x['message'] for x in errors))
        self.assertTrue(any('HTTP 429' in x['message'] for x in errors))

    def test_repeated_expired_results_wait_until_duration_without_commands(self):
        self.config['client'].update(timeout_ms=800,max_action_age_ms=100,retry_interval_ms=50)
        original = self.runner.predict
        def delayed(*args):
            time.sleep(.17)
            return original(*args)
        self.runner.predict = delayed
        path = run(self.config,mock=True,execute=True,duration=.7,no_preview=True)
        summary = json.loads((path/'summary.json').read_text())
        self.assertTrue(summary['success'])
        self.assertTrue(summary['waiting_at_end'])
        self.assertEqual(summary['sent_commands'],0)
        self.assertGreaterEqual(summary['prediction_retries'],2)

    def test_response_expiring_between_arrival_and_control_tick_is_discarded(self):
        from e14.inference_client import InferenceClient
        original = InferenceClient.predict
        calls = 0
        def boundary(client,*args,**kwargs):
            nonlocal calls
            result = original(client,*args,**kwargs)
            calls += 1
            if calls == 1:
                # Valid at arrival (480 ms), expired when polled (>=520 ms).
                result['captured_monotonic'] = time.monotonic()-.52
                result['received_monotonic'] = result['captured_monotonic']+.48
            return result
        self.config['client']['max_action_age_ms'] = 500
        with patch.object(InferenceClient,'predict',boundary):
            path = run(self.config,mock=True,execute=True,duration=.5,no_preview=True)
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(rows[-1]['success'])
        discarded = next(x for x in rows if x['event']=='prediction_discarded')
        self.assertEqual(discarded['reason'],'expired before control tick')
        self.assertNotIn(discarded['request_id'],[x['request_id'] for x in rows if x['event']=='prediction_resumed'])
        self.assertGreater(rows[-1]['sent_commands'],0)

    def test_sensor_failure_during_wait_is_still_terminal(self):
        from e14.policy_control import SimulatedArm
        class BadFeedback(SimulatedArm):
            def __init__(self, config):
                super().__init__(config)
                self.samples = 0
            def sample(self, now, dt):
                self.samples += 1
                feedback, _, snap = super().sample(now,dt)
                if self.samples >= 5:
                    feedback.ready = False
                return feedback, None, snap
        original = self.runner.predict
        def delayed(*args):
            time.sleep(.3)
            return original(*args)
        self.runner.predict = delayed
        with patch('e14.deploy_policy.SimulatedArm',BadFeedback):
            with self.assertRaisesRegex(RuntimeError,'Lost enabled healthy'):
                run(self.config,mock=True,execute=True,duration=.8,no_preview=True)

    def test_operator_stop_while_waiting_cannot_resume_from_pending_result(self):
        original = self.runner.predict
        def delayed(*args):
            time.sleep(.4)
            return original(*args)
        self.runner.predict = delayed
        class StopDashboard:
            url = 'mock-preview'
            def __enter__(self):
                self.start = time.monotonic()
                return self
            def __exit__(self,*_): pass
            def update(self,*_): pass
            def drain(self):
                return ['quit'] if time.monotonic()-self.start>.2 else []
        with patch('e14.teaching_ui.Dashboard',return_value=StopDashboard()):
            path = run(self.config,mock=True,execute=True,duration=2.)
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(x['event']=='operator_stop' for x in rows))
        self.assertFalse(any(x['event']=='prediction_resumed' for x in rows))
        self.assertEqual(rows[-1]['sent_commands'],0)
        self.assertLess(rows[-1]['elapsed_s'],1.)

    def test_hold_failure_is_terminal_instead_of_resuming(self):
        self.config['client']['execution_horizon'] = 2
        original = self.runner.predict
        calls = 0
        def delayed(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(.3)
            return original(*args)
        self.runner.predict = delayed
        with patch.object(PolicyOutput,'hold',side_effect=RuntimeError('synthetic hold failed')):
            with self.assertRaisesRegex(RuntimeError,'synthetic hold failed'):
                run(self.config,mock=True,execute=True,duration=.9,no_preview=True)
        path = next(Path(self.temp.name).glob('policy-*'))
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertFalse(rows[-1]['success'])
        self.assertEqual(sum(x['event']=='prediction_resumed' for x in rows),1)

    def test_model_mismatch_is_fatal_even_in_an_expired_prediction(self):
        from e14.inference_client import InferenceClient
        original = InferenceClient.predict
        def changed(client,*args,**kwargs):
            result = original(client,*args,**kwargs)
            result['response']['metadata']['model_id'] = 'unexpected-model'
            result['captured_monotonic'] -= 2.
            return result
        with patch.object(InferenceClient,'predict',changed):
            with self.assertRaisesRegex(ValueError,'model_id'):
                run(self.config,mock=True,execute=True,duration=.5,no_preview=True)

    def test_recorded_inference_uses_next_state_and_no_hardware(self):
        episode = Path(self.temp.name)/'episode'
        episode.mkdir()
        manifest = dict(format='piper-raw-v1',status='complete',state_names=list(ACTION_NAMES),
                        joint_unit='degree',gripper_unit='meter_full_opening',action_mode='absolute_position',
                        label_semantics='next_state',control_hz=20,frames=4,task='Pick block',
                        cameras=self.config['cameras'])
        (episode/'manifest.json').write_text(json.dumps(manifest))
        rows = []
        for index in range(4):
            images = {}
            for name in self.config['cameras']:
                file = f'{name}-{index}.jpg'
                Image.new('RGB',(64,48),(index,0,0)).save(episode/file)
                images[name] = file
            state = [float(index),80.,-60.,0.,40.,0.,.04]
            rows.append(dict(frame_index=index,state=state,images=images,captured_monotonic=100+index*.05))
        (episode/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        sample = RecordedEpisode(episode,self.config['cameras'],20).sample(1)
        self.assertEqual(sample['next_state'][0],2.)
        report = replay(self.config,episode,[0,2],Path(self.temp.name)/'replay')
        self.assertEqual(report['samples'],2)
        self.assertEqual(report['first_action_mae'][0],1.)
        with self.assertRaises(ValueError):
            RecordedEpisode(episode,self.config['cameras'],20).sample(3)


if __name__ == '__main__':
    unittest.main()
