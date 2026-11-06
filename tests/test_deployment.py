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

    def test_exhausted_horizon_does_not_execute_a_late_response(self):
        self.config['client']['execution_horizon'] = 2
        original = self.runner.predict
        calls = 0
        def delayed(*args):
            nonlocal calls
            calls += 1
            if calls > 1:
                time.sleep(.3)
            return original(*args)
        self.runner.predict = delayed
        with self.assertRaisesRegex(TimeoutError,'horizon'):
            run(self.config,mock=True,execute=True,duration=1.,no_preview=True)
        path = next(Path(self.temp.name).glob('policy-*'))
        rows = [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(x['event']=='prediction' for x in rows),1)
        self.assertTrue(any(x['event']=='measured_joint_hold_requested' for x in rows))
        self.assertFalse(rows[-1]['success'])

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
