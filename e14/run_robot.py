"""E14 supervisor: read-only observe, dry-run inference, or explicit motor execution."""
import argparse
import concurrent.futures
import json
import math
import time
from pathlib import Path

from shared.config import load_config
from shared.protocol import finite_number, validate_response, vector7
from e14.cameras import CameraReader
from e14.inference_client import InferenceClient
from e14.piper_robot import PiperRobot


def capture_observation(robot, cameras, config):
    state = robot.get_observation()
    camera = cameras.read()
    now = time.monotonic()
    timestamps = list(camera['image_timestamps'].values()) + [state['feedback_monotonic']]
    if not timestamps or any(not math.isfinite(t) or t > now for t in timestamps):
        raise ValueError('Invalid observation clock')
    oldest = min(timestamps)
    max_skew = config['collection']['max_sync_skew_ms'] / 1000
    if max(timestamps) - oldest > max_skew:
        raise TimeoutError('Robot/camera sample skew exceeds configured threshold')
    if now - state['feedback_monotonic'] > config['robot']['feedback_max_age_ms'] / 1000:
        raise TimeoutError('Robot observation is stale')
    for name, stamp in camera['image_timestamps'].items():
        if now - stamp > config['cameras'][name].get('max_age_ms', 200) / 1000:
            raise TimeoutError(f'Camera {name} is stale')
    return {
        'state': vector7(state['state']),
        'images': camera['images'],
        'image_timestamps': camera['image_timestamps'],
        # Start the action-age clock at the oldest actual source sample.
        'captured_monotonic': oldest,
        'feedback_monotonic': state['feedback_monotonic'],
    }


def validate_action(action, current_state, config):
    action = vector7(action, 'action')
    current = vector7(current_state)
    options = config['robot']
    limits = list(options['joint_limits_deg']) + [options['gripper_limits_m']]
    deltas = vector7(options['max_step_delta'], 'max_step_delta')
    for i, (target, measured, bounds, delta) in enumerate(zip(action, current, limits, deltas)):
        if len(bounds) != 2 or not bounds[0] <= target <= bounds[1]:
            raise ValueError(f'Action axis {i} exceeds absolute limits')
        if delta <= 0 or abs(target - measured) > delta + 1e-9:
            raise ValueError(f'Action axis {i} exceeds maximum change relative to feedback')
    return action


def select_action_chunk(result, now, hz, max_age_s, execution_horizon):
    age = now - result['captured_monotonic']
    if age < 0 or age > max_age_s:
        raise TimeoutError('Action chunk expired')
    index = int(age * hz)
    received_index = int((result['received_monotonic'] - result['captured_monotonic']) * hz)
    actions = result['response']['actions']
    if index >= len(actions) or index >= received_index + execution_horizon:
        return None
    return actions[index]


def run(config, *, mock=False, execute=False, observe=False, duration=10, task='Pick up the cube and place it in the box.'):
    finite_number(duration, 'duration', positive=True)
    hz = finite_number(config['robot']['control_hz'], 'control_hz', positive=True)
    if execute and observe:
        raise ValueError('--observe and --execute are mutually exclusive')
    if execute and not mock and config['robot'].get('stop_strategy_verified') is not True:
        raise ValueError('First verify the configured stop behavior on the actual arm, then set stop_strategy_verified')
    log_root = Path(config['storage']['log_root'])
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / (time.strftime('%Y%m%d-%H%M%S') + f'-{time.time_ns() % 1000000:06d}-control.jsonl')
    robot = PiperRobot(config, mock=mock)
    cameras = CameraReader(config['cameras'], mock=mock)
    client = None if observe else InferenceClient(config)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='inference')
    future = None
    active = None
    armed = False
    started = time.monotonic()
    last_request = -math.inf
    step = 0
    metadata = None
    failure = None
    with log_path.open('x', encoding='utf-8') as log:
        def event(kind, **kwargs):
            log.write(json.dumps({'event': kind, 'monotonic': time.monotonic(), **kwargs}, allow_nan=False) + '\n')
            log.flush()
        event('start', mock=mock, execute=execute, observe=observe, task=task, control_hz=hz)
        try:
            if client:
                metadata = client.health(require_executable=execute and not mock)
                event('health', metadata=metadata)
                if not mock and metadata.get('backend') == 'mock' and execute:
                    raise ValueError('Mock server cannot control physical hardware')
            robot.connect()
            cameras.connect()
            if execute:
                # Even a partially completed preparation can have enabled motors.
                armed = True
                robot.prepare_control()
            period = 1 / hz
            deadline = time.monotonic()
            started = deadline
            while time.monotonic() - started < duration:
                now = time.monotonic()
                if now - deadline > period:
                    raise TimeoutError('Local control loop missed its deadline')
                observation = capture_observation(robot, cameras, config)
                now = time.monotonic()
                if future is not None and future.done():
                    # Failure never silently keeps an older action chunk running.
                    active = future.result()
                    future = None
                    validate_response(active['response'], active['request_id'], hz, execute and not mock)
                    event('prediction', request_id=active['request_id'],
                          round_trip_ms=active['round_trip_ms'],
                          inference_ms=active['response']['inference_ms'],
                          captured_monotonic=active['captured_monotonic'],
                          actions=active['response']['actions'])
                interval = config['client'].get('inference_interval_steps', 2) / hz
                if client and future is None and now - last_request >= interval:
                    future = executor.submit(client.predict, observation, task, None, execute and not mock)
                    last_request = now
                action = None
                if active:
                    action = select_action_chunk(active, now, hz,
                              config['client']['max_action_age_ms'] / 1000,
                              config['client'].get('execution_horizon', 5))
                    if action is None and execute:
                        raise TimeoutError('Action horizon exhausted before a fresh prediction arrived')
                elif execute and now - started > config['client']['timeout_ms'] / 1000:
                    raise TimeoutError('Initial prediction deadline exceeded')
                if execute and action is not None:
                    validate_action(action, observation['state'], config)
                    robot.send_action(action)
                event('observation', step=step, state=observation['state'],
                      captured_monotonic=observation['captured_monotonic'],
                      image_shapes={k: list(v.shape) for k, v in observation['images'].items()},
                      action=action, sent=bool(execute and action is not None))
                if step % max(1, round(hz)) == 0:
                    print(f"step={step} mode={'mock' if mock else 'real'}/{'execute' if execute else 'observe' if observe else 'dry-run'} state={observation['state']}", flush=True)
                step += 1
                deadline += period
                time.sleep(max(0, deadline - time.monotonic()))
        except BaseException as error:
            failure = error
            event('error', message=str(error), error_type=type(error).__name__)
        finally:
            if armed:
                try:
                    robot.stop()
                    event('stop', strategy=config['robot']['stop_strategy'])
                except Exception as error:
                    event('stop_failed', message=str(error))
                    if failure is None:
                        failure = error
            if future is not None:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            for cleanup in (cameras.close, robot.disconnect):
                try:
                    cleanup()
                except Exception as error:
                    event('cleanup_failed', message=str(error))
                    if failure is None:
                        failure = error
            event('end', steps=step, success=failure is None)
    print('Log:', log_path, flush=True)
    if failure:
        raise failure
    return log_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mock', action='store_true', help='Simulated robot and cameras only')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true', help='Explicitly enable and send checked motor targets')
    group.add_argument('--observe', action='store_true', help='Read hardware/cameras without inference')
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--task', default='Pick up the cube and place it in the box.')
    args = parser.parse_args()
    run(load_config(args.config), mock=args.mock, execute=args.execute, observe=args.observe,
        duration=args.duration, task=args.task)


if __name__ == '__main__':
    main()
