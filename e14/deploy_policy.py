"""Live PiPER inference: read-only by default, explicit --execute for SDK output.

Uses passive CAN feedback, both trained RGB views, asynchronous HTTP inference,
time-aligned action chunks and a rate-limited absolute-position follower.
No ROS/Cartesian inverse kinematics, arming token, reset or automatic homing.
"""
import argparse
import concurrent.futures
from contextlib import ExitStack
import json
import math
from pathlib import Path
import signal
import time

from e14.cameras import CameraReader
from e14.inference_client import InferenceClient
from e14.policy_control import PolicyOutput, PositionFollower, SimulatedArm, guard_config, select_position_mode
from e14.run_robot import capture_observation, select_action_chunk
from shared.config import load_config
from shared.deployment import check_model
from shared.protocol import finite_number, validate_response

TASK = 'Pick up the black foam block and place it in the box.'
PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PiPER SmolVLA 推理</title>
<style>body{font:16px sans-serif;max-width:1200px;margin:24px auto;background:#161b22;color:#eee}img{width:100%}#views{display:flex;gap:12px}figure{width:50%;margin:0}button{padding:12px}pre{white-space:pre-wrap}</style>
<h1>PiPER SmolVLA 推理</h1><p id="message">连接中</p>
<button id="stop">停止本次推理（Q / 空格）</button><p>停止请求会在控制循环中处理；关闭页面不会停止程序。终端 Ctrl+C 同样请求位置保持。</p>
<div id="views"></div><pre id="state"></pre><script>
const token=__TOKEN__,sessionId=__SESSION_ID__;
async function stop(){await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json','X-Teaching-Token':token},body:JSON.stringify({command:'quit'})})}
document.querySelector('#stop').onclick=stop;document.onkeydown=e=>{if(!e.repeat&&(e.code==='KeyQ'||e.code==='Space')){e.preventDefault();stop()}};
async function refresh(){try{const s=await(await fetch('/state',{cache:'no-store'})).json();if(s.session_id!==sessionId){location.reload();return}
document.querySelector('#message').textContent=s.message;document.querySelector('#state').textContent=JSON.stringify(s,null,2);
const v=document.querySelector('#views');if(!v.children.length)for(const n of s.camera_names??[]){const f=document.createElement('figure'),c=document.createElement('figcaption'),i=document.createElement('img');c.textContent=n;i.src='/stream/'+encodeURIComponent(n);f.append(c,i);v.append(f)}}catch(e){document.querySelector('#message').textContent='程序已断开，请查看终端和日志。'}setTimeout(refresh,250)}refresh();</script></html>'''


class Observer:
    def __init__(self, source, config):
        self.source, self.config = source, config
        self.latest = None

    def get_observation(self):
        now = time.monotonic()
        feedback, _, snapshot = self.source.sample(now, 1/self.config['robot']['control_hz'])
        if not 0 <= now-feedback.received <= self.config['robot']['feedback_max_age_ms']/1000:
            raise TimeoutError('CAN feedback is stale or future')
        self.latest = (feedback, snapshot)
        return {'state': snapshot['state'], 'feedback_monotonic': feedback.received}


def run(config, *, execute=False, observe=False, mock=False, duration=30., task=TASK,
        initialize_can=False, preview_port=8767, no_preview=False):
    finite_number(duration, 'duration', positive=True)
    if execute and observe:
        raise ValueError('--execute and --observe are mutually exclusive')
    if mock and initialize_can:
        raise ValueError('Mock mode cannot configure physical CAN')
    if execute and config['robot'].get('enable_gripper') is not True:
        raise ValueError('Pick/place policy execution requires robot.enable_gripper=true')
    follower = PositionFollower(config)  # Validate before touching any device.
    hz, period = config['robot']['control_hz'], follower.period
    if config['client']['max_action_age_ms'] <= 0 or config['client']['execution_horizon'] < 1:
        raise ValueError('Positive action age and execution horizon required')
    root = Path(config['storage']['log_root']) / ('policy-' + time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns()%1000000))
    root.mkdir(parents=True, exist_ok=False)
    client = None if observe else InferenceClient(config)
    metadata = check_model(client.health(require_executable=execute and not mock), config,
                           physical=execute and not mock) if client else None
    with ExitStack() as stack:
        events = stack.enter_context((root/'events.jsonl').open('x', buffering=1))
        def event(kind, **values):
            events.write(json.dumps(dict(event=kind, monotonic=time.monotonic(), **values), allow_nan=False)+'\n')
        event('session', execute=execute, mock=mock, observe=observe, task=task, metadata=metadata,
              config=config, control_hz=hz, controller='bounded_absolute_position_no_cartesian_ik')
        if execute and not mock:
            # Same lock as the working Servo/teleop/replay writer.
            from e14.replay_data import writer_lock
            stack.enter_context(writer_lock())
        if mock:
            source = SimulatedArm(config)
        else:
            from e14.piper_robot import ensure_can
            from servo_bridge.sources import PassiveCan
            ensure_can(config['robot']['can_interface'], initialize=initialize_can)
            source = PassiveCan(config['robot']['can_interface'])
        stack.callback(source.close)
        robot = Observer(source, config)
        end = time.monotonic() + config['robot'].get('connect_timeout_s', 3.)
        while True:
            try:
                robot.get_observation()
                break
            except RuntimeError:
                if time.monotonic() >= end:
                    raise
                time.sleep(.02)
        cameras = stack.enter_context(CameraReader(config['cameras'], mock=mock))
        dashboard = None
        if not no_preview:
            from e14.teaching_ui import Dashboard
            dashboard = stack.enter_context(Dashboard(preview_port, page=PAGE))
            print('Inference preview:', dashboard.url, flush=True)
        output = writer = None
        if execute:
            if mock:
                writer = source
            else:
                from piper_sdk import C_PiperInterface_V2
                from servo_bridge.prepare_control import prepare_control
                from servo_bridge.sdk_worker import SDKJointWriter
                cfg = guard_config(config)
                def snapshot():
                    robot.get_observation()
                    return robot.latest[1]
                prepare_control(cfg, snapshot, C_PiperInterface_V2)
                writer = SDKJointWriter(cfg['can_interface'])
                stack.callback(writer.close)
                snap = snapshot()
                select_position_mode(writer, snap, config)
                event('move_j_selected', speed_percent=config['robot']['speed_percent'],
                      initial_joint_target=snap['state'][:6])
                if not snap['gripper_enabled']:
                    width = snap['state'][6]
                    if not config['robot']['gripper_limits_m'][0] <= width <= config['robot']['gripper_limits_m'][1]:
                        raise ValueError('Current jaw opening is outside limits; no enable command sent')
                    writer.gripper(round(width*1e6), config['robot']['gripper_effort'])
                    event('gripper_enable_at_measured_width', width_m=width)
                    end = time.monotonic()+3.
                    while not snapshot()['gripper_enabled']:
                        if time.monotonic() >= end:
                            raise RuntimeError('Gripper enable feedback timed out')
                        time.sleep(.02)
            output = PolicyOutput(config, writer)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='policy-http')
        future = active = None
        failure = None
        count = sent = predictions = 0
        start = due = time.monotonic()
        last_request = -math.inf
        last_cycle = None
        print(f"Mode: {'SIMULATED ' if mock else ''}{'EXECUTE' if execute else 'OBSERVE' if observe else 'DRY-RUN'}; duration={duration}s", flush=True)
        try:
            while time.monotonic()-start < duration:
                now = time.monotonic()
                if now-due > period:
                    raise TimeoutError('Policy control loop missed its deadline; no catch-up commands sent')
                if dashboard and 'quit' in dashboard.drain():
                    event('operator_stop')
                    break
                observation = capture_observation(robot, cameras, config)
                feedback, snapshot = robot.latest
                now = time.monotonic()
                if execute and (not feedback.ready or snapshot['gripper_faults'] or not snapshot['gripper_enabled']):
                    raise RuntimeError('Lost enabled healthy CAN/MOVE_J or gripper feedback')
                if future and future.done():
                    active = future.result()
                    future = None
                    validate_response(active['response'], active['request_id'], hz, execute and not mock)
                    received_metadata = check_model(active['response']['metadata'], config, physical=execute and not mock)
                    if received_metadata['model_id'] != metadata['model_id']:
                        raise ValueError('Model changed during control session')
                    predictions += 1
                    event('prediction', **active)
                if client and future is None and now-last_request >= config['client']['inference_interval_steps']/hz:
                    future = pool.submit(client.predict, observation, task, None, execute and not mock)
                    last_request = now
                action = command = target = velocity = None
                rejected = None
                if active:
                    action = select_action_chunk(active, now, hz, config['client']['max_action_age_ms']/1000,
                                                 config['client']['execution_horizon'])
                    if action is None:
                        raise TimeoutError('No fresh action horizon; position hold requested')
                elif client and now-start > config['client']['timeout_ms']/1000:
                    raise TimeoutError('Initial prediction deadline exceeded')
                if action is not None:
                    try:
                        dt = period if last_cycle is None else now-last_cycle
                        command, velocity, target = follower.step(action, observation['state'], dt=dt)
                        if execute:
                            output.send(command.tolist(), velocity.tolist(), feedback, snapshot)
                            sent += 1
                    except ValueError as exc:
                        rejected = str(exc)
                        event('action_rejected', message=rejected, raw_action=action, measured=observation['state'])
                        if execute:
                            raise
                def values(x):
                    return None if x is None else x.tolist()
                status = dict(mode='execute' if execute else 'observe' if observe else 'dry-run', mock=mock,
                              model_id=metadata['model_id'] if metadata else None,
                              message='正在执行预测；Q / 空格 / Ctrl+C 请求位置保持并退出' if execute else '只读预览；未向机械臂发送指令',
                              state=observation['state'], raw_action=action, bounded_target=values(target),
                              command=values(command), velocity=values(velocity), rejected=rejected,
                              image_age_ms={n:round((now-t)*1000,1) for n,t in observation['image_timestamps'].items()},
                              predictions=predictions, sent_commands=sent)
                event('cycle', **status)
                if dashboard:
                    dashboard.update(status, observation['images'])
                if count % max(1, round(hz)) == 0:
                    print(f"t={now-start:.1f}s predictions={predictions} sent={sent} state={observation['state']}" +
                          (f' rejected={rejected}' if rejected else ''), flush=True)
                count += 1
                last_cycle = now
                due += period
                time.sleep(max(0., due-time.monotonic()))
        except BaseException as exc:
            failure = exc
            event('error', message=str(exc), error_type=type(exc).__name__)
        finally:
            if output and output.holder.moved:
                try:
                    robot.get_observation()
                    output.hold(robot.latest[0])
                    event('measured_joint_hold_requested', gripper='last command retained')
                except Exception as exc:
                    event('hold_unavailable', message=str(exc))
                    if failure is None:
                        failure = exc
            if future:
                future.cancel()
            # Hold first, then let the bounded HTTP request finish. Otherwise an
            # immediate next session can collide with the preceding server job.
            pool.shutdown(wait=True, cancel_futures=True)
            summary = dict(success=failure is None, mode='execute' if execute else 'observe' if observe else 'dry-run',
                           mock=mock, predictions=predictions, sent_commands=sent, cycles=count,
                           elapsed_s=time.monotonic()-start, error=str(failure) if failure else None,
                           metadata=metadata)
            (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            event('end', **summary)
        print('Inference logs:', root, flush=True)
        if failure:
            raise failure
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.local.inference.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute', action='store_true')
    modes.add_argument('--observe', action='store_true')
    parser.add_argument('--mock', action='store_true', help='Synthetic arm/cameras; never opens CAN or SDK')
    parser.add_argument('--initialize-can', action='store_true')
    parser.add_argument('--duration', type=float, default=30.)
    parser.add_argument('--task', default=TASK)
    parser.add_argument('--preview-port', type=int, default=8767)
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Inference stop signal received')
    signal.signal(signal.SIGTERM, interrupted)
    run(load_config(args.config), execute=args.execute, observe=args.observe, mock=args.mock,
        initialize_can=args.initialize_can, duration=args.duration, task=args.task,
        preview_port=args.preview_port, no_preview=args.no_preview)


if __name__ == '__main__':
    main()
