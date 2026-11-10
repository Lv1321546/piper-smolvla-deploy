"""Live PiPER inference: read-only by default, explicit --execute for SDK output.

Uses passive CAN feedback, both trained RGB views, asynchronous HTTP inference,
time-aligned action chunks and a rate-limited absolute-position follower.
No ROS/Cartesian inverse kinematics, arming token, reset or automatic homing.
"""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import signal
import time

from e14.cameras import CameraReader
from e14.inference_client import retryable_prediction_error
from e14.staggered_inference import StaggeredInference
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
    retry_period = finite_number(config['client'].get('retry_interval_ms', 250),
                                 'client.retry_interval_ms', positive=True) / 1000
    root = Path(config['storage']['log_root']) / ('policy-' + time.strftime('%Y%m%d-%H%M%S') + '-' + str(time.time_ns()%1000000))
    root.mkdir(parents=True, exist_ok=False)
    client = None if observe else StaggeredInference(config)
    metadata = client.health(require_executable=execute and not mock) if client else None
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
        active = None
        failure = None
        count = sent = predictions = 0
        start = due = time.monotonic()
        waiting, waiting_since, waiting_reason = True, start, '等待首次有效预测'
        # Startup has no preceding motion. Later waits require source samples
        # captured AFTER the measured-position hold, not just late arrivals.
        fresh_after = -math.inf
        waits = resumes = retries = discarded = 0
        newest_capture = -math.inf
        last_cycle = None
        def begin_wait(reason, feedback):
            nonlocal active, waiting, waiting_since, waiting_reason, fresh_after, waits
            active = None
            waiting_reason = reason
            if waiting:
                return
            if output and output.holder.moved:
                output.hold(feedback)  # Failure to hold is terminal, never retried.
                event('measured_joint_hold_requested', gripper='last command retained', reason=reason)
            waiting = True
            waiting_since = fresh_after = time.monotonic()
            waits += 1
            event('prediction_wait_started', reason=reason)
            print('预测暂不可用，位置保持并等待新预测：'+reason, flush=True)

        def usable(result, now):
            try:
                return select_action_chunk(result, now, hz,
                    config['client']['max_action_age_ms']/1000, config['client']['execution_horizon'])
            except TimeoutError:
                return None

        event('prediction_wait_started', reason=waiting_reason, initial=True)
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
                # Validate every completion, including older results and errors.
                # Then accept the newest usable source before expiring the old plan.
                candidates = []
                for replica_index, completed in client.completed() if client else []:
                    try:
                        candidate = completed.result()
                    except Exception as exc:
                        if not retryable_prediction_error(exc):
                            raise
                        retries += 1
                        client.defer(replica_index, time.monotonic() + retry_period)
                        event('prediction_retry', message=str(exc), error_type=type(exc).__name__,
                              replica_index=replica_index, replica_url=client.urls[replica_index])
                        if active is None:
                            waiting_reason = '预测请求超时或连接暂不可用，正在重试'
                    else:
                        # Protocol/model errors remain fatal, even for a result
                        # that is too old to move the robot.
                        validate_response(candidate['response'], candidate['request_id'], hz, execute and not mock)
                        received_metadata = check_model(candidate['response']['metadata'], config, physical=execute and not mock)
                        if any(received_metadata.get(key) != metadata.get(key)
                               for key in ('model_id', 'model_sha256', 'action_chunk_steps')):
                            raise ValueError('Model changed during control session')
                        predictions += 1
                        event('prediction', **candidate)
                        candidates.append(candidate)
                for candidate in sorted(candidates, key=lambda item: item['captured_monotonic'], reverse=True):
                    now = time.monotonic()
                    captured = candidate['captured_monotonic']
                    reason = ('source predates hold' if captured < fresh_after else
                              'superseded by newer observation' if captured <= newest_capture else
                              'expired before control tick' if usable(candidate, now) is None else None)
                    if reason:
                        discarded += 1
                        event('prediction_discarded', request_id=candidate['request_id'], reason=reason,
                              replica_index=candidate['replica_index'])
                    else:
                        if waiting:
                            if output:
                                output.resume(follower, observation['state'])
                            else:
                                follower.reset(observation['state'])
                            event('prediction_resumed', request_id=candidate['request_id'],
                                  waited_s=now-waiting_since, captured_monotonic=captured,
                                  fresh_after=None if fresh_after == -math.inf else fresh_after)
                            resumes += 1
                            waiting = False
                            print('已收到新鲜预测，从当前状态继续。', flush=True)
                        active = candidate
                        newest_capture = captured
                action = command = target = velocity = None
                rejected = None
                if active:
                    now = time.monotonic()
                    action = usable(active, now)
                    if action is None:
                        begin_wait('动作已过期或可执行时域已耗尽', feedback)
                if client and observation['captured_monotonic'] >= fresh_after:
                    replica_index = client.submit(observation, task, now, execute and not mock)
                    if replica_index is not None:
                        event('prediction_requested', captured_monotonic=observation['captured_monotonic'],
                              waiting=waiting, replica_index=replica_index,
                              replica_url=client.urls[replica_index])
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
                              message=(('位置保持，等待新预测：' if execute else '只读等待新预测：')+waiting_reason
                                       if client and waiting else
                                       '正在执行预测；Q / 空格 / Ctrl+C 请求位置保持并退出' if execute else '只读预览；未向机械臂发送指令'),
                              waiting_for_prediction=bool(client and waiting),
                              waiting_s=round(now-waiting_since, 3) if client and waiting else 0.,
                              state=observation['state'], raw_action=action, bounded_target=values(target),
                              command=values(command), velocity=values(velocity), rejected=rejected,
                              action_source_monotonic=active['captured_monotonic'] if action is not None else None,
                              action_age_ms=(now-active['captured_monotonic'])*1000 if action is not None else None,
                              active_replica=active['replica_index'] if action is not None else None,
                              image_age_ms={n:round((now-t)*1000,1) for n,t in observation['image_timestamps'].items()},
                              predictions=predictions, sent_commands=sent)
                event('cycle', **status)
                if dashboard:
                    dashboard.update(status, observation['images'])
                if count % max(1, round(hz)) == 0:
                    print(f"t={now-start:.1f}s predictions={predictions} sent={sent} waiting={bool(client and waiting)} state={observation['state']}" +
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
            # Hold first, then let the bounded HTTP request finish. Otherwise an
            # immediate next session can collide with the preceding server job.
            if client:
                client.close()
            summary = dict(success=failure is None, mode='execute' if execute else 'observe' if observe else 'dry-run',
                           mock=mock, predictions=predictions, sent_commands=sent, cycles=count,
                           elapsed_s=time.monotonic()-start, error=str(failure) if failure else None,
                           prediction_waits=waits, prediction_resumes=resumes, prediction_retries=retries,
                           discarded_predictions=discarded, waiting_at_end=bool(client and waiting),
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
