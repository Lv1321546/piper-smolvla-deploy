"""Servo teleoperation + raw collection. Only the child SDK worker writes CAN.

Mock exercises recording/UI without ROS, SDK, CAN, or physical gamepad.
Real collection uses the commissioned integrated-velocity Servo route.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from e14.cameras import CameraReader
from servo_bridge.ui_commands import ControlMailbox
from e14.collect_data import collection_settings
from e14.piper_robot import PiperRobot, ensure_can
from e14.teaching_ui import Dashboard, TeachingSession, PAGE as TEACHING_PAGE

ROOT = Path(__file__).resolve().parents[1]
PAGE = TEACHING_PAGE.replace('PiPER 抓取示教采集', 'PiPER Servo 手柄采集')
PAGE = PAGE.replace('<button data-c="open">O 张开</button><button data-c="close">C 闭合</button>', '<span>夹爪：按住 LB + B/RT 张爪，A/LT 闭爪；实际开口持续记录。</span>')
PAGE = PAGE.replace('<button data-c="home">H 返回并张爪</button>', '')
PAGE = PAGE.replace("s.mock?'模拟设备':'真实设备',", "s.mock?'模拟设备':'真实设备',s.cartesian_mode_label??'',s.cartesian_planner?.message??'',")
PAGE = PAGE.replace('CAN 模式可用 O/C 测试夹爪，此时不要手拖关节。手拖抓取采集需要固件支持示教期间的软件夹爪；录制结束保持夹爪当前状态。', '持续按 LB + 摇杆控制 XYZ，同时按住 RB 为精细档；仅按 RB 不移动，松开 LB 保持。夹爪用 B/RT 张爪、A/LT 闭爪（也要按住 LB）。手柄 X 开始/结束，Y 成功，Back 失败。录制开始不自动张爪或回零。')

LEGACY_PAGE = PAGE
PAGE = PAGE.replace('K 记录初始姿态', 'K 保存参考快照')
PAGE = PAGE.replace('<button data-c="save_home">K 保存参考快照</button>',
    '<button data-c="save_home">K 保存参考快照</button><button data-c="home">H 返回并张爪</button>')
PAGE = PAGE.replace("||(c==='record'&&['review','finished'].includes(status))",
    "||(c==='record'&&['review','finished'].includes(status))||(s.reset_active&&['home','save_home','record'].includes(c))")
PAGE = PAGE.replace("document.querySelector('#stats').textContent=", "document.querySelector('[data-c=finish]').textContent=s.reset_active?'空格 取消复位':'空格 结束';document.querySelector('#stats').textContent=")
PAGE = PAGE.replace('夹爪：按住 LB + B/RT 张爪，A/LT 闭爪；实际开口持续记录。',
    '夹爪：LB+B/RT 张爪，LB+LT 闭爪。长按 A 返回初始姿态，松开 A 停止。')
PAGE = PAGE.replace('夹爪用 B/RT 张爪、A/LT 闭爪（也要按住 LB）。',
    '夹爪用 LB+B/RT 张爪、LB+LT 闭爪。')
PAGE = PAGE.replace('手柄 X 开始/结束，Y 成功，Back 失败。',
    '先将摇杆和扳机回中，长按 A 返回（无需 LB），松开 A 停止。录制外按 H 自动返回并张爪，空格取消复位。手柄 X 开始/结束，Y 成功，Back 失败。')


class PassiveRobot:
    """Collection-only facade: no movement, enable, mode or gripper methods."""
    def __init__(self, config):
        self.config, self.settings, self.mock = config, config['robot'], False
        self.source = None

    def connect(self):
        from servo_bridge.sources import PassiveCan
        self.source = PassiveCan(self.settings['can_interface'])
        end = time.monotonic()+3
        while True:
            try:
                self.get_feedback()
                return
            except RuntimeError:
                if time.monotonic() >= end:
                    raise
                time.sleep(.02)

    def get_feedback(self):
        f, _, snap = self.source.sample(time.monotonic(), .05)
        if time.monotonic()-f.received > self.settings['feedback_max_age_ms']/1000:
            raise RuntimeError('Stale CAN feedback')
        if not f.ready:
            raise RuntimeError('采集需要健康且已使能的 CAN/MOVE_J；不会自动切模式或使能')
        return snap

    def get_observation(self):
        snap = self.get_feedback()
        return {'state':snap['state'], 'feedback_monotonic':snap['feedback_monotonic'],
                'captured_monotonic':time.monotonic()}

    def disconnect(self):
        if self.source:
            self.source.close()


class ServoProcess:
    """Own child processes and stream their audit log; never call motion APIs."""
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.path = args.output / '_servo_sessions' / uuid.uuid4().hex
        self.trace = self.path/'trace/events.jsonl'
        self.processes, self.handles = [], []
        self.reader = None
        self.latest = None
        self.stopped = False
        self.partial = ''
        self.socket_path = '/tmp/piper-servo-sdk.sock'
        self.initial_state = None
        self.profile = {}
        self.control = ControlMailbox(self.path/'ui_command.json')

    def start(self):
        from servo_bridge.core import load_config
        cfg = load_config(self.args.servo_config)
        self.profile = cfg
        if cfg['can_interface'] != self.config['robot']['can_interface']:
            raise ValueError('Servo and collection CAN interfaces differ')
        self.path.mkdir(parents=True, exist_ok=False)
        if self.initial_state is not None:
            (self.path/'initial_state.json').write_text(
                json.dumps(self.initial_state, ensure_ascii=False, indent=2)+'\n')
        try:
            if self.args.execute:
                if Path(self.socket_path).exists():
                    raise RuntimeError('SDK socket already exists; stop the other writer first')
                command = [sys.executable,'-u','-m','servo_bridge.sdk_worker','--execute',
                           '--parent-pid',str(os.getpid())]
                if self.args.servo_config: command += ['--config',str(self.args.servo_config.resolve())]
                self.spawn(command,'worker.log')
                deadline = time.monotonic()+5
                while not Path(self.socket_path).exists():
                    self.alive()
                    if time.monotonic()>deadline: raise RuntimeError('SDK worker startup timed out')
                    time.sleep(.05)
            controller = getattr(self.args, 'servo_controller', 'cartesian')
            runner = 'servo_bridge/run.sh' if controller == 'legacy' else 'servo_bridge/run_cartesian.sh'
            command = ['bash',runner,'--source','can','--input','gamepad',
                       '--gamepad-config',str(self.args.gamepad_config.resolve()),
                       '--execution-mode','integrated-velocity','--duration',str(self.args.run_seconds),
                       '--output',str((self.path/'trace').resolve()),'--parent-pid',str(os.getpid())]
            if controller == 'cartesian':
                command += ['--orientation-mode', getattr(self.args, 'orientation_mode', 'position')]
                if self.initial_state is not None:
                    command += ['--initial-state', str((self.path/'initial_state.json').resolve()),
                                '--control-file', str(self.control.path.resolve())]
            if self.args.servo_config: command += ['--config',str(self.args.servo_config.resolve())]
            if self.args.execute: command += ['--output-socket',self.socket_path]
            self.spawn(command,'runtime.log')
            deadline = time.monotonic()+20
            while self.latest is None:
                self.poll()
                if time.monotonic()>deadline: raise RuntimeError('Servo startup timed out; inspect '+str(self.path))
                time.sleep(.05)
        except BaseException:
            self.close()
            raise

    def send_control(self, command):
        self.alive()
        if self.initial_state is None or getattr(self.args, 'servo_controller', 'cartesian') != 'cartesian':
            raise RuntimeError('此会话没有可用的笛卡尔初始状态')
        return self.control.send(command)

    def spawn(self, command, name):
        f=(self.path/name).open('w'); self.handles.append(f)
        self.processes.append(subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True))

    def alive(self):
        if self.stopped or any(p.poll() is not None for p in self.processes):
            raise RuntimeError('Servo/SDK 已停止；本条数据不可作为完整示教。日志：'+str(self.path))

    def poll(self):
        self.alive()
        if self.reader is None and self.trace.exists(): self.reader=self.trace.open()
        events=[]
        if self.reader:
            chunk=self.partial+self.reader.read()
            lines=chunk.split('\n');self.partial=lines.pop()
            for line in lines:
                if not line: continue
                row=json.loads(line);events.append(row)
                if row['event']=='input': self.latest=row
                if row['event']=='relay_latched_closed': raise RuntimeError(row['reason'])
                if (row['event']=='servo_status' and row['value'] in (2,3,4,5)
                        and not (row['value'] in (2,5) and row.get('recoverable') is True)):
                    raise RuntimeError('Servo halt: '+row['label'])
        if self.latest and time.monotonic()-self.latest['monotonic']>.5:
            raise RuntimeError('Servo telemetry stale')
        return events

    def close(self):
        if self.stopped:return
        self.stopped=True
        # Stop the producer first; ask this session's worker for measured hold.
        if len(self.processes) > (1 if self.args.execute else 0):
            p=self.processes[-1]
            if p.poll() is None: os.killpg(p.pid,signal.SIGINT)
        if self.args.execute and self.processes and self.processes[0].poll() is None:
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as s:
                    s.sendto(json.dumps({'version':1,'monotonic':time.monotonic(),
                        'stop_requested':True,'reason':'collection stopped'}).encode(),self.socket_path)
            except OSError: pass
        for p in reversed(self.processes):
            try:p.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGINT)
                try:p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=3)
        if self.reader:self.reader.close()
        for f in self.handles:f.close()


class MockServo:
    def __init__(self, robot, clock=time.monotonic):
        self.robot,self.clock=robot,clock;self.started=clock();self.stopped=False
        self.trace=None;self.latest=None
    def start(self):pass
    def poll(self):
        if self.stopped:raise RuntimeError('Mock Servo stopped')
        now=self.clock()
        self.robot._backend.state[0]=2*math.sin((now-self.started)*.5)
        self.robot._backend.refresh()
        self.latest={'event':'input','monotonic':now,'active':True,'status':0,'gamepad':{'buttons':{}}}
        return [self.latest]
    def close(self):self.stopped=True


class ServoSession(TeachingSession):
    def __init__(self,args,robot,cameras,dashboard,bridge,**kwargs):
        super().__init__(args,robot,cameras,dashboard,startup_gripper=False,**kwargs)
        self.bridge=bridge;self.old_buttons=None;self.fault=None
        self.reset_pending_seq = 0
        self.message='Servo 手柄已连接。LB + 摇杆移动；B/RT 张爪、A/LT 闭爪（按住 LB）；R 开始录制；不自动张爪或返回。'

    def reset_in_progress(self):
        if self.fault:
            return False
        latest = self.bridge.latest or {}
        pending = self.reset_pending_seq > latest.get('ui_command_seq', 0)
        phase = latest.get('planner', {}).get('reset_state')
        return pending or phase in ('pending', 'returning', 'opening')

    def handle(self,command):
        if command == 'home' and getattr(self.args, 'servo_controller', 'cartesian') == 'cartesian':
            if self.status == 'recording' or self.start_requested:
                self.message = '请先结束录制并标注结果，再按 H 复位。'
                return
            if not self.args.execute or self.args.mock:
                self.message = '自动复位需要真实执行会话。'
                return
            if self.reset_in_progress():
                return
            self.robot.get_feedback()
            self.reset_pending_seq = self.bridge.send_control('reset')
            self.start_requested = False
            self.message = '已请求返回初始位置并张爪，空格取消复位。'
            return
        if self.reset_in_progress() and command in ('finish', 'record', 'save_home'):
            if command == 'finish':
                self.reset_pending_seq = self.bridge.send_control('cancel_reset')
                self.message = '已请求取消复位并保持当前位置。'
            else:
                self.message = '复位进行中，等待完成或按空格取消后再录制。'
            return
        if command in {'open','close','home'}:
            if command == 'home':
                self.message = '此控制模式不提供自动复位。'
            else:
                self.message = '夹爪请用手柄 LB+B/RT 张开、LB+LT 闭合。'
            return
        if command=='record':
            if not (self.args.execute or self.args.mock):
                raise RuntimeError('只读预览不可录制运动示教；实机执行需 --execute')
            if self.fault:raise RuntimeError(self.fault)
            if self.status in {'review','finished','recording'}:return
            self.robot.get_feedback()
            self.start_requested=True
            self._event({'event':'operator_key','command':command,'monotonic':self.clock()})
            return
        super().handle(command)
        if command == 'save_home':
            self.message = '参考姿态和图片已保存。'
            if getattr(self.args, 'servo_controller', 'cartesian') == 'cartesian':
                self.message += '长按 A 返回固定初始姿态。'
        if command=='quit':self.bridge.close()

    def recording_metadata(self):
        return {'mode':'moveit_servo_gamepad','execution_mode':'integrated-velocity',
                'controller':getattr(self.args, 'servo_controller', 'cartesian'),
                'orientation_mode':getattr(self.args, 'orientation_mode', 'position'),
                'initial_state':getattr(self.bridge, 'initial_state', None),
                'teleop_trace':str(self.bridge.trace) if self.bridge.trace else None,
                'execute':self.args.execute,'gripper_control':bool(self.args.execute or self.args.mock),
                'operator_events':'events.jsonl','label_semantics':'next_state'}

    def publish(self,frame_index=None):
        super().publish(frame_index)
        state=dict(self.dashboard.state)
        state['warnings']=[x for x in state['warnings'] if not x.startswith('H 返回未启用')]
        has_home = getattr(self.args, 'servo_controller', 'cartesian') == 'cartesian'
        state['gripper_status'] = ('LB+B/RT 张爪、LB+LT 闭爪；长按 A 返回，松开停止'
                                   if has_home else '手柄 B/RT 张爪、A/LT 闭爪（按住 LB）')
        state['servo']=self.bridge.latest
        state['teleop_execute']=bool(self.args.execute)
        planner = (self.bridge.latest or {}).get('planner', {})
        state['cartesian_planner'] = planner
        mode = getattr(self.args, 'orientation_mode', 'position')
        orientation_limit = math.degrees(getattr(self.bridge, 'profile', {}).get(
            'cartesian_orientation_drift_rad', math.radians(15)))
        state['initial_state'] = getattr(self.bridge, 'initial_state', None)
        state['cartesian_mode_label'] = ('XYZ 旧版固定姿态' if getattr(self.args, 'servo_controller', 'cartesian') == 'legacy'
                                        else 'XYZ v2：保持朝向（2°余量）' if mode == 'hold'
                                        else 'XYZ 位置优先（本段朝向变化 ≤%.0f°）' % orientation_limit)
        if has_home:
            state['message'] = state.get('message', '').replace('A/LT', 'LT')
        if planner.get('message') and self.status != 'recording':
            state['message'] = planner['message']
        if self.status=='recording':
            close_button = 'LT' if has_home else 'A/LT'
            state['message'] = ('正在录制 Servo 手柄示教；空格/X 结束，S/Y 成功，F/Back 失败。'
                                '夹爪 B/RT 张、%s 闭（按住 LB）。' % close_button)
            if has_home:
                state['message'] += '长按 A 返回初始位置，松开停止。'
        if (self.bridge.latest or {}).get('status') in (2,5):
            state['warnings'].append('关节限位/奇异保护保持：demo 仍运行，按住 LB 反向退离。')
        elif planner.get('geometry_pause'):
            state['warnings'].append(planner.get('message', '几何边界保持，请反向退离。'))
        state['reset_active'] = self.reset_in_progress()
        if state['reset_active']:
            phase = planner.get('reset_state')
            state['status_label'] = '张爪中' if phase == 'opening' else '复位返回中'
            state['message'] = planner.get('message', '复位请求已收到。')+' 空格取消复位。'
        elif (self.status != 'recording' and not self.start_requested
              and not (self.bridge.latest or {}).get('active') and planner.get('reset_message')):
            state['message'] = planner['reset_message']
        if self.fault:
            state['warnings'].append(self.fault)
            state['status'] = 'finished'
            state['status_label'] = '控制已停止'
            state['reset_active'] = False
            state['message'] = '采集/控制已停止：'+self.fault
        if not (self.args.execute or self.args.mock):state['warnings'].append('只读预览，运动输出关闭')
        self.dashboard.update(state,(self.camera_frame or {}).get('images',{}))

    def poll(self,frame_index=None):
        try:
            events=self.bridge.poll()
            commands=self.dashboard.drain()
            for event in events:
                self._event({'event':'servo_audit','payload':event})
                if event['event']=='input':
                    buttons=event.get('gamepad',{}).get('buttons',{})
                    if self.old_buttons is None:
                        self.old_buttons=dict(buttons)
                    for key,cmd in {'x':'record','y':'success','back':'failure'}.items():
                        if buttons.get(key) and not self.old_buttons.get(key):
                            commands.append('finish' if key=='x' and (self.status=='recording' or self.start_requested) else cmd)
                    self.old_buttons=dict(buttons)
            for cmd in commands:
                try:self.handle(cmd)
                except (ValueError,RuntimeError,OSError) as exc:self.message=str(exc)
            self.publish(frame_index)
            if self.device_errors:raise RuntimeError('; '.join(self.device_errors))
        except (ValueError,RuntimeError,OSError) as exc:
            self.fault=str(exc);self.start_requested=False;self.bridge.close()
            self.message='采集/控制已停止：'+str(exc)
            if self.status=='recording':raise
            self.quit=True
            self.publish(frame_index)
        return not (self.quit or self.stop_recording)

    def record(self):
        super().record()
        if self.status=='preparing' and self.message.startswith('本条录制失败'):
            self.fault=self.message;self.bridge.close();self.quit=True
            self.message += ' Servo 已退出；处理问题后重新启动采集命令。'

    def run(self):
        end=self.clock()+self.args.run_seconds;period=1/self.robot.settings['control_hz']
        try:
            while not self.quit and self.clock()<end:
                start=self.clock();self.poll()
                if self.start_requested:self.record()
                self.sleep(max(0,period-(self.clock()-start)))
        finally:self.bridge.close()


def run_servo_collection(args):
    config=json.loads(args.config.read_text());collection_settings(config)
    if not math.isfinite(args.duration) or args.duration<=0 or not args.task.strip():
        raise ValueError('A task and positive recording duration are required')
    if not 0<=args.preview_port<=65535:raise ValueError('Invalid preview port')
    if args.interactive:raise ValueError('--teleop servo replaces --interactive')
    if args.mock and args.execute:raise ValueError('--mock cannot be combined with --execute')
    if not math.isfinite(args.run_seconds) or args.run_seconds<=0:raise ValueError('Positive --run-seconds required')
    if args.mock and (args.initialize_can or args.restart_can):raise ValueError('Mock must not configure real CAN')
    robot=PiperRobot(config,mock=True) if args.mock else PassiveRobot(config)
    cameras=CameraReader(config['cameras'],mock=args.mock);bridge=None
    try:
        if args.mock:
            robot.connect();robot.prepare_control(joint_only=True)
        else:
            ensure_can(config['robot']['can_interface'],initialize=args.initialize_can,restart=args.restart_can)
            robot.connect()
        # Camera failure must occur before any worker/Servo output is started.
        cameras.connect();cameras.read()
        bridge=MockServo(robot) if args.mock else ServoProcess(args,config)
        if not args.mock:
            from servo_bridge.core import load_config
            from servo_bridge.start_reference import reference_from_snapshot, load_fixed_reference
            profile = load_config(args.servo_config)
            observed = reference_from_snapshot(robot.get_feedback(), profile, config, now=time.monotonic())
            fixed_home = getattr(args, 'fixed_home', None)
            bridge.initial_state = load_fixed_reference(fixed_home, profile) if fixed_home else observed
        bridge.start()
        page = PAGE if getattr(args, 'servo_controller', 'cartesian') == 'cartesian' else LEGACY_PAGE
        with Dashboard(args.preview_port,page=page) as dashboard:
            print('Servo collection dashboard: '+dashboard.url,flush=True)
            ServoSession(args,robot,cameras,dashboard,bridge).run()
    finally:
        if bridge:bridge.close()
        cameras.close();robot.disconnect()
