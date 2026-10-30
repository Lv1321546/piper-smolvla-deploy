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
from e14.collect_data import collection_settings
from e14.piper_robot import PiperRobot, ensure_can
from e14.teaching_ui import Dashboard, TeachingSession, PAGE as TEACHING_PAGE

ROOT = Path(__file__).resolve().parents[1]
PAGE = TEACHING_PAGE.replace('PiPER 抓取示教采集', 'PiPER Servo 手柄采集')
PAGE = PAGE.replace('<button data-c="open">O 张开</button><button data-c="close">C 闭合</button>', '<span>夹爪开合尚未接入；当前只记录实际开口。</span>')
PAGE = PAGE.replace('<button data-c="home">H 返回并张爪</button>', '')
PAGE = PAGE.replace('CAN 模式可用 O/C 测试夹爪，此时不要手拖关节。手拖抓取采集需要固件支持示教期间的软件夹爪；录制结束保持夹爪当前状态。', '持续按 LB + 摇杆控制 XYZ，RB 精细速度；松开 LB 保持。手柄 X 开始/结束，Y 成功，Back 失败。录制开始不自动张爪或回零。')


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

    def start(self):
        from servo_bridge.core import load_config
        cfg = load_config(self.args.servo_config)
        if cfg['can_interface'] != self.config['robot']['can_interface']:
            raise ValueError('Servo and collection CAN interfaces differ')
        self.path.mkdir(parents=True, exist_ok=False)
        try:
            if self.args.execute:
                from servo_bridge.sdk_worker import validate_arm
                validate_arm(json.loads(self.args.servo_arm_file.read_text()), cfg['can_interface'])
                if Path(self.socket_path).exists():
                    raise RuntimeError('SDK socket already exists; stop the other writer first')
                command = [sys.executable,'-u','-m','servo_bridge.sdk_worker','--execute',
                           '--arm-file',str(self.args.servo_arm_file.resolve()),'--parent-pid',str(os.getpid())]
                if self.args.servo_config: command += ['--config',str(self.args.servo_config.resolve())]
                self.spawn(command,'worker.log')
                deadline = time.monotonic()+5
                while not Path(self.socket_path).exists():
                    self.alive()
                    if time.monotonic()>deadline: raise RuntimeError('SDK worker startup timed out')
                    time.sleep(.05)
            command = ['bash','servo_bridge/run.sh','--source','can','--input','gamepad',
                       '--gamepad-config',str(self.args.gamepad_config.resolve()),
                       '--execution-mode','integrated-velocity','--duration',str(self.args.run_seconds),
                       '--output',str((self.path/'trace').resolve()),'--parent-pid',str(os.getpid())]
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
                if row['event']=='servo_status' and row['value'] in (2,4,5):
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
        self.message='Servo 手柄已连接。LB + 摇杆移动；R 开始录制；不自动张爪或返回。'

    def handle(self,command):
        if command in {'open','close','home'}:
            self.message='此 Servo 入口尚未接入夹爪/自动返回，不会发送这些指令。';return
        if command=='record':
            if not (self.args.execute or self.args.mock):
                raise RuntimeError('只读预览不可录制运动示教；实机执行需 --execute 和现场授权文件')
            if self.fault:raise RuntimeError(self.fault)
            if self.status in {'review','finished','recording'}:return
            self.robot.get_feedback()
            self.start_requested=True
            self._event({'event':'operator_key','command':command,'monotonic':self.clock()})
            return
        super().handle(command)
        if command=='save_home':self.message='参考姿态和图片已保存；本入口不自动返回。'
        if command=='quit':self.bridge.close()

    def recording_metadata(self):
        return {'mode':'moveit_servo_gamepad','execution_mode':'integrated-velocity',
                'teleop_trace':str(self.bridge.trace) if self.bridge.trace else None,
                'execute':self.args.execute,'gripper_control':False,
                'operator_events':'events.jsonl','label_semantics':'next_state'}

    def publish(self,frame_index=None):
        super().publish(frame_index)
        state=dict(self.dashboard.state)
        state['warnings']=[x for x in state['warnings'] if not x.startswith('H 返回未启用')]
        state['gripper_status']='只读开口；尚未接入控制'
        state['servo']=self.bridge.latest
        state['teleop_execute']=bool(self.args.execute)
        if self.status=='recording':
            state['message']='正在录制 Servo 手柄示教；空格/X 结束，S/Y 成功，F/Back 失败。夹爪控制未接入。'
        if self.fault:state['warnings'].append(self.fault)
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
    if args.execute and args.servo_arm_file is None:raise ValueError('--execute requires --servo-arm-file')
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
        bridge.start()
        with Dashboard(args.preview_port,page=PAGE) as dashboard:
            print('Servo collection dashboard: '+dashboard.url,flush=True)
            ServoSession(args,robot,cameras,dashboard,bridge).run()
    finally:
        if bridge:bridge.close()
        cameras.close();robot.disconnect()
