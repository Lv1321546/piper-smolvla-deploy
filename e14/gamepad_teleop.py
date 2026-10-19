"""Gamepad_PiPER integration: controller, live camera, raw demonstrations.

Default is real read-only preview. --mock uses synthetic robot/camera;
--execute explicitly enables real CAN control, subject to existing commissioning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from e14.cameras import CameraReader
from e14.gamepad import PygamePad, DemoPad
from e14.gamepad_control import GamepadControl
from e14.piper_robot import PiperRobot, MockBackend
from e14.teaching_ui import Dashboard, TeachingSession

UPSTREAM_COMMIT = 'aa00c80e100d687a5f4a386b79b83ece06b3a30f'
PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PiPER 手柄遥操作</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{background:#111923;color:#e8eff6;font:16px system-ui;max-width:1160px;margin:24px auto;padding:0 18px}h1{font-size:26px}main{display:grid;grid-template-columns:2fr 1fr;gap:24px}img{width:100%;background:#000;border-radius:12px}button{padding:11px;background:#24435a;color:white;border:1px solid #68869b;border-radius:7px;margin:4px;cursor:pointer}pre,p{white-space:pre-wrap;line-height:1.6}#warning{color:#ffc181}small{color:#adc0d1}#badge{font-size:16px;background:#24435a;padding:7px;border-radius:6px}@media(max-width:780px){main{grid-template-columns:1fr}}</style>
<h1>PiPER 手柄遥操作 <span id="badge">等待设备</span></h1>
<main><section><img src="/stream" alt="相机实时图像"><p id="message"></p><p id="warning"></p>
<small><strong>移动必须持续按住 LB / L1，再拨动摇杆。</strong>先松开所有输入回中；RB 为精细速度。松开 LB 保持位置。Start 请求软件停止；不能代替实体急停。关闭 MuJoCo 后需重启本程序。</small></section>
<section><pre id="pad"></pre><pre id="stats"></pre>
<button data-c="open">O 请求张爪</button><button data-c="close">C 请求闭爪</button>
<button data-c="record">R / X 开始录制</button><button data-c="finish">空格 / X 结束</button>
<button data-c="success">S / Y 成功</button><button data-c="failure">F / Back 失败</button>
<button data-c="unknown">U 暂不标注</button><button data-c="save_home">K 保存参考姿态</button>
<button data-c="quit">Q 退出</button>
<p>末端模式：左摇杆前后 X / 左右 Y，右摇杆上下 Z。默认保持起始姿态；旋转需显式配置。关节模式：两摇杆与方向键控制六关节。</p>
<details><summary>输入与运行详情</summary><pre id="debug"></pre></details></section></main>
<script>const token=__TOKEN__;
async function cmd(command){try{let r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json','X-Teaching-Token':token},body:JSON.stringify({command})});if(!r.ok)throw Error(await r.text())}catch(e){document.querySelector('#message').textContent=e.message}}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>cmd(b.dataset.c));
const keys={o:'open',c:'close',r:'record',' ':'finish',s:'success',f:'failure',u:'unknown',k:'save_home',q:'quit'};
document.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey)return;let c=keys[e.key.toLowerCase()];if(c){e.preventDefault();cmd(c)}});
async function refresh(){try{let s=await(await fetch('/state',{cache:'no-store'})).json();let g=s.gamepad??{};
document.querySelector('#badge').textContent=(s.mock?'模拟设备':s.teleop_execute?'实机执行':'只读预览')+' · '+s.status_label;
document.querySelector('#message').textContent=s.message;
document.querySelector('#warning').textContent=(s.warnings??[]).join('\n');
document.querySelector('#pad').textContent=['手柄：'+(g.name??'等待连接'),'控制模式：'+(s.teleop_mode??''),'LB：'+(s.deadman_active?'允许移动':'未激活'),'控制状态：'+(s.teleop_message??'')].join('\n');
document.querySelector('#stats').textContent=['实际关节：'+(s.joint_angles_deg??[]).map(v=>v.toFixed(2)).join(', ')+' °','实际开口：'+(s.gripper_width_mm??'无新反馈')+' mm','最近记录：'+(s.recorded_frames??0)+' 帧','已标注：'+(s.saved_episodes??0)+' 条',s.episode??''].join('\n');
document.querySelector('#debug').textContent=JSON.stringify(s,null,2)}catch(e){document.querySelector('#message').textContent='程序连接已断开，请检查终端。'}setTimeout(refresh,250)}refresh();</script></html>'''


class GamepadSession(TeachingSession):
    def __init__(self, args, robot, cameras, dashboard, pad, *, clock=time.monotonic, sleep=time.sleep, mirror=None):
        super().__init__(args, robot, cameras, dashboard, clock=clock, sleep=sleep, startup_gripper=False)
        self.pad = pad
        settings = {**robot.config.get('gamepad',{}), 'gripper_speed_m_s':args.gripper_speed}
        self.core = GamepadControl(robot, settings, execute=args.execute or args.mock,
                                   mode=args.mode, clock=clock,
                                   kinematics=mirror.kinematics(robot.settings['joint_limits_deg']) if mirror else None)
        self.pad_state = None
        self.old_buttons = {}
        self.previous_pad_instance = None
        self.last_control_tick = None
        self.mirror = mirror
        self.message = '先松开所有输入回中，再持续按住 LB / L1 并拨动摇杆。不会自动归零或张爪。'

    def handle(self, command):
        if command in {'record','open','close','home'}:
            self._event({'event':'operator_key','command':command,'monotonic':self.clock()})
        if command == 'record':
            if self.status in {'review','finished','recording'}:
                self.message = '先结束或标注上一条录制。'
                return
            if self.core.faulted:
                raise RuntimeError(self.core.message)
            if not self.core.execute:
                raise RuntimeError('只读预览不采集有效演示；使用 --mock 测试或显式 --execute 操作实机。')
            if not self.pad_state or not self.pad_state.connected:
                raise RuntimeError('先连接手柄再开始录制')
            self.robot.check_ready(check_gripper=False)
            self.start_requested = True
            self.message = '开始录制；初始场景和夹爪开口由操作者确认。'
        elif command in {'open','close'}:
            lo,hi = self.robot.settings['gripper_limits_m']
            if not lo <= self.args.close_width < self.args.open_width <= hi:
                raise ValueError('开合目标超出配置行程')
            self.core.request_gripper(command=='open',self.args.open_width,self.args.close_width)
            self.message = '已设置夹爪目标；只有按住 LB 才发送。'
        elif command == 'home':
            self.message = '此手柄入口不自动回零；K 可保存参考姿态，返回请使用已验证的姿态脚本。'
        else:
            super().handle(command)

    def recording_metadata(self):
        data = super().recording_metadata()
        data.update(mode='gamepad', execute=self.core.execute, teleop_mode=self.args.mode,
                    gamepad=asdict(self.pad_state), upstream_commit=UPSTREAM_COMMIT,
                    controller_settings=self.robot.config.get('gamepad',{}),
                    action_note='events.jsonl audits commanded/proposed targets; training labels remain measured next_state')
        return data

    def publish(self, frame_index=None):
        super().publish(frame_index)
        state = dict(self.dashboard.state)
        state['warnings'] = [w for w in state['warnings'] if not w.startswith('H 返回未启用')]
        if not self.core.execute:
            state['warnings'].append('只读预览：不会使能、切模式或发送运动目标。')
        if self.core.faulted:
            state['warnings'].append(self.core.message)
        state.update(gamepad=asdict(self.pad_state) if self.pad_state else {},
                     teleop_mode=self.args.mode, teleop_execute=self.core.execute,
                     deadman_active=self.core.active, teleop_message=self.core.message,
                     proposed_action=self.core.last_proposal)
        self.dashboard.update(state, (self.camera_frame or {}).get('images',{}))

    def poll(self, frame_index=None):
        self.pad_state = self.pad.poll()
        commands = self.dashboard.drain()
        if self.previous_pad_instance != self.pad_state.instance:
            self.old_buttons = dict(self.pad_state.buttons)
            self.previous_pad_instance = self.pad_state.instance
        for key,command in {'x':'record','y':'success','back':'failure'}.items():
            if self.pad_state.buttons[key] and not self.old_buttons.get(key,False):
                if key=='x' and (self.status=='recording' or self.start_requested):
                    command='finish'
                commands.append(command)
        self.old_buttons = dict(self.pad_state.buttons)
        for command in commands:
            try:
                self.handle(command)
            except (ValueError,RuntimeError,OSError,KeyError) as exc:
                self.message=str(exc)
        error = None
        now = self.clock()
        if not self.quit and not self.stop_recording and (self.last_control_tick is None or now-self.last_control_tick>=.8/self.robot.settings['control_hz']):
            try:
                event = self.core.tick(self.pad_state)
                # IK/send takes time. Throttle from completion so returning from
                # the recorder cannot squeeze two sends into one control period.
                self.last_control_tick=self.clock()
                if event:
                    event['input']=asdict(self.pad_state)
                    self._event(event)
                if self.mirror:
                    self.mirror.update(self.robot.get_feedback()['state'])
            except (ValueError,RuntimeError,OSError) as exc:
                error=str(exc)
                self.core.stop(error)
        self.publish(frame_index)
        if self.device_errors:
            self.core.stop('; '.join(self.device_errors))
        if self.core.faulted:
            self.start_requested=False
        if self.status=='recording' and (error or self.core.faulted or self.device_errors):
            raise RuntimeError(error or self.core.message)
        return not (self.quit or self.stop_recording)

    def record(self):
        super().record()
        if self.status=='preparing' and self.message.startswith('本条录制失败'):
            self.core.stop('录制失败，遥操作停止')

    def run(self):
        period=1/self.robot.settings['control_hz']
        until=self.clock()+self.args.run_seconds if self.args.run_seconds else float('inf')
        try:
            while not self.quit and self.clock()<until:
                started=self.clock()
                self.poll()
                if self.start_requested:
                    self.record()
                self.sleep(max(0,period-(self.clock()-started)))
        finally:
            self.core.stop('采集程序退出')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--gamepad-config',type=Path,help='Optional separate JSON with a gamepad section')
    p.add_argument('--mock',action='store_true')
    p.add_argument('--demo',action='store_true',help='Synthetic input; requires --mock')
    p.add_argument('--execute',action='store_true',help='Explicit real control; requires verified stop strategy')
    p.add_argument('--mode',choices=['joint','cartesian'],default='cartesian')
    p.add_argument('--initialize-can',action='store_true')
    p.add_argument('--restart-can',action='store_true')
    p.add_argument('--preview-port',type=int,default=8766)
    p.add_argument('--output',type=Path,default=Path('data/piper_raw_gamepad'))
    p.add_argument('--home-file',type=Path,default=Path('outputs/teaching_home.json'))
    p.add_argument('--task',default='Pick up the black foam block and drop it onto the mouse pad.')
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--duration',type=float,default=30.)
    p.add_argument('--open-width',type=float,default=.06)
    p.add_argument('--close-width',type=float,default=0.)
    p.add_argument('--gripper-speed',type=float,default=.02)
    p.add_argument('--run-seconds',type=float,default=0.,help='Optional dashboard runtime limit')
    p.add_argument('--mujoco',action='store_true',help='Mesh-free kinematic MuJoCo preview, mock only')
    p.add_argument('--mujoco-scene',type=Path,help='Optional kinematic MuJoCo mirror, mock only')
    p.add_argument('--mujoco-headless',action='store_true')
    args=p.parse_args(argv)
    if args.demo and not args.mock:p.error('--demo requires --mock; synthetic input cannot drive a real arm')
    if args.restart_can and not args.initialize_can:p.error('--restart-can requires --initialize-can')
    if args.episodes<1 or not math.isfinite(args.run_seconds) or args.run_seconds<0:p.error('Positive episode count and finite nonnegative runtime required')
    if (args.mujoco or args.mujoco_scene) and not args.mock:p.error('MuJoCo mirror is a mock-only preview')
    if args.mujoco_headless and not (args.mujoco or args.mujoco_scene):p.error('--mujoco-headless requires --mujoco or --mujoco-scene')
    if not math.isfinite(args.duration) or args.duration<=0 or not args.task.strip():p.error('Need a task and positive finite duration')
    if not 1<=args.preview_port<=65535:p.error('Invalid preview port')
    config=json.loads(args.config.read_text())
    if args.gamepad_config:
        extra=json.loads(args.gamepad_config.read_text())['gamepad']
        config['gamepad']={**config.get('gamepad',{}),**extra}
    robot=PiperRobot(config,mock=args.mock)
    lo,hi=robot.settings['gripper_limits_m']
    if not lo<=args.close_width<args.open_width<=hi:p.error('Gripper goals outside configured limits')
    # Validate control settings and the URDF before touching CAN or enabling motors.
    GamepadControl(robot,{**config.get('gamepad',{}),'gripper_speed_m_s':args.gripper_speed},mode=args.mode)
    cameras=CameraReader(config['cameras'],mock=args.mock)
    pad=DemoPad() if args.demo else PygamePad(config.get('gamepad'))
    mirror=session=None
    try:
        first=pad.poll()
        if args.execute and not args.mock and not first.connected:
            raise RuntimeError('没有检测到可用手柄；未连接或使能机械臂。先运行 python -m e14.gamepad --monitor')
        if args.mujoco or args.mujoco_scene:
            from e14.gamepad_sim import MujocoMirror
            from e14.install_mujoco_model import MODEL_ROOT
            scene=args.mujoco_scene
            if scene is None and (MODEL_ROOT/'scene.xml').is_file():scene=MODEL_ROOT/'scene.xml'
            print('MuJoCo model:',scene or 'built-in simplified preview',flush=True)
            mirror=MujocoMirror(scene,headless=args.mujoco_headless)
        if args.mock:
            backend=MockBackend()
            backend.state=[0.,75.,-75.,0.,45.,0.,.06]
            robot._backend=backend
        robot.connect(initialize_can=args.initialize_can,restart_can=args.restart_can)
        if args.execute or args.mock:robot.prepare_control(joint_only=True)
        cameras.connect()
        with Dashboard(args.preview_port,page=PAGE) as dashboard:
            print(f'Gamepad dashboard: {dashboard.url}\nMode: {args.mode}; execute={args.execute or args.mock}; mock={args.mock}',flush=True)
            session=GamepadSession(args,robot,cameras,dashboard,pad,mirror=mirror)
            session.run()
    finally:
        if session is None and robot._prepared and (args.execute or args.mock):
            robot.stop()
        if mirror:mirror.close()
        pad.close();cameras.close();robot.disconnect()


if __name__=='__main__':
    main()
