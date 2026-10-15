"""Local, single-camera teaching dashboard. HTTP threads never command hardware.

O/C command only the gripper. K records a reference pose; H explicitly returns
the six arm joints outside recording using the existing motion checks. Neither
startup nor starting an episode changes the arm's control mode.
"""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time

from PIL import Image

from e14.collect_data import record_episode, write_json
from e14.piper_robot import PiperRobot
from e14.cameras import CameraReader


COMMANDS = {"open", "close", "record", "finish", "success", "failure", "unknown", "save_home", "home", "quit"}

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>PiPER 采集</title>
<style>body{background:#10151d;color:#e8edf5;font:16px system-ui;margin:24px auto;max-width:1120px;padding:0 16px}h1{font-size:24px}main{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:20px}img{width:100%;background:#000;border-radius:10px}button{background:#24374c;color:white;border:1px solid #526981;border-radius:7px;padding:12px;margin:4px;cursor:pointer}button:disabled{opacity:.4;cursor:default}#message{white-space:pre-wrap;color:#ffd88a;min-height:64px}pre{white-space:pre-wrap;font:14px monospace}#badge{padding:8px;border-radius:6px;background:#24374c}small{color:#a8b8c8}@media(max-width:760px){main{grid-template-columns:1fr}}</style>
<h1>PiPER 抓取示教采集 <span id="badge">连接中</span></h1>
<main><section><img src="/stream" alt="腕上相机实时画面"><p id="message"></p><small>按键仅在此页面获得焦点时有效。Q 退出软件不会自动松爪；不是硬件急停。</small></section>
<section><div><button data-c="open">O 张开</button><button data-c="close">C 闭合</button></div>
<div><button data-c="record">R 开始录制</button><button data-c="finish">空格 结束</button></div>
<div><button data-c="success">S 成功并保存</button><button data-c="failure">F 失败并保存</button><button data-c="unknown">U 暂不标注</button></div>
<div><button data-c="save_home">K 记录初始姿态</button><button data-c="home">H 返回并张爪</button></div>
<button data-c="quit">Q 退出</button><pre id="stats"></pre><details><summary>运行详情</summary><pre id="debug"></pre></details>
<small>先手动调整到能看清方块和鼠标垫的位置。录制前自动张爪并检查真实反馈；录制结束保持夹爪当前状态。</small></section></main>
<script>const token=__TOKEN__;let status='';
async function command(c){try{let r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json','X-Teaching-Token':token},body:JSON.stringify({command:c})});if(!r.ok)throw Error(await r.text());}catch(e){document.querySelector('#message').textContent=e.message}}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>command(b.dataset.c));
const keys={o:'open',c:'close',r:'record',' ':'finish',s:'success',f:'failure',u:'unknown',k:'save_home',h:'home',q:'quit'};
document.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||['INPUT','TEXTAREA'].includes(e.target.tagName))return;const c=keys[e.key.toLowerCase()];if(c){e.preventDefault();command(c)}});
async function refresh(){try{let r=await fetch('/state',{cache:'no-store'});let s=await r.json();status=s.status;document.querySelector('#badge').textContent=s.status_label;document.querySelector('#badge').style.background=status==='recording'?'#a82035':'#24374c';document.querySelector('#message').textContent=s.message;
document.querySelector('#stats').textContent=[s.mock?'模拟设备':'真实设备','实际开口：'+s.gripper_width_mm+' mm','目标开口：'+(s.gripper_target_mm??'保持')+' mm','当前/最近记录：'+s.recorded_frames+' 帧','本次已标注：'+s.saved_episodes+' 条','图像更新于：'+Math.max(...Object.values(s.image_age_ms))+' ms 前',s.episode?'记录：'+s.episode:''].join('\n');document.querySelector('#debug').textContent=JSON.stringify(s,null,2);document.querySelectorAll('button').forEach(b=>{let c=b.dataset.c;b.disabled=(['home','save_home','record'].includes(c)&&['recording','returning'].includes(status))||(c==='record'&&['review','finished'].includes(status))});}
catch(e){document.querySelector('#message').textContent='采集程序连接已断开，请检查终端。'}setTimeout(refresh,250)}refresh();</script></html>'''


class Dashboard:
    """Loopback-only preview and bounded intent queue; no global keyboard hooks."""
    def __init__(self, port=8766):
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("preview_port must be 0..65535")
        self.commands = queue.Queue(maxsize=32)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.image = None
        self.state = {"status": "preparing", "status_label": "准备中", "message": "等待设备"}
        self.token = secrets.token_urlsafe(24)
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, code, content, kind="text/plain; charset=utf-8"):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)

            def do_GET(self):
                if self.path == "/":
                    self.reply(200, PAGE.replace("__TOKEN__", json.dumps(dashboard.token)).encode(), "text/html; charset=utf-8")
                elif self.path == "/state":
                    with dashboard.lock:
                        state = dict(dashboard.state)
                    self.reply(200, json.dumps(state, ensure_ascii=False, allow_nan=False).encode(), "application/json")
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while not dashboard.stop.is_set():
                            with dashboard.lock:
                                rgb = dashboard.image
                            if rgb is not None:
                                stream = io.BytesIO()
                                Image.fromarray(rgb).save(stream, format="JPEG", quality=80)
                                data = stream.getvalue()
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                                self.wfile.flush()
                            dashboard.stop.wait(.1)
                    except (BrokenPipeError, ConnectionResetError, TimeoutError):
                        pass
                else:
                    self.reply(404, b"Not found")

            def do_POST(self):
                origin = self.headers.get("Origin")
                if (self.path != "/command" or self.headers.get("X-Teaching-Token") != dashboard.token
                        or (origin and origin not in {dashboard.url, dashboard.url.replace("127.0.0.1", "localhost")})):
                    self.reply(403, b"Forbidden")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 1024:
                        raise ValueError("Invalid body size")
                    command = json.loads(self.rfile.read(length))["command"]
                    if command not in COMMANDS:
                        raise ValueError("Unknown command")
                    dashboard.commands.put_nowait(command)
                except (ValueError, KeyError, TypeError, queue.Full):
                    self.reply(400, b"Invalid command or queue full")
                    return
                self.reply(202, b"Queued")

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def update(self, state, images):
        with self.lock:
            self.state = dict(state)
            self.image = next(iter(images.values())) if images else None

    def drain(self):
        result = []
        while len(result) < 32:
            try:
                result.append(self.commands.get_nowait())
            except queue.Empty:
                break
        return result


class GripperControl:
    def __init__(self, robot, open_width=.06, close_width=0., speed=.03, *, clock=time.monotonic):
        self.robot, self.clock = robot, clock
        low, high = robot.settings["gripper_limits_m"]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (open_width, close_width, speed)) or not low <= close_width < open_width <= high or speed <= 0):
            raise ValueError("Need configured limits <= close_width < open_width <= limits and positive gripper_speed")
        self.open_width, self.close_width, self.speed = open_width, close_width, speed
        self.target = None
        self.last_command = None
        self.last_sent_at = None
        self.requested_at = None
        self.events = deque(maxlen=200)

    def request(self, opening):
        self.target = self.open_width if opening else self.close_width
        self.last_command = self.robot.get_feedback()["state"][6]
        self.last_sent_at = None
        self.requested_at = self.clock()
        self.events.append({"event": "gripper_request", "monotonic": self.requested_at,
                            "target_m": self.target})

    def tick(self):
        if self.target is None:
            return
        now = self.clock()
        period = 1 / self.robot.settings["control_hz"]
        if self.last_sent_at is not None and now - self.last_sent_at < period * .8:
            return
        # Cap each target increment even if the operator/UI loop was delayed.
        step = min(self.speed * period, self.robot.settings["max_step_delta"][6])
        width = self.last_command + max(-step, min(step, self.target - self.last_command))
        event = self.robot.command_gripper(width)
        self.last_command, self.last_sent_at = event["commanded_width_m"], now
        self.events.append({"event": "gripper_command", **event})

    def is_open(self, feedback):
        return feedback["gripper_enabled"] and abs(feedback["state"][6] - self.open_width) <= .002

    def suspend(self):
        self.target = None


class TeachingSession:
    def __init__(self, args, robot, cameras, dashboard, *, clock=time.monotonic, sleep=time.sleep):
        if not math.isfinite(args.duration) or args.duration <= 0 or not args.task.strip():
            raise ValueError("Need a task and positive duration")
        self.args, self.robot, self.cameras, self.dashboard = args, robot, cameras, dashboard
        self.clock, self.sleep = clock, sleep
        self.gripper = GripperControl(robot, args.open_width, args.close_width, args.gripper_speed, clock=clock)
        self.status, self.message = "preparing", "手动摆好全局视角；正在张爪。O 张开 / C 闭合；R 开始。"
        self.quit = False
        self.start_requested = False
        self.stop_recording = False
        self.requested_outcome = None
        self.completed = 0
        self.last_frame_count = 0
        self.review_path = None
        self.home_target = None
        self.home_started = None
        self.home_after_status = None
        self.event_stream = None
        self.active_path = None
        self.latest = None
        self.gripper.request(True)

    def _event(self, event):
        if self.event_stream is not None:
            self.event_stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.event_stream.flush()

    def _save_home(self):
        feedback = self.robot.get_feedback()
        # The saved pose must also be a legal future joint target.
        self.robot._validate_target(feedback["state"])
        path = self.args.home_file
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"schema_version": 1, "joint_unit": "degree", "joint_angles": feedback["state"][:6],
                          "camera_names": list(self.cameras.config), "created_unix_s": time.time(),
                          "note": "Operator-selected global view, not a calibrated Cartesian pose."})
        for name, rgb in self.cameras.read()["images"].items():
            Image.fromarray(rgb).save(path.parent / f"{path.stem}-{name}.png")
        self.message = f"初始姿态已记录：{path}。H 可在录制外返回，R 不要求与它完全一致。"

    def _start_home(self):
        data = json.loads(self.args.home_file.read_text())
        if data.get("schema_version") != 1 or data.get("joint_unit") != "degree":
            raise ValueError("Initial pose needs schema_version=1 and degree units")
        joints = data.get("joint_angles")
        if not isinstance(joints, list) or len(joints) != 6:
            raise ValueError("Initial pose needs six joint_angles")
        feedback = self.robot.get_feedback()
        self.robot._validate_target(joints + [feedback["state"][6]])
        if not self.robot.mock and feedback["ctrl_mode"] == 2:
            raise RuntimeError("返回前请手动退出按键示教；程序不复位、不强制切出示教。")
        if self.robot.mock:
            self.robot._backend.mode = 0  # Simulate the operator leaving button teaching.
            self.robot._backend.refresh()
        self.robot.prepare_control()
        self.gripper.suspend()
        self.home_after_status = self.status
        self.home_target, self.home_started = joints, self.clock()
        self.status, self.message = "returning", "返回初始姿态中。Q/空格取消并调用已配置停止；完成后重新按实体按钮进入拖动示教。"

    def _home_tick(self):
        if self.clock() - self.home_started > 60:
            raise TimeoutError("返回初始姿态超过 60 秒")
        feedback = self.robot.check_ready()
        current = feedback["state"]
        if max(abs(a - b) for a, b in zip(current[:6], self.home_target)) <= .5:
            self.home_target = None
            self.status = self.home_after_status
            self.gripper.request(True)
            self.message = "已返回并正在张爪。请用实体按钮进入拖动示教，确认画面后按 R；不要直接拖动 CAN 位置控制中的关节。"
            return
        target = [q + max(-delta * .5, min(delta * .5, goal - q))
                  for q, goal, delta in zip(current[:6], self.home_target, self.robot.settings["max_step_delta"][:6])]
        self.robot.send_action(target + [current[6]])

    def handle(self, command):
        if command not in COMMANDS:
            raise ValueError("Unknown teaching command")
        self._event({"event": "operator_key", "command": command, "monotonic": self.clock()})
        if self.status == "returning":
            if command in {"quit", "finish"}:
                self.robot.stop()
                self.home_target = None
                self.status, self.message = self.home_after_status, "返回已停止；按停止策略处理实机后再继续。"
                self.quit = command == "quit"
            else:
                self.message = "返回中只接受空格取消或 Q 退出。"
            return
        if command in {"open", "close"}:
            self.gripper.request(command == "open")
        elif command == "quit":
            self.quit, self.stop_recording, self.start_requested = True, True, False
        elif command in {"save_home", "home"}:
            if self.status == "recording":
                self.message = "先结束录制再记录/返回初始姿态，避免将复位混进训练数据。"
            elif command == "save_home":
                self._save_home()
            else:
                self._start_home()
        elif command == "record":
            if self.status in {"review", "finished", "recording"}:
                self.message = "先标注上一条结果，或本次计划条数已完成。"
                return
            feedback = self.robot.get_feedback()
            if not self.robot.mock and feedback["ctrl_mode"] != 2:
                raise RuntimeError("请先用实体按钮进入拖动示教，再开始手拖录制；程序不会自动切模式。")
            self.gripper.request(True)
            self.start_requested = True
            self.message = "开始请求已收到，等待真实张爪反馈；确认视野里有方块和鼠标垫。"
        elif command in {"finish", "success", "failure", "unknown"}:
            if self.status == "recording":
                self.stop_recording = True
                self.requested_outcome = command if command != "finish" else None
            elif self.status == "review" and command != "finish":
                self._label(command)
            elif command == "finish":
                self.start_requested = False

    def _label(self, outcome):
        path = self.review_path / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest["status"] != "complete":
            raise RuntimeError("不能把不完整录制标注为成功")
        manifest["task_outcome"] = outcome
        write_json(path, manifest)
        self.completed += 1
        self.review_path = None
        self.status = "finished" if self.completed >= self.args.episodes else "preparing"
        self.message = f"已标记 {outcome}；本次已保存 {self.completed}/{self.args.episodes} 条。准备下一条前摆回场景、张爪。"
        self.start_requested = False

    def publish(self, frame_index=None):
        feedback = self.robot.get_feedback()
        camera_frame = self.cameras.read()
        self.latest = feedback
        warning = ""
        if (self.gripper.target == self.gripper.open_width and not self.gripper.is_open(feedback)
                and self.clock() - self.gripper.requested_at > 5):
            warning = "张爪反馈未达到目标；可能是示教固件不接受软件夹爪命令。不会自动复位或切模式，请先检查夹爪。"
        labels = {"preparing": "准备中", "recording": "录制中", "review": "待标注", "returning": "返回中", "finished": "本次采集完成"}
        image_ages = {name: round((self.clock() - t) * 1000, 1) for name, t in camera_frame["image_timestamps"].items()}
        self.dashboard.update({"status": self.status, "status_label": labels[self.status],
                               "message": warning or self.message, "mock": self.args.mock,
                               "task": self.args.task, "frame_index": frame_index,
                               "recorded_frames": frame_index if frame_index is not None else self.last_frame_count,
                               "episode": str(self.active_path or self.review_path or ""),
                               "saved_episodes": self.completed, "image_age_ms": image_ages,
                               "joint_angles_deg": feedback["state"][:6],
                               "gripper_width_mm": round(feedback["state"][6] * 1000, 2),
                               "gripper_target_mm": None if self.gripper.target is None else round(self.gripper.target * 1000, 2),
                               "gripper_enabled": feedback["gripper_enabled"], "ctrl_mode": feedback["ctrl_mode"],
                               "home_file": str(self.args.home_file)}, camera_frame["images"])

    def poll(self, frame_index=None):
        for command in self.dashboard.drain():
            try:
                self.handle(command)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                self.message = str(exc)
                self._event({"event": "command_rejected", "command": command, "message": str(exc), "monotonic": self.clock()})
        if self.quit or self.stop_recording:
            # Do not send another ramp target after an operator stop/quit key.
            pass
        elif self.home_target is not None:
            self._home_tick()
        else:
            self.gripper.tick()
        while self.gripper.events:
            self._event(self.gripper.events.popleft())
        self.publish(frame_index)
        return not (self.stop_recording or self.quit)

    def record(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        indices = [int(p.name[8:]) for p in self.args.output.glob("episode_*") if p.is_dir() and p.name[8:].isdigit()]
        path = self.args.output / f"episode_{max(indices, default=-1) + 1:06d}"
        self.start_requested, self.stop_recording, self.requested_outcome = False, False, None
        self.status, self.active_path = "recording", path
        self.message = "正在录制。O 张开 / C 闭合；空格结束后标注，或 S/F 直接结束并标注。"
        print(f"Recording {path}; gripper keys active, no arm-joint commands", flush=True)
        try:
            metadata = {"mode": "interactive_gripper", "initial_joint_angles_deg": self.latest["state"][:6],
                        "initial_gripper_width_m": self.latest["state"][6], "home_file": str(self.args.home_file),
                        "open_width_m": self.args.open_width, "close_width_m": self.args.close_width,
                        "gripper_effort_sdk": self.robot.settings["gripper_effort"],
                        "operator_events": "events.jsonl"}
            # record_episode creates the directory, so open the audit stream on its first tick.
            def before_sample(index):
                if self.event_stream is None:
                    self.event_stream = (path / "events.jsonl").open("x", encoding="utf-8")
                return self.poll(index)
            manifest = record_episode(self.robot, self.cameras, path, self.args.task, self.robot.config,
                                      self.args.duration, mock=self.args.mock, outcome="unknown",
                                      before_sample=before_sample, metadata=metadata, clock=self.clock, sleep=self.sleep)
            self.review_path, self.status = path, "review"
            self.last_frame_count = manifest['frames']
            self.message = f"已录 {manifest['frames']} 帧，夹爪保持当前状态。S 成功 / F 失败 / U 暂不标注。"
            if self.requested_outcome:
                self._label(self.requested_outcome)
        finally:
            if self.event_stream is not None:
                self.event_stream.close()
                self.event_stream = None
            self.active_path = None
            self.stop_recording = False

    def run(self):
        period = 1 / self.robot.settings["control_hz"]
        try:
            while not self.quit:
                tick = self.clock()
                self.poll()
                if self.start_requested and self.gripper.is_open(self.latest):
                    self.record()
                self.sleep(max(0, period - (self.clock() - tick)))
        finally:
            self.gripper.suspend()
            if self.home_target is not None:
                self.robot.stop()


def run_interactive(args):
    config = json.loads(args.config.read_text())
    robot = PiperRobot(config, mock=args.mock)
    cameras = CameraReader(config["cameras"], mock=args.mock)
    try:
        robot.connect(initialize_can=args.initialize_can, restart_can=args.restart_can)
        if args.mock:
            robot._backend.mode = 2  # Simulate button teaching; no physical mode command.
            robot._backend.refresh()
        cameras.connect()
        with Dashboard(args.preview_port) as dashboard:
            print(f"Teaching dashboard: {dashboard.url}\nO=open C=close R=record Space=finish S/F/U=outcome K=save initial H=return Q=quit", flush=True)
            TeachingSession(args, robot, cameras, dashboard).run()
    finally:
        cameras.close()
        robot.disconnect()
