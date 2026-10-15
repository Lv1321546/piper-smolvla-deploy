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
<button data-c="quit">Q 退出</button><pre id="stats"></pre><p id="warnings" style="white-space:pre-wrap;color:#ffb5b5"></p><details><summary>运行详情</summary><pre id="debug"></pre></details>
<small>CAN 模式可用 O/C 测试夹爪，此时不要手拖关节。手拖抓取采集需要固件支持示教期间的软件夹爪；录制结束保持夹爪当前状态。</small></section></main>
<script>const token=__TOKEN__;let status='';
async function command(c){try{let r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json','X-Teaching-Token':token},body:JSON.stringify({command:c})});if(!r.ok)throw Error(await r.text());}catch(e){document.querySelector('#message').textContent=e.message}}
document.querySelectorAll('button').forEach(b=>b.onclick=()=>command(b.dataset.c));
const keys={o:'open',c:'close',r:'record',' ':'finish',s:'success',f:'failure',u:'unknown',k:'save_home',h:'home',q:'quit'};
document.addEventListener('keydown',e=>{if(e.repeat||e.ctrlKey||e.metaKey||e.altKey||['INPUT','TEXTAREA'].includes(e.target.tagName))return;const c=keys[e.key.toLowerCase()];if(c){e.preventDefault();command(c)}});
async function refresh(){try{let r=await fetch('/state',{cache:'no-store'});let s=await r.json();status=s.status;document.querySelector('#badge').textContent=s.status_label;document.querySelector('#badge').style.background=status==='recording'?'#a82035':'#24374c';document.querySelector('#message').textContent=s.message;
document.querySelector('#stats').textContent=[s.mock?'模拟设备':'真实设备',s.ctrl_mode_label??'','实际开口：'+(s.gripper_width_mm??'无新反馈')+' mm','目标开口：'+(s.gripper_target_mm??'未发送新目标')+' mm','夹爪：'+(s.gripper_status??'等待反馈'),'当前/最近记录：'+s.recorded_frames+' 帧','本次已标注：'+s.saved_episodes+' 条','图像更新于：'+Math.max(0,...Object.values(s.image_age_ms??{}))+' ms 前',s.episode?'记录：'+s.episode:''].join('\n');document.querySelector('#warnings').textContent=(s.warnings??[]).join('\n');document.querySelector('#debug').textContent=JSON.stringify(s,null,2);document.querySelectorAll('button').forEach(b=>{let c=b.dataset.c;b.disabled=(['home','save_home','record'].includes(c)&&['recording','returning'].includes(status))||(c==='record'&&['review','finished'].includes(status))});}
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
    # Measurement tolerance only. Never expand command limits or rewrite logs.
    FEEDBACK_TOLERANCE_M = .003

    def __init__(self, robot, open_width=.06, close_width=0., speed=.03, *, clock=time.monotonic):
        self.robot, self.clock = robot, clock
        self.teaching_gripper_supported = robot.config.get("collection", {}).get("gripper_in_teaching_supported", False)
        if not isinstance(self.teaching_gripper_supported, bool):
            raise ValueError("collection.gripper_in_teaching_supported must be boolean")
        low, high = robot.settings["gripper_limits_m"]
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (open_width, close_width, speed)) or not low <= close_width < open_width <= high or speed <= 0):
            raise ValueError("Need configured limits <= close_width < open_width <= limits and positive gripper_speed")
        self.open_width, self.close_width, self.speed = open_width, close_width, speed
        self.target = None
        self.last_command = None
        self.last_sent_at = None
        self.requested_at = None
        self.initial_width = None
        self.response_seen = False
        self.error = None
        self.events = deque(maxlen=200)

    def _feedback(self):
        feedback = self.robot.get_feedback()
        if not self.robot.mock and feedback["ctrl_mode"] == 2 and not self.teaching_gripper_supported:
            raise RuntimeError("实体示教下的软件夹爪未验证兼容，已禁止发送。退出实体示教并选择 CAN 后可测试 O/C。手拖抓取采集需兼容固件或改用遥操作。")
        return feedback

    def request(self, opening):
        self.suspend()  # A rejected replacement must not leave an old ramp running.
        self.error = None
        measured = self._feedback()["state"][6]
        low, high = self.robot.settings["gripper_limits_m"]
        if (not math.isfinite(measured)
                or not low - self.FEEDBACK_TOLERANCE_M <= measured <= high + self.FEEDBACK_TOLERANCE_M):
            raise ValueError(f"夹爪反馈 {measured * 1000:g} mm 超出范围及 3 mm 测量容差；暂停夹爪，请检查行程/零点。")
        self.target = self.open_width if opening else self.close_width
        self.initial_width = measured
        self.response_seen = False
        self.last_command = max(low, min(high, measured))
        self.last_sent_at = None
        self.requested_at = self.clock()
        self.events.append({"event": "gripper_request", "monotonic": self.requested_at,
                            "target_m": self.target, "measured_width_m": measured,
                            "ramp_start_m": self.last_command})

    def fail(self, error):
        self.suspend()
        self.error = str(error)
        self.events.append({"event": "gripper_suspended", "message": self.error, "monotonic": self.clock()})

    def tick(self):
        if self.target is None:
            return
        now = self.clock()
        period = 1 / self.robot.settings["control_hz"]
        if self.last_sent_at is not None and now - self.last_sent_at < period * .8:
            return
        feedback = self._feedback()
        measured = feedback["state"][6]
        low, high = self.robot.settings["gripper_limits_m"]
        if (not math.isfinite(measured)
                or not low - self.FEEDBACK_TOLERANCE_M <= measured <= high + self.FEEDBACK_TOLERANCE_M):
            raise RuntimeError("夹爪反馈超出范围及 3 mm 测量容差，已暂停夹爪。")
        # A successful send is not actuator acknowledgement. A contact while
        # closing can stop motion above zero, so do not require a zero reading.
        self.response_seen |= feedback["gripper_enabled"] and abs(measured - self.initial_width) >= .0005
        arrived = feedback["gripper_enabled"] and abs(measured - self.target) <= .002
        if self.last_sent_at is not None and not arrived:
            elapsed = now - self.requested_at
            opening_timeout = abs(self.target - self.initial_width) / self.speed + 3.
            if ((elapsed > 2 and not self.response_seen)
                    or (self.target == self.open_width and elapsed > opening_timeout)):
                raise RuntimeError("夹爪未响应/未达到张开目标；已暂停发送并取消等待录制。示教模式可能忽略软件夹爪命令；闭合时也可能已接触物体。请检查实机，确认后再按 O/C 重试。")
        # Cap each target increment even if the operator/UI loop was delayed.
        step = min(self.speed * period, self.robot.settings["max_step_delta"][6])
        width = max(low, min(high, self.last_command + max(-step, min(step, self.target - self.last_command))))
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
        self.camera_frame = None
        self.device_errors = []
        try:
            self.gripper.request(True)
        except (ValueError, RuntimeError, OSError) as exc:
            self.gripper.fail(exc)
            self.message = "夹爪准备失败；画面和姿态记录仍可使用，请查看下方提示。"

    def _event(self, event):
        if self.event_stream is not None:
            self.event_stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.event_stream.flush()

    def _save_home(self):
        feedback = self.robot.get_feedback()
        # Capturing an observed reference is not a motion request. In particular
        # a noisy gripper boundary must not prevent saving six joint angles.
        joints = feedback["state"][:6]
        if len(joints) != 6 or any(not math.isfinite(q) for q in joints):
            raise ValueError("初始姿态需要六个有效关节角度")
        outside = [i + 1 for i, (q, (low, high)) in enumerate(zip(joints, self.robot.settings["joint_limits_deg"]))
                   if not low <= q <= high]
        path = self.args.home_file
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"schema_version": 1, "joint_unit": "degree", "joint_angles": joints,
                          "joints_outside_target_limits": outside,
                          "camera_names": list(self.cameras.config), "created_unix_s": time.time(),
                          "note": "Operator-selected global view, not a calibrated Cartesian pose."})
        for name, rgb in self.cameras.read()["images"].items():
            Image.fromarray(rgb).save(path.parent / f"{path.stem}-{name}.png")
        self.message = f"初始姿态已记录：{path}。H 可在录制外返回，R 不要求与它完全一致。"
        if outside:
            self.message += f"关节 {outside} 的原始反馈超出目标范围，返回前需要重新调整并记录。"

    def _start_home(self):
        data = json.loads(self.args.home_file.read_text())
        if data.get("schema_version") != 1 or data.get("joint_unit") != "degree":
            raise ValueError("Initial pose needs schema_version=1 and degree units")
        joints = data.get("joint_angles")
        if not isinstance(joints, list) or len(joints) != 6:
            raise ValueError("Initial pose needs six joint_angles")
        feedback = self.robot.get_feedback()
        self.robot._validate_target(joints + [self.robot.settings["gripper_limits_m"][0]])
        if not self.robot.mock and feedback["ctrl_mode"] == 2:
            raise RuntimeError("返回前请手动退出按键示教；程序不复位、不强制切出示教。")
        if not self.robot.mock and not self.robot.settings["stop_strategy_verified"]:
            raise RuntimeError("H 返回尚不可用：robot.stop_strategy_verified=false。需现场验证所选停止方式后才能启用六关节返回；K 保存和相机预览可正常使用。")
        if self.robot.mock:
            self.robot._backend.mode = 0  # Simulate the operator leaving button teaching.
            self.robot._backend.refresh()
        self.robot.prepare_control()
        self.gripper.suspend()
        self.start_requested = False
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
                try:
                    self.robot.stop()
                finally:
                    # Even a failed stop must latch off further home targets.
                    self.home_target = None
                    self.gripper.suspend()
                    self.start_requested = False
                    self.status = self.home_after_status
                    self.quit = command == "quit"
                self.message = "已发送返回停止请求；请确认实机已停止再继续。"
            else:
                self.message = "返回中只接受空格取消或 Q 退出。"
            return
        if command in {"open", "close"}:
            self.gripper.request(command == "open")
            if command == "close":
                self.start_requested = False
            self.message = ("已请求张开" if command == "open" else "已请求闭合") + "；以下方实际开口变化为准。"
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
                if not self.gripper.teaching_gripper_supported:
                    raise RuntimeError("当前 CAN 模式可测试 O/C，但本采集流程按手拖设计；软件夹爪不兼容实体示教时不能录制完整抓取。需改用遥操作采集或取得兼容固件，不要在 CAN 下直接手拖。")
                raise RuntimeError("请先用实体按钮进入拖动示教，再开始手拖录制；程序不会自动切模式。")
            self.start_requested = True
            try:
                self.gripper.request(True)
            except (ValueError, RuntimeError, OSError):
                self.start_requested = False
                raise
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
        self.device_errors = []
        try:
            feedback = self.robot.get_feedback()
        except (ValueError, RuntimeError, OSError) as exc:
            feedback = None
            self.device_errors.append(f"机械臂反馈不可用：{exc}")
        try:
            self.camera_frame = self.cameras.read()
        except (ValueError, RuntimeError, OSError) as exc:
            self.device_errors.append(f"相机画面无新数据（显示的是最后一帧）：{exc}")
        camera_frame = self.camera_frame or {"images": {}, "image_timestamps": {}}
        self.latest = feedback
        warnings = list(self.device_errors)
        if self.gripper.error:
            warnings.append(self.gripper.error)
        if feedback and feedback["ctrl_mode"] == 2 and not self.robot.mock:
            warnings.append("实体示教模式；当前固件不响应 O/C 时，不能用手拖完成键控夹爪抓取采集。H 返回需先退出示教。")
        if not self.robot.mock and not self.robot.settings["stop_strategy_verified"]:
            warnings.append("H 返回未启用：尚未现场验证停止策略。K 只保存姿态，不移动。")
        mode = None if feedback is None else feedback["ctrl_mode"]
        grip_status = ("已暂停，请处理提示后按 O/C 重试" if self.gripper.error else
                       "已观测到开口变化" if self.gripper.response_seen else "等待执行反馈")
        labels = {"preparing": "准备中", "recording": "录制中", "review": "待标注", "returning": "返回中", "finished": "本次采集完成"}
        image_ages = {name: round((self.clock() - t) * 1000, 1) for name, t in camera_frame["image_timestamps"].items()}
        self.dashboard.update({"status": self.status, "status_label": labels[self.status],
                               "message": self.message, "warnings": warnings, "mock": self.args.mock,
                               "task": self.args.task, "frame_index": frame_index,
                               "recorded_frames": frame_index if frame_index is not None else self.last_frame_count,
                               "episode": str(self.active_path or self.review_path or ""),
                               "saved_episodes": self.completed, "image_age_ms": image_ages,
                               "joint_angles_deg": None if feedback is None else feedback["state"][:6],
                               "gripper_width_mm": None if feedback is None else round(feedback["state"][6] * 1000, 2),
                               "gripper_target_mm": None if self.gripper.target is None else round(self.gripper.target * 1000, 2),
                               "gripper_status": grip_status,
                               "gripper_enabled": False if feedback is None else feedback["gripper_enabled"], "ctrl_mode": mode,
                               "ctrl_mode_label": {1: "CAN 控制", 2: "实体示教模式", None: "无新状态反馈"}.get(mode, f"控制模式 {mode}"),
                               "home_file": str(self.args.home_file)}, camera_frame["images"])

    def poll(self, frame_index=None):
        commands = self.dashboard.drain()
        for command in commands:
            try:
                self.handle(command)
            except (ValueError, RuntimeError, OSError, KeyError) as exc:
                self.message = str(exc)
                if command in {"open", "close", "record"}:
                    self.start_requested = False
                    self.gripper.fail(exc)
                self._event({"event": "command_rejected", "command": command, "message": str(exc), "monotonic": self.clock()})
        control_error = None
        try:
            if self.quit or self.stop_recording:
                # Do not send another ramp target after an operator stop/quit key.
                pass
            elif self.home_target is not None:
                self._home_tick()
            else:
                self.gripper.tick()
        except (ValueError, RuntimeError, OSError) as exc:
            control_error = str(exc)
            self.start_requested = False
            self.gripper.fail(exc)
            if not commands:
                self.message = "控制已暂停，实时预览保留；详情见下方提示。"
            if self.home_target is not None:
                try:
                    self.robot.stop()
                except (RuntimeError, OSError) as stop_exc:
                    self.message += f"返回停止指令也未确认：{stop_exc}。请在实机处理。"
                self.home_target = None
                self.status = self.home_after_status
        while self.gripper.events:
            self._event(self.gripper.events.popleft())
        self.publish(frame_index)
        if self.device_errors:
            self.gripper.suspend()
            self.start_requested = False
        if self.status == "recording" and (control_error or self.gripper.error or self.device_errors):
            raise RuntimeError(control_error or self.gripper.error or "; ".join(self.device_errors))
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
        except (ValueError, RuntimeError, OSError) as exc:
            self.last_frame_count = 0
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                failed = json.loads(manifest_path.read_text())
                self.last_frame_count = failed.get("frames", 0)
                failed.update(status="failed", task_outcome="unknown", error=f"{type(exc).__name__}: {exc}")
                write_json(manifest_path, failed)
            self.gripper.suspend()
            self.review_path = None
            self.status = "preparing"
            self.message = f"本条录制失败，未计入成功数据：{exc}。文件保留在 {path}；预览继续，处理后可按 R 重试。"
            print(self.message, flush=True)
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
                if self.start_requested and self.latest and not self.device_errors and self.gripper.is_open(self.latest):
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
