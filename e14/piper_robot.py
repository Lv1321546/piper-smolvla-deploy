"""PiPER SDK / LeRobot adapter: connect() only reads, control is explicit.

Seven values: joint1..joint6 in degrees, gripper_width in metres (full opening).
Every joint-pair/status/enable/gripper CAN frame has its own monotonic arrival
stamp. Reading an SDK cache does not renew a frame's freshness. Stop behaviour
needs physical commissioning; no method sends reset or automatically disables.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time

try:
    from lerobot.robots.robot import Robot as _Robot
    from lerobot.robots.config import RobotConfig as _RobotConfig
except ImportError:
    class _Robot:
        pass

    @dataclass(kw_only=True)
    class _RobotConfig:
        id: str | None = None
        calibration_dir: Path | None = None


ACTION_NAMES = tuple(f"joint{i}" for i in range(1, 7)) + ("gripper_width",)
JOINT_FRAMES = (0x2A5, 0x2A6, 0x2A7)
REQUIRED_FRAMES = JOINT_FRAMES + (0x2A8, 0x2A1) + tuple(range(0x261, 0x267))


@dataclass(kw_only=True)
class LeRobotPiperAdapterConfig(_RobotConfig):
    deployment: dict = field(default_factory=dict)
    mock: bool = False


# Preserve the old import while making LeRobot's Config-name based factory select
# LeRobotPiperAdapter, whose flat scalar API matches hw_to_dataset_features.
PiperRobotConfig = LeRobotPiperAdapterConfig


if hasattr(_RobotConfig, "register_subclass"):
    _RobotConfig.register_subclass("piper_smolvla")(PiperRobotConfig)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    return float(value)


def ensure_can(interface, *, initialize=False, restart=False):
    """Only change SocketCAN settings with the caller's explicit permission."""
    if not isinstance(interface, str) or not interface or interface.startswith("-"):
        raise ValueError("robot.can_interface must be an interface name")
    if restart and not initialize:
        raise ValueError("restart_can requires initialize_can")

    def inspect():
        result = subprocess.run(
            ["ip", "-details", "-json", "link", "show", "dev", interface],
            check=True, capture_output=True, text=True,
        )
        entries = json.loads(result.stdout)
        if len(entries) != 1 or entries[0].get("linkinfo", {}).get("info_kind") != "can":
            raise RuntimeError(f"{interface} is not a SocketCAN interface")
        return entries[0]

    info = inspect()
    detail = info["linkinfo"].get("info_data", {})
    if detail.get("state") in {"BUS-OFF", "ERROR-PASSIVE"} and not restart:
        raise RuntimeError(f"{interface}: {detail.get('state')}; inspect wiring/power before explicit restart")
    if ("UP" in info.get("flags", []) and detail.get("bittiming", {}).get("bitrate") == 1_000_000
            and not restart):
        return info
    if not initialize:
        raise RuntimeError(f"{interface} is not UP at 1 Mbps; explicit --initialize-can is needed")
    prefix = [] if os.geteuid() == 0 else ["sudo"]
    subprocess.run(prefix + ["ip", "link", "set", "dev", interface, "down"], check=True)
    subprocess.run(prefix + ["ip", "link", "set", "dev", interface, "up", "type", "can", "bitrate", "1000000"], check=True)
    current = inspect()
    detail = current["linkinfo"].get("info_data", {})
    if ("UP" not in current.get("flags", []) or detail.get("bittiming", {}).get("bitrate") != 1_000_000
            or detail.get("state") in {"BUS-OFF", "ERROR-PASSIVE"}):
        raise RuntimeError("CAN configuration did not produce healthy UP / 1 Mbps state")
    return current


class FeedbackStore:
    """Thread-safe V2 raw feedback snapshots. Standard IDs; no master/slave offsets."""
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.frames = {}

    def ingest(self, can_id, data, *, received_at=None):
        if can_id not in REQUIRED_FRAMES or len(data) != 8:
            return
        stamp = self.clock() if received_at is None else received_at
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError("CAN receipt timestamp must be finite")
        with self.lock:
            self.frames[can_id] = (bytes(data), float(stamp))

    def clear(self):
        with self.lock:
            self.frames.clear()

    def snapshot(self):
        with self.lock:
            missing = [hex(i) for i in REQUIRED_FRAMES if i not in self.frames]
            if missing:
                raise RuntimeError("Missing feedback frames: " + ", ".join(missing))
            frames = dict(self.frames)
        state = []
        for can_id in JOINT_FRAMES:
            data = frames[can_id][0]
            state.extend(int.from_bytes(data[i:i + 4], "big", signed=True) / 1000 for i in (0, 4))
        grip = frames[0x2A8][0]
        state.append(int.from_bytes(grip[:4], "big", signed=True) / 1_000_000)
        status = frames[0x2A1][0]
        motors = [frames[i][0][5] for i in range(0x261, 0x267)]
        return {
            "state": state,
            "frame_timestamps": {hex(i): frames[i][1] for i in REQUIRED_FRAMES},
            "feedback_monotonic": min(frames[i][1] for i in REQUIRED_FRAMES),
            "ctrl_mode": status[0], "arm_status": status[1], "mode_feed": status[2],
            "err_code": int.from_bytes(status[6:8], "big"),
            "enabled": [bool(code & 0x40) for code in motors],
            "motor_faults": [code & 0xBF for code in motors],
            "gripper_enabled": bool(grip[6] & 0x40), "gripper_faults": grip[6] & 0x3F,
        }


class MockBackend:
    """Synthetic hardware only. freeze_feedback() simulates a stalled receive bus."""
    def __init__(self, clock=time.monotonic, state_rate=0):
        self.store = FeedbackStore(clock)
        self.state = [0., 0., -30., 0., 30., 0., .04]
        self.mode = 0
        self.arm_status = 0
        self.enabled = False
        self.gripper_enabled = False
        self.frozen = False
        self.commands = []
        self._previous = clock()
        self.state_rate = state_rate

    def refresh(self):
        if self.frozen:
            return
        stamp = self.store.clock()
        self.state[0] += (stamp - self._previous) * self.state_rate
        self._previous = stamp
        for index, can_id in enumerate(JOINT_FRAMES):
            values = self.state[index * 2:index * 2 + 2]
            raw = b"".join(round(v * 1000).to_bytes(4, "big", signed=True) for v in values)
            self.store.ingest(can_id, raw, received_at=stamp)
        grip = round(self.state[6] * 1_000_000).to_bytes(4, "big", signed=True)
        self.store.ingest(0x2A8, grip + b"\0\0" + bytes([0x40 if self.gripper_enabled else 0, 0]), received_at=stamp)
        self.store.ingest(0x2A1, bytes([self.mode, self.arm_status, 1, 0, 0, 0, 0, 0]), received_at=stamp)
        for can_id in range(0x261, 0x267):
            self.store.ingest(can_id, bytes([0, 0, 0, 0, 0, 0x40 if self.enabled else 0, 0, 0]), received_at=stamp)

    def connect(self):
        self.refresh()

    def disconnect(self):
        pass

    def mode_control(self, speed):
        self.commands.append(("mode", speed))
        self.mode = 1
        self.refresh()

    def enable(self):
        self.commands.append(("enable",))
        self.enabled = True
        self.refresh()

    def joint(self, values):
        self.commands.append(("joint", list(values)))
        self.state[:6] = [v / 1000 for v in values]
        self.refresh()

    def gripper(self, width, effort):
        self.commands.append(("gripper", width, effort))
        self.state[6] = width / 1_000_000
        self.gripper_enabled = True
        self.refresh()

    def emergency_stop(self):
        self.commands.append(("emergency_stop",))
        self.arm_status = 1
        self.refresh()

    def freeze_feedback(self):
        self.frozen = True


class _SDKBackend:
    def __init__(self, can_interface, clock):
        from piper_sdk import C_PiperInterface_V2
        self.store = FeedbackStore(clock)
        store = self.store

        class TrackedInterface(C_PiperInterface_V2):
            # Avoid getting a previous untracked object from the SDK singleton.
            _instances = {}
            _lock = threading.Lock()

            def ParseCANFrame(self, frame):
                if (frame is None or frame.is_error_frame or frame.is_remote_frame
                        or frame.is_extended_id or len(frame.data) != 8):
                    return
                received_at = store.clock()
                super().ParseCANFrame(frame)
                store.ingest(frame.arbitration_id, bytes(frame.data), received_at=received_at)

        self.arm = TrackedInterface(can_interface, start_sdk_joint_limit=False, start_sdk_gripper_limit=False)
        bus = self.arm.GetCanBus()
        original_send = bus.SendCanMessage

        def checked_send(*args, **kwargs):
            result = original_send(*args, **kwargs)
            if result != bus.CAN_STATUS.SEND_MESSAGE_SUCCESS:
                raise RuntimeError(f"PiPER CAN send failed: {result}")
            return result

        bus.SendCanMessage = checked_send

    def refresh(self):
        pass  # Only ParseCANFrame records real receipt times.

    def connect(self):
        self.store.clear()
        self.arm.ConnectPort(piper_init=False)

    def disconnect(self):
        # SDK 0.6.2 DisconnectPort does not stop the monitor and its read-thread
        # join has a misspelled attribute check. Set the known stop events first
        # and join both workers after closing; never reset/disable the arm.
        prefix = "_C_PiperInterface_V2__"
        for name in ("read_can_stop_event", "can_monitor_stop_event"):
            event = getattr(self.arm, prefix + name, None)
            if event is not None:
                event.set()
        try:
            self.arm.DisconnectPort()
        finally:
            for name in ("can_deal_th", "can_monitor_th"):
                thread = getattr(self.arm, prefix + name, None)
                if thread is not None and thread is not threading.current_thread() and thread.is_alive():
                    thread.join(timeout=.5)

    def mode_control(self, speed):
        self.arm.MotionCtrl_2(0x01, 0x01, speed, 0x00)

    def enable(self):
        self.arm.EnableArm(7)

    def joint(self, values):
        self.arm.JointCtrl(*values)

    def gripper(self, width, effort):
        self.arm.GripperCtrl(width, effort, 0x01, 0x00)

    def emergency_stop(self):
        self.arm.MotionCtrl_1(0x01, 0x00, 0x00)


class PiperRobot(_Robot):
    """Deployment API; LeRobotPiperAdapter exposes LeRobot's flat scalar interface."""
    name = "piper_smolvla"
    config_class = PiperRobotConfig

    def __init__(self, config, mock=False, *, backend=None, clock=time.monotonic):
        configured_id = None
        if isinstance(config, PiperRobotConfig):
            configured_id = config.id
            mock = bool(mock or config.mock)
            config = config.deployment
        self.config = copy.deepcopy(config)
        self.settings = self.config.get("robot", {})
        self.mock = bool(mock)
        self._clock = clock
        self._backend = backend
        self._connected = False
        self._prepared = False
        self._last_target = None
        self._last_sent = None
        self._command_lock = threading.RLock()
        self.id = configured_id or self.settings.get("id", "piper")
        self.robot_type = self.name
        self._validate_settings()

    def _validate_settings(self):
        defaults = {
            "can_interface": "can0", "control_hz": 20., "feedback_max_age_ms": 300.,
            "joint_limits_deg": [[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-180, 180]],
            "gripper_limits_m": [0, .08], "max_step_delta": [1.] * 6 + [.002],
            "speed_percent": 10, "gripper_effort": 1000, "stop_strategy": "emergency_stop",
            "stop_strategy_verified": False, "enable_gripper": True,
            "feedback_limit_tolerance_deg": 1.,
        }
        if self.mock:
            self.settings = {**defaults, **{k: v for k, v in self.settings.items() if v is not None}}
        for name in defaults:
            if name not in self.settings or self.settings[name] is None:
                if name == "feedback_limit_tolerance_deg":
                    self.settings[name] = 1.
                else:
                    raise ValueError(f"Missing required configuration robot.{name}")
        s = self.settings
        for key in ("control_hz", "feedback_max_age_ms"):
            _number(s[key], f"robot.{key}", positive=True)
        if not isinstance(s["joint_limits_deg"], list) or len(s["joint_limits_deg"]) != 6:
            raise ValueError("robot.joint_limits_deg needs six [min,max] pairs")
        for pair in list(s["joint_limits_deg"]) + [s["gripper_limits_m"]]:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("Limits must be [min,max] pairs")
            if _number(pair[0], "lower limit") >= _number(pair[1], "upper limit"):
                raise ValueError("Lower limit must be smaller than upper limit")
        if s["gripper_limits_m"][0] < 0 or s["gripper_limits_m"][1] > .1:
            raise ValueError("gripper_limits_m must be full opening metres in [0,.1]")
        self._vector(s["max_step_delta"], "robot.max_step_delta")
        if any(v <= 0 for v in s["max_step_delta"]):
            raise ValueError("max_step_delta entries must be positive")
        if not isinstance(s["speed_percent"], int) or isinstance(s["speed_percent"], bool) or not 1 <= s["speed_percent"] <= 100:
            raise ValueError("speed_percent must be integer 1..100")
        if not isinstance(s["gripper_effort"], int) or isinstance(s["gripper_effort"], bool) or not 0 <= s["gripper_effort"] <= 5000:
            raise ValueError("gripper_effort must be integer SDK units 0..5000")
        if s["stop_strategy"] not in {"hold", "emergency_stop"}:
            raise ValueError("stop_strategy must be hold or emergency_stop")
        for key in ("stop_strategy_verified", "enable_gripper"):
            if not isinstance(s[key], bool):
                raise ValueError(f"robot.{key} must be boolean")
        if _number(s["feedback_limit_tolerance_deg"], "feedback_limit_tolerance_deg") < 0:
            raise ValueError("feedback_limit_tolerance_deg must be >= 0")
        if tuple(s.get("state_action_order", ACTION_NAMES)) != ACTION_NAMES:
            raise ValueError(f"state_action_order must be {ACTION_NAMES}")
        for key, required in (("joint_unit", "degree"), ("gripper_unit", "meter_full_opening"), ("action_mode", "absolute_position")):
            if s.get(key, required) != required:
                raise ValueError(f"robot.{key} must be {required}")

    @staticmethod
    def _vector(values, name):
        if isinstance(values, dict):
            if set(values) != set(ACTION_NAMES):
                raise ValueError(f"{name} needs exactly keys {ACTION_NAMES}")
            values = [values[key] for key in ACTION_NAMES]
        if not isinstance(values, (list, tuple)) or len(values) != 7:
            raise ValueError(f"{name} needs seven values: {ACTION_NAMES}")
        return [_number(value, name) for value in values]

    @property
    def observation_features(self):
        return {"state": (7,), "captured_monotonic": float, "feedback_monotonic": float}

    @property
    def action_features(self):
        return {name: float for name in ACTION_NAMES}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True  # Uses factory zero positions, never changes calibration.

    def calibrate(self):
        raise RuntimeError("Factory-zero calibration needs the official PiPER tools")

    def configure(self):
        """Validated in __init__; prepare_control() is explicitly required for control."""

    def connect(self, calibrate=False, *, initialize_can=False, restart_can=False):
        if calibrate:
            raise RuntimeError("Automatic hardware calibration is unsupported")
        if self._connected:
            return
        if self._backend is None:
            if self.mock:
                self._backend = MockBackend(self._clock, self.settings.get("mock_state_rate_deg_s", 0))
            else:
                ensure_can(self.settings["can_interface"], initialize=initialize_can, restart=restart_can)
                self._backend = _SDKBackend(self.settings["can_interface"], self._clock)
        try:
            self._backend.connect()
            self._connected = True
            self._wait(lambda snapshot: True, timeout=float(self.settings.get("connect_timeout_s", 3)))
        except BaseException as error:
            try:
                self._backend.disconnect()
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Backend disconnect also failed: {cleanup_error}")
            finally:
                self._connected = False
                self._prepared = False
            raise

    def _snapshot(self):
        if not self._connected:
            raise RuntimeError("PiPER is not connected")
        self._backend.refresh()
        snap = self._backend.store.snapshot()
        now = self._clock()
        stale = [key for key, stamp in snap["frame_timestamps"].items()
                 if not math.isfinite(stamp) or now - stamp > self.settings["feedback_max_age_ms"] / 1000 or stamp > now]
        if stale:
            raise RuntimeError("Stale feedback frames: " + ", ".join(stale))
        return snap

    def _wait(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        error = None
        while time.monotonic() < deadline:
            try:
                snapshot = self._snapshot()
                if predicate(snapshot):
                    return snapshot
            except RuntimeError as exc:
                error = exc
            time.sleep(.01)
        raise RuntimeError(f"Feedback wait timed out: {error or 'condition not confirmed'}")

    @staticmethod
    def _healthy(snap):
        if snap["arm_status"] != 0 or snap["err_code"] or any(snap["motor_faults"]) or snap["gripper_faults"]:
            raise RuntimeError(f"PiPER error: arm={snap['arm_status']} code={snap['err_code']} motors={snap['motor_faults']} gripper={snap['gripper_faults']}")

    def get_observation(self):
        snap = self._snapshot()
        return {"state": list(snap["state"]), "captured_monotonic": self._clock(),
                "feedback_monotonic": snap["feedback_monotonic"]}

    def get_lerobot_observation(self):
        return dict(zip(ACTION_NAMES, self.get_observation()["state"]))

    def get_feedback(self):
        """Fresh feedback for the operator UI; reading never commands an actuator."""
        return self._snapshot()

    def command_gripper(self, width, *, effort=None):
        """Explicit manual gripper command, independent of six-joint control.

        Does not select CAN mode, enable arm joints, reset, clear faults or set
        zero. Some firmware ignores gripper commands in button teaching mode;
        callers must check measured opening rather than assume acceptance.
        ``enable_gripper`` applies to policy execution, not this explicit API.
        """
        with self._command_lock:
            width = _number(width, "gripper width")
            low, high = self.settings["gripper_limits_m"]
            if not low <= width <= high:
                raise ValueError(f"Gripper width {width} outside [{low},{high}] metres")
            effort = self.settings["gripper_effort"] if effort is None else effort
            if isinstance(effort, bool) or not isinstance(effort, int) or not 0 < effort <= 5000:
                raise ValueError("Manual gripper effort must be integer SDK units 1..5000")
            snap = self._snapshot()
            if (snap["ctrl_mode"] not in (1, 2) or snap["arm_status"] not in (0, 0x0B)
                    or snap["err_code"] or any(snap["motor_faults"]) or snap["gripper_faults"]):
                raise RuntimeError("Manual gripper needs healthy CAN/teaching feedback; no mode switch is sent")
            raw = round(width * 1_000_000)
            if not low <= raw / 1_000_000 <= high:
                raise ValueError("Rounded gripper command exceeds configured limits")
            self._backend.gripper(raw, effort)
            return {"commanded_width_m": raw / 1_000_000, "effort_sdk": effort,
                    "command_monotonic": self._clock(), "feedback_before_m": snap["state"][6]}

    def prepare_control(self, *, joint_only=False):
        """Select CAN/MOVE_J and enable, never reset or send a joint movement goal.

        Gripper enable uses GripperCtrl at its current opening; this is still an
        actuator command, covered by the explicit physical commissioning gate.
        """
        with self._command_lock:
            self._prepared = False
            if not self.mock and not self.settings["stop_strategy_verified"]:
                raise RuntimeError("Real control requires robot.stop_strategy_verified=true after commissioning")
            snap = self._snapshot()
            self._healthy(snap)
            if snap["ctrl_mode"] == 2:
                raise RuntimeError("Exit teaching mode with the physical button first; no reset is sent")
            prior = snap["frame_timestamps"]["0x2a1"]
            self._backend.mode_control(self.settings["speed_percent"])
            snap = self._wait(lambda s: s["ctrl_mode"] == 1 and s["mode_feed"] == 1
                              and s["frame_timestamps"]["0x2a1"] > prior, 3.)
            self._healthy(snap)
            if not all(snap["enabled"]):
                self._backend.enable()
                snap = self._wait(lambda s: all(s["enabled"]), 5.)
            if not joint_only and self.settings["enable_gripper"] and not snap["gripper_enabled"]:
                width = snap["state"][6]
                lo, hi = self.settings["gripper_limits_m"]
                if not lo <= width <= hi:
                    raise RuntimeError("Current gripper opening is outside configured bounds")
                self._backend.gripper(round(width * 1_000_000), self.settings["gripper_effort"])
                self._wait(lambda s: s["gripper_enabled"], 3.)
            self._last_target = None
            self._last_sent = None
            self.check_ready(check_gripper=not joint_only)
            self._prepared = True

    def check_ready(self, *, check_gripper=True):
        snap = self._snapshot()
        self._healthy(snap)
        if snap["ctrl_mode"] != 1 or snap["mode_feed"] != 1 or not all(snap["enabled"]):
            raise RuntimeError("PiPER needs CAN control, MOVE_J and all six motors enabled")
        if check_gripper and self.settings["enable_gripper"] and not snap["gripper_enabled"]:
            raise RuntimeError("Gripper is not enabled")
        tolerance = self.settings["feedback_limit_tolerance_deg"]
        for name, value, (low, high) in zip(ACTION_NAMES[:6], snap["state"][:6], self.settings["joint_limits_deg"]):
            if not low - tolerance <= value <= high + tolerance:
                raise RuntimeError(f"{name} feedback {value} outside configured limits plus {tolerance} degree tolerance")
        if check_gripper and not self.settings["gripper_limits_m"][0] - 1e-6 <= snap["state"][6] <= self.settings["gripper_limits_m"][1] + 1e-6:
            raise RuntimeError("Gripper feedback outside configured metre limits")
        return snap

    @property
    def last_joint_target(self):
        """Copy of the last successfully transmitted teleop/policy joint goal."""
        with self._command_lock:
            return None if self._last_target is None else list(self._last_target[:6])

    def send_joint_action(self, joints):
        """Six-joint teleop goal, never sends or silently enables the gripper.

        Independent manual gripper commands use command_gripper(). This lets
        joint motion tolerate measured jaw boundary offsets without treating
        the raw jaw reading as a new seven-dimensional actuator target.
        """
        with self._command_lock:
            if not self._prepared:
                raise RuntimeError("Call prepare_control() explicitly before motion")
            snap = self.check_ready(check_gripper=False)
            if not isinstance(joints, (list, tuple)) or len(joints) != 6:
                raise ValueError("Six joint targets required")
            target = [_number(v, 'joint target') for v in joints]
            sent = [round(v * 1000) / 1000 for v in target]
            for values in (target, sent):
                for v, (lo,hi) in zip(values, self.settings['joint_limits_deg']):
                    if not lo <= v <= hi:
                        raise ValueError("Teleop joint target outside configured limits")
                references = [('实测角度的跟踪差',snap['state'])]
                if self._last_target is not None:
                    references.append(('上一目标的步长',self._last_target))
                for label,ref in references:
                    errors = [f'J{i}: {label} {abs(a-b):.3f}° > {limit:.3f}° '
                              f'(目标 {a:.3f}°, 参考 {b:.3f}°)'
                              for i,(a,b,limit) in enumerate(zip(values,ref,self.settings['max_step_delta'][:6]),1)
                              if abs(a-b)>limit+1e-9]
                    if errors:
                        raise ValueError('Teleop joint target exceeds per-cycle tracking/step limit: '+ '; '.join(errors))
            if (self._last_sent is not None and sent != self._last_target[:6]
                    and self._clock()-self._last_sent < .8/self.settings['control_hz']):
                raise RuntimeError("Teleop target arrived faster than control_hz")
            try:
                self._backend.joint([round(v*1000) for v in sent])
            except BaseException:
                self._prepared = False
                try:
                    self.stop()
                except Exception:
                    pass
                raise
            self._last_target = sent + [snap['state'][6]]
            self._last_sent = self._clock()
            return sent

    def _validate_target(self, action):
        target = self._vector(action, "action")
        bounds = self.settings["joint_limits_deg"] + [self.settings["gripper_limits_m"]]
        for name, value, (low, high) in zip(ACTION_NAMES, target, bounds):
            if not low <= value <= high:
                raise ValueError(f"{name} target {value} outside [{low},{high}]; no silent clipping")
        return target

    def send_action(self, action):
        with self._command_lock:
            if not self._prepared:
                raise RuntimeError("Call prepare_control() explicitly before motion")
            snap = self.check_ready()
            target = self._validate_target(action)
            for reference in [snap["state"]] + ([self._last_target] if self._last_target else []):
                for name, value, current, delta in zip(ACTION_NAMES, target, reference, self.settings["max_step_delta"]):
                    if abs(value - current) > delta + 1e-9:
                        raise ValueError(f"{name}: step {value-current:g} exceeds max_step_delta {delta}")
            if not self.settings["enable_gripper"] and abs(target[6] - snap["state"][6]) > 1e-6:
                raise RuntimeError("Gripper command requested with enable_gripper=false")
            raw = [round(v * 1000) for v in target[:6]] + [round(target[6] * 1_000_000)]
            sent = [v / 1000 for v in raw[:6]] + [raw[6] / 1_000_000]
            self._validate_target(sent)
            for reference in [snap["state"]] + ([self._last_target] if self._last_target else []):
                if any(abs(value - current) > delta + 1e-9
                       for value, current, delta in zip(sent, reference, self.settings["max_step_delta"])):
                    raise ValueError("SDK integer rounding would exceed configured max_step_delta")
            # Caller owns the periodic loop. Reject rapid changed commands that
            # would otherwise bypass a per-cycle delta limit by busy-looping.
            if (self._last_sent is not None and sent != self._last_target
                    and self._clock() - self._last_sent < .8 / self.settings["control_hz"]):
                raise RuntimeError("Action arrived faster than configured control_hz")
            try:
                self._backend.joint(raw[:6])
                if self.settings["enable_gripper"]:
                    self._backend.gripper(raw[6], self.settings["gripper_effort"])
            except BaseException:
                self._prepared = False
                try:
                    self.stop()  # Joint CAN frames are not atomic: attempt configured stop.
                except Exception:
                    pass
                raise
            self._last_target = sent
            self._last_sent = self._clock()
            return dict(zip(ACTION_NAMES, sent)) if isinstance(action, dict) else sent

    def stop(self):
        """Latch execution off. Never disable/reset or recover an emergency stop.

        emergency_stop: MotionCtrl_1(0x01,0,0); official PiPER documentation
        warns this allows damped falling, NOT position holding. hold: one
        position goal from fresh healthy feedback, never a stale SDK cache.
        Neither guarantees stopping on a broken CAN bus; both require physical
        verification. Never silently fall back from hold to emergency_stop.
        """
        with self._command_lock:
            self._prepared = False
            if not self._connected:
                return
            if not self.mock and not self.settings["stop_strategy_verified"]:
                raise RuntimeError("Stop strategy has not been physically verified")
            if self.settings["stop_strategy"] == "emergency_stop":
                self._backend.emergency_stop()
                return
            snap = self.check_ready()
            # check_ready permits small measured zero-boundary offsets. A hold
            # goal still has to be legal, so correct only that tolerated offset
            # and reject any correction larger than the normal per-cycle delta.
            bounds = self.settings["joint_limits_deg"] + [self.settings["gripper_limits_m"]]
            target = [max(low, min(high, value)) for value, (low, high) in zip(snap["state"], bounds)]
            raw = [round(v * 1000) for v in target[:6]] + [round(target[6] * 1_000_000)]
            sent = [v / 1000 for v in raw[:6]] + [raw[6] / 1_000_000]
            self._validate_target(sent)
            if any(abs(value - current) > delta + 1e-9
                   for value, current, delta in zip(sent, snap["state"], self.settings["max_step_delta"])):
                raise RuntimeError("Boundary correction for hold exceeds max_step_delta")
            self._backend.joint(raw[:6])
            if self.settings["enable_gripper"]:
                self._backend.gripper(raw[6], self.settings["gripper_effort"])

    def disconnect(self):
        with self._command_lock:
            try:
                if self._connected:
                    self._backend.disconnect()
            finally:
                self._connected = False
                self._prepared = False
                self._last_target = None
                self._last_sent = None
            # Closing a read-only connection must not disable, reset or stop.


class LeRobotPiperAdapter(PiperRobot):
    """Flat scalar API for LeRobot's standard robot/dataset tools."""
    @property
    def observation_features(self):
        return {name: float for name in ACTION_NAMES}

    def get_observation(self):
        return dict(zip(ACTION_NAMES, PiperRobot.get_observation(self)["state"]))

    def get_lerobot_observation(self):
        return self.get_observation()


def main():
    parser = argparse.ArgumentParser(description="Read-only PiPER inspection; never enables motors")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--initialize-can", action="store_true")
    parser.add_argument("--restart-can", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.restart_can and not args.initialize_can:
        parser.error("--restart-can requires --initialize-can")
    config = json.loads(Path(args.config).read_text())
    robot = PiperRobot(config, mock=args.mock)
    try:
        robot.connect(initialize_can=args.initialize_can, restart_can=args.restart_can)
        for _ in range(args.samples):
            print(json.dumps(robot.get_observation(), ensure_ascii=False), flush=True)
            time.sleep(1 / robot.settings["control_hz"])
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
