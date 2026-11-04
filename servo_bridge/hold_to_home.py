"""Hold A to return through the existing guarded JointJog execution path."""
import json
import math
from pathlib import Path
import time

import numpy as np

from servo_bridge.cartesian_policy import CartesianPolicy
from servo_bridge.core import vector
from servo_bridge.joint_policy import JointPolicy


class HoldToHomePolicy:
    def __init__(self, cfg, mapping=None, home_q=None, *, clock=time.monotonic):
        self.cfg, self.clock = cfg, clock
        self.manual = CartesianPolicy(cfg, mapping)
        self.home_gate = JointPolicy(cfg, mapping)
        self.home_q = vector(home_q).copy()
        self.limits = self.home_gate.limits
        self.margin = self.manual.joint_margin_rad
        if (np.any(self.home_q < self.limits[:, 0]+self.margin) or
                np.any(self.home_q > self.limits[:, 1]-self.margin)):
            raise ValueError('返回初始姿态必须位于关节软限位内，请重新记录初始状态')
        defaults = dict(home_hold_s=.6, home_speed_deg_s=cfg['joint_speed_deg_s'],
                        home_acceleration_deg_s2=cfg['joint_acceleration_deg_s2'], home_tolerance_deg=.3)
        for key, default in defaults.items():
            value = cfg.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
            setattr(self, key.removeprefix('home_'), float(value))
        self.speed = math.radians(self.speed_deg_s)
        self.acceleration = math.radians(self.acceleration_deg_s2)
        self.tolerance = math.radians(self.tolerance_deg)
        if self.speed > self.home_gate.speed or self.acceleration > self.home_gate.acceleration:
            raise ValueError('返回速度/加速度不能超过手柄关节控制限制')
        self.velocity = np.zeros(6)
        self.home_requested_at = None
        self.home_active = self.active = self.pause = False
        self.gripper_allowed = True
        self.state = 'manual'
        self.instance = None
        self.a_released = False
        self.manual_rearm_required = False
        self.diagnostics = {}
        self.reset_state = 'idle'
        self.auto_reset = False
        self.reset_gripper_target = None
        self.opening_started = None
        self.reset_message = ''

    @property
    def faulted(self):
        return self.home_gate.faulted or self.manual.faulted

    @property
    def orientation(self):
        return self.manual.orientation

    @property
    def reset_busy(self):
        return self.auto_reset

    def request_reset(self):
        if self.auto_reset:
            return False
        self.auto_reset = True
        self.reset_state = 'pending'
        self.reset_message = ''
        self.reset_gripper_target = None
        self.opening_started = None
        self.home_requested_at = self.clock()
        self.state = 'pending'
        self.manual_rearm_required = True
        return True

    def cancel_reset(self, reason='自动复位已取消，保持当前位置。', *, failed=False):
        self.auto_reset = False
        self.reset_state = 'failed' if failed else 'cancelled'
        self.reset_message = reason
        self.reset_gripper_target = None
        self.home_requested_at = None
        self.state = 'manual'
        self.manual_rearm_required = True
        self.a_released = False
        return self._hold(reason)

    def reset_motion(self):
        self.velocity[:] = 0
        self.manual.reset_motion()

    def _hold(self, message):
        self.active = self.home_active = False
        self.reset_motion()
        self.diagnostics = dict(message=message, home_state=self.state,
                                home_active=False, achieved_twist=[0.]*6)
        return np.zeros(6)

    def step(self, pad, measured_rotation, dt, *, reference_may_advance=True,
             measured_q=None, reference_q=None, home_may_start=True, measured_width=None):
        buttons = pad.get('buttons', {})
        a = bool(buttons.get('a'))
        home_pad = {**pad, 'buttons': {**buttons, 'a': False, 'lb': a or self.auto_reset}}
        # Reuse the same normalized-input, device identity and fault latch.
        self.home_gate.step(home_pad, measured_rotation, dt)
        if pad.get('instance') != self.instance:
            self.instance = pad.get('instance')
            self.a_released = False
        self.pause = False
        if self.faulted or not pad.get('connected') or pad.get('error'):
            if self.auto_reset:
                return self.cancel_reset('自动复位因手柄断连、设备变化或停止键而中止。', failed=True)
            self.state = 'blocked'
            self.gripper_allowed = False
            self.home_requested_at = None
            self.a_released = False
            return self._hold('手柄断连、设备变化或停止键触发，保持当前位置。')
        values = vector([pad.get('axes', {}).get(key, 0.) for key in
                         ('left_x', 'left_y', 'right_x', 'right_y', 'left_trigger', 'right_trigger')])
        neutral = np.max(abs(values)) < .05 and not np.any(pad.get('hat', [0, 0])) and not buttons.get('b')
        if self.auto_reset:
            return self._step_reset(pad, measured_rotation, dt, neutral=neutral,
                measured_q=measured_q, reference_q=reference_q,
                home_may_start=home_may_start, measured_width=measured_width)
        if not a:
            self.a_released = bool(neutral)
            self.home_requested_at = None
            if self.state != 'manual':
                self.state = 'manual'
                self.manual_rearm_required = True
                return self._hold('A 已松开，保持当前位置；松开 LB 并回中后继续普通控制。')
            if self.manual_rearm_required:
                if not buttons.get('lb') and neutral:
                    self.manual_rearm_required = False
                self.gripper_allowed = False
                return self._hold('返回已停止，松开 LB 并将输入回中后继续。')
            self.gripper_allowed = True
            output = self.manual.step(pad, measured_rotation, dt,
                reference_may_advance=reference_may_advance,
                measured_q=measured_q, reference_q=reference_q)
            self.active = self.manual.active
            self.home_active = False
            self.pause = self.manual.pause
            self.diagnostics = {**self.manual.diagnostics, 'home_state': 'manual', 'home_active': False}
            return output
        self.gripper_allowed = False
        self.manual_rearm_required = True
        # Stop Cartesian input immediately, including during the long-press wait.
        self.manual.step({**pad, 'buttons': {**buttons, 'a': False, 'lb': False}},
                         measured_rotation, dt, measured_q=measured_q, reference_q=reference_q)
        if self.state in ('blocked', 'reached'):
            return self._hold('已到达初始位置，松开 A。' if self.state == 'reached'
                              else '请松开 A，将摇杆/扳机回中后重新长按。')
        if not neutral:
            self.state = 'blocked'
            return self._hold('返回时请将摇杆、方向键和扳机回中；松开 A 后重试。')
        if self.home_requested_at is None:
            if not self.a_released or not self.home_gate.active:
                self.state = 'blocked'
                return self._hold('请先松开 A，再长按返回。')
            self.home_requested_at = self.clock()
            self.a_released = False
            self.state = 'pending'
        if (self.clock()-self.home_requested_at < self.hold_s or
                self.state != 'returning' and not home_may_start):
            return self._hold('持续按住 A，等待原运动停止后返回。')
        q = vector(measured_q)
        reference = q if reference_q is None else vector(reference_q)
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError('返回时实测关节超出模型范围')
        if max(np.max(abs(q-self.home_q)), np.max(abs(reference-self.home_q))) <= self.tolerance:
            self.state = 'reached'
            return self._hold('已到达初始位置，松开 A。')
        return self._return_velocity(q, reference, dt, fine=bool(buttons.get('rb')))

    def _return_velocity(self, q, reference, dt, *, fine=False):
        error = self.home_q-reference
        lookahead = max(.35, 2*self.cfg.get('execution_velocity_time_constant_s', .15)+2*dt)
        speed = self.speed*.5 if fine else self.speed
        desired = np.clip(error/lookahead, -speed, speed)
        desired = np.sign(desired)*np.minimum(abs(desired), np.sqrt(self.acceleration*abs(error)))
        velocity = self.velocity + np.clip(desired-self.velocity, -self.acceleration*dt, self.acceleration*dt)
        low_distance = np.maximum(0., np.minimum(q, reference)-self.limits[:, 0]-self.margin)
        high_distance = np.maximum(0., self.limits[:, 1]-self.margin-np.maximum(q, reference))
        low = -np.minimum(np.sqrt(self.acceleration*low_distance), low_distance/lookahead)
        high = np.minimum(np.sqrt(self.acceleration*high_distance), high_distance/lookahead)
        bounded = np.clip(velocity, low, high)
        if np.max(abs(bounded-self.velocity)) > self.acceleration*dt+1e-8:
            self.pause = True
            return self._hold('返回接近关节边界，已保持当前位置。')
        self.velocity = bounded
        self.state = 'returning'
        self.active = self.home_active = True
        self.diagnostics = dict(message='按住 A 返回初始关节位置；松开即停止。',
            home_state=self.state, home_active=True,
            home_error_deg=float(np.degrees(np.max(abs(self.home_q-q)))),
            home_target_deg=np.degrees(self.home_q).tolist())
        return bounded.copy()


    def _step_reset(self, pad, measured_rotation, dt, *, neutral, measured_q,
                    reference_q, home_may_start, measured_width):
        buttons = pad.get('buttons', {})
        self.gripper_allowed = False
        if not neutral or any(buttons.get(key) for key in ('a', 'b', 'lb', 'rb')):
            return self.cancel_reset('检测到手柄操作，自动复位已取消；回中后继续控制。')
        self.manual.step({**pad, 'buttons': {**buttons, 'a': False, 'lb': False}},
                         measured_rotation, dt, measured_q=measured_q, reference_q=reference_q)
        if not self.home_gate.active:
            return self.cancel_reset('请先将手柄回中，再按 H 复位。')
        q = vector(measured_q)
        reference = q if reference_q is None else vector(reference_q)
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError('自动复位时实测关节超出模型范围')
        error = max(np.max(abs(q-self.home_q)), np.max(abs(reference-self.home_q)))
        if self.reset_state in ('pending', 'returning'):
            if self.reset_state == 'pending' and not home_may_start:
                return self._hold('正在停止原运动，随后返回初始位置。')
            if error <= self.tolerance:
                self.reset_state = 'opening'
                self.state = 'opening_pending'
                self.home_requested_at = self.clock()
                return self._hold('已回到初始位置，准备张开夹爪。')
            self.reset_state = 'returning'
            velocity = self._return_velocity(q, reference, dt)
            if self.pause:
                return self.cancel_reset('自动返回受关节边界限制，已停止。', failed=True)
            self.diagnostics['message'] = '正在返回初始位置；空格取消复位。'
            return velocity
        if error > self.tolerance:
            return self.cancel_reset('张爪阶段关节偏离初始位置，已停止复位。', failed=True)
        if self.state == 'opening_pending' and not home_may_start:
            return self._hold('关节已到位，等待稳定后张爪。')
        if (isinstance(measured_width, bool) or not isinstance(measured_width, (int, float)) or
                not math.isfinite(measured_width) or not 0 <= measured_width <= .07):
            raise ValueError('自动张爪需要模型范围内的新鲜夹爪反馈')
        target = min(.07, self.cfg['gripper_limits_m'][1])
        if self.opening_started is None:
            self.opening_started = self.clock()
        elapsed = self.clock()-self.opening_started
        if elapsed >= .1 and abs(measured_width-target) <= .0005:
            self.auto_reset = False
            self.reset_state = 'complete'
            self.reset_message = '已回到初始位置并张爪到最大，可准备下一条录制。'
            self.state = 'manual'
            self.home_requested_at = None
            self.reset_gripper_target = None
            return self._hold(self.reset_message)
        if elapsed > 5:
            return self.cancel_reset('夹爪未在 5 秒内张开到最大，复位已停止。', failed=True)
        self.reset_gripper_target = target
        self.state = 'opening'
        self.active = True
        self.home_active = False
        self.velocity[:] = 0
        self.diagnostics = dict(message='关节已到位，夹爪正在张开；空格取消复位。',
                                home_state='opening', home_active=False)
        return np.zeros(6)


def load_home_reference(path):
    reference = json.loads(Path(path).read_text())
    if reference.get('schema_version') != 1 or reference.get('joint_unit') != 'degree':
        raise ValueError('初始状态文件需要 schema_version=1 和 degree 单位')
    return np.radians(vector(reference.get('joint_angles')))
