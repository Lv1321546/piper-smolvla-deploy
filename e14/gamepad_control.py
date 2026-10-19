"""Bounded gamepad -> joint targets, using Gamepad_PiPER's input conventions."""
import math
import time
import numpy as np

from e14.gamepad_kinematics import PiperKinematics, rotation


class GamepadControl:
    def __init__(self, robot, settings=None, *, execute=False, mode='cartesian', clock=time.monotonic, kinematics=None):
        self.robot, self.clock, self.execute, self.mode = robot, clock, execute, mode
        s = settings or {}
        self.linear_speed = s.get('linear_speed_m_s', .04)
        self.angular_speed = s.get('angular_speed_rad_s', .15)
        self.joint_speed = s.get('joint_speed_deg_s', 8.)
        self.grip_speed = s.get('gripper_speed_m_s', .02)
        self.radius = s.get('workspace_radius_m', .15)
        self.allow_rotation = s.get('allow_rotation', True)
        self.invert_y = s.get('invert_y', True)
        self.invert_pitch = s.get('invert_pitch', True)
        for v in (self.linear_speed,self.angular_speed,self.joint_speed,self.grip_speed,self.radius):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError('Positive finite gamepad speeds/workspace radius required')
        if (not isinstance(self.allow_rotation,bool) or not isinstance(self.invert_y,bool)
                or not isinstance(self.invert_pitch,bool)
                or mode not in {'cartesian','joint'}):
            raise ValueError('Invalid gamepad control mode/rotation setting')
        if kinematics is not None and not robot.mock:
            raise ValueError('Preview-model kinematics is restricted to the mock robot')
        self.kin = kinematics if kinematics is not None else PiperKinematics(robot.settings['joint_limits_deg'])
        self.active = self.neutral_seen = self.sent_joints = self.faulted = False
        self.instance = None
        self.velocity = np.zeros(6)
        self.orientation = None
        self.origin = None
        self.gripper_goal = None
        self.message = '先松开 LB 并让摇杆、扳机回中，再按住 LB 操作。'
        self.last_proposal = None
        self.last_time = None

    def _hold(self):
        if self.execute and self.sent_joints:
            state = self.robot.check_ready(check_gripper=False)['state']
            joints = [max(lo,min(hi,q)) for q,(lo,hi) in zip(state[:6],self.robot.settings['joint_limits_deg'])]
            self.robot.send_joint_action(joints)

    def stop(self, reason):
        if self.faulted:
            return  # A latched fault must not repeat emergency commands every UI poll.
        self.active = self.neutral_seen = False
        self.faulted = True
        self.gripper_goal = None
        self.velocity[:] = 0
        self.message = reason + '；已锁住遥操作，恢复实机状态后重新启动程序。'
        if self.robot.mock:
            self.message = reason + '；模拟控制已锁定，请重新启动本程序。'
        if self.execute and self.sent_joints:
            try:
                self.robot.stop()
            except (RuntimeError,OSError) as exc:
                self.message += f' 停止请求未确认：{exc}'

    def request_gripper(self, opening, open_width, close_width):
        self.gripper_goal = open_width if opening else close_width

    def tick(self, pad, *, dt=None):
        now = self.clock()
        period = 1 / self.robot.settings['control_hz']
        dt = period if dt is None else dt
        if self.last_time is not None and now-self.last_time > period*2.5:
            self.stop('控制循环超时')
        self.last_time = now
        if self.faulted:
            return None
        if (not pad.connected or pad.error or not math.isfinite(pad.timestamp)
                or not 0 <= now-pad.timestamp <= .3):
            if self.active:
                self.stop('手柄断开或输入过期')
            else:
                self.neutral_seen = False
                self.message = pad.error or '等待手柄连接'
            return None
        if not 0 < dt <= period*1.5:
            self.stop('无效的控制周期')
            return None
        if self.instance != pad.instance:
            if self.active:
                self.stop('手柄设备已更换')
                return None
            self.instance, self.neutral_seen = pad.instance, False
        values = list(pad.axes.values())
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>1 for v in values):
            self.stop('无效手柄输入')
            return None
        if pad.buttons['start']:
            self.stop('操作者按下 Start 停止')
            return None
        neutral = all(abs(v)<.05 for v in values) and pad.hat==(0,0) and not (pad.buttons['a'] or pad.buttons['b'])
        if not pad.buttons['lb']:
            if self.active:
                self._hold()
                self.gripper_goal = None
            self.active = False
            self.velocity[:] = 0
            self.neutral_seen = neutral
            self.message = '已保持当前位置；按住 LB 允许操作。' if neutral else '先松开摇杆、扳机回中。'
            return None
        if not self.active and not self.neutral_seen:
            self.message = '需要先松开 LB，摇杆和扳机回中，再按住 LB。'
            return None
        feedback = self.robot.get_feedback()
        if self.execute:
            feedback = self.robot.check_ready(check_gripper=False)
        actual = np.array(feedback['state'][:6],float)
        q = np.radians(actual)
        pose = self.kin.forward(q)
        if not self.active:
            self.orientation = pose[:3,:3].copy()
            if self.origin is None:
                self.origin = pose[:3,3].copy()
        self.active = True
        fine = .25 if pad.buttons['rb'] else 1.
        axes = pad.axes
        controls = np.array([axes['left_x'],-axes['left_y'],-axes['right_y'],pad.hat[0],pad.hat[1],axes['right_x']])
        limits = np.array(self.robot.settings['max_step_delta'][:6]) * .75
        if self.mode == 'joint':
            demand = controls*self.joint_speed*fine
            self.velocity += np.clip(demand-self.velocity,-30*dt,30*dt)
            target = np.degrees(np.clip(q+np.radians(np.clip(self.velocity*dt,-limits,limits)),self.kin.low,self.kin.high))
        else:
            y_sign = -1 if self.invert_y else 1
            pitch_sign = -1 if self.invert_pitch else 1
            demand = np.r_[np.array([-axes['left_y'],y_sign*axes['left_x'],-axes['right_y']])*self.linear_speed,
                           np.array([pad.hat[0],pitch_sign*pad.hat[1],axes['right_x']])*self.angular_speed]*fine
            if not self.allow_rotation:
                demand[3:] = 0
            accel = np.array([.1,.1,.1,.5,.5,.5])
            self.velocity += np.clip(demand-self.velocity,-accel*dt,accel*dt)
            position = pose[:3,3]+self.velocity[:3]*dt
            if np.max(np.abs(position-self.origin))>self.radius:
                self._hold(); self.message='到达相对工作区边界；请反向移动。'
                return None
            # A valid Cartesian increment can still require more than the
            # permitted joint step. Reduce the increment rather than alternating
            # between a full-speed goal and a hold during coupled rotations.
            answer = None
            for motion_scale in (1., .5, .25, .125):
                position = pose[:3,3]+self.velocity[:3]*dt*motion_scale
                angular = self.velocity[3:]*dt*motion_scale
                next_orientation = self.orientation.copy()
                if np.linalg.norm(angular)>1e-12:
                    next_orientation = rotation(angular,np.linalg.norm(angular)) @ next_orientation
                answer = self.kin.inverse(position,next_orientation,q,np.radians(limits))
                if answer is not None:
                    break
            if answer is None:
                self._hold(); self.message='局部逆解未收敛，保持当前位置；尝试反向移动或关节模式。'
                self.velocity[:] = 0
                return None
            self.orientation = next_orientation
            target = np.degrees(answer)
        # Jaw goals are based on fresh measured opening, not an accumulating
        # open-loop target. This keeps a lagging/blocked jaw from running away.
        jaw = feedback['state'][6]
        lo,hi = self.robot.settings['gripper_limits_m']
        if not lo-.003 <= jaw <= hi+.003:
            raise RuntimeError('夹爪反馈超出范围及 3 mm 测量容差')
        grip_axis = axes['right_trigger']-axes['left_trigger']
        if pad.buttons['a']: grip_axis=-1.
        if pad.buttons['b']: grip_axis=1.
        if abs(grip_axis)>.05:
            self.gripper_goal = None
            jaw_target = max(lo,min(hi,jaw+grip_axis*self.grip_speed*dt))
        elif self.gripper_goal is not None:
            jaw_target = max(lo,min(hi,jaw+max(-self.grip_speed*dt,min(self.grip_speed*dt,self.gripper_goal-jaw))))
        else:
            jaw_target = None
        if jaw_target is not None:
            correction = abs(jaw-max(lo,min(hi,jaw)))
            if abs(jaw_target-jaw)>self.robot.settings['max_step_delta'][6]+correction+1e-9:
                raise ValueError('夹爪目标变化超过每周期限值')
        self.last_proposal = target.tolist()+[jaw if jaw_target is None else jaw_target]
        event = {'event':'gamepad_target','monotonic':now,'mode':self.mode,'execute':self.execute,
                 'target':self.last_proposal,'measured_before':feedback['state']}
        if self.execute:
            if np.max(np.abs(target-actual))>1e-6 or np.max(np.abs(self.velocity))>1e-6:
                self.robot.send_joint_action(target.tolist())
                self.sent_joints = True
            if jaw_target is not None:
                self.robot.command_gripper(jaw_target)
        self.message = '按住 LB 遥操作中；RB 精细速度。' if self.execute else '只读预览：显示拟发送目标，未发送运动指令。'
        return event
