"""Bounded gamepad -> joint targets, using Gamepad_PiPER's input conventions."""
import math
import time
import numpy as np

from e14.gamepad_kinematics import PiperKinematics, rotation

INPUT_DIRECTIONS = {'left_x':1, 'left_y':1, 'right_x':1, 'right_y':1,
                    'hat_x':1, 'hat_y':1}


class GamepadControl:
    def __init__(self, robot, settings=None, *, execute=False, mode='cartesian', clock=time.monotonic, kinematics=None):
        self.robot, self.clock, self.execute, self.mode = robot, clock, execute, mode
        s = settings or {}
        self.linear_speed = s.get('linear_speed_m_s', .12)
        self.angular_speed = s.get('angular_speed_rad_s', .45)
        self.joint_speed = s.get('joint_speed_deg_s', 24.)
        self.grip_speed = s.get('gripper_speed_m_s', .05)
        self.radius = s.get('workspace_radius_m', .15)
        self.allow_rotation = s.get('allow_rotation', True)
        self.cartesian_reference = s.get('cartesian_reference', 'integrated')
        if not isinstance(self.cartesian_reference,str) or self.cartesian_reference not in {'measured','integrated'}:
            raise ValueError('cartesian_reference must be measured or integrated')
        self.invert_y = s.get('invert_y', True)
        self.invert_pitch = s.get('invert_pitch', True)
        directions = s.get('input_directions', {})
        if (not isinstance(directions,dict) or set(directions)-set(INPUT_DIRECTIONS)
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or v not in (-1,1)
                       for v in directions.values())):
            raise ValueError('input_directions needs known input names with +1 or -1 values')
        self.input_directions = {**INPUT_DIRECTIONS, **directions}
        for v in (self.linear_speed,self.angular_speed,self.joint_speed,self.grip_speed,self.radius):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
                raise ValueError('Positive finite gamepad speeds/workspace radius required')
        if (not isinstance(self.allow_rotation,bool) or not isinstance(self.invert_y,bool)
                or not isinstance(self.invert_pitch,bool)
                or mode not in {'cartesian','joint'}):
            raise ValueError('Invalid gamepad control mode/rotation setting')
        if kinematics is not None and not robot.mock:
            raise ValueError('Preview-model kinematics is restricted to the mock robot')
        if self.grip_speed/robot.settings['control_hz'] > robot.settings['max_step_delta'][6]+1e-9:
            raise ValueError('gripper_speed_m_s / control_hz exceeds robot.max_step_delta[6]; adjust the gripper step limit or speed')
        self.kin = kinematics if kinematics is not None else PiperKinematics(robot.settings['joint_limits_deg'])
        self.active = self.neutral_seen = self.sent_joints = self.faulted = False
        self.instance = None
        self.velocity = np.zeros(6)
        self.orientation = None
        self.position_goal = None
        self.joint_goal = None
        self.origin = None
        self.gripper_goal = None
        self.last_proposed_joints = None
        self.cartesian_scale = None
        self.diagnostics = {}
        self.message = '先松开 LB 并让摇杆、扳机回中，再按住 LB 操作。'
        self.last_proposal = None
        self.last_time = None

    def input_mapping(self):
        """Effective directions, after both input signs and legacy axis settings."""
        d = self.input_directions
        if self.mode=='joint':
            names = [('左摇杆向右','J1',d['left_x']), ('左摇杆向前','J2',d['left_y']),
                     ('右摇杆向上','J3',d['right_y']), ('方向键向右','J4',d['hat_x']),
                     ('方向键向上','J5',d['hat_y']), ('右摇杆向右','J6',d['right_x'])]
        else:
            names = [('左摇杆向右','Y',d['left_x']*(-1 if self.invert_y else 1)),
                     ('左摇杆向前','X',d['left_y']), ('右摇杆向上','Z',d['right_y']),
                     ('方向键向右','Roll',d['hat_x']),
                     ('方向键向上','Pitch',d['hat_y']*(-1 if self.invert_pitch else 1)),
                     ('右摇杆向右','Yaw',d['right_x'])]
        return {key:f'{axis}{"+" if sign>0 else "−"}' for key,axis,sign in names}

    def _hold(self):
        self.joint_goal = self.position_goal = None
        self.last_proposed_joints = None
        self.cartesian_scale = None
        if self.execute and self.sent_joints:
            state = self.robot.check_ready(check_gripper=False)['state']
            joints = [max(lo,min(hi,q)) for q,(lo,hi) in zip(state[:6],self.robot.settings['joint_limits_deg'])]
            self.last_proposed_joints = self.robot.send_joint_action(joints)

    def stop(self, reason):
        if self.faulted:
            return  # A latched fault must not repeat emergency commands every UI poll.
        self.active = self.neutral_seen = False
        self.faulted = True
        self.gripper_goal = None
        self.velocity[:] = 0
        self.joint_goal = self.position_goal = None
        self.last_proposed_joints = None
        self.cartesian_scale = None
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
        self.diagnostics = {}
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
            self.position_goal = pose[:3,3].copy()
            self.joint_goal = actual.copy()
            if self.origin is None:
                self.origin = pose[:3,3].copy()
        self.active = True
        fine = .25 if pad.buttons['rb'] else 1.
        axes = {key:value*self.input_directions.get(key,1) for key,value in pad.axes.items()}
        hat = (pad.hat[0]*self.input_directions['hat_x'], pad.hat[1]*self.input_directions['hat_y'])
        controls = np.array([axes['left_x'],-axes['left_y'],-axes['right_y'],hat[0],hat[1],axes['right_x']])
        sender_limits = np.array(self.robot.settings['max_step_delta'][:6],float)
        limits = sender_limits * .75
        previous = self.robot.last_joint_target if self.execute else self.last_proposed_joints
        previous = actual.copy() if previous is None else np.array(previous,float)
        # Both constraints are checked again by send_joint_action. A solution
        # within +/-0.75 degrees of feedback may still jump 1.5 degrees from
        # the previous target after a reversal with motor lag. Plan within the
        # intersection, retaining headroom for rounding/new feedback at send.
        previous_limits = limits
        if self.mode=='cartesian' and self.cartesian_reference=='measured':
            # Match historical valid steps: the send guard permits 1 degree,
            # not 0.75 degrees, relative to the prior sent target. Reserve one
            # SDK quantization unit while retaining the feedback headroom.
            previous_limits = np.maximum(limits,sender_limits-.001)
        low = np.maximum.reduce([np.degrees(self.kin.low),actual-limits,previous-previous_limits])
        high = np.minimum.reduce([np.degrees(self.kin.high),actual+limits,previous+previous_limits])
        if np.any(low>high):
            raise RuntimeError('实测角度与上一目标已超出共同允许范围；需要检查跟踪状态')
        cartesian_waiting = False
        if self.mode == 'joint':
            demand = controls*self.joint_speed*fine
            self.velocity += np.clip(demand-self.velocity,-30*dt,30*dt)
            # Integrate the bounded command reference, not the measured angle.
            # Small measured-relative steps can remain below a real driver's
            # deadband forever. Still bound lead against fresh feedback and
            # the previous goal, so a stuck motor cannot accumulate far goals.
            target = np.clip(self.joint_goal+np.clip(self.velocity*dt,-limits,limits),low,high)
            self.joint_goal = target.copy()
        else:
            y_sign = -1 if self.invert_y else 1
            pitch_sign = -1 if self.invert_pitch else 1
            demand = np.r_[np.array([-axes['left_y'],y_sign*axes['left_x'],-axes['right_y']])*self.linear_speed,
                           np.array([hat[0],pitch_sign*hat[1],axes['right_x']])*self.angular_speed]*fine
            if not self.allow_rotation:
                demand[3:] = 0
            accel = np.array([.1,.1,.1,.5,.5,.5])
            self.velocity += np.clip(demand-self.velocity,-accel*dt,accel*dt)
            if self.position_goal is None:
                self.position_goal = pose[:3,3].copy()
                self.orientation = pose[:3,:3].copy()
            # The historical real-arm controller (3e30c52) starts each XYZ
            # increment at measured FK. Joint deadband integration was later
            # extended to Cartesian goals without a hardware comparison.
            reference = pose[:3,3] if self.cartesian_reference=='measured' else self.position_goal
            position = reference+self.velocity[:3]*dt
            self.diagnostics = {'cartesian_reference':self.cartesian_reference,
                                'measured_joints_deg':actual.tolist(),
                                'measured_tcp_m':pose[:3,3].tolist(),
                                'reference_tcp_m':reference.tolist(),
                                'requested_tcp_m':position.tolist(),
                                'previous_sent_joints_deg':previous.tolist(),
                                'joint_bounds_deg':np.c_[low,high].tolist()}
            if np.max(np.abs(position-self.origin))>self.radius:
                self._hold(); self.message='到达相对工作区边界；请反向移动。'
                return None
            # A valid Cartesian increment can still require more than the
            # permitted joint step. Reduce the increment rather than alternating
            # between a full-speed goal and a hold during coupled rotations.
            answer = None
            for motion_scale in (1., .5, .25, .125, .0625, .03125, .015625, .0078125):
                position = reference+self.velocity[:3]*dt*motion_scale
                angular = self.velocity[3:]*dt*motion_scale
                next_orientation = self.orientation.copy()
                if np.linalg.norm(angular)>1e-12:
                    next_orientation = rotation(angular,np.linalg.norm(angular)) @ next_orientation
                answer = self.kin.inverse(position,next_orientation,q,np.radians(limits),
                                          joint_bounds=np.radians(np.c_[low,high]))
                if answer is not None:
                    break
            if answer is None:
                # A lagging motor can temporarily exhaust the allowed lead.
                # Retracting to measured q here cancels the pending movement
                # and creates a move/retract oscillation. Leave the successful
                # bounded goal in place and retry one increment next tick;
                # do not accumulate unaccepted Cartesian demand.
                target = np.clip(previous,low,high)
                cartesian_waiting = True
                self.cartesian_scale = 0.
            else:
                self.orientation = next_orientation
                self.position_goal = position.copy()
                target = np.degrees(answer)
                self.cartesian_scale = motion_scale
            self.diagnostics.update(accepted=not cartesian_waiting,scale=self.cartesian_scale,
                                    target_joints_deg=target.tolist())
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
                self.last_proposed_joints = self.robot.send_joint_action(target.tolist())
                self.sent_joints = True
            if jaw_target is not None:
                self.robot.command_gripper(jaw_target)
        else:
            self.last_proposed_joints = target.tolist()
        self.message = '按住 LB 遥操作中；RB 精细速度。' if self.execute else '只读预览：显示拟发送目标，未发送运动指令。'
        if cartesian_waiting:
            self.message = '笛卡尔增量暂受跟踪/步长/姿态约束；保留上一有效目标，等待反馈，未累积新目标。可反向或用 RB 精细档。'
        elif self.mode == 'cartesian' and self.cartesian_scale < 1.:
            self.message = f'末端增量受限，本周期采用 {self.cartesian_scale:.1%}；跟踪延迟或当前姿态会限制速度，未放宽关节限制。'
        return event
