"""Launcher takeover contracts; fake SDK and passive snapshots, no CAN writes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from servo_bridge.core import load_config
from servo_bridge.prepare_control import inspect_snapshot, main, prepare_control


ROOT = Path(__file__).resolve().parents[2]


class PrepareControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT/'servo_bridge/config.joint.example.json')
        self.now = 100.
        self.calls = []
        self.state = dict(state=[-5.6,90.2,-35.5,-17.4,51.2,-25.5,.04],
                          ctrl_mode=2,mode_feed=1,teach_status=2,arm_status=0,
                          err_code=0,enabled=[True]*6,motor_faults=[0]*6,gripper_faults=0)
        self.respond = True
        self.on_sleep = None
        owner = self
        class Arm:
            def ConnectPort(self, **kwargs):
                owner.calls.append(('connect', kwargs))
            def JointCtrl(self, *q):
                owner.calls.append(('joint', q))
            def MotionCtrl_2(self, *args):
                owner.calls.append(('mode', args))
                if owner.respond:
                    owner.state.update(ctrl_mode=args[0], mode_feed=args[1])
            def EnableArm(self, number):
                owner.calls.append(('enable', number))
                if owner.respond:
                    owner.state['enabled'] = [True]*6
            def DisconnectPort(self):
                owner.calls.append(('disconnect',))
        self.factory = Mock(return_value=Arm())

    def read(self):
        snap = deepcopy(self.state)
        snap.update(feedback_monotonic=self.now, frame_timestamps={'0x2a1':self.now})
        return snap

    def sleep(self, duration):
        self.now += duration
        if self.on_sleep:
            self.on_sleep()

    def run_prepare(self, **kwargs):
        return prepare_control(self.cfg, self.read, self.factory,
                               clock=lambda:self.now, sleep=self.sleep, **kwargs)

    def test_already_ready_never_constructs_sdk(self):
        self.state['ctrl_mode'] = 1
        self.assertFalse(self.run_prepare())
        self.factory.assert_not_called()

    def test_finished_teaching_preloads_current_goal_before_low_speed_can(self):
        self.assertTrue(self.run_prepare())
        self.assertEqual(self.calls[0], ('connect', dict(piper_init=False,start_thread=False)))
        self.assertEqual(self.calls[1], ('joint',(-5600,90200,-35500,-17400,51200,-25500)))
        self.assertEqual(self.calls[2], ('mode',(1,1,5,0)))
        self.assertEqual([x[1][2] for x in self.calls if x[0]=='mode'],[5,70])
        self.assertNotIn('enable',[x[0] for x in self.calls])
        self.assertEqual(self.calls[-1],('disconnect',))

    def test_standby_disabled_enables_only_six_arm_motors(self):
        self.state.update(ctrl_mode=0,teach_status=0,mode_feed=0,enabled=[False]*6)
        self.assertTrue(self.run_prepare())
        self.assertIn(('enable',7),self.calls)
        names=[x[0] for x in self.calls]
        self.assertLess(names.index('joint'),names.index('mode'))
        self.assertLess(names.index('mode'),names.index('enable'))
        self.assertFalse(set(names)-{'connect','joint','mode','enable','disconnect'})

    def test_recording_or_replaying_teaching_never_constructs_sdk(self):
        for teach in (1,3):
            self.state['teach_status']=teach
            with self.subTest(teach=teach),self.assertRaisesRegex(RuntimeError,'尚未结束'):
                self.run_prepare()
        self.factory.assert_not_called()

    def test_bad_feedback_faults_and_other_controller_never_construct_sdk(self):
        for change in (dict(ctrl_mode=3),dict(arm_status=1),dict(err_code=1),
                       dict(motor_faults=[0,0,1,0,0,0]),dict(gripper_faults=1),
                       dict(state=[-5.6,90.2,-35.5,-17.4,51.2,-25.5,.1025])):
            original=deepcopy(self.state)
            self.state.update(change)
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.run_prepare()
            self.state=original
        self.factory.assert_not_called()
        snapshot=self.read();snapshot['feedback_monotonic']-=.2
        with self.assertRaisesRegex(RuntimeError,'反馈已过期'):
            inspect_snapshot(snapshot,self.cfg,now=self.now)

    def test_timeout_does_not_restore_fast_mode(self):
        self.respond=False
        with self.assertRaisesRegex(RuntimeError,'超时'):self.run_prepare(timeout=.25)
        self.assertTrue(all(x[1][2]==5 for x in self.calls if x[0]=='mode'))
        self.assertEqual(self.calls[-1],('disconnect',))

    def test_old_status_frame_cannot_confirm_takeover(self):
        read=self.read
        def frozen_status():
            snap=read()
            snap['frame_timestamps']['0x2a1']=100.
            return snap
        with self.assertRaisesRegex(RuntimeError,'超时'):
            prepare_control(self.cfg,frozen_status,self.factory,clock=lambda:self.now,
                            sleep=self.sleep,timeout=.25)
        self.assertFalse(any(x[0]=='mode' and x[1][2]==70 for x in self.calls))

    def test_joint_movement_during_takeover_aborts_and_holds_current(self):
        self.on_sleep=lambda:self.state['state'].__setitem__(0,-3.6)
        with self.assertRaisesRegex(RuntimeError,'偏移超过'):self.run_prepare()
        joints=[x for x in self.calls if x[0]=='joint']
        self.assertEqual(joints[-1][1][0],-3600)
        self.assertFalse(any(x[0]=='mode' and x[1][2]==70 for x in self.calls))

    def test_jaw_change_aborts_without_any_jaw_command(self):
        self.on_sleep=lambda:self.state['state'].__setitem__(6,.05)
        with self.assertRaisesRegex(RuntimeError,'夹爪反馈变化'):self.run_prepare()
        self.assertFalse(any(x[0]=='gripper' for x in self.calls))

    def test_interruption_disconnects_and_holds_only_arm(self):
        def interrupted():raise KeyboardInterrupt()
        self.on_sleep=interrupted
        with self.assertRaises(KeyboardInterrupt):self.run_prepare()
        self.assertEqual(self.calls[-1],('disconnect',))
        self.assertEqual(self.calls[-2][0],'joint')

    def test_check_and_preflight_remain_passive(self):
        for mode in ('--check','--preflight'):
            with self.subTest(mode=mode):
                source=Mock()
                source.sample.return_value=(None,None,self.read())
                sdk=Mock()
                with patch('servo_bridge.sources.PassiveCan',return_value=source), \
                        patch.dict('sys.modules',piper_sdk=SimpleNamespace(C_PiperInterface_V2=sdk)), \
                        patch('servo_bridge.prepare_control.time.monotonic',return_value=self.now):
                    args=['--config',str(ROOT/'config.example.json'),'--profile',
                          str(ROOT/'servo_bridge/config.joint.example.json'),mode]
                    if mode=='--check':
                        with self.assertRaisesRegex(RuntimeError,'只读检查'):main(args)
                    else:main(args)
                sdk.assert_not_called()
                source.close.assert_called_once()


if __name__=='__main__':unittest.main()
