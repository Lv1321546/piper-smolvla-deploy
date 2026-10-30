"""Synthetic gamepad/CAN tests. No real controller or robot is accessed."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from e14.cameras import CameraReader
from e14.export_dataset import load_raw_episode, build_samples
from e14.gamepad import PadState, PygamePad, deadzone
from e14.gamepad_control import GamepadControl
from e14.gamepad_kinematics import PiperKinematics, rotation
from e14.gamepad_teleop import GamepadSession, main
from e14.piper_robot import PiperRobot, MockBackend
from tests.test_teaching_ui import Clock, FakeDashboard

ROOT=Path(__file__).resolve().parents[1]


class ModelInstallerTests(unittest.TestCase):
    def test_cached_install_verifies_assets_and_does_not_download_again(self):
        from e14.install_mujoco_model import install,FILES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('e14.install_mujoco_model.urllib.request.urlopen',
                       side_effect=lambda url,timeout:io.BytesIO(url.encode())) as fetch:
                scene=install(root)
                self.assertTrue(scene.is_file())
                self.assertEqual(fetch.call_count,len(FILES))
                fetch.reset_mock();self.assertEqual(install(root),scene)
                fetch.assert_not_called()
                (root/'LICENSE').write_text('modified')
                with self.assertRaisesRegex(ValueError,'changed'):install(root)
                fetch.assert_not_called()

    def test_install_does_not_overwrite_a_different_existing_asset(self):
        from e14.install_mujoco_model import install
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'LICENSE').write_text('existing license')
            with patch('e14.install_mujoco_model.urllib.request.urlopen',
                       side_effect=lambda url,timeout:io.BytesIO(url.encode())):
                with self.assertRaises(FileExistsError):install(root)
            self.assertEqual((root/'LICENSE').read_text(),'existing license')


class JoystickAdapterTests(unittest.TestCase):
    def make_pad(self):
        # Match pygame.joystick.Joystick's public interface: intentionally no
        # get_attached(), which belongs to pygame._sdl2.controller.Controller.
        joy=SimpleNamespace(get_init=lambda:True,init=lambda:None,
                            get_name=lambda:'test DS4',get_guid=lambda:'ds4',
                            get_instance_id=lambda:7,get_numaxes=lambda:6,
                            get_numbuttons=lambda:13,get_numhats=lambda:1,
                            get_axis=lambda i:[0.,0.,-1.,0.,0.,-1.][i],
                            get_button=lambda i:i==9,get_hat=lambda i:(0,0))
        events=[]
        pg=SimpleNamespace(error=RuntimeError,JOYDEVICEREMOVED=1,
                           event=SimpleNamespace(get=lambda:list(events),pump=lambda:None),
                           joystick=SimpleNamespace(get_count=lambda:1,Joystick=lambda i:joy))
        pad=PygamePad.__new__(PygamePad)
        pad.pg=pg;pad.clock=lambda:1.;pad.joy=None;pad.index=0
        pad.guid=pad.selected_guid=None;pad.zone=.12
        pad.trigger_min=-1.;pad.trigger_max=1.
        profile=json.loads((ROOT/'gamepad.ds4.example.json').read_text())['gamepad']
        pad.axis_map=profile['axis_map'];pad.button_map=profile['button_map']
        return pad,events,pg

    def test_repeated_polls_use_joystick_api_and_ds4_stop_mapping(self):
        pad,_,_=self.make_pad()
        first=pad.poll();second=pad.poll()
        self.assertTrue(first.connected and second.connected)
        self.assertTrue(second.buttons['start'])
        self.assertFalse(second.buttons['back'])
        self.assertEqual(second.axes['left_trigger'],0.)
        self.assertEqual(second.axes['right_trigger'],0.)

    def test_removal_event_returns_disconnected_instead_of_stale_input(self):
        pad,events,pg=self.make_pad()
        self.assertTrue(pad.poll().connected)
        pg.joystick.get_count=lambda:0
        events.append(SimpleNamespace(type=1,instance_id=7))
        self.assertFalse(pad.poll().connected)


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        config=json.loads((ROOT/'config.mock.json').read_text())
        self.kin=PiperKinematics(config['robot']['joint_limits_deg'])
        self.q=np.radians([12.,75.,-75.,10.,45.,-15.])

    def test_jacobian_matches_independent_finite_differences(self):
        pose,j=self.kin.forward(self.q,jacobian=True)
        eps=1e-6
        for index in range(6):
            q=self.q.copy();q[index]+=eps
            other=self.kin.forward(q)
            np.testing.assert_allclose((other[:3,3]-pose[:3,3])/eps,j[:3,index],atol=1e-6)
            dr=other[:3,:3]@pose[:3,:3].T
            angular=np.array([dr[2,1]-dr[1,2],dr[0,2]-dr[2,0],dr[1,0]-dr[0,1]])/(2*eps)
            np.testing.assert_allclose(angular,j[3:,index],atol=1e-6)

    def test_reachable_xyz_targets_preserve_orientation_and_step_bounds(self):
        pose=self.kin.forward(self.q)
        delta=np.radians(np.ones(6)*.75)
        for xyz in np.eye(3)*.0005:
            result=self.kin.inverse(pose[:3,3]+xyz,pose[:3,:3],self.q,delta)
            self.assertIsNotNone(result)
            self.assertTrue(np.all(abs(result-self.q)<=delta+1e-12))
            np.testing.assert_allclose(self.kin.forward(result)[:3,3],pose[:3,3]+xyz,atol=8e-5)
            np.testing.assert_allclose(self.kin.forward(result)[:3,:3],pose[:3,:3],atol=.001)

    def test_unreachable_and_half_turn_targets_do_not_return_false_solutions(self):
        pose=self.kin.forward(self.q)
        self.assertIsNone(self.kin.inverse([5,5,5],pose[:3,:3],self.q,np.ones(6)*.01))
        self.assertIsNone(self.kin.inverse(pose[:3,3],rotation([1,0,0],np.pi)@pose[:3,:3],self.q,np.ones(6)*.01))
        with self.assertRaises(ValueError):
            self.kin.inverse(pose[:3,3],np.zeros((3,3)),self.q,np.ones(6)*.01)

    def test_inverse_respects_an_additional_previous_target_window(self):
        destination=self.q.copy();destination[0]+=.008
        pose=self.kin.forward(destination)
        bounds=np.c_[self.q-.02,self.q+.02]
        bounds[0,1]=self.q[0]+.004
        result=self.kin.inverse(pose[:3,3],pose[:3,:3],self.q,np.ones(6)*.02,joint_bounds=bounds)
        self.assertIsNone(result)
        bounds[0,1]=self.q[0]+.012
        result=self.kin.inverse(pose[:3,3],pose[:3,:3],self.q,np.ones(6)*.02,joint_bounds=bounds)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result>=bounds[:,0]-1e-12))
        self.assertTrue(np.all(result<=bounds[:,1]+1e-12))

    def test_mujoco_joint_mirror_matches_urdf_including_jaw(self):
        from e14.gamepad_sim import MujocoMirror
        mirror=MujocoMirror(headless=True)
        self.addCleanup(mirror.close)
        mirror.update(np.r_[np.degrees(self.q),.06])
        np.testing.assert_allclose(mirror.data.site('flange').xpos,self.kin.forward(self.q)[:3,3],atol=1e-12)
        self.assertAlmostEqual(mirror.data.joint('joint7').qpos[0],.03)
        self.assertAlmostEqual(mirror.data.joint('joint8').qpos[0],-.03)
        before=mirror.data.body('link7').xpos.copy()
        mirror.update(np.r_[np.degrees(self.q),.02])
        self.assertGreater(np.linalg.norm(before-mirror.data.body('link7').xpos),.01)

    def test_preview_model_jacobian_matches_finite_differences(self):
        from e14.gamepad_sim import MujocoMirror
        mirror=MujocoMirror(headless=True)
        self.addCleanup(mirror.close)
        limits=np.degrees(np.c_[self.kin.low,self.kin.high])
        kin=mirror.kinematics(limits)
        pose,j=kin.forward(self.q,jacobian=True)
        np.testing.assert_allclose(pose,self.kin.forward(self.q),atol=1e-12)
        for i in range(6):
            q=self.q.copy();q[i]+=1e-6
            other=kin.forward(q)
            np.testing.assert_allclose((other[:3,3]-pose[:3,3])/1e-6,j[:3,i],atol=1e-6)
        result=kin.inverse(pose[:3,3]+[.0005,0,0],pose[:3,:3],self.q,np.ones(6)*.01)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(kin.forward(result)[:3,3],pose[:3,3]+[.0005,0,0],atol=8e-5)


class GamepadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.clock=Clock()
        self.config=json.loads((ROOT/'config.mock.json').read_text())
        self.config['robot']['enable_gripper']=False
        self.config['cameras']['front'].update(width=16,height=12)
        self.backend=MockBackend(self.clock)
        self.backend.state=[0.,75.,-75.,0.,45.,0.,.06]
        self.robot=PiperRobot(self.config,mock=True,backend=self.backend,clock=self.clock)
        self.robot.connect()
        original=self.backend.mode_control
        def mode(speed):
            self.clock.sleep(.001);original(speed)
        self.backend.mode_control=mode
        self.robot.prepare_control(joint_only=True)
        self.backend.commands.clear()
        self.core=GamepadControl(self.robot,execute=True,clock=self.clock)

    def tearDown(self):
        self.robot.disconnect();self.tmp.cleanup()

    def state(self, *, lb=False, **axes):
        state=PadState(True,self.clock(),'test','mock',1)
        state.buttons['lb']=lb;state.axes.update(axes)
        return state

    def advance(self, **kwargs):
        self.clock.sleep(.05)
        return self.core.tick(self.state(**kwargs))

    def activate(self):
        self.core.tick(self.state())
        self.advance(lb=True)

    def test_deadzone_is_continuous_and_normalized(self):
        self.assertEqual(deadzone(.05),0)
        self.assertEqual(deadzone(-.05),0)
        self.assertAlmostEqual(deadzone(.56),.5)
        self.assertEqual(deadzone(1),1)
        with self.assertRaises(ValueError):deadzone(float('nan'))

    def test_startup_held_lb_requires_neutral_release(self):
        self.core.tick(self.state(lb=True,left_y=-.8))
        self.assertFalse(self.core.active)
        self.assertFalse(self.backend.commands)
        self.advance(lb=False,left_y=-.8)
        self.advance(lb=True,left_y=-.8)
        self.assertFalse(self.core.active)
        self.advance(lb=False)
        self.advance(lb=True,left_y=-.3)
        self.assertTrue(self.core.active)
        self.assertTrue(any(c[0]=='joint' for c in self.backend.commands))

    def test_cartesian_input_moves_the_requested_axis_with_bounded_joints(self):
        self.activate()
        before=self.robot.get_feedback()['state']
        pose=self.core.kin.forward(np.radians(before[:6]))
        event=self.advance(lb=True,left_y=-.5)
        after=self.robot.get_feedback()['state']
        moved=self.core.kin.forward(np.radians(after[:6]))
        self.assertGreater(moved[0,3],pose[0,3])
        self.assertTrue(all(abs(a-b)<=1 for a,b in zip(before[:6],after[:6])))
        self.assertTrue(event['execute'])
        self.assertFalse(any(c[0] in {'mode','enable','gripper'} for c in self.backend.commands))

    def test_left_stick_right_restored_toward_negative_y(self):
        self.activate()
        before=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.advance(lb=True,left_x=.5)
        after=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertLess(after[1,3],before[1,3]-.0001)
        np.testing.assert_allclose(after[[0,2],3],before[[0,2],3],atol=2e-5)

    def test_real_profile_cruise_speed_is_three_centimetres_per_second(self):
        self.assertAlmostEqual(self.core.linear_speed,.12)
        self.core.linear_speed=.03
        self.activate()
        for _ in range(12):self.advance(lb=True,left_y=-1.)
        before=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.advance(lb=True,left_y=-1.)
        after=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertAlmostEqual((after[0,3]-before[0,3])/.05,.03,delta=.001)

    def assert_rotation_input(self, expected_axis, *, hat=(0,0), direction=1, **axes):
        self.activate()
        before=self.core.kin.forward(np.radians(self.backend.state[:6]))
        state=self.state(lb=True,**axes);state.hat=hat
        self.clock.sleep(.05);self.core.tick(state)
        after=self.core.kin.forward(np.radians(self.backend.state[:6]))
        dr=after[:3,:3]@before[:3,:3].T
        angular=np.array([dr[2,1]-dr[1,2],dr[0,2]-dr[2,0],dr[1,0]-dr[0,1]])/2
        self.assertGreater(direction*angular[expected_axis],.0008)
        np.testing.assert_allclose(np.delete(angular,expected_axis),0,atol=6e-5)
        np.testing.assert_allclose(after[:3,3],before[:3,3],atol=2e-5)

    def test_right_stick_horizontal_controls_yaw(self):
        self.assert_rotation_input(2,right_x=1.)

    def test_dpad_up_controls_pitch(self):
        self.assert_rotation_input(1,hat=(0,1),direction=-1)

    def test_dpad_down_controls_positive_pitch(self):
        self.assert_rotation_input(1,hat=(0,-1),direction=1)

    def test_dpad_right_controls_roll(self):
        self.assert_rotation_input(0,hat=(1,0),direction=1)

    def test_right_stick_up_restored_toward_positive_z(self):
        self.activate()
        before=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.advance(lb=True,right_y=-.5)
        after=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertGreater(after[2,3],before[2,3]+.0001)
        np.testing.assert_allclose(after[:2,3],before[:2,3],atol=2e-5)

    def test_rotation_can_still_be_disabled_in_config(self):
        self.core=GamepadControl(self.robot,{'allow_rotation':False},execute=True,clock=self.clock)
        self.activate()
        state=self.state(lb=True,right_x=1.);state.hat=(1,1)
        self.clock.sleep(.05);self.core.tick(state)
        self.assertFalse(self.backend.commands)

    def test_rotation_reduces_cartesian_step_to_keep_joint_motion_bounded(self):
        self.core=GamepadControl(self.robot,{'angular_speed_rad_s':.4},execute=True,clock=self.clock)
        self.activate()
        start=self.core.kin.forward(np.radians(self.backend.state[:6]))
        emitted=0
        for _ in range(15):
            before=list(self.backend.state[:6])
            if self.advance(lb=True,right_x=1.) is not None:emitted+=1
            self.assertTrue(all(abs(a-b)<=.751 for a,b in zip(before,self.backend.state[:6])))
        end=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertEqual(emitted,15)
        np.testing.assert_allclose(end[:3,3],start[:3,3],atol=1e-4)
        self.assertFalse(self.core.faulted)

    def test_cartesian_reversals_with_motor_lag_bound_both_feedback_and_previous_target(self):
        # This seed/lag profile reproduced a 1.023-degree previous-goal jump
        # while every joint was within 0.75 degrees of measured feedback.
        self.backend.state=[-13.545,18.499,-19.196,7.014,44.821,-10.388,.0003]
        self.backend.refresh()
        self.core=GamepadControl(self.robot,{'linear_speed_m_s':.03,'angular_speed_rad_s':.15},
                                 execute=True,clock=self.clock)
        desired=np.array(self.backend.state[:6],float)
        original_refresh=self.backend.refresh
        last=[self.clock()]
        sent=[]
        def refresh():
            if self.clock()>last[0]:
                q=np.array(self.backend.state[:6])
                self.backend.state[:6]=(q+np.clip((desired-q)*.35,-.5,.5)).tolist()
                last[0]=self.clock()
            original_refresh()
        def joint(raw):
            target=np.array(raw)/1000
            actual=np.array(self.backend.state[:6])
            self.assertTrue(np.all(np.abs(target-actual)<=1.+1e-9))
            if sent:self.assertTrue(np.all(np.abs(target-sent[-1])<=1.+1e-9))
            sent.append(target.copy());desired[:]=target
        self.backend.refresh=refresh;self.backend.joint=joint
        self.activate()
        random=np.random.default_rng(19)
        for tick in range(350):
            if tick%5==0:inputs=random.choice([-1.,0.,1.],size=6)
            self.clock.sleep(.05)
            state=self.state(lb=True,left_x=inputs[0],left_y=inputs[1],right_y=inputs[2],right_x=inputs[3])
            state.hat=tuple(inputs[4:].astype(int))
            self.core.tick(state)
        self.assertGreater(len(sent),100)
        self.assertFalse(self.core.faulted)
        self.advance(lb=False)

    def test_sender_still_rejects_excessive_step_and_tracking_with_joint_details(self):
        self.backend.joint=lambda raw:self.backend.commands.append(('joint',list(raw)))
        initial=self.backend.state[:6].copy()
        first=initial.copy();first[0]=.7
        self.robot.send_joint_action(first)
        copy_of_target=self.robot.last_joint_target;copy_of_target[0]=99.
        self.assertEqual(self.robot.last_joint_target,first)
        self.clock.sleep(.05)
        target=initial.copy();target[0]=-.6
        before=len(self.backend.commands)
        with self.assertRaisesRegex(ValueError,'J1: 上一目标的步长'):
            self.robot.send_joint_action(target)
        self.assertEqual(len(self.backend.commands),before)
        self.backend.state[0]=2.;self.backend.refresh()
        with self.assertRaisesRegex(ValueError,'J1: 实测角度的跟踪差'):
            self.robot.send_joint_action(first)
        self.assertEqual(len(self.backend.commands),before)

    def test_limited_cartesian_step_retains_pending_target_instead_of_retracting(self):
        self.backend.joint=lambda raw:self.backend.commands.append(('joint',list(raw)))
        self.core.linear_speed=.03;self.activate()
        self.advance(lb=True,left_y=-1.)
        pending=self.robot.last_joint_target
        reference=self.core.position_goal.copy()
        # Exhausted lead/unreachable increment is not an LB release. The
        # pending goal must survive, without growing a queue of future poses.
        with patch.object(self.core.kin,'inverse',return_value=None):
            for _ in range(20):
                self.advance(lb=True,left_y=-1.)
                np.testing.assert_allclose(self.robot.last_joint_target,pending,atol=1e-12)
                np.testing.assert_allclose(self.core.position_goal,reference,atol=1e-12)
                self.assertEqual(self.core.cartesian_scale,0.)
        self.assertFalse(self.core.faulted)
        self.assertGreater(self.core.velocity[0],0)
        self.advance(lb=False)
        self.assertIsNone(self.core.position_goal)
        np.testing.assert_allclose(self.robot.last_joint_target,self.backend.state[:6],atol=1e-12)

    def test_near_singular_y_motion_uses_smaller_steps_and_makes_monotone_progress(self):
        initial=[-6.282,16.686,-15.465,3.904,27.699,1.874,.0004]
        self.backend.state=initial.copy();self.backend.refresh()
        self.core=GamepadControl(self.robot,{'linear_speed_m_s':.03},execute=True,clock=self.clock)
        desired=np.array(initial[:6]);last=[self.clock()]
        refresh_original=self.backend.refresh
        sent=[];positions=[];scales=[]
        def refresh():
            if self.clock()>last[0]:
                q=np.array(self.backend.state[:6])
                self.backend.state[:6]=(q+np.clip((desired-q)*.35,-.5,.5)).tolist()
                last[0]=self.clock()
            refresh_original()
        def joint(raw):
            target=np.array(raw)/1000
            self.assertTrue(np.all(np.abs(target-self.backend.state[:6])<=1.+1e-9))
            if sent:self.assertTrue(np.all(np.abs(target-sent[-1])<=1.+1e-9))
            sent.append(target.copy());desired[:]=target
        self.backend.refresh=refresh;self.backend.joint=joint
        self.activate()
        start=self.core.kin.forward(np.radians(initial[:6]))
        for _ in range(100):
            self.advance(lb=True,left_x=1.)
            positions.append(self.core.kin.forward(np.radians(self.backend.state[:6]))[:3,3])
            scales.append(self.core.cartesian_scale)
        positions=np.array(positions)
        self.assertLess(positions[-1,1]-start[1,3],-.002)
        self.assertLess(np.max(np.diff(positions[:,1])),1e-5)
        np.testing.assert_allclose(positions[:,[0,2]],np.tile(start[[0,2],3],(100,1)),atol=5e-5)
        final=self.core.kin.forward(np.radians(self.backend.state[:6]))
        np.testing.assert_allclose(final[:3,:3],start[:3,:3],atol=2e-4)
        self.assertTrue(any(0.<scale<.125 for scale in scales))
        self.assertFalse(self.core.faulted)

    def test_delayed_x_motion_in_a_bent_pose_continues_without_retracting(self):
        self.core.linear_speed=.03
        desired=np.array(self.backend.state[:6]);last=[self.clock()]
        refresh_original=self.backend.refresh
        def refresh():
            if self.clock()>last[0]:
                q=np.array(self.backend.state[:6])
                self.backend.state[:6]=(q+np.clip((desired-q)*.2,-.5,.5)).tolist()
                last[0]=self.clock()
            refresh_original()
        self.backend.refresh=refresh
        self.backend.joint=lambda raw:desired.__setitem__(slice(None),np.array(raw)/1000)
        self.activate()
        start=self.core.kin.forward(np.radians(self.backend.state[:6]))[:3,3]
        positions=[]
        for _ in range(60):
            self.advance(lb=True,left_y=-1.)
            positions.append(self.core.kin.forward(np.radians(self.backend.state[:6]))[:3,3])
        positions=np.array(positions)
        self.assertGreater(positions[-1,0]-start[0],.02)
        self.assertGreater(np.min(np.diff(positions[:,0])),-1e-5)
        np.testing.assert_allclose(positions[:,1:],np.tile(start[1:],(60,1)),atol=1e-4)
        self.assertFalse(self.core.faulted)

    def test_joint_mode_preserves_other_joints(self):
        self.core.mode='joint';self.activate()
        before=self.robot.get_feedback()['state']
        self.advance(lb=True,left_x=.5)
        after=self.robot.get_feedback()['state']
        self.assertGreater(after[0],before[0])
        self.assertEqual(after[1:],before[1:])

    def use_motor_deadband(self, degrees=.08):
        def joint(raw):
            self.backend.commands.append(('joint',list(raw)))
            for i,value in enumerate(raw):
                target=value/1000
                if abs(target-self.backend.state[i])>degrees:
                    self.backend.state[i]=target
            self.backend.refresh()
        self.backend.joint=joint

    def test_slow_joint_input_crosses_driver_deadband(self):
        self.use_motor_deadband()
        self.core.mode='joint';self.core.joint_speed=1.;self.activate()
        initial=self.backend.state.copy()
        for _ in range(10):self.advance(lb=True,left_x=1.)
        self.assertGreater(self.backend.state[0]-initial[0],.3)
        self.assertEqual(self.backend.state[1:],initial[1:])

    def test_stalled_joint_reference_is_bounded_and_cleared_on_release(self):
        self.backend.joint=lambda raw:self.backend.commands.append(('joint',list(raw)))
        self.core.mode='joint';self.core.joint_speed=1.;self.activate()
        for _ in range(40):self.advance(lb=True,left_x=1.)
        targets=[c[1][0]/1000 for c in self.backend.commands if c[0]=='joint']
        self.assertGreater(max(targets),.7)
        self.assertLessEqual(max(abs(q) for q in targets),.75)
        self.advance(lb=False)
        self.assertIsNone(self.core.joint_goal)
        self.advance(lb=True,left_x=1.)
        self.assertLessEqual(abs(self.backend.commands[-1][1][0]),50)

    def test_last_four_reversals_are_undone_in_joint_mode(self):
        self.core.mode='joint';self.activate()
        for kwargs,hat,index in [({'left_x':1.},(0,0),0),({'right_y':-1.},(0,0),2),
                                  ({},(1,0),3),({},(0,1),4)]:
            self.advance(lb=False)
            before=self.backend.state[:6].copy()
            state=self.state(lb=True,**kwargs);state.hat=hat
            self.clock.sleep(.05);self.core.tick(state)
            self.assertGreater(self.backend.state[index],before[index])
            np.testing.assert_allclose(np.delete(self.backend.state[:6],index),np.delete(before,index),atol=1e-12)

    def test_cartesian_front_back_holds_height_but_joint2_motion_does_not(self):
        initial=self.backend.state.copy()
        start=self.core.kin.forward(np.radians(initial[:6]))
        self.activate()
        heights=[]
        for _ in range(12):
            self.advance(lb=True,left_y=-1.)
            heights.append(self.core.kin.forward(np.radians(self.backend.state[:6]))[2,3])
        end=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertGreater(end[0,3]-start[0,3],.002)
        self.assertLess(max(abs(z-start[2,3]) for z in heights),.0001)
        self.advance(lb=False)
        self.backend.state=initial.copy();self.backend.refresh()
        self.robot._last_target=self.robot._last_sent=None
        self.core=GamepadControl(self.robot,execute=True,mode='joint',clock=self.clock)
        self.activate()
        for _ in range(12):self.advance(lb=True,left_y=-1.)
        end=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertGreater(abs(end[2,3]-start[2,3]),.001)

    def test_gripper_speed_is_fifty_mm_per_second_with_matching_step_limit(self):
        self.assertAlmostEqual(self.core.grip_speed,.05)
        self.activate()
        before=self.backend.state[6]
        self.advance(lb=True,left_trigger=1.)
        self.assertAlmostEqual(before-self.backend.state[6],.0025)
        self.assertFalse(self.core.faulted)

    def test_gripper_step_mismatch_and_invalid_directions_fail_before_control(self):
        self.robot.settings['max_step_delta'][6]=.002
        with self.assertRaisesRegex(ValueError,'max_step_delta'):
            GamepadControl(self.robot)
        for directions in [{'left_x':True},{'hat_x':0},{'unknown':1}]:
            with self.assertRaisesRegex(ValueError,'input_directions'):
                GamepadControl(self.robot,{'input_directions':directions})

    def test_slow_cartesian_reference_crosses_driver_deadband(self):
        self.use_motor_deadband()
        self.core.linear_speed=.0015;self.activate()
        initial=self.core.kin.forward(np.radians(self.backend.state[:6]))
        for _ in range(40):self.advance(lb=True,left_y=-1.)
        final=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertGreater(final[0,3]-initial[0,3],.001)
        self.assertFalse(self.core.faulted)
        self.advance(lb=False)
        self.assertIsNone(self.core.position_goal)

    def test_fine_speed_does_not_discard_small_stick_motion(self):
        self.activate()
        before=self.core.kin.forward(np.radians(self.backend.state[:6]))
        state=self.state(lb=True,left_y=-.2);state.buttons['rb']=True
        self.clock.sleep(.05);self.core.tick(state)
        after=self.core.kin.forward(np.radians(self.backend.state[:6]))
        self.assertGreater(after[0,3]-before[0,3],.00002)
        self.assertTrue(any(c[0]=='joint' for c in self.backend.commands))

    def test_read_only_preview_never_sends_even_gripper_commands(self):
        self.core.execute=False
        self.activate()
        for _ in range(5):self.advance(lb=True,left_y=-.5,right_trigger=1.)
        self.assertFalse(self.backend.commands)
        self.assertIsNotNone(self.core.last_proposal)
        self.core.stop('quit')
        self.assertFalse(self.backend.commands)

    def test_release_lb_holds_once_then_sends_no_more_goals(self):
        self.activate();self.advance(lb=True,left_y=-.5)
        self.advance(lb=False)
        commands=list(self.backend.commands)
        self.advance(lb=False);self.advance(lb=False)
        self.assertEqual(commands,self.backend.commands)
        self.assertFalse(self.core.active)
        self.assertFalse(any(c[0]=='emergency_stop' for c in commands))

    def test_disconnect_after_motion_stops_and_never_auto_rearms(self):
        self.activate();self.advance(lb=True,left_y=-.5)
        self.clock.sleep(.05)
        self.core.tick(PadState(timestamp=self.clock()))
        self.assertTrue(self.core.faulted)
        self.assertIn(('emergency_stop',),self.backend.commands)
        before=list(self.backend.commands)
        self.advance(lb=False);self.advance(lb=True,left_y=-.5)
        self.assertEqual(before,self.backend.commands)
        self.core.stop('repeated fault')
        self.assertEqual(before,self.backend.commands)

    def test_stale_input_and_delayed_loop_latch_stop(self):
        self.activate();self.advance(lb=True,left_y=-.5)
        stale=self.state(lb=True);stale.timestamp-=.5
        self.clock.sleep(.05);self.core.tick(stale)
        self.assertTrue(self.core.faulted)
        self.assertIn(('emergency_stop',),self.backend.commands)

    def test_device_swap_during_motion_latches_stop(self):
        self.activate();self.advance(lb=True,left_y=-.5)
        changed=self.state(lb=True);changed.instance=2
        self.clock.sleep(.05);self.core.tick(changed)
        self.assertTrue(self.core.faulted)

    def test_manual_gripper_works_without_policy_gripper_and_never_opens_on_exit(self):
        self.activate()
        self.advance(lb=True,left_trigger=1.)
        self.assertLess(self.backend.state[6],.06)
        self.assertTrue(all(c[0]=='gripper' for c in self.backend.commands))
        before=list(self.backend.commands)
        self.core.stop('quit')
        self.assertEqual(before,self.backend.commands)

    def test_joint_only_api_does_not_turn_noisy_gripper_feedback_into_a_goal(self):
        self.backend.state[6]=.1006;self.backend.refresh()
        target=self.backend.state[:6];target[0]+=.1
        self.robot.send_joint_action(target)
        self.assertEqual(self.backend.commands,[('joint',[100,75000,-75000,0,45000,0])])
        self.assertEqual(self.backend.state[6],.1006)
        with self.assertRaises(ValueError):self.robot.send_joint_action([15,75,-75,0,45,0])

    def test_real_prepare_keeps_existing_commissioning_gate(self):
        self.robot.mock=False;self.robot.settings['stop_strategy_verified']=False
        with self.assertRaisesRegex(RuntimeError,'stop_strategy_verified'):
            self.robot.prepare_control(joint_only=True)
        self.assertFalse(self.backend.commands)

    def test_preview_kinematics_cannot_replace_real_control_kinematics(self):
        self.robot.mock=False
        with self.assertRaisesRegex(ValueError,'mock'):
            GamepadControl(self.robot,kinematics=self.core.kin)

    def make_session(self, mode='cartesian'):
        class Pad:
            def __init__(inner):inner.current=None
            def poll(inner):
                s=copy.deepcopy(inner.current) if inner.current else self.state()
                s.timestamp=self.clock();return s
        pad=Pad()
        cameras=CameraReader(self.config['cameras'],mock=True,clock=self.clock);cameras.connect()
        self.addCleanup(cameras.close)
        dashboard=FakeDashboard()
        args=SimpleNamespace(open_width=.06,close_width=0.,gripper_speed=.02,duration=.5,task='Pick foam',
                             home_file=self.root/'home.json',output=self.root/'raw',mock=True,execute=False,
                             episodes=2,mode=mode,run_seconds=0)
        session=GamepadSession(args,self.robot,cameras,dashboard,pad,clock=self.clock,sleep=self.clock.sleep)
        return session,dashboard,pad,args

    def test_session_startup_and_held_record_button_do_not_autostart_or_move(self):
        session,dashboard,pad,args=self.make_session()
        pad.current=self.state();pad.current.buttons['x']=True
        session.poll()
        self.assertFalse(session.start_requested)
        self.assertFalse(self.backend.commands)
        self.assertFalse(args.output.exists())

    def test_dashboard_joint_mode_explains_arc_motion_and_shows_effective_directions(self):
        session,dashboard,_,_=self.make_session(mode='joint')
        session.poll()
        self.assertTrue(any('J2' in warning and 'cartesian' in warning for warning in dashboard.state['warnings']))
        self.assertEqual(dashboard.state['teleop_mapping']['左摇杆向右'],'J1+')
        self.assertEqual(dashboard.state['teleop_mapping']['右摇杆向上'],'J3+')

    def test_gripper_config_is_used_unless_cli_explicitly_overrides_it(self):
        profile=self.root/'gamepad.json'
        profile.write_text(json.dumps({'gamepad':{'gripper_speed_m_s':.03}}))
        for extra,expected in [([],.03),(['--gripper-speed','.04'],.04)]:
            with patch('e14.gamepad_teleop.GamepadControl',wraps=GamepadControl) as core, \
                 patch('e14.gamepad_teleop.PiperRobot.connect',side_effect=RuntimeError('before hardware')), \
                 patch('e14.gamepad_teleop.PygamePad'):
                with self.assertRaisesRegex(RuntimeError,'before hardware'):
                    main(['--config',str(ROOT/'config.mock.json'),'--gamepad-config',str(profile),'--mock',*extra])
                self.assertAlmostEqual(core.call_args.args[1]['gripper_speed_m_s'],expected)

    def test_session_spacing_includes_solver_and_send_time(self):
        session,dashboard,pad,args=self.make_session()
        original=session.core.tick
        with patch.object(session.core,'tick',side_effect=lambda s:(self.clock.sleep(.005),original(s))[1]) as tick:
            session.poll()
            self.clock.sleep(.036)
            session.poll()
            self.assertEqual(tick.call_count,1)
            self.clock.sleep(.005)
            session.poll()
            self.assertEqual(tick.call_count,2)

    def test_gamepad_recording_remains_next_state_exportable_with_audit(self):
        session,dashboard,pad,args=self.make_session()
        session.poll()
        self.clock.sleep(.05)
        pad.current=self.state(lb=True,left_y=-.3,left_trigger=.2)
        session.poll()
        dashboard.scheduled=[[],[],[],['success']]
        session.record()
        path=args.output/'episode_000000'
        episode=load_raw_episode(path)
        samples=list(build_samples(episode))
        self.assertEqual(episode['manifest']['task_outcome'],'success')
        self.assertEqual(episode['manifest']['operator_session']['mode'],'gamepad')
        self.assertEqual(session.completed,1)
        np.testing.assert_allclose(samples[0]['action'],episode['rows'][1]['state'])
        events=[json.loads(v) for v in (path/'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(v['event']=='gamepad_target' for v in events))

    def test_recording_failure_stops_control_and_does_not_become_success(self):
        session,dashboard,pad,args=self.make_session()
        session.poll()
        self.backend.frozen=True;self.clock.sleep(.4)
        session.record()
        manifest=json.loads((args.output/'episode_000000/manifest.json').read_text())
        self.assertEqual(manifest['status'],'failed')
        self.assertTrue(session.core.faulted)
        self.assertEqual(session.completed,0)

    def test_synthetic_input_cannot_drive_real_hardware(self):
        with self.assertRaises(SystemExit):main(['--config','missing.json','--demo','--execute'])

    def test_bad_arguments_fail_before_robot_or_pad_connection(self):
        with patch('e14.gamepad_teleop.PiperRobot.connect') as connect, patch('e14.gamepad_teleop.PygamePad') as pad:
            for options in [['--duration','nan'],['--open-width','1'],['--gripper-speed','-1']]:
                with self.assertRaises((SystemExit,ValueError)):
                    main(['--config',str(ROOT/'config.mock.json'),'--execute',*options])
            connect.assert_not_called();pad.assert_not_called()


if __name__=='__main__':unittest.main()
