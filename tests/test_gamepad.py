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

    def test_joint_mode_preserves_other_joints(self):
        self.core.mode='joint';self.activate()
        before=self.robot.get_feedback()['state']
        self.advance(lb=True,left_x=.5)
        after=self.robot.get_feedback()['state']
        self.assertGreater(after[0],before[0])
        self.assertEqual(after[1:],before[1:])

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

    def make_session(self):
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
                             episodes=2,mode='cartesian',run_seconds=0)
        session=GamepadSession(args,self.robot,cameras,dashboard,pad,clock=self.clock,sleep=self.clock.sleep)
        return session,dashboard,pad,args

    def test_session_startup_and_held_record_button_do_not_autostart_or_move(self):
        session,dashboard,pad,args=self.make_session()
        pad.current=self.state();pad.current.buttons['x']=True
        session.poll()
        self.assertFalse(session.start_requested)
        self.assertFalse(self.backend.commands)
        self.assertFalse(args.output.exists())

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
