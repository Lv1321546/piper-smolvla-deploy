"""Virtual-clock replay tests: no CAN, SDK device, ROS, or camera access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from e14.collect_data import ACTION_NAMES
from e14.replay_joint import load_episode, timeline, replay, execute, main, alignment_plan
from servo_bridge.core import load_config
ACK = 'legacy ignored acknowledgement'

ROOT = Path(__file__).resolve().parents[1]


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.cfg = load_config(ROOT/'servo_bridge/config.joint.example.json')
        self.manifest = dict(format='piper-raw-v1', status='complete', state_names=ACTION_NAMES,
                             joint_unit='degree', gripper_unit='meter_full_opening', frames=3,
                             action_mode='absolute_position', label_semantics='next_state',
                             control_hz=20, mock=False, operator_session=dict(mode='moveit_servo_joint_gamepad'))
        self.rows = [dict(frame_index=i, captured_monotonic=100+t,
                         state=[t*10,90,-40,0,20,0,.02+t*.02]) for i,t in enumerate([0,.049,.1])]
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))
        (self.path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def test_irregular_original_timestamps_and_gripper_interpolation(self):
        _, t, q = load_episode(self.path, self.cfg)
        now, sent = [5.], []
        result = replay(t,q,lambda v:sent.append((now[0]-5,v.copy())),
                        clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
        self.assertAlmostEqual(result['elapsed_s'], .1)
        self.assertAlmostEqual(result['speed'], 1)
        np.testing.assert_allclose(sent[-1][1],q[-1])
        np.testing.assert_allclose([offset for offset,_ in sent],np.arange(6)*.02)
        for offset, target in sent:
            self.assertAlmostEqual(target[0],offset*10)
            self.assertAlmostEqual(target[6],.02+offset*.02)

    def test_alignment_for_reported_pose_is_slow_bounded_and_exact(self):
        current=np.array([-5.618,90.165,-35.496,-17.353,51.199,-25.459,.0526])
        target=np.array([-5.545,90.206,-35.536,-17.353,51.21,-25.559,.0399])
        t,q=alignment_plan(current,target,self.cfg)
        np.testing.assert_allclose(q[0],current)
        np.testing.assert_allclose(q[-1],target)
        v=np.diff(q,axis=0)/np.diff(t)[:,None]
        self.assertLessEqual(float(np.max(abs(v[:,:6]))),2.+1e-8)
        self.assertLessEqual(float(np.max(abs(v[:,6]))),.005+1e-8)
        self.assertGreaterEqual(t[-1],3.81-1e-8)
        self.assertTrue(np.all(q>=np.minimum(current,target)-1e-10))
        self.assertTrue(np.all(q<=np.maximum(current,target)+1e-10))

    def test_alignment_accepts_large_valid_moves_but_rejects_limits(self):
        q=np.array([0,90,-40,0,20,0,.02])
        target=np.array([100,160,-150,80,-50,100,.065])
        t,positions=alignment_plan(q,target,self.cfg)
        np.testing.assert_allclose(positions[-1],target)
        v=np.diff(positions,axis=0)/np.diff(t)[:,None]
        self.assertLessEqual(float(np.max(abs(v[:,:6]))),2.+1e-8)
        self.assertLessEqual(float(np.max(abs(v[:,6]))),.005+1e-8)
        for index,delta in [(0,151),(6,.06),(1,100)]:
            target=q.copy();target[index]+=delta
            with self.assertRaises(ValueError):alignment_plan(q,target,self.cfg)

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_alignment_finishes_before_original_speed_replay(self, robot, writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        current=q[0].copy();current[6]+=.0127
        robot.return_value.get_feedback.side_effect=lambda:{'state':current.tolist()}
        def joint(raw):current[:6]=np.asarray(raw)/1000
        def gripper(raw,effort):current[6]=raw/1e6
        writer.return_value.joint.side_effect=joint
        writer.return_value.gripper.side_effect=gripper
        phases=[]
        def virtual_replay(stamps,states,send,**kwargs):
            phases.append((stamps.copy(),states.copy(),current.copy()))
            now=[0.]
            return replay(stamps,states,send,clock=lambda:now[0],
                          sleep=lambda dt:now.__setitem__(0,now[0]+dt),**kwargs)
        with patch('e14.replay_joint.replay',side_effect=virtual_replay),patch('builtins.print'):
            execute(self.path,self.manifest,t,q,cfg,self.cfg,arm,align_start=True)
        self.assertEqual(len(phases),2)
        self.assertGreater(phases[0][0][-1],3.8)
        np.testing.assert_allclose(phases[1][2],q[0],atol=1e-6)
        np.testing.assert_allclose(phases[1][0],t)
        np.testing.assert_allclose(current,q[-1],atol=1e-6)

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_alignment_out_of_limit_start_rejects_before_writer_creation(self,robot,writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        q[0,0]=151
        with self.assertRaises(ValueError):
            execute(self.path,self.manifest,t,q,cfg,self.cfg,arm,align_start=True)
        writer.assert_not_called()

    def test_stall_stops_without_burst_catchup(self):
        _,t,q=load_episode(self.path,self.cfg)
        now,count=[0.],[0]
        def send(_):
            count[0]+=1
            now[0]+=.08
        with self.assertRaisesRegex(RuntimeError,'发送超时'):
            replay(t,q,send,clock=lambda:now[0],sleep=lambda dt:now.__setitem__(0,now[0]+dt))
        self.assertEqual(count[0],1)

    def test_late_wakeup_sends_no_late_target(self):
        _,t,q=load_episode(self.path,self.cfg)
        now,sent=[0.],[]
        def sleep(dt):now[0]+=dt+.1
        with self.assertRaisesRegex(RuntimeError,'调度超时'):
            replay(t,q,lambda v:sent.append(v),clock=lambda:now[0],sleep=sleep)
        self.assertEqual(len(sent),1)

    def test_corrupt_unit_frame_count_mode_and_timestamp_rejected(self):
        for key,value in [('joint_unit','radian'),('status','failed'),('frames',4)]:
            with self.subTest(key=key):
                old=self.manifest[key];self.manifest[key]=value;self.save()
                with self.assertRaises(ValueError):load_episode(self.path,self.cfg)
                self.manifest[key]=old
        self.rows[1]['captured_monotonic']=self.rows[0]['captured_monotonic'];self.save()
        with self.assertRaises(ValueError):load_episode(self.path,self.cfg)

    def test_limits_nan_speed_and_gap_rejected(self):
        original=copy.deepcopy(self.rows)
        for state_index,value in [(0,151),(0,float('nan')),(0,4),(6,.06)]:
            self.rows=copy.deepcopy(original);self.rows[1]['state'][state_index]=value;self.save()
            with self.assertRaises(ValueError):load_episode(self.path,self.cfg)
        self.rows=copy.deepcopy(original);self.rows[1]['captured_monotonic']=100.001;self.save()
        with self.assertRaises(ValueError):load_episode(self.path,self.cfg)

    def test_offline_check_never_constructs_robot_or_writer(self):
        with patch('e14.joint_collect.PassiveRobot') as robot, patch('servo_bridge.sdk_worker.SDKJointWriter') as writer:
            main([str(self.path),'--profile',str(ROOT/'servo_bridge/config.joint.example.json'),'--root',str(self.path/'empty')])
            robot.assert_not_called();writer.assert_not_called()

    def hardware_setup(self, mock_robot, mock_writer):
        _,t,q=load_episode(self.path,self.cfg)
        arm=dict(operator_present=True,model_and_control_point_checked=True,
                 measured_joint_hold_checked=True,only_one_can_writer=True,
                 acknowledgement=ACK,can_interface='can0')
        arm_path=self.path/'arm.json';arm_path.write_text(json.dumps(arm))
        config=dict(robot=dict(can_interface='can0',feedback_max_age_ms=150))
        mock_robot.return_value.get_feedback.return_value={'state':q[0].tolist()}
        return t,q,arm_path,config

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_start_mismatch_blocks_all_writes(self, robot, writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        q[:,0]+=10
        with self.assertRaisesRegex(RuntimeError,'首帧'):
            execute(self.path,self.manifest,t,q,cfg,self.cfg,arm)
        writer.assert_not_called();robot.return_value.disconnect.assert_called_once()

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_partial_send_failure_requests_hold_and_closes(self, robot, writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        writer.return_value.joint.side_effect=[OSError('CAN failure'),None]
        with self.assertRaises(OSError):
            execute(self.path,self.manifest,t,q,cfg,self.cfg,arm)
        self.assertEqual(writer.return_value.joint.call_count,2)
        writer.return_value.close.assert_called_once();robot.return_value.disconnect.assert_called_once()

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_successful_hardware_adapter_preserves_units_and_final_target(self, robot, writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        current=q[0].copy()
        robot.return_value.get_feedback.side_effect=lambda:{'state':current.tolist()}
        def joint(raw):current[:6]=np.asarray(raw)/1000
        def gripper(raw,effort):current[6]=raw/1e6
        writer.return_value.joint.side_effect=joint
        writer.return_value.gripper.side_effect=gripper
        with patch('builtins.print'):
            execute(self.path,self.manifest,t,q,cfg,self.cfg,arm)
        np.testing.assert_allclose(current,q[-1],atol=1e-6)
        self.assertEqual(writer.return_value.joint.call_args.args[0],[1000,90000,-40000,0,20000,0])
        self.assertEqual(writer.return_value.gripper.call_args.args[0],22000)
        writer.return_value.close.assert_called_once()

    @patch('servo_bridge.sdk_worker.SDKJointWriter')
    @patch('e14.joint_collect.PassiveRobot')
    def test_mock_episode_cannot_execute(self,robot,writer):
        t,q,arm,cfg=self.hardware_setup(robot,writer)
        self.manifest['mock']=True
        with self.assertRaises(ValueError):execute(self.path,self.manifest,t,q,cfg,self.cfg,arm)
        writer.assert_not_called();robot.assert_not_called()


if __name__=='__main__':unittest.main()
