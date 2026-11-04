import copy
import unittest
from servo_bridge.feedback_diagnostics import frame_ages,stale_description,readiness_description


class FeedbackDiagnosticsTests(unittest.TestCase):
    def test_old_motor_frame_not_hidden_by_fresh_joint_frames(self):
        snapshot={'frame_timestamps':{'0x2a5':10.,'0x261':9.8}}
        before=copy.deepcopy(snapshot)
        details=frame_ages(snapshot,10.01,.15)
        self.assertEqual(details['stale_frame_ids'],['0x261'])
        self.assertEqual(details['frame_age_ms']['0x261'],210.)
        self.assertIn('0x261',stale_description(details))
        self.assertEqual(snapshot,before)

    def test_future_timestamp_is_explicitly_invalid(self):
        details=frame_ages({'frame_timestamps':{'0x2a5':11.}},10.,.15)
        self.assertEqual(details['stale_frame_ids'],['0x2a5'])

    def test_missing_audit_timestamps_does_not_invent_freshness(self):
        details=frame_ages(None,10.,.15)
        self.assertEqual(details['frame_age_ms'],{})
        self.assertIn('no per-frame',stale_description(details))

    def test_finished_teaching_is_still_not_can_control(self):
        snapshot=dict(ctrl_mode=2,mode_feed=1,teach_status=2,enabled=[True]*6,
                      arm_status=0,err_code=0,motor_faults=[0]*6)
        message=readiness_description(snapshot)
        self.assertIn('尚未切入 CAN 控制',message)
        self.assertIn('示教记录已结束',message)
        self.assertNotIn('未使能关节',message)
        self.assertNotIn('存在机械臂/电机故障',message)

    def test_disabled_joint_and_wrong_move_mode_reported_separately(self):
        snapshot=dict(ctrl_mode=1,mode_feed=0,teach_status=0,
                      enabled=[True,False,True,True,True,True],
                      arm_status=0,err_code=0,motor_faults=[0]*6)
        message=readiness_description(snapshot)
        self.assertIn('未使能关节=[2]',message)
        self.assertIn('不是 MOVE_J',message)
        self.assertNotIn('尚未切入 CAN 控制',message)


if __name__=='__main__':unittest.main()
