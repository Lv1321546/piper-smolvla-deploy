"""Menu commands and process lifecycle; fake runner/SDK only."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch,MagicMock

from e14.collect_data import ACTION_NAMES
from e14.replay_menu import scan,menu,replay_command,launch

ROOT=Path(__file__).resolve().parents[1]


class ReplayMenuTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=ROOT/'config.joint.dual.example.json'
        self.profile=ROOT/'servo_bridge/config.joint.example.json'
        self.sources=[(self.root,self.config,'双相机')]
        for n,status in [(0,'failed'),(1,'complete')]:
            path=self.root/f'episode_{n:06d}';path.mkdir()
            manifest=dict(format='piper-raw-v1',status=status,state_names=ACTION_NAMES,
                          joint_unit='degree',gripper_unit='meter_full_opening',frames=3,
                          action_mode='absolute_position',label_semantics='next_state',
                          control_hz=20,mock=False,operator_session=dict(mode='moveit_servo_joint_gamepad'),
                          task_outcome='success' if status=='complete' else 'unknown',task='pick',cameras={'front':{},'astra':{}})
            (path/'manifest.json').write_text(json.dumps(manifest))
            rows=[dict(frame_index=i,captured_monotonic=100+i*.05,state=[i*.1,90,-40,0,20,0,.02]) for i in range(3)]
            (path/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def tearDown(self):self.temp.cleanup()

    def run_menu(self,choices,check_only=False):
        calls=[];output=[];answers=iter(choices)
        menu(self.sources,self.profile,check_only=check_only,input_fn=lambda _:next(answers),
             output=output.append,runner=lambda cmd:calls.append(cmd) or 0)
        return calls,output

    def test_list_includes_failed_but_only_complete_is_eligible(self):
        records=scan(self.sources)
        self.assertEqual(len(records),2)
        self.assertFalse(records[0].eligible);self.assertTrue(records[1].eligible)
        self.assertAlmostEqual(records[1].duration,.1)

    def test_selection_uses_exact_episode_and_correct_config(self):
        calls,output=self.run_menu(['v 2','1','2','q'])
        self.assertEqual(len(calls),1)
        cmd=calls[0]
        self.assertIn(str(self.root/'episode_000001'),cmd)
        self.assertIn(str(self.config),cmd)
        self.assertIn('--execute',cmd)
        self.assertTrue(any('禁止' in line for line in output))

    def test_check_only_and_c_command_never_request_hardware(self):
        for choices,check in [(['2','q'],True),(['c 2','q'],False)]:
            calls,_=self.run_menu(choices,check)
            self.assertEqual(len(calls),1)
            self.assertNotIn('--execute',calls[0])

    def test_bad_input_refresh_and_details_cannot_start_robot(self):
        calls,_=self.run_menu(['0','99','bad','r','v 1','q'])
        self.assertEqual(calls,[])

    def test_file_change_revalidated_before_launch(self):
        record=scan(self.sources)[1]
        (record.path/'frames.jsonl').unlink()
        with self.assertRaises(OSError):replay_command(record,self.profile)

    @patch('e14.replay_menu.subprocess.Popen')
    def test_ctrl_c_is_forwarded_once_and_waits_for_hold_cleanup(self,popen):
        process=MagicMock();popen.return_value=process
        process.wait.side_effect=[KeyboardInterrupt(),0];process.poll.return_value=None
        self.assertEqual(launch(['fake-python']),130)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(process.wait.call_count,2)

    @patch('e14.replay_menu.subprocess.Popen')
    def test_unconfirmed_child_exit_blocks_more_replays(self,popen):
        process=MagicMock();popen.return_value=process;process.poll.return_value=None
        process.wait.side_effect=[KeyboardInterrupt(),subprocess.TimeoutExpired(['fake'],10)]
        with self.assertRaisesRegex(RuntimeError,'未确认退出'):launch(['fake'])


if __name__=='__main__':unittest.main()
