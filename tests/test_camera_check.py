"""Bounded disposable camera preflight; simulated captures/processes only."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from e14.camera_check import check_cameras


class CameraCheckTests(unittest.TestCase):
    def test_mock_reader_runs_in_real_child_without_hardware(self):
        check_cameras(Path('config.mock.json'), timeout=10, mock=True)

    def test_blocked_worker_is_killed_and_reaped_on_timeout(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_file = Path(folder) / 'pid'
            program = ('import os,time; from pathlib import Path; '
                       f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(20)')
            with patch('e14.camera_check.worker_command', return_value=[sys.executable, '-c', program]):
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, '子进程已终止'):
                    check_cameras('unused.json', timeout=.5)
                self.assertLess(time.monotonic() - started, 5)
            self.assertTrue(pid_file.is_file())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)

    def test_worker_failure_is_not_reported_as_success(self):
        with patch('e14.camera_check.subprocess.run', return_value=SimpleNamespace(returncode=2)) as run:
            with self.assertRaisesRegex(RuntimeError, '退出码 2'):
                check_cameras('example.json', timeout=12)
            self.assertEqual(run.call_args.kwargs['timeout'], 12)

    def test_invalid_timeout_cannot_launch_worker(self):
        with patch('e14.camera_check.subprocess.run') as run:
            for timeout in (True, 0, -1, float('inf'), float('nan')):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    check_cameras('example.json', timeout=timeout)
            run.assert_not_called()
