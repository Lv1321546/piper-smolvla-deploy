"""Exercise launcher routes using a Python stub, with no hardware commands."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CartesianLauncherTests(unittest.TestCase):
    def test_check_only_and_fixed_start_routes(self):
        if Path('/tmp/piper-servo-sdk.sock').exists():
            self.skipTest('A live worker owns the startup socket')
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            stub = folder/'python-stub'
            stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
            stub.chmod(0o755)
            pgrep = folder/'pgrep'
            pgrep.write_text('#!/usr/bin/env bash\nexit 1\n')
            pgrep.chmod(0o755)
            for check in (True, False):
                log = folder/('check.log' if check else 'normal.log')
                env = {**os.environ, 'PIPER_DEMO_PYTHON': str(stub),
                       'PIPER_DEMO_CONFIG': str(ROOT/'config.joint.dual.example.json'),
                       'PATH': str(folder)+os.pathsep+os.environ['PATH'],
                       'PIPER_DEMO_OUTPUT': str(folder/'data'), 'CALL_LOG': str(log)}
                env.pop('SAFE_POSE', None)
                result = subprocess.run(['bash', str(ROOT/'start_demo.sh')]+(['--check'] if check else []),
                                        cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                calls = log.read_text()
                self.assertIn('-m servo_bridge.start_reference', calls)
                self.assertIn('-m e14.camera_check', calls)
                if check:
                    self.assertNotIn('--execute', calls)
                    self.assertNotIn('-m servo_bridge.joint_start_pose', calls)
                    self.assertNotIn('--output', calls)
                else:
                    self.assertIn('-m servo_bridge.joint_start_pose', calls)
                    self.assertIn('--fixed-home config.cartesian.home.local.json', calls)
                    self.assertIn('--preview-port 8768', calls)
                    self.assertIn('-m e14.collect_data', calls)
                    self.assertIn('--task Pick up the black foam block and place it in the box.', calls)


if __name__ == '__main__':
    unittest.main()
