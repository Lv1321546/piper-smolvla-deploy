"""XYZ v2 routing, metadata telemetry and boundary-vs-fault distinction."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from e14.servo_collect import ServoProcess


ROOT = Path(__file__).resolve().parents[1]


class CartesianCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.args = SimpleNamespace(output=Path(self.tmp.name), execute=False,
            servo_config=ROOT/'servo_bridge/config.cartesian.example.json', run_seconds=10,
            gamepad_config=ROOT/'gamepad.local.servo.json', servo_controller='cartesian',
            orientation_mode='hold')
        self.config = json.loads((ROOT/'config.mock.json').read_text())
        self.bridge = ServoProcess(self.args, self.config)

    def tearDown(self):
        self.bridge.close()
        self.tmp.cleanup()

    def write_events(self, events):
        self.bridge.trace.parent.mkdir(parents=True, exist_ok=True)
        self.bridge.trace.write_text(''.join(json.dumps(r)+'\n' for r in events))

    def test_cartesian_page_explains_a_return_and_lt_close(self):
        from e14.servo_collect import PAGE, LEGACY_PAGE
        self.assertIn('长按 A', PAGE)
        self.assertIn('无需 LB', PAGE)
        self.assertNotIn('A/LT', PAGE)
        self.assertIn('A/LT', LEGACY_PAGE)
        self.assertIn('K 保存参考快照', PAGE)

    def test_routes_to_new_runtime_and_preserves_orientation_choice(self):
        commands = []
        def spawn(command, name):
            commands.append(command)
            self.write_events([dict(event='input', monotonic=time.monotonic(),
                                    planner={'message': 'XYZ v2'})])
        with patch.object(self.bridge, 'spawn', side_effect=spawn):
            self.bridge.start()
        self.assertEqual(commands[0][:2], ['bash', 'servo_bridge/run_cartesian.sh'])
        self.assertIn('--orientation-mode', commands[0])
        self.assertEqual(commands[0][commands[0].index('--orientation-mode')+1], 'hold')

    def test_home_reference_is_saved_and_forwarded_to_cartesian_runtime(self):
        reference = dict(schema_version=1, joint_unit='degree',
                         joint_angles=[0., 80., -60., 0., 40., 0.], gripper_width_m=.04)
        self.bridge.initial_state = reference
        commands = []
        def spawn(command, name):
            commands.append(command)
            self.write_events([dict(event='input', monotonic=time.monotonic())])
        with patch.object(self.bridge, 'spawn', side_effect=spawn):
            self.bridge.start()
        command = commands[0]
        path = Path(command[command.index('--initial-state')+1])
        self.assertEqual(json.loads(path.read_text()), reference)
        control_path = Path(command[command.index('--control-file')+1])
        self.assertEqual(control_path, self.bridge.control.path.resolve())
        self.bridge.send_control('reset')
        self.assertEqual(json.loads(control_path.read_text())['command'], 'reset')

    def test_legacy_route_keeps_original_runtime(self):
        self.args.servo_controller = 'legacy'
        commands = []
        def spawn(command, name):
            commands.append(command)
            self.write_events([dict(event='input', monotonic=time.monotonic())])
        with patch.object(self.bridge, 'spawn', side_effect=spawn):
            self.bridge.start()
        self.assertEqual(commands[0][:2], ['bash', 'servo_bridge/run.sh'])
        self.assertNotIn('--orientation-mode', commands[0])

    def test_explicit_v2_joint_boundary_is_recoverable(self):
        self.write_events([dict(event='servo_status', value=5, label='JOINT_BOUND', recoverable=True),
                           dict(event='input', monotonic=time.monotonic(), status=5)])
        self.assertEqual(len(self.bridge.poll()), 2)
        self.assertEqual(self.bridge.latest['status'], 5)

    def test_collision_faults_are_never_recoverable(self):
        for status in (3, 4):
            self.write_events([dict(event='servo_status', value=status, label='FAULT', recoverable=True)])
            if self.bridge.reader:
                self.bridge.reader.close()
                self.bridge.reader = None
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                self.bridge.poll()

    def test_geometric_halts_require_explicit_recoverable_marker(self):
        for controller in ('legacy', 'cartesian'):
            for status in (2, 5):
                for recoverable in (False, True):
                    self.args.servo_controller = controller
                    self.write_events([dict(event='servo_status', value=status,
                        label='GEOMETRY', recoverable=recoverable)])
                    if self.bridge.reader:
                        self.bridge.reader.close()
                        self.bridge.reader = None
                    with self.subTest(controller=controller, status=status, recoverable=recoverable):
                        if recoverable:
                            self.assertEqual(len(self.bridge.poll()), 1)
                        else:
                            with self.assertRaises(RuntimeError):
                                self.bridge.poll()


if __name__ == '__main__':
    unittest.main()
