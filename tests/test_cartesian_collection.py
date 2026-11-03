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
            gamepad_config=ROOT/'gamepad.servo.example.json', servo_controller='cartesian',
            orientation_mode='hold')
        self.config = json.loads((ROOT/'config.mock.json').read_text())
        self.bridge = ServoProcess(self.args, self.config)

    def tearDown(self):
        self.bridge.close()
        self.tmp.cleanup()

    def write_events(self, events):
        self.bridge.trace.parent.mkdir(parents=True, exist_ok=True)
        self.bridge.trace.write_text(''.join(json.dumps(r)+'\n' for r in events))

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

    def test_collision_and_singularity_faults_are_never_recoverable(self):
        for status in (2, 4):
            self.write_events([dict(event='servo_status', value=status, label='FAULT', recoverable=True)])
            if self.bridge.reader:
                self.bridge.reader.close()
                self.bridge.reader = None
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                self.bridge.poll()

    def test_legacy_boundary_still_latches_and_unmarked_boundary_is_fault(self):
        for controller, recoverable in [('legacy', True), ('cartesian', False)]:
            self.args.servo_controller = controller
            self.write_events([dict(event='servo_status', value=5, label='JOINT_BOUND', recoverable=recoverable)])
            if self.bridge.reader:
                self.bridge.reader.close()
                self.bridge.reader = None
            with self.subTest(controller=controller), self.assertRaises(RuntimeError):
                self.bridge.poll()


if __name__ == '__main__':
    unittest.main()
