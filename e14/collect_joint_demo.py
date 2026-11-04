"""Independent command-line entry for joint gamepad control and raw collection."""
import argparse
from pathlib import Path

from e14.joint_collect import run_servo_collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config.local.json'))
    parser.add_argument('--output', type=Path, default=Path('data/piper_raw_joint'))
    parser.add_argument('--task', default='Push the black foam block.')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--duration', type=float, default=600., help='单条录制时长上限（秒），可用 X 提前结束')
    parser.add_argument('--run-seconds', type=float, default=1500., help='整个手柄控制会话的时长上限（秒）')
    parser.add_argument('--preview-port', type=int, default=8766)
    parser.add_argument('--servo-config', type=Path, default=(Path('servo_bridge/config.joint.local.json') if Path('servo_bridge/config.joint.local.json').is_file() else Path('servo_bridge/config.joint.example.json')))
    parser.add_argument('--servo-arm-file', type=Path, help=argparse.SUPPRESS)  # Legacy option, ignored.
    parser.add_argument('--gamepad-config', type=Path, default=(Path('gamepad.local.joint.json') if Path('gamepad.local.joint.json').is_file() else Path('gamepad.joint.example.json')))
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--home-file', type=Path, default=Path('outputs/joint_demo/reference.json'))
    # TeachingSession's shared read-only/recording interface.
    parser.set_defaults(interactive=False, initialize_can=False, restart_can=False,
                        open_width=.06, close_width=0., gripper_speed=.1)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    run_servo_collection(args)


if __name__ == '__main__':
    main()
