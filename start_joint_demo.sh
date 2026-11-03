#!/usr/bin/env bash
# Independent joint demo. Does not edit or invoke start_demo.sh.
# ./start_joint_demo.sh           preflight -> 5% startup pose -> joint collection
# ./start_joint_demo.sh --check   passive preflight only; no CAN commands or file writes
# ./start_joint_demo.sh --mock    UI/collection simulation; no CAN, ROS or gamepad
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PY=${JOINT_DEMO_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
CONFIG=${JOINT_DEMO_CONFIG:-config.local.json}
PROFILE=${JOINT_DEMO_PROFILE:-servo_bridge/config.joint.local.json}
if [[ ! -f "$PROFILE" && -z "${JOINT_DEMO_PROFILE:-}" ]]; then
  PROFILE=servo_bridge/config.joint.example.json
fi
PAD_FILE=gamepad.local.joint.json
[[ -f "$PAD_FILE" ]] || PAD_FILE=gamepad.joint.example.json
ARM_FILE=outputs/joint_demo/arm.local.json
OUTPUT=${JOINT_DEMO_OUTPUT:-data/piper_raw_joint}
TASK=${TASK:-"Push the black foam block."}
EPISODES=${EPISODES:-3}
DURATION=${DURATION:-30}
RUN_SECONDS=${RUN_SECONDS:-600}
PREVIEW_PORT=${PREVIEW_PORT:-8766}
SAFE_POSE=${SAFE_POSE:-"-5.6 90.2 -35.5 -17.4 51.2 -25.5"}
CHECK_ONLY=0
MOCK=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --mock) MOCK=1 ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done
if [[ "$MOCK" == 1 && "$CHECK_ONLY" == 1 ]]; then
  echo "--check 检查实机；模拟界面请单独用 --mock。" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then echo "找不到 Python：$PY" >&2; exit 1; fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  export SDL_VIDEODRIVER=dummy
fi

if [[ "$MOCK" == 1 ]]; then
  exec "$PY" -m e14.collect_joint_demo --config "$CONFIG" --mock \
    --output "${OUTPUT}_mock" --task "$TASK" --episodes "$EPISODES" \
    --duration "$DURATION" --run-seconds "$RUN_SECONDS" --preview-port "$PREVIEW_PORT"
fi

echo "== 1/5 检查重复控制进程和残留 socket =="
if pgrep -f '[e]14.collect_data|[e]14.collect_joint_demo|[e]14.gamepad_teleop|[s]ervo_bridge.sdk_worker|[s]ervo_bridge.runtime|[s]ervo_bridge.joint_runtime|[s]ervo_bridge.joint_start_pose' >/dev/null; then
  echo "已有控制/采集程序；请在原终端退出后重试。" >&2
  exit 1
fi
if [[ -e /tmp/piper-servo-sdk.sock ]]; then
  echo "存在 /tmp/piper-servo-sdk.sock；请确认旧 worker 已正常退出，脚本不会自动删除。" >&2
  exit 1
fi

echo "== 2/5 只读检查配置、起始姿态、授权、CAN/MOVE_J/使能和相机 =="
"$PY" - "$CONFIG" "$PROFILE" "$RUN_SECONDS" "$SAFE_POSE" <<'PY'
import json
import math
from pathlib import Path
import sys
import time

from e14.cameras import CameraReader
from e14.joint_collect import PassiveRobot
from servo_bridge.core import CommandGuard, load_config
from servo_bridge.joint_policy import JointPolicy
from servo_bridge.joint_start_pose import validate_pose
from servo_bridge.sdk_worker import validate_arm

config = json.loads(Path(sys.argv[1]).read_text())
cfg = load_config(sys.argv[2])
JointPolicy(cfg)
target, margins, condition, tcp = validate_pose(sys.argv[4], cfg)
print('起始姿态：', target.tolist())
print('条件数 %.1f；最小关节限位余量 %.1f°' % (condition, min(margins)))
runtime = float(sys.argv[3])
if not math.isfinite(runtime) or runtime <= 0:
    raise ValueError('RUN_SECONDS 必须是正数')
if cfg['can_interface'] != config['robot']['can_interface']:
    raise ValueError('CAN 接口配置不一致')
if cfg['joint_limits_deg'] != config['robot']['joint_limits_deg']:
    raise ValueError('关节配置与采集配置不一致，请先核对两个配置文件')
# Reuse only previously recorded commissioning acknowledgements, not their
# expiry. No original authorization file is changed.
arm = json.loads(Path('outputs/servo_bridge/arm.local.json').read_text())
arm['expires_unix_s'] = int(time.time() + runtime + 120)
validate_arm(arm, cfg['can_interface'])
robot, cameras = PassiveRobot(config), CameraReader(config['cameras'])
try:
    robot.connect()
    state = robot.get_feedback()['state']
    import numpy as np
    q = np.radians(state[:6])
    bounds = CommandGuard(cfg).limits
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        raise ValueError('当前角度超出配置/模型范围，请先人工处理')
    print('当前关节角：', [round(v, 2) for v in state[:6]])
    print('当前夹爪开口：%.1f mm' % (state[6] * 1000))
    cameras.connect()
    cameras.read()
    print('CAN/MOVE_J/使能、反馈和相机正常；未发送运动、模式或使能命令。')
finally:
    cameras.close()
    robot.disconnect()
PY

if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "自检完成。未启动控制，未刷新授权，未移动机械臂。"
  exit 0
fi

echo "== 3/5 生成独立模型参数和本次授权 =="
/usr/bin/python3 -m servo_bridge.prepare --config "$PROFILE" \
  --mesh-source models/piper_servo/meshes --output models/piper_joint_servo
"$PY" - "$ARM_FILE" "$RUN_SECONDS" "$PROFILE" <<'PY'
import json
from pathlib import Path
import sys
import time
from servo_bridge.core import load_config
from servo_bridge.sdk_worker import validate_arm
arm = json.loads(Path('outputs/servo_bridge/arm.local.json').read_text())
arm['expires_unix_s'] = int(time.time() + float(sys.argv[2]) + 120)
validate_arm(arm, load_config(sys.argv[3])['can_interface'])
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(arm, indent=2) + '\n')
PY

echo "== 4/5 低速移动到起始姿态 =="
"$PY" -m servo_bridge.joint_start_pose --config "$CONFIG" --profile "$PROFILE" \
  --safe-pose "$SAFE_POSE" --arm-file "$ARM_FILE" --execute

echo "== 5/5 启动关节直控采集 =="
echo "左杆左右 J1，左杆前后 J2，右杆上下 J3，方向键左右 J4，上下 J5，右杆左右 J6。"
echo "按住 LB 允许；同时按住 RB 为精细档（仅按 RB 不移动）；B/RT 张爪；A/LT 闭爪；X 开始/结束；Y 成功；Back 失败。"
echo "控制页面：http://127.0.0.1:$PREVIEW_PORT"
exec "$PY" -m e14.collect_joint_demo --config "$CONFIG" --execute \
  --gamepad-config "$PAD_FILE" \
  --servo-config "$PROFILE" --servo-arm-file "$ARM_FILE" \
  --output "$OUTPUT" --task "$TASK" --episodes "$EPISODES" \
  --duration "$DURATION" --run-seconds "$RUN_SECONDS" --preview-port "$PREVIEW_PORT"
