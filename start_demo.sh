#!/usr/bin/env bash
# PiPER 双 RGB 笛卡尔手柄采集。
# ./start_demo.sh          自检 -> CAN接管 -> 返回固定起点 -> 手柄采集
# ./start_demo.sh --check  相机和反馈只读检查
# 固定起点保存在 config.cartesian.home.local.json；启动、A 和 H 共用此目标
# TASK / EPISODES / DURATION / RUN_SECONDS / CAMERA_CHECK_TIMEOUT
# EPISODES=0 / RUN_SECONDS=0 表示不限条数 / 不限会话时长（默认）
# PIPER_DEMO_CONFIG / PIPER_SERVO_CONFIG / PIPER_DEMO_OUTPUT / PREVIEW_PORT
# XYZ_ORIENTATION=position|hold；XYZ_CONTROLLER=cartesian|legacy
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

PY=${PIPER_DEMO_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
CONFIG=${PIPER_DEMO_CONFIG:-config.cartesian.dual.local.json}
XYZ_CONTROLLER=${XYZ_CONTROLLER:-cartesian}
XYZ_ORIENTATION=${XYZ_ORIENTATION:-position}
case "$XYZ_CONTROLLER" in
  cartesian) DEFAULT_PROFILE=servo_bridge/config.cartesian.local.json ;;
  legacy) DEFAULT_PROFILE=servo_bridge/config.local.json ;;
  *) echo "XYZ_CONTROLLER 必须是 cartesian 或 legacy" >&2; exit 2 ;;
esac
SERVO_CONFIG=${PIPER_SERVO_CONFIG:-$DEFAULT_PROFILE}
if [[ ! -f "$SERVO_CONFIG" && -z "${PIPER_SERVO_CONFIG:-}" ]]; then
  case "$XYZ_CONTROLLER" in
    cartesian) SERVO_CONFIG=servo_bridge/config.cartesian.example.json ;;
    legacy) SERVO_CONFIG=servo_bridge/config.servo.example.json ;;
  esac
fi
case "$XYZ_ORIENTATION" in
  position|hold) ;;
  *) echo "XYZ_ORIENTATION 必须是 position 或 hold" >&2; exit 2 ;;
esac
PAD_FILE=gamepad.local.servo.json
[[ -f "$PAD_FILE" ]] || PAD_FILE=gamepad.servo.example.json
OUTPUT=${PIPER_DEMO_OUTPUT:-data/piper_raw_cartesian_dual}
PREVIEW_PORT=${PREVIEW_PORT:-8768}
FIXED_HOME=${PIPER_FIXED_HOME:-config.cartesian.home.local.json}
TASK=${TASK:-"Pick up the black foam block and place it in the box."}
EPISODES=${EPISODES:-0}
DURATION=${DURATION:-120}
RUN_SECONDS=${RUN_SECONDS:-0}
CAMERA_CHECK_TIMEOUT=${CAMERA_CHECK_TIMEOUT:-45}
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    --switch-can) ;; # Compatibility: a normal start already includes takeover.
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "未知参数：$arg" >&2; exit 2 ;;
  esac
done
fail() { echo "[错误] $*" >&2; exit 1; }
[[ -x "$PY" ]] || fail "找不到 Python：$PY"
[[ -f "$CONFIG" ]] || fail "缺少双相机配置 $CONFIG"
[[ -f "$SERVO_CONFIG" ]] || fail "缺少伺服配置 $SERVO_CONFIG"
[[ -f "$FIXED_HOME" ]] || fail "缺少固定起点文件 $FIXED_HOME"

echo "== 1/5 检查重复控制进程和残留 socket =="
if pgrep -f '[e]14.collect_data|[e]14.collect_joint_demo|[e]14.replay_joint|[e]14.gamepad_teleop|[s]ervo_bridge.sdk_worker|[s]ervo_bridge.runtime|[s]ervo_bridge.joint_runtime|[s]ervo_bridge.cartesian_runtime|[s]ervo_bridge.joint_start_pose|[s]ervo_bridge.prepare_control' >/dev/null; then
  fail "已有控制/采集程序，请先在原终端退出"
fi
[[ ! -e /tmp/piper-servo-sdk.sock ]] || fail "存在 SDK socket，请确认旧 worker 已退出"

echo "== 2/5 只读检查当前实测状态和控制配置 =="
"$PY" - "$SERVO_CONFIG" "$RUN_SECONDS" "$XYZ_ORIENTATION" <<'PY'
import math
import sys
from servo_bridge.core import load_config
cfg = load_config(sys.argv[1])
cfg['cartesian_orientation_mode'] = sys.argv[3]
if not math.isfinite(float(sys.argv[2])) or float(sys.argv[2]) < 0:
    raise ValueError('RUN_SECONDS 必须是非负数，0 表示不限时长')
PY
"$PY" -m servo_bridge.start_reference --config "$CONFIG" --profile "$SERVO_CONFIG"
SAFE_POSE=$("$PY" -m servo_bridge.start_reference --config "$CONFIG" --profile "$SERVO_CONFIG" --fixed-home "$FIXED_HOME" --print-joints)
"$PY" - "$SERVO_CONFIG" "$SAFE_POSE" <<'PY'
import sys
from servo_bridge.core import load_config
from servo_bridge.joint_start_pose import validate_pose
q, margins, condition, _ = validate_pose(sys.argv[2], load_config(sys.argv[1]))
print('固定起始目标（度）：', q.tolist())
print('条件数 %.1f；最小关节余量 %.1f°' % (condition, min(margins)))
PY
"$PY" -m servo_bridge.prepare_control --config "$CONFIG" --profile "$SERVO_CONFIG" --preflight

echo "== 3/5 预检两路 RGB 相机 =="
"$PY" -m e14.camera_check --config "$CONFIG" --timeout "$CAMERA_CHECK_TIMEOUT"
if [[ "$CHECK_ONLY" == 1 ]]; then
  echo "只读自检完成。"
  exit 0
fi

echo "== 4/5 从当前姿态接管 CAN =="
"$PY" -m servo_bridge.prepare_control --config "$CONFIG" --profile "$SERVO_CONFIG" --execute
"$PY" -m servo_bridge.joint_start_pose --config "$CONFIG" --profile "$SERVO_CONFIG" \
    --safe-pose "$SAFE_POSE" --tolerance-deg 0.3 --execute

echo "== 5/5 加载固定初始状态并启动双 RGB 采集 =="
"$PY" -m servo_bridge.start_reference --config "$CONFIG" --profile "$SERVO_CONFIG" \
  --fixed-home "$FIXED_HOME" --output "$OUTPUT/initial_state.json"
echo "控制页面：http://127.0.0.1:$PREVIEW_PORT"
echo "LB + 摇杆控制 XYZ；LB+RB 为精细档；方向键/右杆左右调整朝向。"
echo "朝向模式：$XYZ_ORIENTATION；X 开始/结束，Y 成功，Back 失败。"
if [[ "$XYZ_CONTROLLER" == cartesian ]]; then
  echo "长按 A 约 0.6 秒以普通档速度返回固定初始位置，松开 A 停止；返回时摇杆和扳机需回中。"
  echo "夹爪用 LB+B/RT 张开、LB+LT 闭合；页面按 H 一键返回并张爪，空格取消复位。"
fi
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  export SDL_VIDEODRIVER=dummy
fi
exec "$PY" -m e14.collect_data --config "$CONFIG" --teleop servo --execute \
  --servo-config "$SERVO_CONFIG" --servo-controller "$XYZ_CONTROLLER" \
  --orientation-mode "$XYZ_ORIENTATION" --gamepad-config "$PAD_FILE" \
  --output "$OUTPUT" --task "$TASK" --episodes "$EPISODES" --duration "$DURATION" \
  --run-seconds "$RUN_SECONDS" --preview-port "$PREVIEW_PORT" --fixed-home "$FIXED_HOME"
