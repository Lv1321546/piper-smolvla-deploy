#!/usr/bin/env bash
# PiPER Servo 手柄采集一键启动脚本
#
# 用法:
#   ./start_demo.sh               # 自检 -> 移动到安全姿态 -> 启动
#   ./start_demo.sh --check       # 只自检（不移动、不启动）
#   ./start_demo.sh --switch-can  # 示教模式下自动切 CAN（会回到上次 CAN 目标位置）
#
# 可用环境变量覆盖:
#   SAFE_POSE  安全起始关节角（度，6 个值，空格分隔）
#   TASK / EPISODES / DURATION / RUN_SECONDS / CAMERA_CHECK_TIMEOUT
#   XYZ_ORIENTATION=position|hold；XYZ_CONTROLLER=cartesian|legacy
set -u
cd -- "$(dirname -- "$0")"

PY=${PIPER_DEMO_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
CONFIG=config.local.json
ARM_FILE=outputs/servo_bridge/arm.local.json
SERVO_CONFIG=servo_bridge/config.local.json
XYZ_CONTROLLER=${XYZ_CONTROLLER:-cartesian}
XYZ_ORIENTATION=${XYZ_ORIENTATION:-position}
case "$XYZ_CONTROLLER" in
  cartesian) SERVO_CONFIG=servo_bridge/config.cartesian.local.json ;;
  legacy) ;;
  *) echo "XYZ_CONTROLLER 必须是 cartesian 或 legacy" >&2; exit 2 ;;
esac
case "$XYZ_ORIENTATION" in
  position|hold) ;;
  *) echo "XYZ_ORIENTATION 必须是 position 或 hold" >&2; exit 2 ;;
esac
# Prefer local device profiles; fall back to versioned examples on a fresh clone.
if [ ! -f "$SERVO_CONFIG" ]; then
  case "$XYZ_CONTROLLER" in
    cartesian) SERVO_CONFIG=servo_bridge/config.cartesian.example.json ;;
    legacy) SERVO_CONFIG=servo_bridge/config.servo.example.json ;;
  esac
fi
PAD_FILE=gamepad.local.servo.json
[ -f "$PAD_FILE" ] || PAD_FILE=gamepad.servo.example.json
OUTPUT=data/piper_raw_servo

# 安全起始姿态（度）。当前值 cond≈11.3、关节余量≥12°，抖动更小:
SAFE_POSE=${SAFE_POSE:-"-5.6 90.2 -35.5 -17.4 51.2 -25.5"}

TASK=${TASK:-"Push the black foam block."}
EPISODES=${EPISODES:-3}
DURATION=${DURATION:-30}
RUN_SECONDS=${RUN_SECONDS:-600}
CAMERA_CHECK_TIMEOUT=${CAMERA_CHECK_TIMEOUT:-45}

CHECK_ONLY=0
SWITCH_CAN=0
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    --switch-can) SWITCH_CAN=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

fail() { echo "[错误] $*" >&2; exit 1; }
ok() { echo "[OK] $*"; }

[ -x "$PY" ] || fail "找不到 Python: $PY"
[ -f "$CONFIG" ] || fail "缺少 $CONFIG"
[ -f "$ARM_FILE" ] || fail "缺少授权文件 $ARM_FILE"
[ -f "$SERVO_CONFIG" ] || fail "缺少伺服配置 $SERVO_CONFIG"

echo "== 1/7 检查是否已有采集进程 =="
if pgrep -f '[e]14.collect_data|[e]14.collect_joint_demo|[e]14.gamepad_teleop|[s]ervo_bridge.sdk_worker|[s]ervo_bridge.runtime|[s]ervo_bridge.joint_runtime|[s]ervo_bridge.cartesian_runtime|[s]ervo_bridge.joint_start_pose' >/dev/null 2>&1; then
  fail "已有采集程序在运行，请先在原终端按 Q 或 Ctrl+C 停止"
fi
ok "无重复进程"
if [ -e /tmp/piper-servo-sdk.sock ]; then
  fail "存在 SDK socket；请确认旧 worker 已退出，不自动删除运行锁或 socket"
fi

echo "== 2/7 检查相机 =="
"$PY" -m e14.camera_check --config "$CONFIG" --timeout "$CAMERA_CHECK_TIMEOUT" \
  || fail "相机检查失败/超时；请查看上方具体原因，确认其他程序已退出后重试"
ok "相机正常"

echo "== 3/7 检查 CAN 数据 =="
"$PY" - <<'PY' || fail "CAN 无数据：检查机械臂电源、急停按钮、CAN 线"
import socket, struct, time, sys
s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
s.bind(('can0',)); s.settimeout(0.5)
end = time.time() + 3
while time.time() < end:
    try:
        d = s.recv(16)
    except socket.timeout:
        continue
    cid = struct.unpack('=I', d[:4])[0] & 0x1FFFFFFF
    if cid == 0x2A1:
        print('CAN status frame OK')
        sys.exit(0)
print('no 0x2A1 frame')
sys.exit(1)
PY
ok "CAN 正常"

echo "== 4/7 检查模式 / 使能 / 目标姿态 =="
"$PY" - "$SWITCH_CAN" "$SAFE_POSE" "$CHECK_ONLY" <<'PY' || fail "机械臂状态不适合启动，请按上面的提示处理后重试"
import json
import socket
import struct
import sys
import time

import numpy as np

from e14.gamepad_kinematics import PiperKinematics

switch_can = sys.argv[1] == '1'
check_only = sys.argv[3] == '1'
if check_only and switch_can:
    print('--check 不允许 --switch-can，不发送切模式命令')
    sys.exit(1)
try:
    safe = [float(v) for v in sys.argv[2].split()]
except ValueError:
    print('SAFE_POSE 解析失败')
    sys.exit(1)
if len(safe) != 6:
    print('SAFE_POSE 必须是 6 个角度值')
    sys.exit(1)

FRAMES = (0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A8, 0x261, 0x262, 0x263, 0x264, 0x265, 0x266)


def read_frames(n=3.0):
    s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind(('can0',)); s.settimeout(0.5)
    frames = {}
    end = time.time() + n
    while time.time() < end and not all(c in frames for c in FRAMES):
        try:
            d = s.recv(16)
        except socket.timeout:
            continue
        cid = struct.unpack('=I', d[:4])[0] & 0x1FFFFFFF
        if cid in FRAMES:
            frames[cid] = d[8:16]
    s.close()
    return frames


def sdk_mode(move_mode):
    from piper_sdk import C_PiperInterface_V2
    arm = C_PiperInterface_V2('can0', start_sdk_joint_limit=False, start_sdk_gripper_limit=False)
    try:
        arm.ConnectPort(piper_init=False)
        time.sleep(0.3)
        arm.MotionCtrl_2(0x01, move_mode, 70, 0x00)
        time.sleep(1.0)
    finally:
        try:
            arm.DisconnectPort()
        except Exception:
            pass


def joints_of(frames):
    q = []
    for cid in (0x2A5, 0x2A6, 0x2A7):
        b = frames.get(cid)
        if b is None:
            return None
        q += [int.from_bytes(b[i:i + 4], 'big', signed=True) / 1000 for i in (0, 4)]
    return q


def describe(frames):
    st = frames.get(0x2A1)
    if st is None:
        return None
    motors = [frames[c][5] for c in range(0x261, 0x267) if c in frames]
    grip = frames.get(0x2A8)
    return dict(ctrl=st[0], arm_status=st[1], mode_feed=st[2],
                err=int.from_bytes(st[6:8], 'big'),
                enabled=[bool(code & 0x40) for code in motors],
                faults=[code & 0x3F for code in motors],
                grip_enabled=bool(grip[6] & 0x40) if grip else None,
                grip_faults=(grip[6] & 0x3F) if grip else None)


frames = read_frames()
info = describe(frames)
if info is None:
    print('无法读取控制状态')
    sys.exit(1)

if info['ctrl'] == 2 and switch_can:
    print('示教模式 -> 请求 CAN 模式（机械臂会回到上次 CAN 目标位置，请远离工作区）')
    sdk_mode(0x01)
    frames = read_frames()
    info = describe(frames)

if info['ctrl'] != 1:
    print('当前是示教模式(2)。请按实体按钮退出示教后重试；')
    print('或运行 ./start_demo.sh --switch-can（会回到上次 CAN 目标位置）')
    sys.exit(1)

if info['mode_feed'] != 1:
    if check_only:
        print('当前不是 MOVE_J；只读自检不发送切换命令，请先处理模式后重试')
        sys.exit(1)
    print('mode_feed=%d（非 MOVE_J）-> 请求 CAN+MOVE_J' % info['mode_feed'])
    sdk_mode(0x01)
    frames = read_frames()
    info = describe(frames)

print('ctrl_mode=%d arm_status=%d mode_feed=%d err_code=%d'
      % (info['ctrl'], info['arm_status'], info['mode_feed'], info['err']))
print('enabled=%s faults=%s gripper_enabled=%s gripper_faults=%s'
      % (info['enabled'], info['faults'], info['grip_enabled'], info['grip_faults']))

if info['ctrl'] != 1 or info['mode_feed'] != 1:
    print('无法进入 CAN + MOVE_J')
    sys.exit(1)
if info['arm_status'] != 0 or info['err'] != 0:
    print('机械臂处于停止/故障状态（arm_status 或 err_code 非 0），请先在现场处理')
    sys.exit(1)
if not all(info['enabled']):
    print('电机未全部使能，请先完成使能')
    sys.exit(1)
if any(info['faults']) or (info['grip_faults'] or 0):
    print('存在电机/夹爪故障位，请先处理')
    sys.exit(1)

# 目标姿态必须安全
cfg = json.load(open('config.local.json'))
kin = PiperKinematics(cfg['robot']['joint_limits_deg'])
limits = np.array(cfg['robot']['joint_limits_deg'], float)
T, J = kin.forward(np.radians(safe), jacobian=True)
cond = float(np.linalg.cond(J))
margins = [min(safe[i] - limits[i, 0], limits[i, 1] - safe[i]) for i in range(6)]
print('安全姿态 %s' % [round(v, 1) for v in safe])
print('  tcp_m=%s cond=%.1f joint_margins=%s'
      % (np.round(T[:3, 3], 3).tolist(), cond, [round(m, 1) for m in margins]))
if cond > 25:
    print('起始姿态条件数 >25，请修改脚本顶部的 SAFE_POSE，选择更易活动的起始姿态')
    sys.exit(1)
if min(margins) < 5:
    print('安全姿态有关节离限位不足 5°，请修改脚本顶部的 SAFE_POSE')
    sys.exit(1)
if cond > 15:
    print('提示：起始姿态条件数 >15，灵活性较低，建议调整起始姿态')

q_now = joints_of(frames)
if q_now is not None:
    delta = max(abs(a - b) for a, b in zip(q_now, safe))
    print('当前角度 %s，与安全姿态最大差 %.1f°'
          % ([round(v, 1) for v in q_now], delta))
PY
ok "模式 / 使能 / 目标姿态正常"

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "== 自检完成：未移动、未刷新授权、未启动控制 =="
  exit 0
fi

echo "== 5/7 刷新输出授权 =="
"$PY" - "$RUN_SECONDS" <<'PY' || fail "授权刷新失败：检查授权文件的确认项"
import json
import sys
import time

p = 'outputs/servo_bridge/arm.local.json'
d = json.load(open(p))
need = ('operator_present', 'model_and_control_point_checked',
        'measured_joint_hold_checked', 'only_one_can_writer')
missing = [k for k in need if d.get(k) is not True]
if missing:
    print('以下安全确认项不是 true：', ', '.join(missing))
    sys.exit(1)
runtime = float(sys.argv[1])
if not 0 < runtime < float('inf'):
    raise ValueError('RUN_SECONDS 必须是正数')
d['expires_unix_s'] = int(time.time() + runtime + 180)
json.dump(d, open(p, 'w'), indent=2)
print('授权有效至', time.strftime('%H:%M:%S', time.localtime(d['expires_unix_s'])))
PY

echo "== 6/7 以 5% 速度移动到安全起始姿态 =="
"$PY" -m servo_bridge.joint_start_pose --config "$CONFIG" --profile "$SERVO_CONFIG" \
  --safe-pose "$SAFE_POSE" --arm-file "$ARM_FILE" --execute \
  || fail "起始移动未完成；已取消启动，请检查现场状态"

echo "== 7/7 启动 Servo 采集 =="
echo "控制页面: http://127.0.0.1:8766"
echo "按住 LB + 摇杆移动 XYZ；同时按住 RB 为精细档；仅按 RB 不移动。"
echo "XYZ 控制器：$XYZ_CONTROLLER；姿态模式：$XYZ_ORIENTATION（position 允许小幅朝向变化，hold 保持朝向）。"
if [ -e /tmp/piper-servo-sdk.sock ]; then
  fail "存在 SDK socket；请确认旧 worker 已退出，不自动删除运行锁或 socket"
fi
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  export SDL_VIDEODRIVER=dummy
fi
exec "$PY" -m e14.collect_data \
  --config "$CONFIG" \
  --teleop servo --execute \
  --servo-arm-file "$ARM_FILE" \
  --servo-config "$SERVO_CONFIG" \
  --servo-controller "$XYZ_CONTROLLER" --orientation-mode "$XYZ_ORIENTATION" \
  --gamepad-config "$PAD_FILE" \
  --output "$OUTPUT" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --duration "$DURATION" \
  --run-seconds "$RUN_SECONDS"
