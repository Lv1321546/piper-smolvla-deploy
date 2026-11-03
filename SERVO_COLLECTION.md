# Servo 手柄控制与数据采集

当前 `start_demo.sh` 默认使用重写后的 XYZ v2，配置为 `servo_bridge/config.cartesian.local.json`。
位置优先、保持朝向、旧版对比和验证记录见 [XYZ_V2.md](XYZ_V2.md)。以下旧路线说明保留作参考。

入口已接入 `e14.collect_data --teleop servo`。本次只完成代码和离线测试，未连接实机测试采集。

## 功能与边界

- ROS 子进程独占手柄输入，使用已验证的 `integrated-velocity` 执行层；独立 SDK worker 是唯一运动写入者。
- 采集进程只读真实 CAN 状态与相机，浏览器实时显示画面。PNG 保存不会在 Servo 控制循环内执行。
- LB + 摇杆移动 XYZ；同时按住 LB + RB 为精细档，仅按 RB 不移动；松开 LB 保持。R 或手柄 X 开始录制，空格或 X 结束；S/Y 成功，F/Back 失败，U 暂不标注；K 仅保存参考姿态及图片；Q 退出。
- 夹爪由手柄控制：按住 LB 后，B / RT 张爪、A / LT 闭爪；页面 O/C 按钮不显示。实际开口作为第七维状态记录。H 自动返回尚未接入。开始录制不自动张爪，不自动返回，不要求固定起始视角。
- 相机失效、反馈过期、Servo 停止/失联或录制节拍丢失会终止控制与当前录制。不完整 episode 保留为 failed，不能标为成功。处理后重新启动命令，不自动重启控制。
- 正常退出先停止 ROS 生产目标，再请求 worker 实测位置保持；子进程监测父采集进程是否还在。无自动软急停、复位、失能或模式切换。
- 轻微抖动会进入下一状态动作标签；当前功能可用于下一步现场联调，还不表示完整抓取采集已验证。

## 先用模拟环境查看界面（不需要实机或手柄）

```bash
conda activate lerobot
cd /home/eureka/piper_ws/smolvla_deploy
python -m e14.collect_data \
  --config config.mock.json --teleop servo --mock \
  --output data/mock_raw_servo \
  --task "Push the block." --episodes 3 --duration 10 --run-seconds 300
```

打开终端打印的 `http://127.0.0.1:8766`，点击 R 开始，结束后标注。mock 只生成带时间戳的模拟关节和图像，用于验证采集流程，不运行真实 ROS/SDK/手柄，不能用于训练真实策略或证明实机运动质量。

## 稍后现场联调

使用既有 `config.local.json` 的相机设备、分辨率、状态顺序与采样率，不需要另外填写保存路径到 JSON。相机名称继续沿用配置（例如 front，即使物理上装在腕部）；暂不改名以免破坏已有模型映射。

### 一键启动的精细档与相机自检

`start_demo.sh` 的 XYZ 控制保持 LB 允许运动，同时按住 LB + RB 才进入精细档；只按 RB 不移动。
本机 `servo_bridge/config.local.json` 使用 `fine_speed_scale: 0.5`：普通档平移 60 mm/s、
旋转上限约 17.2°/s；精细档平移 30 mm/s、旋转上限约 8.6°/s（满输入、稳定后）。
夹爪仍为目标开口速度 100 mm/s，不受 RB 缩放。

原自检在主线程直接调用 `cv2.VideoCapture` / `cap.read()`，没有整个调用的超时保护；
若驱动在第一次读取中阻塞，后续三次重试无法执行。现在自检调用 `e14.camera_check`，
在独立子进程中复用与关节 demo 和正式采集相同的 `CameraReader`，校验图像尺寸、格式和新鲜度。
总超时默认 45 秒，超时会终止检查子进程并报错，机械臂不会进入后续启动移动。
可通过 `CAMERA_CHECK_TIMEOUT=60 ./start_demo.sh` 调整自检总时间。
正式采集仍会重新打开相机；硬件、带宽、驱动或其他程序占用设备的问题仍需处理。

关节 demo 的 24°/s / 8°/s 与 XYZ 平移的 mm/s 不能直接等同，末端速度与姿态、力臂和多个关节配合有关。
两者执行层的关节速度/加速度上限相同；XYZ 模式还保留奇异性和姿态保持约束，因此不能保证所有姿态下与关节直控一样快。

1. 按 [MOVEIT_SERVO.md](MOVEIT_SERVO.md) 完成 CAN/MOVE_J 和关节使能。实机输出仅需 `--execute`，无需授权文件或续期。采集入口不自动做这些操作。需要启用电脑 CAN 时可以添加 `--initialize-can`，它不是电机使能。
2. 关闭单独运行的旧遥操作、Servo runtime 和 worker。新采集命令会自行管理自己的两端子进程，不要再启动第二个 worker。
3. 启动命令（会允许 LB 手柄控制实机）：

```bash
conda activate lerobot
cd /home/eureka/piper_ws/smolvla_deploy
python -m e14.collect_data \
  --config config.local.json --teleop servo --execute \
  --gamepad-config gamepad.local.servo.json \
  --output data/piper_raw_servo \
  --task "Push the black foam block into the target area." \
  --episodes 3 --duration 30 --run-seconds 600
```

`gamepad.local.servo.json` 对应系统 ROS Python 的 SDL GUID，不能直接替换成 Conda 手柄配置。每条最多 30 秒，整个进程最多 600 秒；临近会话结束不要再开始新一条。抓取任务还需要后续接入并验证夹爪，当前先用移动/推块联调。

不加 `--execute` 时只读预览、审计手柄，不输出运动，也不允许录制运动示教。不要同时传 `--interactive`；它是旧手拖入口。

## 文件保存位置

以上实机命令写到 `/home/eureka/piper_ws/smolvla_deploy/data/piper_raw_servo/`：

```text
episode_000000/
  manifest.json          # 任务、结果、mock 标记、控制方式、时间规范
  frames.jsonl           # 实测六关节角度+夹爪宽度、图像路径、时间戳
  images/<camera>/       # RGB PNG
  events.jsonl           # 本条录制期间的操作键及 Servo 审计
_servo_sessions/<id>/
  runtime.log
  worker.log             # 仅 --execute 时有
  trace/events.jsonl     # 完整会话输入、候选目标、执行目标、反馈
```

每次录制沿用递增 episode 编号，不覆盖已有片段。相机和反馈继续执行新鲜度、时间对齐和固定节拍检查。动作标签仍由导出器使用 **state[t+1]** 生成，不把 Servo 候选目标或虚拟轨迹当作已执行动作。复制到 NAS 时保留整个 raw 根目录，包括 `_servo_sessions`；manifest 还记录本机 trace 路径作诊断索引。

完成至少两条成功示教后，原导出命令继续适用：

```bash
python -m e14.export_dataset \
  --raw-root data/piper_raw_servo \
  --output data/piper_dataset --repo-id local/piper_servo
```

先联调少量数据再批量录制，检查抖动、图像模糊、夹爪状态和真实运动方向。模拟数据与真实数据分目录保存。
