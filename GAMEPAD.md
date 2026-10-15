# Gamepad_PiPER 手柄遥操作与采集

功能在 E14 上运行，服务器不参与手柄控制。用于在 CAN 模式下操作六关节、夹爪并录制演示，不需要先训练 SmolVLA。不要同时启动策略执行程序。

采用“独立 clone 留作参考 + 项目内选择性集成”：[上游 Gamepad_PiPER](https://github.com/kehuanjack/Gamepad_PiPER) 固定在 `aa00c80e100d687a5f4a386b79b83ece06b3a30f`，参考目录 `/home/eureka/piper_ws/_references/Gamepad_PiPER`。运行时不依赖该目录。保留 MIT 许可证、URDF 和改编的 Linux 输入映射，见 `third_party/gamepad_piper/README.md`。未安装上游旧 NumPy 限制、Trac-IK、ROS 或自动回零入口。

## 先运行模拟演示

已在 E14 的 lerobot 环境安装 pygame 2.6.1。重新安装环境时执行 `python -m pip install -r requirements-gamepad.txt`。

```bash
conda activate lerobot
cd /home/eureka/piper_ws/smolvla_deploy
python -m e14.gamepad_teleop --config config.mock.json --mock --demo \
  --output data/mock_gamepad --episodes 2 --duration 10
```

打开终端打印的地址，默认 `http://127.0.0.1:8766`。模拟手柄每 12 秒循环：回中、按住 LB 向前、松开、向后并闭爪。网页图像是模拟相机的彩色测试图，不是工作台或抓取成功的证据。

点击 R 开始，空格提前结束，S 标为成功。duration 是每条最长时间，episodes 是本次标注条数；未完成上一条标注不能开始下一条。Q 退出。

加 `--mujoco` 可打开简化机械臂的三维窗口：

```bash
python -m e14.gamepad_teleop --config config.mock.json --mock --demo --mujoco
```

简化模型用同一 URDF 的关节变换生成，不依赖外部 meshes。它镜像模拟关节，只调用 mj_forward，不运行物体动力学、碰撞规划或抓取仿真；网页相机仍是测试图。已有完整场景可用 `--mujoco-scene /绝对路径/scene.xml` 替代，需含 joint1–joint8，且其 include/mesh 路径必须有效。模型手指范围 0–70 mm，超出的镜像值只裁剪显示，不改变原始反馈或控制目标。

## 手柄到场后先核对输入

优先 USB 数据线连接，选 XInput / Xbox 兼容模式。轴号需按这只手柄在 Ubuntu 下的实际输出确认。

```bash
cp gamepad.example.json gamepad.local.json
python -m e14.gamepad --config gamepad.local.json --monitor --duration 30
```

此命令只读手柄，不导入 PiPER SDK、不打开 CAN。逐个操作输入，核对 raw_axes、raw_buttons、归一化 axes/buttons。扳机松开必须接近 0；若原始轴松开为 0、按满为 1，设置 trigger_min=0、trigger_max=1。默认是松开 -1、按满 1。轴号、按钮号、死区和可选 GUID 写在 gamepad.local.json，该文件被 Git 忽略。

让真实手柄控制模拟机械臂：去掉 demo，保留 mock。

```bash
python -m e14.gamepad_teleop --config config.mock.json \
  --gamepad-config gamepad.local.json --mock --mujoco
```

先松开 LB，让摇杆、方向键和扳机回中，再按住 LB。启动时已按住 LB 不会立即运动。选择手柄后绑定 GUID；活动期间断连或设备实例更换会锁住控制，需退出后重新启动。

## 按键

| 输入 | 功能 |
|---|---|
| 按住 LB | 允许六关节/夹爪操作；松开后发送一次当前位置保持目标，停止新增夹爪目标 |
| 按住 RB | 在 LB 基础上采用 25% 精细速度 |
| 左摇杆前后 / 左右 | 末端模式 X / Y；方向以 URDF base_link 为准，不是相机坐标 |
| 右摇杆上下 | 末端模式 Z 升降 |
| LT / RT | 夹爪逐步闭合 / 张开 |
| A / B | 闭合 / 张开的替代按钮 |
| X | 开始录制，再按结束 |
| Y / Back | 成功 / 失败并保存，录制中先结束 |
| Start | 锁住遥操作并请求配置的软件停止，不能代替实体急停 |
| 网页 O / C | 设置夹爪目标，仍须按住 LB 才发送 |
| 网页 R / 空格 / S / F / U | 开始 / 结束 / 成功 / 失败 / 暂不标注 |
| 网页 K | 保存当前参考姿态及图像，不移动 |
| 网页 Q | 退出，不自动张爪、失能、复位或归零 |

默认 cartesian 模式保持每次按下 LB 时的末端朝向，线速度 0.02 m/s。局部逆解失败时保持位置，不累积跳到其他解。相对工作区在首次激活时建立，XYZ 各 ±0.15 m，仅是附加范围限制，不是障碍物检测。

`--mode joint` 调六关节：左摇杆左右→J1、上下→J2，右摇杆上下→J3，方向键左右→J4、上下→J5，右摇杆左右→J6。默认速度 8°/s，最大单步和跟踪差沿用 robot 配置。

仅 `gamepad.allow_rotation=true` 启用末端姿态操作：方向键上下→绕 X、左右→绕 Y，右摇杆左右→绕 Z，均在基座系。默认关闭，需另外验证方向。

## 实机入口与当前边界

只读预览真实机械臂、相机和手柄：

```bash
python -m e14.gamepad_teleop --config config.local.json \
  --gamepad-config gamepad.local.json --initialize-can
```

不加 execute 时不使能、不切模式、不发运动目标，也不录制有效遥操作数据。正式操作在上面加 `--execute`，会选择 CAN/MOVE_J、使能六关节，不自动归零、返回或张爪。须先用实体按钮退出示教。

停止策略与恢复方式现场验收后，才由操作者在本地 robot 配置记录 stop_strategy_verified=true；本次没有替用户开启该开关，也没有执行实机运动。没有真实手柄，轴映射、基座方向、URDF 与硬件一致性、跟踪和停止效果尚未现场验收。运动学预览不能验证桌面、自碰撞或线缆干涉。

此入口不自动 H 返回；通过手柄移到能看见方块和鼠标垫的位置、张爪，再录制。CAN/相机过期、控制超时、Start 或活动时手柄断连，会停止当前录制并锁住控制；停止请求只发送一次，恢复硬件后重新启动，不自动清错恢复。

## 数据位置与导出

output 相对代码目录，例如 data/mock_gamepad/episode_000000/：

- manifest.json：任务、结果、mock 标记、手柄模式/配置和上游版本。
- frames.jsonl：时间戳、六关节角度和完整夹爪开口。
- images/front/*.png：图像，真实运行时为腕上相机。
- events.jsonl：按键、手柄输入及拟发送/发送目标，供审查。

至少录制并标注两条成功演示，才能分开训练集和验证集：

```bash
python -m e14.export_dataset --raw-root data/mock_gamepad \
  --output data/mock_gamepad_dataset --repo-id local/piper_gamepad_mock
```

训练标签仍是 action[t] = measured_state[t+1]，不直接以控制命令替代反馈。失败/未标注片段按原规则处理。mock 数据只验管线，不能生成可部署的真实 PiPER 模型。

模块：gamepad 读输入；gamepad_kinematics 用 NumPy 做 FK/Jacobian/局部 DLS IK；gamepad_control 做限速、LB/断连状态机、独立夹爪；gamepad_teleop 接现有预览和采集；gamepad_sim 提供可选 MuJoCo 关节镜像。原有策略推理和手拖入口保留。
