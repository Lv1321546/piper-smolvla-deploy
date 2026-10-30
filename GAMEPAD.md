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

加 `--mujoco` 打开三维窗口。本机现已安装松灵官方 PiPER 模型，该选项会优先使用它：

```bash
python -m e14.gamepad_teleop --config config.mock.json --mock --demo --mujoco
```

模型来源是[松灵官方 piper_ros](https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description/mujoco_model/piper_description.xml)，固定版本 ac41fcbcdda598f01b51cf6175ed9a24d0dacadc。XML、10 个 STL 和 MIT 许可证保持原样，放在 models/piper_official；scene.xml 是本项目添加的地面、相机和物体包装场景。source_manifest.json 记录来源和文件哈希。无需安装 ROS 或整个 piper_ros。

新电脑或重新 clone 后下载一次（模型目录被 Git 忽略）：

```bash
python -m e14.install_mujoco_model
```

已安装文件会核对哈希，文件不同则报错，不覆盖。没有安装时 --mujoco 仍可使用内置简化模型，终端会打印当前模型路径。使用其他模型可指定 `--mujoco-scene /绝对路径/scene.xml`，需含 joint1–joint8、link6 末端，以及有效的 include/mesh 路径。

仿真末端 IK 使用当前显示模型的 FK/Jacobian 与限位，实机仍沿用原 URDF 和控制检查。窗口镜像模拟关节，只调用 mj_forward，不运行物体动力学、碰撞规划或抓取仿真；网页相机仍是测试图。官方模型手指范围 0–70 mm，超出的镜像值只裁剪显示，不改变原始反馈或控制目标。

移动时必须持续按住 LB/L1，再拨动摇杆；只拨动摇杆不会移动。先松开所有输入回中，再按下 LB。关闭三维窗口后当前控制会锁定：退出该终端程序后重新运行。使用此入口自己打开的窗口，手柄状态才会同步进去。

## 手柄到场后先核对输入

优先 USB 数据线连接，选 XInput / Xbox 兼容模式。轴号需按这只手柄在 Ubuntu 下的实际输出确认。

2026-10-19 E14 实际连接识别为 Sony Computer Entertainment Wireless Controller（6 轴、13 按钮、1 方向键）。已从 SDL 读取映射；该兼容模式的 X/Y 为 b3/b2，Back/Start 为 b8/b9，与 Xbox 默认表不同。对应模板是 gamepad.ds4.example.json，本机 gamepad.local.json 已填写并绑定当前 GUID。真实逐键操作仍需核对；仅识别静止输入不代表所有按键已通过测试。

已修正真实手柄第二次轮询崩溃的问题：Joystick 使用 [get_init 与 JOYDEVICEREMOVED](https://www.pygame.org/docs/ref/joystick.html) 检查初始化和拔出，不再误用 Controller 的 get_attached。真实静止输入已连续读取成功。

```bash
cp gamepad.example.json gamepad.local.json
python -m e14.gamepad --config gamepad.local.json --monitor --duration 30
```

此命令只读手柄，不导入 PiPER SDK、不打开 CAN。逐个操作输入，核对 raw_axes、raw_buttons、归一化 axes/buttons。扳机松开必须接近 0；若原始轴松开为 0、按满为 1，设置 trigger_min=0、trigger_max=1。默认是松开 -1、按满 1。轴号、按钮号、死区和可选 GUID 写在 gamepad.local.json，该文件被 Git 忽略。

已有本地配置时跳过 cp，避免覆盖；Sony/DS4 兼容模式首次建立配置时用 gamepad.ds4.example.json。切换 USB/无线/兼容模式后 GUID 可能改变，应重新确认输入和映射。

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
| 左摇杆前后 / 左右 | 末端模式 X / Y；向前对应 +X，向右对应 -Y，以当前控制模型的基座系为准 |
| 右摇杆上下 | 末端模式 Z 升降；向上对应 +Z |
| 右摇杆左右 | 末端 Yaw，绕基座 Z 轴；向右为正 |
| 方向键上下 | 末端 Pitch，绕基座 Y 轴；向上为负、向下为正 |
| 方向键左右 | 末端 Roll，绕基座 X 轴；向右为正、向左为负 |
| LT / RT | 夹爪逐步闭合 / 张开 |
| A / B | 闭合 / 张开的替代按钮 |
| X | 开始录制，再按结束 |
| Y / Back | 成功 / 失败并保存，录制中先结束 |
| Start | 锁住遥操作并请求配置的软件停止，不能代替实体急停 |
| 网页 O / C | 设置夹爪目标，仍须按住 LB 才发送 |
| 网页 R / 空格 / S / F / U | 开始 / 结束 / 成功 / 失败 / 暂不标注 |
| 网页 K | 保存当前参考姿态及图像，不移动 |
| 网页 Q | 退出，不自动张爪、失能、复位或归零 |

当前本机 teleop 回到低速对照：平移 0.01 m/s（10 mm/s）、旋转 0.05 rad/s、关节 3°/s，硬件速度比例 5%。这是用户反馈实机 XY 问题仍存在后的回退，尚待现场确认恢复。普通示例的速度仍较高，实机排查请使用下方本机 teleop 两份配置。

夹爪保留上一轮 0.05 m/s，表示完整开口宽度的变化速度；20 Hz 时每周期 2.5 mm，robot.max_step_delta[6] 保持至少 0.0025 m。本轮不再提高夹爪速度或力矩。

按住 LB 时可同时平移与旋转；没有旋转输入则保持当前目标朝向。RB 精细档仍为机械臂速度的 25%。加速、逐关节步长与反馈跟踪限制继续生效，接近限位或奇异姿态时未必能达到设定速度。gripper_speed_m_s 从配置读取，只有显式传 --gripper-speed 才覆盖它；配置速度超过每周期允许值会在连接硬件前报错。

平移/旋转增量超过当前关节单步能力时，最多缩小到本周期增量的 1/128；仍无局部逆解则保留上一有效有界目标，等待跟踪，不撤回实测位置、不积累未接受的增量。网页显示缩步比例或等待原因。松开 LB 仍会请求实测位置保持并清空参考。相对工作区在首次激活时建立，XYZ 各 ±0.15 m，仅是附加范围限制，不是障碍物检测。

本机 `gamepad.cartesian_reference="measured"` 恢复早期以实测末端位置为起点的 XYZ 增量；`integrated` 会沿上一目标位置积分，保留供离线比较。发送层跟踪差/目标步长上限仍为原配置值。实测参考模式在规划时保留实测附近 75% 的余量，上一发送目标窗口留出 SDK 取整余量，不再额外将其压成原值的 75%。48 帧历史输出对照通过不代表实机已恢复。

保持末端朝向的 XYZ 运动在接近奇异姿态时可能很慢，缩步无法消除运动学限制。先退出程序，用已掌握的方式调整为肘部明显弯曲、腕部轴线不过分对齐的姿态，再退出实体示教并重新启动；不会自动改变腕部朝向来换取平移速度。

逆解同时约束目标相对实测角度、上一条成功发送目标的变化，避免电机跟踪稍慢时反向操作造成目标跳步。发送前仍重新检查新鲜反馈；真正超限时会锁住遥操作，并显示具体 J1–J6、跟踪差或上一目标步长。先松开 LB、查看反馈和现场状态，再退出并重新启动；不要通过提高限制值消除提示。

`--mode joint` 调六关节：左摇杆向右→J1 正向、向前→J2 正向，右摇杆向上→J3 正向，方向键向右→J4 正向、向上→J5 正向，右摇杆向右→J6 正向。最大单步和跟踪差沿用 robot 配置。**该模式直接转动关节，末端会沿弧线运动；需要末端水平前后移动时必须用 --mode cartesian。** 页面会显示中文模式、有效映射及关节模式提示。

本机配置和示例均启用 allow_rotation。按用户更正，撤销上一轮四组反转，gamepad.input_directions 六项全部恢复 +1，joint/cartesian 均恢复上一轮修改前的方向。更早已设置的 invert_y=true / invert_pitch=true 保留，未一并撤销。需要单独改方向时，可将 input_directions 相应项在 +1/-1 间切换；页面显示各层相乘后的最终有效方向。所有末端旋转均为基座系 X/Y/Z 轴的增量。配置修改后需重启程序生效。

## 实机入口与当前边界

只读预览真实机械臂、相机和手柄：

```bash
python -m e14.gamepad_teleop --config config.local.json \
  --gamepad-config gamepad.local.json --initialize-can
```

不加 execute 时不使能、不切模式、不发运动目标，也不录制有效遥操作数据。正式操作在上面加 `--execute`，会选择 CAN/MOVE_J、使能六关节，不自动归零、返回或张爪。须先用实体按钮退出示教。

只测机械臂和手柄、未接摄像头时，可加 `--no-camera`。页面显示占位测试图并标明相机未接；此时禁用录制和带参考图的姿态保存。默认仍只读，不能靠这个参数绕过执行门槛。

停止策略与恢复方式现场验收后，才在对应本地配置记录 stop_strategy_verified=true。2026-10-19 首次有界 J1 测试失败并触发下落；随后完成原位保持、J1 小幅运动及修复后的真实手柄试验。J1 实际变化约 4.56°，其余关节反馈不变，hold 后稳定。原配置仍保留 false，另外保存经测试的低速 hold 本地配置。基座方向、URDF 与硬件一致性、其余关节、LB 释放分支和故障处理尚未完整验收。运动学预览不能验证桌面、自碰撞或线缆干涉。

本机使用末端平移/旋转控制（已接机械臂和手柄、未接相机）：

```bash
python -m e14.gamepad_teleop --config config.local.teleop.json \
  --gamepad-config gamepad.local.teleop.json --no-camera --mode cartesian --execute
```

该配置当前为平移 10 mm/s、旋转 0.05 rad/s、关节模式 3°/秒；RB 精细速度 25%、硬件速度比例 5%，停止策略为 hold，平移参考为 measured。需要单关节调试时才改成 --mode joint。原位保持与 J1 已实测，本次低速回退的 XYZ 效果仍待现场核对；软件测试不能替代现场验证。进入实体示教后，仍须先退出并确认 CAN 模式。不接相机时不采集训练数据。

**PiPER 的 emergency_stop 不能当作位置保持。** 官方[停止说明](https://github.com/agilexrobotics/piper_ros/blob/noetic/README%28EN%29.md)注明会带阻尼下落；[服务代码](https://github.com/agilexrobotics/piper_ros/blob/noetic/src/piper/scripts/piper_ctrl_single_node.py)调用的正是 `MotionCtrl_1(0x01,0,0)`。`hold` 是另一条保持位置目标的路径，已在本次空载姿态下验证；其他姿态和故障状态未完整验收。不能只放宽阈值或改 verified 来替代实测，也不自动复位。

此入口不自动 H 返回；通过手柄移到能看见方块和鼠标垫的位置、张爪，再录制。CAN/相机过期、控制超时、Start 或活动时手柄断连，会停止当前录制并锁住控制；停止请求只发送一次，恢复硬件后重新启动，不自动清错恢复。

关节模式保留连续积分参考，确保很小的摇杆输入可以累积到电机能响应的幅度。Cartesian 使用本机配置选择的实测参考；两种模式的发送目标都经过跟踪和步长检查。松开 LB 会清空参考并请求实测位置保持，下一次激活从当前反馈重新开始。

实机 `--execute` 即使不录制，也会在 `outputs/teleop_trace/时间-随机编号.jsonl` 保存每周期手柄输入、实测角度、规划参考/目标、成功发送目标、耗时和错误。终端和网页显示实际日志路径，以及版本 `20261019-measured-baseline`、平移参考和速度。可用 `--trace 路径.jsonl` 指定文件，已有文件不会覆盖。这是排障日志，不是训练数据；不需要相机或按录制键。

独立的只读 CAN 诊断可在停止状态运行，不要求使能或进入 CAN 控制模式：

```bash
python -m e14.diagnose_can --config config.local.json --duration 5
```

默认生成 `outputs/can_diagnostic_时间戳.jsonl`，包含原始 CAN 帧、主机接收时间、总线时间戳、反馈快照、结束或失败摘要。可以在另一个终端与后续人工测试同时运行，以保留操作前后的原始记录。它只接收，不启用或重启 CAN、不查询固件、不切模式、使能、停止或复位；记录到的控制帧可能来自其他总线用户。CAN 接口须已启用，已有日志不会覆盖。

安装版 PiPER SDK 的 SocketCAN 发送端使用 local_loopback=False，因此同机被动监听不能记录到这些本机发送帧。复测还须在发送调用处记录 ID、payload 和返回值，不能把被动日志没有控制帧当作没有发送控制指令的证明。

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
