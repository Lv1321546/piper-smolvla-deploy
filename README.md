# PiPER + SmolVLA：E14 / 服务器部署

代码入口已经实现；默认只读或 dry-run。模拟链路可以直接运行。真实策略训练和实机效果要用真实数据验证，当前没有 PiPER 微调模型。

## 放在哪里

- E14：/home/eureka/piper_ws/smolvla_deploy
- 服务器：/home/lvzx/workspace/piper_smolvla_deploy
- NAS 模型、数据、输出：/mnt/nas_ailab_434/Personal_File/eureka
- 一路固定 RGB 相机，命名 front。真实设备尚未填写，不会自动拿笔记本摄像头冒充固定相机。

所有命令都在代码根目录运行，使用 python -m，避免模块导入路径错误。

## 文件与作用

| 文件 | 运行端 | 功能 |
|---|---|---|
| e14/piper_robot.py | E14 | 只读反馈、逐 CAN 帧时效、单位转换、显式准备与发送、停止 |
| e14/cameras.py | E14 | 后台最新 RGB 帧、采集时刻、尺寸与超时检查 |
| e14/inference_client.py | E14 | 图像编码、HTTP 请求、编号/频率/维度与过期检查 |
| e14/run_robot.py | E14 | observe、dry-run、异步推理与固定频率执行闭环 |
| e14/collect_data.py | E14 | 只读示教采集，保存图像、状态和实际时间戳 |
| e14/export_dataset.py | E14 或服务器 | LeRobot 格式导出、next-state 标签、训练/验证分集 |
| server/policy_runner.py | 服务器 | 常驻模型、完整处理器、模型/统计/schema 校验 |
| server/inference_server.py | 服务器 | localhost HTTP health/predict，单请求推理、不累积动作请求 |
| server/train_policy.py | 服务器 | 官方 LeRobot 训练入口、离线验证和部署 manifest |
| shared/protocol.py | 两端 | 七维顺序、单位、协议、图片编解码 |
| config.example.json | 两端 | 真实配置模板，需要填写相机与模型路径 |
| config.mock.json | 两端 | 模拟机械臂/相机/策略配置 |

## 1. 先跑两端模拟链路

服务器终端（使用 VS Code Remote SSH，或自行 ssh Supermicro）：

    conda activate eureka312
    cd /home/lvzx/workspace/piper_smolvla_deploy
    python -m server.inference_server --config config.mock.json --mock

E14 的另一个终端建立隧道，保持运行：

    ssh -N -L 127.0.0.1:8080:127.0.0.1:8080 Supermicro

E14 运行：

    conda activate lerobot
    cd /home/eureka/piper_ws/smolvla_deploy
    python -m e14.run_robot --config config.mock.json --mock --duration 5

这时两个硬件输入和服务器策略全部是模拟值。加 --execute 也只会执行模拟机械臂，服务器返回的 mock 标记不能用于真实机械臂。

    python -m e14.run_robot --config config.mock.json --mock --execute --duration 5

日志位于 outputs/*-control.jsonl：含观测、动作、请求、耗时、异常和停止记录。运行前可用 curl http://127.0.0.1:8080/health 查看服务状态。

## 2. 接真实硬件，只读取

复制配置：

    cp config.example.json config.local.json

编辑 cameras.front.device，优先填写 /dev/v4l/by-id/... 稳定设备名；确认 width/height/fps。
核对 robot 的关节范围、夹爪完整开口宽度范围、每周期允许变化量。配置中数值是启动示例，不代表已实机标定。

只读机械臂：

    python -m e14.piper_robot --config config.local.json --samples 10

如果 CAN 尚未启用，在同一命令加 --initialize-can；只配置电脑网络接口，不使能电机。
只有确认接线和供电后才考虑 --initialize-can --restart-can。该选项不会复位机械臂控制器。

相机拍一帧：

    python -m e14.cameras --config config.local.json --output outputs/camera_check

合并只读观测，不访问服务器：

    python -m e14.run_robot --config config.local.json --observe --duration 5

connect 不切换控制模式、不使能、不发送目标。相机时刻是 cap.read 返回的主机接收时刻，不是传感器曝光时刻；真实 USB 缓冲与延迟仍需核对。

## 3. 采集与导出

腕上相机抓取任务可使用交互采集：

    python -m e14.collect_data --config config.local.json --interactive --initialize-can \
      --output data/piper_raw_pickdrop_v1 --episodes 50 --duration 30 \
      --task "Pick up the black foam block and drop it onto the mouse pad."

终端会打印本机预览地址（默认 http://127.0.0.1:8766），用浏览器打开。需要关闭之前的相机拍照程序，避免占用同一设备。
交互模式在 CAN 模式下会尝试使能并逐步张开夹爪；它不是只读模式，此时不要直接手拖关节。实体示教下能读角度，不代表固件同时支持软件夹爪控制。当前 S-V1.8-3 实机已对照验证：实体示教下 98–100 mm 指令无效，退出示教并选择 CAN 后同样指令有效。因此当前设备不能直接按“手拖 + O/C”流程采集抓取任务，需要兼容固件或 CAN 模式遥操作（后者尚未实现）。

| 按键 | 操作 |
|---|---|
| O / C | 夹爪张开 / 闭合，目标平滑变化；页面显示实际宽度 |
| K | 记录当前六关节作为初始姿态，并保存参考图片，不移动 |
| H | 在录制外返回初始姿态，完成后张爪 |
| R | 等待夹爪实际张开后开始录制，不自动移动到初始姿态 |
| 空格 | 提前结束录制，进入结果标注；返回过程中则取消并调用配置的停止策略 |
| S / F / U | 标注成功 / 失败 / 暂不标注；录制中按 S/F/U 会先结束再标注 |
| Q | 退出采集；不会自动松爪，不能代替实体急停 |

按键只在预览网页获得焦点时有效，也可以点击对应按钮。页面约 10 FPS 显示最新图像，原始采集仍按 robot.control_hz（默认 20 Hz）保存。
实时图像读取和显示不依赖 opencv GUI，预览服务只监听本机回环地址。
--duration 在交互模式中是每条的最长时间；手动结束仍保持均匀采样。没有标注上一条结果前不会开始下一条。
每条开始前人工确认视野；可以适度改变初始姿态，程序不要求与 K 记录的角度完全相同。

初始姿态默认保存在 outputs/teaching_home.json，参考图片位于同目录 teaching_home-front.png。
可用 --home-file 指定其他任务的初始姿态。H 使用原有运动检查：实机须先人工退出示教、完成停止策略验收后配置 robot.stop_strategy_verified=true；不会发送复位或强制退出按键示教。
H 返回后关节处于位置控制，需要重新按实体按钮进入拖动示教再录下一条。录制期间禁止返回。
这些六关节运动只由 H 触发；启动、R、O、C 不发送六关节运动目标。
初始姿态不是避障规划，返回路径必须在实际空旷工作区核对。

夹爪参数默认 --open-width 0.06 --close-width 0 --gripper-speed 0.03（完整开口米、米/秒）；力矩沿用 robot.gripper_effort（SDK 单位，默认示例 500）。
robot.enable_gripper 控制策略执行的夹爪动作，不禁用用户显式选择的交互夹爪控制。
SDK 可以发送独立 GripperCtrl，但当前已测固件在实体按钮示教模式下忽略它。交互采集器默认不再向真实示教模式发送夹爪目标，也不进入抓取录制，仍提供相机预览、状态读取及 K 姿态保存。CAN 模式下可用 O/C 验证夹爪，但当前手拖采集流程不允许在 CAN 下录制；不会自动切模式、复位或清错。
仅在其他固件已实测支持该组合后，才可在 collection 中添加 "gripper_in_teaching_supported": true；改这个配置不会使不兼容固件获得该能力。模拟模式的 O/C 与录制功能照常用于软件验证。
先在空夹爪下验证 O/C 和实际开口变化，再采抓取数据；闭合目标不是“抓取成功”的自动判定。

夹爪边界反馈允许最多 3 mm 的测量偏差，仅用于交互夹爪的爬升起点，例如 102.1 mm 的反馈先按 100 mm 起点计算合法指令，目标仍严格限制在配置范围内。页面与原始数据保留实际反馈；这不是重设零点，也不扩大机械行程。偏差更大时暂停夹爪并提示，不直接关闭预览。
发送后超过 2 秒仍未观测到执行反馈，或张开超过预计时间仍不到位，会暂停重复发送、取消待开始的录制，按 O/C 才重试。闭合接触物体后不要求达到 0 mm；完全没有变化也可能是已经接触物体，需操作者检查。
录制中发生控制/反馈/相机错误，该条标记 failed，不能导出为有效训练数据；预览和按钮仍保留。K 保存的是六关节原始参考角度，不再因夹爪反馈偏差失败。姿态超出目标范围时只保存并提示，H 仍检查目标和运动条件。

如果实体示教时 O/C 始终不动，先解决控制方式再批量采集：向厂商核对当前固件是否支持“六关节手拖 + 软件夹爪”，或使用 CAN 模式下的遥操作。不能把切换 CAN 并使能后的机械臂继续当作可手拖机械臂。本项目尚未实现手柄遥操作；仅修复采集器无法保证现有固件支持混合控制。官方同类讨论见 [piper_sdk issue #11](https://github.com/agilexrobotics/piper_sdk/issues/11)。

原始目录仍使用 manifest.json、frames.jsonl、images/front/*.png；交互模式额外写 events.jsonl 保存按键和夹爪目标，operator_session 保存初始关节状态。
训练标签仍是下一帧真实反馈，events.jsonl 不会替代标签。录制结束及退出时不会自动开爪，防止持物时意外掉落。

不连接实机的交互练习：

    python -m e14.collect_data --config config.mock.json --mock --interactive \
      --output data/mock_interactive --episodes 2 --duration 10 \
      --task "Pick up the black foam block and drop it onto the mouse pad."

先用模拟数据学习流程：

    python -m e14.collect_data --config config.mock.json --mock \
      --output data/mock_raw --task "Pick up the cube and place it in the box." \
      --episodes 3 --duration 2 --outcome success

真实示教时去掉 --mock，换 config.local.json 和独立输出目录。操作者需要先按 PiPER 按钮进入适合的示教状态；脚本本身只读。
--outcome success 是操作者的明确标注，录制正常完成不等于任务成功。不确定时保留默认 unknown。
该采集入口为固定时长，每条结束后自动开始下一条；真实采集初期建议 --episodes 1，摆好场景后逐条录制。

导出：

    python -m e14.export_dataset --raw-root data/mock_raw \
      --output data/mock_dataset --repo-id local/piper_mock

只有 complete 且成功的演示默认纳入，unknown 需 --include-unknown 明确放行；失败演示不会自动当成功训练数据。
图像先用 PNG / LeRobot 嵌入图像，避免视频编码依赖。后续需要大规模视频存储时再扩展。

动作语义是 action[t] = state[t+1]，表示下一采样时刻的位置目标。
它不是采集过程中实际发出的 SDK 命令，最后一个原始状态没有下一帧标签，因此不导出。
缺帧、重复反馈、重复相机帧或不均匀采样会被拒绝，不会偷偷补成恒定频率。
输出包含 split.json、train_stats.json、deployment_schema.json、export_manifest.json；训练/验证按 episode 划分，统计量只取训练集。
模拟数据带 mock 标记，只用于验证数据管线，不会生成可执行的真实 PiPER 模型。

上面的模拟导出目录是 `data/mock_dataset`，只用于本地流程验证，暂时不需要上传服务器。
下面的 `data/piper_dataset` 指后续真实数据导出时选用的目录；必须先生成该目录，不能直接接在模拟采集命令之后执行。

真实导出数据上传 NAS（从 E14 发起，目录必须是自己的）：

    ssh Supermicro 'mkdir -p /mnt/nas_ailab_434/Personal_File/eureka/datasets'
    scp -r data/piper_dataset Supermicro:/mnt/nas_ailab_434/Personal_File/eureka/datasets/

## 4. 服务器训练入口

在服务器使用真实数据先检查计划：

    python -m server.train_policy \
      --dataset-root /mnt/nas_ailab_434/Personal_File/eureka/datasets/piper_dataset \
      --base-model /mnt/nas_ailab_434/Personal_File/eureka/models/smolvla_bundle/smolvla_base \
      --vlm-path /mnt/nas_ailab_434/Personal_File/eureka/models/smolvla_bundle/smolvlm \
      --output-dir /mnt/nas_ailab_434/Personal_File/eureka/outputs/piper_run_001 \
      --steps 5000 --batch-size 8

默认只验证并打印计划，不启动训练。准备好真实数据和已分配 GPU 后，加 --execute；在命令前用 CUDA_VISIBLE_DEVICES=已分配卡号 选择 GPU。
服务器读取数据集时还会生成 Arrow 缓存，建议在运行前设置：

    export HF_HOME=/mnt/nas_ailab_434/Personal_File/eureka/huggingface
    export HF_DATASETS_CACHE="$HF_HOME/datasets"

步数和 batch_size 只是起点，需要按实际数据和验证结果调整。输出目录必须是新目录。

六维基础模型的 state/action 投影有更大的 padding 维度；脚本检查真实权重投影形状，再配置 PiPER 七维和真实相机，用官方训练循环微调。
不会把 PiPER 七维输入裁成六维，也不会把修改 JSON 当作训练完成。
训练使用 train_stats.json，不把验证集统计量用于归一化。
训练结束后进行 held-out 数值验证，生成 deployment_manifest.json 和 deployment_stats.json；数值有限和离线误差不证明实机任务成功。

## 5. PiPER 模型服务与实机闭环

将服务器配置中的 model_path 指向训练输出的 checkpoints/last/pretrained_model；
vlm_path 指向本地 smolvlm；camera_map 与数据集实际相机匹配（默认 front -> observation.images.front）。
先启动真实模型服务：

    python -m server.inference_server --config config.local.json

E14 先 dry-run：真实观测传到服务器，只保存动作。

    python -m e14.run_robot --config config.local.json --duration 5

物理执行需显式 --execute，并满足真实 PiPER 模型、反馈/相机时效、单位/频率匹配与已核对的停止配置。
执行开关以命令行 --execute 为准；没有该参数始终不发送动作，配置中的 dry_run 仅为默认意图说明。
prepare_control 会切到 CAN/MOVE_J 并使能；夹爪控制必须按实际需要配置 enable_gripper=true，夹爪力矩单位为 SDK 的 0.001 N·m。
示教模式必须先按机械臂按钮退出。不会自动复位、清错或断使能。

    python -m e14.run_robot --config config.local.json --execute --duration 2

默认 stop_strategy=emergency_stop：结束或异常时发送 MotionCtrl_1(0x01,0,0)。
这是锁停，不会自动发送恢复/复位；后续恢复应按实际固件和官方流程处理。
stop_strategy=hold 仅允许用新鲜、健康的当前反馈发送保持目标。两种停止方式都需在真实机械臂核对后才设置 stop_strategy_verified=true。
CAN 断开时软件无法保证停止；硬件停止方式由现场操作者掌握。

动作分块按 E14 本机时间对齐：跳过网络往返期间已经过去的动作，新块原子替换旧块；超时/块耗尽/反馈过期会停止，旧块不会在重连后继续执行。
真实控制时推理和本地控制分线程；不会让控制循环阻塞等待 HTTP 返回。

## 测试与限制

    python -m unittest discover -s tests -v

测试包含模拟 SDK/CAN、相机阻塞/超时、HTTP 协议、数据真实格式写入与读取、训练计划和完整模拟闭环。HTTP 测试需允许 localhost socket。
依赖沿用现有环境：E14 lerobot、服务器 eureka312；基础包 numpy、Pillow、LeRobot，真实相机 OpenCV，数据导出需 datasets/pyarrow，训练需 SmolVLA extras。
测试写缓存时可设置 HF_HOME=/tmp/piper-test-hf，避免污染原有缓存。

本次没有自动运行真实机械臂或训练。未配置真实相机设备、尚无 PiPER 任务数据/微调模型，实机成功率和真实停止行为不能通过模拟测试替代。
审查和测试记录见 REVIEW.md。

## GitHub 版本管理

已接入私有仓库 [piper-smolvla-deploy](https://github.com/Lv1321546/piper-smolvla-deploy)。两端同步方式见 [GIT_WORKFLOW.md](GIT_WORKFLOW.md)。
