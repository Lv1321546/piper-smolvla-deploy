# PiPER SmolVLA 真机推理

本入口使用已经训练并完成离线选模的 **4,000 步 checkpoint**。E14 读取机械臂与两路 RGB，服务器执行 SmolVLA，E14 生成受限的位置指令。默认只读推理；只有 `--execute` 会切换 CAN/MOVE_J、使能并输出。没有授权文件、定时授权、自动回零或自动张爪。

## 代码位置

- E14 仓库：`/home/eureka/piper_ws/smolvla_deploy`。
- Inspur 独立部署快照：`/home/lvzx/piper_smolvla_work/code/deploy_20261106`。保留训练快照与训练结果。
- 模型：`/mnt/nas_ailab_434/Personal_File/eureka/piper_smolvla/archive/pickplace_20261105_lrfix_2156/selected`。
- E14 环境：`/home/eureka/miniconda3/envs/lerobot/bin/python`。
- Inspur 环境：`/home/lvzx/.conda/envs/piper_smolvla_20261105/bin/python`。

服务器只需要模型、LeRobot 和 CUDA；E14 不加载模型、不需要 GPU，也不需要为这条推理链启动 ROS/MoveIt。

## 1. 在服务器启动模型

从 E14 用 `ssh Inspur` 或 VS Code 进入服务器后：

```bash
cd /home/lvzx/piper_smolvla_work/code/deploy_20261106
PIPER_GPU=1 bash scripts/start_policy_server.sh
```

默认使用物理 GPU1。脚本启动前检查所选卡的计算进程和显存占用；被占用时退出，不结束其他任务。RLT 继续使用物理 GPU0。

服务器校验 checkpoint、统计与预处理文件的 SHA256，严格恢复权重，执行两次预热后监听 `127.0.0.1:8081`。看到 `Listening ... backend=smolvla execution_ready=True` 再继续。首次加载需要读取 NAS 上约 1.2 GB 的权重。`execution_ready` 表示模型与协议可用，实机状态仍由 E14 检查。

如需退出 SSH 后保留服务，先 `tmux new -s piper_inference`，再执行上述启动命令。`Ctrl+B`、`D` 脱离 tmux；`tmux attach -t piper_inference` 返回；服务器终端 `Ctrl+C` 停止服务并释放显卡。

其他服务器可通过 `PIPER_SERVER_PYTHON` 指定兼容环境、通过 `PIPER_GPU` 指定空闲物理卡。无需更改 NVIDIA 驱动或既有训练环境。

## 2. E14 建立 SSH 隧道

新开 E14 终端，保持此命令运行：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
python3 scripts/inference_tunnel.py Inspur
```

它复用当前 SSH 的主机、用户名、密钥与 `lab-desktop` 跳板，只转发推理端口。临时配置排除原 SSH 别名中的代理反向转发，避免其他电脑已占用该转发时冲突。不复制私钥，不把服务开放到局域网。

隧道终端 `Ctrl+C` 关闭连接。它只承载 HTTP；不会给机械臂下指令。

## 3. 首先用已有数据核对推理

这一命令完全不访问机械臂、CAN 或真实相机：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
conda activate lerobot
python -m e14.replay_inference \
  --config config.inference.example.json \
  --episode data/piper_raw_cartesian_dual/episode_000005 \
  --indices 0,100,300 \
  --output outputs/inference/replay_check_001
```

输出目录须不存在，避免覆盖旧结果。`summary.json` 记录真实模型身份、HTTP 往返与推理耗时；`predictions.jsonl` 保存完整的 `[50,7]` 动作、输入状态与下一帧示教状态。历史时间戳只用于审计，不能当作当前反馈时间戳。

## 4. 真机只读预览与预测

保持服务器和隧道运行，在 E14 新开一个终端：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
bash start_policy_inference.sh --observe --initialize-can --duration 10
```

只读模式不调用 SDK、不会使能或切换机械臂模式。`--initialize-can` 仅在需要时把电脑 CAN 接口设置为 UP / 1 Mbps，不复位机械臂；ERROR-PASSIVE / BUS-OFF 会报错，不自动重启 CAN。

随后运行真实图像和模型的预测，仍不发送动作：

```bash
bash start_policy_inference.sh --duration 30
```

浏览器打开 **http://127.0.0.1:8767**：显示两路实时图像、当前状态、原始预测、受限目标与计划指令。`sent_commands=0` 表示没有硬件输出。终端会打印日志目录。

首次运行会创建 `config.local.inference.json`。默认已填写本次采集使用的设备路径和模型，不需要手工填写数据标签或训练统计。只需核对两台相机的实际身份：

| 名称 | 训练时含义 | 输入 |
| --- | --- | --- |
| front | Dabai DC1，腕上主视角 | 640×480 RGB |
| astra | Astra Pro，外部辅助视角 | 640×480 RGB |

保持相机安装位置和视角与采集时一致。不能只接一路后用黑图代替另一路。若 USB 设备身份变化，只改 local 配置中的 `cameras.<name>.device`。

## 5. 后续现场执行

先退出手柄遥操作、采集或回放进程，将方块、盒子与起始视角恢复到采集时相近状态。实机短测入口：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
bash start_policy_inference.sh --execute --initialize-can --duration 10
```

任务文本默认与训练一致：`Pick up the black foam block and place it in the box.`。需要时用 `--task` 显式设置。

`--execute` 复用现有接管顺序：先请求当前关节位置保持，再确认 CAN/MOVE_J 与六关节使能；随后固定使用已用于手柄链路的 70% MOVE_J 模式速度，实际目标变化仍受下文的低速跟随器限制。夹爪未使能时只以当前开口使能，不自动张开。实体示教记录尚未结束会报错。无自动回零或移动到预设起点。

页面 **Q / 空格 / 停止按钮**，终端 **Ctrl+C**，或运行时限到达，均结束本次执行并请求实测关节位置保持，保留夹爪已有目标。网页关闭本身不会停止程序；软件保持不是硬件急停。反馈失效时不能保证通过 CAN 保持，日志会明确记录 `hold_unavailable`，不会改发让关节变软的停止或失能指令。

## 本机夹爪行程实测（2026-11-06）

本机固件通过夹爪参数查询返回 `max_range_config=100`。独立实机测试仅发送夹爪指令，以约 5 mm/s 从当前开口闭合到 0 mm，再张开到 100 mm；末端各保持 2 秒，最后 1 秒稳定反馈分别为 **0.5 mm** 和 **99.4 mm**。测试无故障，六关节反馈未变化，没有更改零点或固件参数。

本机 `config.local.inference.json` 与本设备示例 `config.inference.example.json` 的 `robot.gripper_limits_m` 因此设为 `[0, 0.1]`（米，夹爪总开口）。这是经过测试的**指令范围**；反馈端点偏差单独记录，不将 0.5/99.4 mm 当成新的零点，也不改变模型动作单位或训练统计。原有 1 mm 模型预测边界容差、限速和限加速度继续生效。模拟配置仍独立保留原 70 mm 范围。

此前 71.375 mm 的模型预测落在实测硬件控制范围内，不再因错误的 70 mm 上限而退出。训练集只覆盖约 46.3–69.6 mm 开口；扩展物理范围不代表模型学会了 100 mm 开口下的任务，推理前仍应恢复采集时的场景与开口。

测试证据：`/home/eureka/Documents/Codex/科研项目起步/outputs/gripper_range_20261106/run-20261106-171042/summary.json` 与同目录 `events.jsonl`。这些数值来自 CAN 编码器反馈，并非卡尺测量。

## 动作如何执行

- 模型输出固定顺序：J1..J6 的角度（度）与夹爪**总开口**（米）。训练标签是下一时刻实测状态，部署继续使用绝对位置，不解释为速度、增量或弧度。
- 每次预测为 50 步、20 Hz。HTTP 请求在后台运行，E14 本地循环不等待网络。根据 E14 自己的采样与接收时间选择当前对应的 chunk 索引；它不是零延迟执行，也不是完整的 RTC 算法。
- 默认动作最大年龄 1,000 ms，接收后最多使用 12 步，再由更新的预测替换。失去新预测、超时、模型身份变化或本地调度超限会结束执行；不继续消费陈旧动作。
- 原始预测先检查物理限位、与实测状态的距离。只容许配置的小边界噪声修正（关节 0.1°、开口 1 mm），原始值与修正值同时保留。其余异常会拒绝。只读模式继续显示拒绝原因，执行模式结束并请求保持。
- 本地位置跟随器上限为每关节 3°/s、15°/s²，夹爪 20 mm/s、80 mm/s²；内部预留量化和计时余量。独立 SDK guard 再检查整数化后的关节指令、跟踪偏差、速度与加速度。不会把较远的模型目标直接跳发给机械臂。
- 这条链不再进行 Cartesian 逆解，也不做碰撞路径规划。它执行模型给出的关节与夹爪位置。

代码入口：`e14/deploy_policy.py`（循环/预览）、`e14/policy_control.py`（位置跟随/SDK guard）、`server/policy_runner.py`（模型/预处理）、`server/inference_server.py`（HTTP）。原手柄和旧 `e14.run_robot` 入口保留。

## 日志、切换模型与回退

每次运行保存至 E14 `outputs/inference/policy-<日期时间>/`：

- `events.jsonl`：配置、模型身份、完整预测、计划/实际发送计数、拒绝原因与保持结果。
- `summary.json`：本次模式、预测次数、发送次数、结束状态。

如切换为备选 2,000 步模型，同时修改配置中的 `server.model_path` 与 `inference.expected_model_id`；只改其一会被身份检查拒绝。相机、动作顺序和统计不能随意替换。服务器预处理来自 checkpoint 中被校验的文件及训练集统计。

回退：先终止本次推理并检查位置保持，再退出推理服务/隧道；之后使用原来的 `start_demo.sh` 等遥操作入口。新入口与当前 Servo/Cartesian 手柄、采集和回放共享 `/tmp/piper-servo-sdk.sock.lock`，不能同时输出。旧 `e14.run_robot` 和仓库外的个人运动脚本也应退出，它们未接入该锁。不要手工删除锁来绕过正在运行的控制程序。

## 验证范围

本次只做模拟硬件、真实 checkpoint 和录制图像的网络推理验证，没有发送实机运动、夹爪、使能、模式或复位指令。离线通过不代表抓取成功率；现场仍需先看只读预测，再完成小幅输出和抓取测试。
