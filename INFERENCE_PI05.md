# PiPER π₀.₅：SuperSvr 三卡错峰推理

2026-11-10 已部署至 SuperSvr 的物理 GPU 0、1、2，各占约 9677 MiB 显存，其余 5 张卡未使用。每张卡独立运行同一模型；本机错峰发请求，不进行模型跨卡切分。

## 本机入口

`config.pi05.json` 已启用三个副本，原启动命令不变。真实传感器只读检查：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
bash start_policy_inference.sh --config config.pi05.json --duration 30
```

`DRY-RUN`、`sent=0` 表示不发送动作。模型与动作检查通过、现场起点与场景就绪后，短时实机运动入口为：

```bash
bash start_policy_inference.sh --config config.pi05.json --execute --duration 10
```

本次部署仅做真实传感器只读验证，没有发送 SDK 使能、夹爪或运动命令。物理运动中的轨迹衔接、保持效果和抓取成功率尚未验证。程序不会自动回到示教起点。

## 三端口 SSH 隧道

正式隧道已运行，本次启动 PID 为 `55099`，重启后 PID 会变化。若连接断开，在单独终端运行并保持终端开启：

```bash
cd /home/eureka/piper_ws/smolvla_deploy
/home/eureka/miniconda3/envs/lerobot/bin/python -u scripts/inference_tunnel.py SuperSvr --local-port 8085 --remote-port 8085 --replicas 3
```

该命令同时转发 8085、8086、8087。启动脚本不自动建立或重连隧道。只转发 8085 将不能通过当前三副本启动检查。

```bash
curl --fail http://127.0.0.1:8085/health
curl --fail http://127.0.0.1:8086/health
curl --fail http://127.0.0.1:8087/health
```

三份健康信息必须一致，包括模型标识、权重 SHA256、相机、单位、控制频率和动作块长度。启动时要求三个端点均健康；运行中的超时、暂时连接失败和 408/429 则按副本分别退避重试，不阻塞其他副本。

## 当前控制规则

- `client.replica_urls` 为本机 8085、8086、8087；每副本最多一个在途请求，无观测请求队列，空闲时取最新观测。
- `inference_interval_steps=4`：20 Hz 控制循环中以 200 ms 为最小请求间隔；调度不会补发突发请求，实测结果间隔中位约 250 ms。
- `max_action_age_ms=1000`、`timeout_ms=1200`、`execution_horizon=12` 保持原值。仍按最早源主机时间戳选动作，禁止过期动作输出。
- 每个控制周期先验证所有已完成请求，再选择观测时间最新且可用的结果。乱序返回的旧观测不覆盖新计划；旧请求的模型或协议错误仍会终止本次运行。
- 有可用新计划时直接接续，不重置跟随器速度。仅在真的进入保持后恢复，才从实测状态重启跟随器；保持之前采集的在途结果仍被丢弃。
- 若所有结果均不可用，仍按原逻辑保持并等待。CAN、相机新鲜度、目标边界、目标距离、速度、加速度和跟踪误差检查均保留。

这一方案提高新结果的到达频率，不缩短单次视觉反馈延迟，也没有启用 RTC 或动作平均。

## 模型、服务和资源

SSH：`ssh SuperSvr`。

服务目录：`/home/lvzx/piper_pi05_work/deploy_supersvr_20261110`。

模型目录：`/home/lvzx/piper_pi05_work/runs/main_20261106_132250/checkpoints/032000/pretrained_model`。

模型标识：`piper-pi05-main_20261106_132250-updates008000`。

权重 SHA256：`adf5f263f8b0518f82c62d627270ff17a793e9a2e05994f798c310131dd941b2`。

| 物理卡 | GPU UUID | 端口 | 配置 | 本次启动 PID |
|---|---|---|---|---|
| 0 | GPU-f9be6157-04a3-93b8-a425-9ee2f24271de | 8085 | config.pi05.json | 1040951 |
| 1 | GPU-eb049849-f110-c16f-0877-3b1df9cadb12 | 8086 | config.gpu1.json | 1062044 |
| 2 | GPU-67d7ec8a-4a79-52a8-4c82-ca3a00da5e61 | 8087 | config.gpu2.json | 1062045 |

各副本内都映射为 `cuda:0`。环境仍为 `/home/lvzx/piper_pi05_work/env/bin/python`，使用原已验证依赖、10 次去噪、BF16、未编译。输入为 Astra 外部 RGB、Dabai 腕部 RGB（各 640×480）、七维状态和任务文字；输出 50×7 个绝对目标，六关节为度，夹爪为米制完整开口宽度，20 Hz 下覆盖 2.5 秒预测。

## 服务重启

服务当前已经运行。仅在相应进程退出后启动对应副本，避免重复加载。登录 SuperSvr，设置环境：

```bash
cd /home/lvzx/piper_pi05_work/deploy_supersvr_20261110
export PYTHONPATH="$PWD:/home/lvzx/piper_pi05_work/code/lerobot"
export HF_HOME=/home/lvzx/piper_pi05_work/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

只重启已退出的那一项：

```bash
CUDA_VISIBLE_DEVICES=GPU-f9be6157-04a3-93b8-a425-9ee2f24271de nohup /home/lvzx/piper_pi05_work/env/bin/python -u -m server.inference_server --config config.pi05.json --warmup 3 > service.gpu0.restart.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-eb049849-f110-c16f-0877-3b1df9cadb12 nohup /home/lvzx/piper_pi05_work/env/bin/python -u -m server.inference_server --config config.gpu1.json --warmup 3 > service.gpu1.restart.log 2>&1 &
CUDA_VISIBLE_DEVICES=GPU-67d7ec8a-4a79-52a8-4c82-ca3a00da5e61 nohup /home/lvzx/piper_pi05_work/env/bin/python -u -m server.inference_server --config config.gpu2.json --warmup 3 > service.gpu2.restart.log 2>&1 &
```

日志均在服务目录。启动完成以三个端口的健康检查通过为准。

## 本次验证

31 项针对性测试通过，覆盖调度不排队、乱序、到期边界接续、副本超时隔离、模型不一致拒绝、模型失败停止、保持/恢复、传感器失效、操作员停止、HTTP 整体截止及原有速度/加速度约束。

相同示教片段、各 15 秒的无硬件输出比较：

| 指标 | 独立单副本 | 三副本 |
|---|---:|---:|
| 收到预测 | 27 | 64 |
| 首包后进入等待 | 9 | 0 |
| HTTP 重试 | 0 | 0 |
| 动作检查拒绝 | 0 | 0 |
| 使用动作的最大观测年龄 | 965 ms | 764 ms |

首轮对照曾遇到 GPU 0 入口的 429 并发占用，未用于上表比较；上表单副本改用独立 GPU 1，完整原始记录均保留。

正式入口的真实双相机＋CAN 只读验证：30 秒、600 个周期、126 份预测（每卡 42 份），首包后等待 0 次、重试 0 次、动作拒绝 0 次，控制命令 0 条。观测至返回中位 506 ms、样本 P95 550 ms、最大 558 ms；结果间隔中位 250 ms、最大 301 ms；使用动作的最大观测年龄 788 ms。首包约 0.55 秒等待属于正常启动等待。

计时基于主机接收时间戳，不包含未标定的相机曝光与 USB 缓冲延迟。有限时长验证不能保证未来网络故障或负载变化时永不等待；失去新鲜预测仍会保持。

证据目录：`/home/eureka/Documents/Codex/科研项目起步/outputs/pi05_multigpu_20261110`，包括 `recorded_comparison_isolated.json`、`live_summary.json`、逐周期日志、原文件备份 `before/`、安装哈希 `installed_files.json`、副本及隧道进程记录。
