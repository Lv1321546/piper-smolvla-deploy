# PiPER 双相机抓取放置：SmolVLA 首轮训练实施计划

日期：2026-11-05（Asia/Shanghai）。用于交接实现；本轮仅检查并制定计划，没有启动训练、预留 GPU 或控制机械臂。

## 1. 目标与范围

用现有 50 条全部有效的示教，微调 `lerobot/smolvla_base`，交付可重复训练、可恢复、可离线比较和可加载部署的 PiPER 模型。

- 接受用户对 50 条数据有效性的确认；不再人工筛选成功率，不删除、裁剪、平滑或重标注这些示教。
- 仍做机器可验证的完整性检查：文件存在、帧数、时间、维度、单位、相机映射及可解码性。若出现错误，修复格式/导出问题，不能静默丢弃 episode 凑数。
- 首轮单卡、双相机、固定数据划分、固定 seed、官方默认微调范围。
- 本计划实施阶段只涉及数据、训练和离线推理；实机闭环验收作为独立后续阶段，不在训练完成后自动发送机械臂指令。
- 不升级或重装现有 LeRobot 环境作为默认操作，不改机械臂控制代码。

## 2. 已完成的资源检查

检查时间：2026-11-05 20:33 左右，中国时间。显存为 nvidia-smi 输出，单位 MiB；快照不代表资源预约。

| 服务器 | GPU | 型号 | 总显存 | 已用显存 | 利用率（首次快照） | 计算进程 | 判断 |
|---|---:|---|---:|---:|---:|---|---|
| Supermicro | 0 | L40 | 46068 | 10675 | 82% | 有 | 占用 |
| Supermicro | 1 | L40 | 46068 | 21439 | 0% | 有，2 个 | 占用，不能仅看利用率 |
| Supermicro | 2 | L40 | 46068 | 10 | 0% | 未列出 | 本次首选 |
| Inspur | 0 | RTX A6000 | 49140 | 48200 | 21% | 有 | 占用 |
| Inspur | 1 | RTX A6000 | 49140 | 47839 | 91% | 有 | 占用 |

Supermicro 的第二次检查中 GPU 2 仍为 10 MiB / 0%，没有计算进程。

首选 GPU UUID：`GPU-4662787b-485a-0e8f-e4b2-dd47352ed69c`。

执行前必须重新运行两项只读查询：

```bash
ssh -o ClearAllForwardings=yes Supermicro 'nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv'
```

连续观察约 10 秒，确认目标 GPU 未出现计算进程、显存仅为系统基线且利用率接近 0。若实验室有调度/预约机制，遵守其安排。若 GPU 2 被占用，检查其他 GPU；都忙则完成导出和代码准备，等待资源，不能终止他人进程或在已占满的卡上尝试。

训练进程使用 UUID 隔离：

```bash
export CUDA_VISIBLE_DEVICES=GPU-4662787b-485a-0e8f-e4b2-dd47352ed69c
```

进程内该卡显示为 `cuda:0`。正式训练、验证和显存测试均只使用这一张卡。第一轮不用 DDP，也不跨两台服务器训练。

## 3. 环境、代码与存储

已确认 Supermicro：

- Python：`/home/lvzx/.conda/envs/eureka312/bin/python`，3.12.14。
- torch：2.11.0+cu128；torchvision：0.26.0+cu128。
- LeRobot：0.6.2；transformers：5.5.4。
- 项目：`/home/lvzx/workspace/piper_smolvla_deploy`，当前提交短号 `b94f91c`。
- E14 项目：`/home/eureka/piper_ws/smolvla_deploy`，当前 HEAD `c60f0635bfe3aa442b36e393ae28179004b8d2da`。
- 两端代码提交不同。实施者须检查 git status、远端和差异后同步本次训练相关代码，记录实际提交及未提交补丁；不得直接覆盖对方本地改动。
- 已核实下列模型目录存在，但尚未重新逐文件校验完整性或加载权重：
  - `/mnt/nas_ailab_434/Personal_File/eureka/models/smolvla_bundle/smolvla_base`
  - `/mnt/nas_ailab_434/Personal_File/eureka/models/smolvla_bundle/smolvlm`

存储快照：

| 位置 | 剩余空间 | 用法 |
|---|---:|---|
| Supermicro 系统盘（含 /home） | 约 9.3 GB | 只保留已有环境与代码；禁止放新数据、大缓存、checkpoint |
| `/mnt/hdd_toshiba_1` | 约 191 GB | 首选训练工作盘，属于 HDD，不声称是 SSD |
| `/mnt/nas_ailab_434` | 约 29 TB | 原始数据、最终模型及报告归档 |

建议新建以下用户专属目录（父目录已存在；子目录尚未创建，需检查权限和空间）：

```text
/mnt/hdd_toshiba_1/lvzx/piper_smolvla/
  raw/pickplace_dual_v1/
  datasets/pickplace_dual_v1/
  cache/huggingface/
  cache/torch/
  tmp/
  runs/smoke_<timestamp>/
  runs/baseline_seed42_<timestamp>/
  reports/
```

NAS 归档根目录：`/mnt/nas_ailab_434/Personal_File/eureka/piper_smolvla/`。
若 HDD 子目录不能写，不更改整个共享盘权限，改用 NAS 中上述用户专属目录并测量数据加载性能。
设置 `HF_HOME`、`HF_DATASETS_CACHE`、`TORCH_HOME`、`TMPDIR` 到工作盘相应子目录；所有输出路径使用绝对路径。保留至少约 50 GB 可用空间作为第一轮预算，并按一次实际 checkpoint 大小估算全程需求。模型权重可从既有 NAS bundle 离线读取；若加载明显受网络影响，再复制到工作盘并校验摘要。

## 4. 数据定义与固定划分

原始目录：`/home/eureka/piper_ws/smolvla_deploy/data/piper_raw_cartesian_dual`。
实测大小约 4.0 GB；50 条，31,000 个原始帧，20 Hz；两路图像均为 640×480 RGB。

- `front`：Dabai DC1 随臂主视角。
- `astra`：Astra Pro 辅助视角。
- 文本统一沿用：`Pick up the black foam block and place it in the box.`
- 状态顺序：`joint1, joint2, joint3, joint4, joint5, joint6, gripper_width`。
- 关节单位 degree；夹爪单位 meter_full_opening；动作类型 absolute_position。
- observation[t] 对应真实 state[t] 和同期两路图像；action[t] 为真实 state[t+1]。
- 每条最后一个原始帧没有下一状态标签，导出时省略。预期导出 **30,950 行**，仍为 50 条 episode。
- 原始图像按既有导出器保留 JPEG 编码内容嵌入 parquet；不为了训练重复有损压缩，也不合并摄像头。

固定按 episode 排序后，使用已有 `episode_split(count=50, validation_fraction=0.2, seed=42)`：

- 训练：40 条，预期 **24,969 行**。
- 验证：10 条，预期 **5,981 行**。
- 验证 episode_index：`[5,17,20,25,27,39,45,46,47,49]`。
- `export_manifest.json` 保存排序后的索引与原始目录映射，核对上述编号确实指向这些完整 episode。
- 不按帧随机拆分；训练统计量仅来自 40 条训练数据，不能混入验证集。
- 这 10 条用于选模型，不宣称是独立最终测试集。最终能力用之后新做的实机试验评估。

实施时先用 rsync 传完整 raw 根目录（包含 manifest、frames、images、events 和 `_servo_sessions`），记录源文件大小/摘要清单，传输后核对。然后在服务器导出：

```bash
cd /home/lvzx/workspace/piper_smolvla_deploy
/home/lvzx/.conda/envs/eureka312/bin/python -m e14.export_dataset \
  --raw-root /mnt/hdd_toshiba_1/lvzx/piper_smolvla/raw/pickplace_dual_v1 \
  --output /mnt/hdd_toshiba_1/lvzx/piper_smolvla/datasets/pickplace_dual_v1 \
  --repo-id local/piper_pickplace_dual_v1 --validation-fraction 0.2 --seed 42
```

这是现有 CLI 支持的命令；实施前需先完成路径创建、传输和环境设置。导出目录必须是新目录，不能覆盖旧导出。

验收：50 条全部导出、excluded 为空、上述行数/划分一致、图像可解码、两相机键与顺序固定、所有训练和验证样本使用同一套 train_stats。不能将 mock、推块任务或旧鼠标垫任务混入。

## 5. 第一轮确定配置

| 配置 | 第一轮值 |
|---|---|
| 初始化 | 完整 SmolVLA base pretrained 权重；不是从零训练 |
| 输入 | 两路图像 + 7 维状态 + 原始任务文本 |
| 输出 | 7 维绝对位置动作，内部 padded dimensions 保留 base 的 32 |
| freeze_vision_encoder | true |
| train_expert_only / train_state_proj | true / true；记录实际 requires_grad 参数清单与数量 |
| empty_cameras | 0，PiPER 双相机不能误用 SO100 的 processor/camera keys |
| Aloha 相关转换 | 全部 false |
| 图像尺寸 | 原始 640×480；沿用 base 的 512×512 等比 padding 预处理 |
| 额外图像增强 | 首轮关闭额外随机裁剪、翻转及颜色增强 |
| chunk_size / n_obs_steps | 50 / 1（50 步约 2.5 秒） |
| Batch size | **16**，单卡、无梯度累积；首轮固定，不中途变更 |
| 精度 | 沿用已下载 base 的 `use_amp=false`，先用这一基线验证稳定性 |
| Optimizer | AdamW；lr=1e-4，betas=(0.9,0.95)，eps=1e-8，weight_decay=1e-10 |
| 梯度裁剪 | norm=10，记录实际梯度范数 |
| 最大更新步数 | 20,000 optimizer steps |
| Scheduler | cosine；warmup=1,000；衰减终点=20,000；末端 lr=2.5e-6 |
| seed | 42（Python、NumPy、PyTorch、数据 loader；记录确定性设置） |
| num_workers | 4，记录数据等待时间；只在明确 I/O 瓶颈时调整 |
| log/save/eval | 每 50 步记录；每 2,000 步保存和验证 |
| 上传 | push_to_hub=false；不自动发布数据/模型 |

第一轮先保持配置少变。若 batch16 在双相机实际训练中 OOM，先确认冻结和训练参数正确、没有其他进程；仍不够才改为 batch8 并重新开始一个记录完整配置的新 run。不要修改已跑 run 的 batch 后直接续训。batch32、BF16 是后续吞吐优化对照，不是这次必须实现的主路线。

20,000×16 / 24,969 ≈ 12.8 次训练样本抽取覆盖量，作为理解步数的近似，不等同于 12.8 次独立环境经历。

时间预算：用预热后的 100 步中位耗时估计 `20,000×秒/步 + 验证 + 保存`，报告显存峰值和数据等待。未经本机实测不承诺几小时完成。

## 6. 先补齐的代码工作（交给实施模型）

复用现有 `e14/export_dataset.py`、`server/train_policy.py`、`server/policy_runner.py` 和官方 LeRobot trainer，避免另写一套优化器循环。

### 6.1 配置和模型加载

- 为训练封装补充可持久化的实验配置，以及 seed、lr、warmup/decay、save/eval frequency、resume、精度和 augmentation 的显式配置途径。
- 核对实际安装 LeRobot 0.6.2 的 API；不能直接照最新 main 文档猜参数名。保存完整解析后的配置、pip freeze、项目和 LeRobot git commit、base 权重摘要。
- 强校验完整 pretrained 权重加载、7→32 padding 和双相机特征；打印 missing/unexpected keys，不能悄悄随机初始化动作投影或加载错误的 SO100 processor。
- 核对 `load_vlm_weights=False` 在本地完整 checkpoint 恢复场景的含义：避免重复下载并不意味着允许随机 VLM。验证 checkpoint 中 VLM 和动作专家权重都被恢复。
- 归一化器/反归一化器使用 train_stats；导出配置相机顺序固定，例如明确 `front, astra`，训练与推理一致，不能依赖一个位置排序、另一个名称排序。

### 6.2 动作块和 episode 边界

- 用实际 dataloader 检查：第 k 个预测目标为 state[t+k+1]，没有再次错位一帧。
- 50 步 chunk 不得跨 episode。遵守安装版本的末尾 padding/mask 语义；确认损失正确处理无效填充，离线指标必须忽略不存在的真实未来帧。
- 确认 padding 到 32 后，只有真实 7 维参与对应的物理误差统计；不得拿 padded 维度稀释误差。

### 6.3 中间验证与恢复

现有脚本 `eval_steps=0`，最终只抽约 20 帧并导出 last；不足以完成本计划。

- 每 2,000 步验证并保存指标；优先使用安装版本支持的 hooks/评估接口，若没有可用接口，以官方生成的 checkpoint 加独立离线评估器实现。
- 不另占他人 GPU。验证可在同卡训练暂停时执行，或训练完后顺序评估保存的 checkpoint；不能两个大模型实例挤爆同卡。
- 保存 optimizer、scheduler、更新步数、RNG 及可用的数据采样状态；明确安装版本能否精确恢复采样顺序。
- 增加 resume 路径，处理当前“output_dir 已存在即拒绝”逻辑；新建 run 与恢复 run 分开，恢复不能重置 scheduler 或覆盖日志。
- 选择 best checkpoint 后，针对它生成 deployment_manifest；不能仍然无条件把 last 当成最终部署模型。

### 6.4 启动与交付

- 新增单卡训练启动脚本：GPU 空闲预检、UUID 隔离、工作盘 cache/tmp、离线模型路径、独立日志、tmux 或等效可断线运行方式。
- 记录只停止本次训练进程的命令；不使用 killall/python 或终止其他用户进程。
- 输出可复现 run 配置、数据/代码/权重摘要、机器资源快照、训练与验证曲线、选模报告、恢复命令、部署 bundle。
- 训练 CLI 的新参数要先实现并检查 `--help` 再写入最终操作文档；本计划不把未实现的 CLI 当成可执行命令。

## 7. 离线评价和模型选择

每次验证：

1. 对固定验证样本计算与训练一致的损失；评估禁用训练增强，model.eval、no_grad。
2. 对每条验证示教均匀选取最多 50 个固定锚点（共约 500 个），固定生成噪声 seed；保存锚点索引，以便所有 checkpoint 完全一致。验证 RNG 不应改变后续训练 RNG。
3. 输出第一步及 5、10、25、50 步 horizon 的误差。关节分别以度报告，夹爪以毫米报告；标注每个 horizon 的有效样本数量。
4. 先对每个 episode 计算指标，再对 10 条取等权均值，并同时提供逐 episode 表，避免长 episode 主导结果。
5. 对同一批锚点计算“未来始终等于当前状态”的保持基线，特别比较 10/25/50 步，防止模型靠预测不动得到漂亮的单步误差。
6. 导出示例：图像、真实/预测夹爪曲线、六关节曲线、预测越界比例与动作块内变化；不把预测的后续相邻点当作实机已执行结果。

选择规则：优先按固定验证集的、排除 padding 后的训练目标损失比较；同时检查反归一化动作有限、关节/夹爪误差和保持基线。如果指标相互冲突，保存前两名候选，不宣称已有实机最优模型。
最终候选增加全部 5,981 行的验证损失检查；保留 selected、runner-up、last 和至少一个可恢复 checkpoint。磁盘足够时首轮保留全部 10 个 checkpoint，禁止无报告自动删除唯一可恢复模型。

不在此阶段规定未经实测的“MAE 小于某数就部署成功”。离线低误差不等价于抓取成功。

## 8. 分阶段执行和验收

### A. 数据与环境就绪

- 重新查 GPU；完成代码对齐、目录、环境与权重完整性确认。
- 传输约 4 GB 数据并导出；验收 50 episodes / 30,950 rows / 40+10 分集。
- 保存原始数据清单、train_stats 和 schema 摘要。

### B. 离线功能测试

- 测试分集无交集、训练统计不受验证数据影响、两相机顺序一致、7 维单位不变、chunk 不跨 episode、padding mask、保存重载和 resume。
- 对断点恢复、checkpoint 选取和 manifest 匹配做针对性测试。
- 不运行实机或伪装 mock 数据为训练数据。

### C. 300 步真实数据短跑

- 在已空闲 GPU 2 上运行实际模型训练 300 步，使用正式模型和两路图像。
- 使用正式 20k scheduler 配置；300 是短跑停止点，不要把 cosine 总长度改成 300 后冒充正式实验。
- 在第 200 步保存并恢复到 300，核对 optimizer step、lr、配置、统计量和日志连续。
- 检查有限 loss/梯度、参数确有更新、冻结参数没有被优化、checkpoint 可重载并生成 `[B,50,7]` 的有效动作（若底层输出 padded32，后处理明确取真实7维）。
- 报告稳定期秒/步、峰值显存、loader 时间、一次保存大小、预测与保持基线；短跑不要求证明任务成功。

### D. 正式基线

- 从基础权重和 seed42 重新初始化独立目录，batch16，最多20k步。
- 采用20k scheduler，每2k保存评估；在4k检查数值/文件/数据流问题。出现 NaN、异常加载或不可恢复错误则终止并诊断，不调高阈值掩盖。
- 正常完成后生成选模报告和候选部署 bundle。
- 第一轮不自动启动多个超参实验，也不因有空卡就扩大范围。

### E. 部署前离线验证与后续实机

- 使用现有 PolicyRunner 加载选定 checkpoint，在本批验证图像/状态上执行离线请求，验证双相机键、反归一化、7维输出和 manifest 哈希。
- 保存适配实际环境路径的处理器与模型配置；确认迁移后不依赖训练机临时目录。若仍依赖本地 VLM/tokenizer bundle，在交付清单明确包含和定位它。
- 记录单次推理延迟分位数。chunk_size=50 不意味着实机连续开环执行2.5秒；后续可试每次执行5–10步再更新观察，但需用真实推理/通信延迟选择。
- 本轮交付止于可加载的离线模型。之后现场做少量候选比较，再固定配置做至少20次试验，报告抓取、抬起、放入盒子和总成功率；这才是实机性能证据。

## 9. 交付清单与完成标准

实施完成应提供：

- 一条数据导出命令、一条短跑命令、一条正式训练命令、一条恢复命令及停止命令。
- 精确数据目录与50条映射、split、train_stats、export_manifest、deployment_schema。
- 已测试代码、环境版本清单、源代码提交与变更摘要。
- smoke 报告（含恢复检查）、正式完整日志、checkpoint列表和验证比较表。
- selected 模型及配套配置、processors、deployment_stats、deployment_manifest、必要tokenizer/VLM资源清单。
- `TRAINING_REPORT.md`：训练了多少步、用了哪张卡、耗时/显存、选了哪个checkpoint、依据是什么、还有什么未在实机验证。

用户下一轮可直接交给实施模型：

> 按本计划自主完成数据导出、训练工具补齐、300步短跑及恢复验证，再运行单张空闲GPU上的20k步基线，完成离线评估和模型交付。接受50条示教全部有效，不重新开展人工筛选。启动前重新检查GPU和存储，不占用他人显卡，不覆盖已有数据/实验，不发送机械臂指令。常规配置按计划执行；遇到资源阻塞先完成其他独立工作。每阶段报告真实证据，离线结果不能称为实机任务已解决。

## 10. 参考及版本原则

- 官方训练教程：https://huggingface.co/docs/lerobot/smolvla
- 基础模型配置：https://huggingface.co/lerobot/smolvla_base/blob/main/config.json
- 官方配置源码：https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/configuration_smolvla.py

官方教程以约50条示教为起点、给出20k步示例；本计划选batch16并以实机环境已有LeRobot 0.6.2为准。上游 main 可能继续变化，实施时读取实际安装代码和已下载模型配置，避免把其他版本参数直接混入。
