# 离线滤波与实机回放

`e14.filter_episode` 只处理文件，不连接 CAN、不启动 SDK、不发送实机命令。只依赖已有 NumPy，不需要安装 SciPy。
原始实测数据保留；输出为独立的 `piper-replay-v1` 派生目标，供 `e14.replay_data` 读取。

## 生成滤波轨迹

```bash
conda activate lerobot
cd /home/eureka/piper_ws/smolvla_deploy
python -m e14.filter_episode \
  --episode data/piper_raw_servo/episode_000000 \
  --output data/piper_filtered_replay/episode_000000_sg7 \
  --window 7 --order 2
```

输出目录必须不存在，不能等于原 episode 或位于原 episode 内。重复试验请换新目录。

- 六关节采用 SG 式局部多项式最小二乘，使用实际采集时间而非假设完全均匀；默认 7 点、二阶。
- 第一帧和最后一帧精确保留，附近平滑过渡；601 帧的数量、时间顺序和录制时长不变。
- 夹爪开口逐帧保留，不滤波、不裁剪，避免改变开合时机。
- 不做删除停顿、重新定时采样或自动提高回放限速。报告比较使用 `--speed 1`，实际回放仍由回放器限速。
- 生成前检查模型关节限位和滤波偏差。默认最大关节偏差 1°、法兰采样位置偏差 5 mm、朝向采样偏差 2°；超限拒绝输出，不自动放宽。
- 关节偏差包含三次插值之间的连续多项式极值；法兰检查在每段四等分位置进行，是 base_link→link6 模型采样，不是标定过的夹爪尖端误差或碰撞检查。

更轻的滤波可使用 `--window 5`，或在同一窗口下设置 `--strength 0.5`（保留一半的平滑修正）。
窗口须为奇数、不大于帧数；阶数至少为 1、小于窗口。

保护接触/释放等时间段（下例仅说明参数格式，时间段需按你的视频选择）：

```bash
python -m e14.filter_episode \
  --episode data/piper_raw_servo/episode_000000 \
  --output data/piper_filtered_replay/episode_000000_protected \
  --window 7 --protect 10:12 --protect 20:22
```

单位是距离本条第一帧的秒数。保护区内原始关节角精确保留，区外做平滑过渡；夹爪始终不处理。
有任务意义的快速修正和接触动作不应仅因频率较高就被删除。

## 查看报告

```text
输出目录/
  manifest.json          # replay_only、参数、源文件路径及 SHA256
  frames.jsonl           # target_state 派生目标；不是实测 state
  filter_report.json     # 前后计划时长、插值峰值、关节和法兰偏差
  comparison.csv         # 时间、六关节原始值及滤波值
  flange_deviation.csv   # 采样法兰位置和朝向偏差
```

先打开 `filter_report.json`，再查看 CSV 曲线。报告峰值加速度是插值参考的计算值，不是实机加速度测量。
若原始夹爪开口不在 profile 范围内，报告 `optional_gripper_replay.valid=false`；关节滤波仍可生成，但加 `--replay-gripper` 会被拒绝，不会把超范围值强行裁成合法值。

本条记录用默认 7 点参数：计划播放约 **194.8→105.7 秒**，连续插值最大关节偏差约 **0.184°**，法兰最大采样位置偏差约 **1.45 mm**、朝向偏差约 **0.352°**。这是离线对比，不是实机去抖效果保证。

## 用现有回放器播放

先只检查，不连接硬件：

```bash
python -m e14.replay_data \
  --episode data/piper_filtered_replay/episode_000000_sg7 \
  --speed 1
```

需要实际运动时，先退出其他手柄/采集/控制程序，确保 CAN、机械臂控制模式和电机使能已经正常，再运行：

```bash
python -m e14.replay_data \
  --episode data/piper_filtered_replay/episode_000000_sg7 \
  --speed 1 --execute
```

它会低速到首帧，然后回放；与原回放器共用跟踪、速度、加速度和反馈检查，没有碰撞规划。
默认不操作夹爪；如需连夹爪一起重放，显式追加 `--replay-gripper`。
Ctrl+C 取消并在反馈仍有效时请求位置保持。路径、抓取/松开效果仍需现场确认，滤波后不能沿用原任务成功标注。

## 与训练数据的区别

此输出不复制旧图像、不将滤波目标标为新的实测反馈，也不声称平滑轨迹已成功执行。
它与原始采集格式不同，现有训练导出器会拒绝读取，防止“旧视频 + 假定的新状态”被误当作真实示教。
如要用于 SmolVLA 标签去噪，应另做保持观测不变的标签派生和训练对照；需要强力修改轨迹时，先验证回放再重新采集匹配的视频、反馈和命令。
