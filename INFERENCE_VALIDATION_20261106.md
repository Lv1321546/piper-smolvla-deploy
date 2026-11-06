# 推理部署验证记录（2026-11-06）

## 实现与位置

E14：`/home/eureka/piper_ws/smolvla_deploy`。

Inspur 独立部署快照：`/home/lvzx/piper_smolvla_work/code/deploy_20261106`。

默认模型为 NAS `archive/pickplace_20261105_lrfix_2156/selected`，模型身份 `piper-smolvla-baseline_seed42_pickplace_20261105_lrfix_2156-step004000`。模型 SHA256 为 `66521f7ee57bd907cc1b9ea894adabeff3135eb1f95e4c6965702ecb2e90d652`。

## 已完成的验证

1. **E14 回归：38 项通过**。覆盖模拟闭环、双相机协议/模型身份、280 ms 延迟重规划、请求绝对超时、服务失败后的保持、动作时域耗尽后不执行迟到响应、限速/限加速度、整数 SDK 精度、卡住反馈拒绝累积、夹爪完整开合端点，以及位置目标先于 MOVE_J 模式选择。
2. **Inspur 实际环境回归：同一组 38 项通过**。测试进程设置 `CUDA_VISIBLE_DEVICES=""`；机器人、相机与推理策略均为测试替身，不访问物理硬件。
3. **真实 checkpoint 加载**：在当时空闲的 Inspur 物理 GPU1 上通过模型、配置、统计与处理器文件 hash 检查，严格加载并预热。第一次约 1,274 ms、第二次约 283.5 ms。服务绑定回环 `127.0.0.1:8081`。
4. **真实网络链路**：E14 → Windows 台式机 SSH 跳板 → Inspur HTTP。使用真实录制 episode_000005 的第 0、100、300 帧双 RGB 与关节状态，全部返回有限的 `[50,7]` 动作。推理中位约 279.7 ms；从 E14 开始编码到接收响应的中位约 387.8 ms。
5. **连续客户端链路**：真实模型 + 模拟机械臂/相机，20 Hz、60 个本地循环，收到 7 个预测，正常结束，`sent_commands=0`。它验证接口和时序，不是视觉任务仿真。
6. **真实数据范围核对**：31,000 个原始反馈样本均位于部署的物理目标范围内。关节实测最小值为 `[-47.102,72.477,-118.489,-21.970,25.693,-77.592]`°，最大值为 `[48.279,120.398,-33.268,24.634,64.245,79.213]`°；总开口范围 `[0.0463,0.0696]` m。
7. Python 编译检查和 Bash 语法检查通过。没有覆盖原训练/手柄改动，也没有更改正在运行的 RLT 作业。

回归命令：

```bash
python -m unittest tests.test_deployment tests.test_protocol_runtime \
  tests.test_client_deadline tests.test_integration tests.test_server -v
```

本地证据保存在 `outputs/inference/verification_20261106/`；服务器测试与临时服务日志在 `/home/lvzx/piper_smolvla_work/reports/deploy_20261106/`。临时真实模型推理服务在验证后关闭，后续按 `INFERENCE.md` 启动。

## 验证边界

本次没有发送任何真实关节、夹爪、使能、模式切换或复位命令，也未打开物理 CAN 接口。`--execute` 分支仅用模拟机械臂/假 SDK 验证。

这些结果证明模型可以在目标服务器加载、处理真实录制双图像，并通过现有跳板返回有效动作，且软件控制层具备检查与拒绝路径；尚未证明新入口在实机上的运动平滑度、停止效果或抓取成功率。现场下一步为真实相机/反馈只读预览，然后小幅输出测试与任务评估。完整命令见 `INFERENCE.md`。
