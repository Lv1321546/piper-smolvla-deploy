# NAS 项目导入说明

本次从 NAS 的 `Personal_File/eureka/piper-smolvla-deploy` 导入源码，保留 E14 的 Git 历史、本地设备配置、数据、模型和 outputs；没有运行实机控制。

## 配置与启动

- 两个启动脚本默认使用 `$HOME/miniconda3/envs/lerobot/bin/python`，可分别用 `PIPER_DEMO_PYTHON` 或 `JOINT_DEMO_PYTHON` 覆盖。
- `config.local.json`、`gamepad.local*.json` 和 `servo_bridge/config*.local.json` 均不上传。已有文件不会被覆盖。
- 仓库包含 NAS 控制参数的 `servo_bridge/config.{servo,joint,cartesian}.example.json`，以及 `gamepad.{servo,joint}.example.json`。这些是参数模板，不包含输出授权或相机设备路径。
- 启动脚本优先读取对应 local 文件，缺少时读取 example；关节采集 CLI 也支持此回退。example 的手柄映射仍需按实际设备核对。
- 离线测试改读版本化 example，不再依赖某台电脑未提交的 local 配置。控制算法、限速和保护阈值未因导入而改动。

原有操作说明见 [XYZ_V2.md](XYZ_V2.md)、[JOINT_DEMO.md](JOINT_DEMO.md) 和 [SERVO_COLLECTION.md](SERVO_COLLECTION.md)。启动 demo 默认会执行起始姿态运动；本次导入检查不调用这些实机入口。

## 导入验证

- LeRobot 环境：`python -m unittest discover -s tests -v`，173 项通过。
- ROS Humble 系统 Python：`python3 -m unittest discover -s servo_bridge/tests -v`，68 项通过。
- 两个启动脚本和两个新增 ROS wrapper 的 `bash -n` 通过；38 个 Python 源文件语法检查通过。
- 验证仅使用模拟/假设备；未启动真实 CAN 控制或移动机械臂。
- 导入前源码提交为 `fd906ab`；额外源码备份在本机 `/tmp/piper-before-nas-import-20261103.tar.gz`（临时备份，不保证重启后保留）。
