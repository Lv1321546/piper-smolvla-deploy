# 同学代码手动合并：2026-11-04

目标仓库：`/home/eureka/piper_ws/smolvla_deploy`。分支：`codex/merge-colleague-20261104`。改动保留在工作区，未提交、未推送。
合并前 HEAD：`01d2706718d6a0144b3c2e25186ce5059e1a1c3e`；压缩包 HEAD：`fd906ab`；包内未提交和新增文件一并纳入。
输入 ZIP SHA256：`0a17d74176864c7714a4c32a498fce4762a43a7dd6ba52cd7fd039655f081a95`。

## 合并方法与冲突处理

以共同提交 fd906ab 做三方比较；共同提交中不存在的同名新增文件还对照本地 NAS 导入版本 fa929d7。
暂存中发现 11 个文本冲突文件，已逐项处理；没有导入压缩包的 .git 配置、hooks、IDE 缓存、实验数据或输出。

- 纳入双 RGB 相机与分视角预览、JPEG 图像存储和导出、关节平滑回放及终端选择器。
- 纳入几何边界保持/反向退离、反馈诊断和启动 CAN 接管模块。
- 保留本地轨迹过滤、原始数据回放，以及所有入口无需授权 JSON/续期的行为；新入口同样遵循 --execute 输出开关。
- 保留现有本地设备配置。双相机配置以 config.joint.dual.example.json 提供，device 留空，使用前复制并填写。
- 启动路径改为当前用户 HOME 下的环境路径，保留本地 profile/gamepad 示例回退及覆盖选项。
- 修复相机重连前清除停止标志可能重新激活旧线程的问题；修复启动脚本中 profile 传递冲突。
- 测试改用已版本化示例，修复 ROS 消息测试与新版 GeometryRecovery 接口的冲突。

## 验证

在临时完整代码副本中：LeRobot 环境 tests 242 项通过；ROS Humble 系统环境 servo_bridge/tests 98 项通过，总计 340 项。
Python 语法和 shell 语法检查通过；未发送实机运动、模式、使能、夹爪指令。
测试模拟 SDK、CAN 和相机，HTTP 测试只使用本机回环地址。离线通过不代表新 CAN 接管/回放/双相机已在此 E14 实机验证。
系统 SciPy 对现有 NumPy 版本有兼容性提示，本次没有修改依赖环境。

## 使用注意

start_demo.sh / start_joint_demo.sh 的正常启动现在包含 CAN 接管，可能切模式并使能，然后移动到起始姿态；--check 保持只读。
start_joint_dual_demo.sh 使用独立双相机本地配置，真实启动未填写时明确提示退出，--mock 自动使用示例；不覆盖 config.local.json。
原始数据回放 e14.replay_data 与同学的关节平滑回放 e14.replay_joint 保留为不同入口，详见各自文档。

## 备份与追溯

合并前被修改文件备份：`/home/eureka/piper_ws/merge_backups/20261104-colleague/before-files.tar.gz`。
文件清单、新增文件、原分支、原 HEAD 和本地配置 SHA256：`/home/eureka/piper_ws/merge_backups/20261104-colleague/manifest.json`。
如需回退，先停止所有控制程序、保存本次后续改动，再依据 manifest 恢复备份并处理新增文件；不自动删除数据。

本次应用 50 个新增或修改文件（不含本报告）：

- `DUAL_CAMERA_DEMO.md`
- `JOINT_DEMO.md`
- `JOINT_REPLAY.md`
- `MOVEIT_SERVO.md`
- `SERVO_COLLECTION.md`
- `XYZ_V2.md`
- `config.joint.dual.example.json`
- `e14/camera_check.py`
- `e14/cameras.py`
- `e14/collect_data.py`
- `e14/collect_joint_demo.py`
- `e14/export_dataset.py`
- `e14/joint_collect.py`
- `e14/piper_robot.py`
- `e14/replay_joint.py`
- `e14/replay_menu.py`
- `e14/replay_smoothing.py`
- `e14/servo_collect.py`
- `e14/teaching_ui.py`
- `replay_joint_demo.sh`
- `replay_joint_dual_demo.sh`
- `servo_bridge/cartesian_runtime.py`
- `servo_bridge/execution.py`
- `servo_bridge/feedback_diagnostics.py`
- `servo_bridge/geometry_recovery.py`
- `servo_bridge/joint_policy.py`
- `servo_bridge/joint_runtime.py`
- `servo_bridge/prepare_control.py`
- `servo_bridge/runtime.py`
- `servo_bridge/sdk_worker.py`
- `servo_bridge/tests/test_feedback_diagnostics.py`
- `servo_bridge/tests/test_geometry_hold.py`
- `servo_bridge/tests/test_geometry_recovery.py`
- `servo_bridge/tests/test_joint_policy.py`
- `servo_bridge/tests/test_joint_trajectory_cadence.py`
- `servo_bridge/tests/test_prepare_control.py`
- `servo_bridge/tests/test_ros_messages.py`
- `start_demo.sh`
- `start_joint_demo.sh`
- `start_joint_dual_demo.sh`
- `start_replay_menu.sh`
- `tests/test_camera_check.py`
- `tests/test_cartesian_collection.py`
- `tests/test_dual_camera.py`
- `tests/test_hardware.py`
- `tests/test_image_encoding.py`
- `tests/test_joint_collection.py`
- `tests/test_replay_joint.py`
- `tests/test_replay_menu.py`
- `tests/test_replay_smoothing.py`
