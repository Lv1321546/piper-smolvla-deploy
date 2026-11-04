# Dabai + Astra Pro 双 RGB 相机关节示教

## 启动

```bash
cd /home/eureka/piper_ws/piper-smolvla-deploy
./start_joint_dual_demo.sh
```

控制页面：http://127.0.0.1:8767 。同一页面并排显示两路独立图像。
启动顺序与关节 demo 相同：相机预检、只读机械臂检查、必要时接管 CAN/MOVE_J 和使能六关节、以 5% MOVE_J 到原起始姿态、
启动关节控制。键位、LB/RB 档位、起始姿态及运动限制沿用原关节 demo。
先用实体按钮结束示教记录；不需要再运行官方运动 demo。接管不操作夹爪、不复位，`--check` 始终只读。接管步骤的限制与实机验证说明见 `JOINT_DEMO.md`。
单条默认上限 600 秒，整个会话默认上限 1500 秒。
键盘快捷键在浏览器控制页面获得焦点时有效：R 开始、空格结束、S 标成功、F 标失败、Q 退出。
终端窗口不接收这些录制快捷键。页面按钮也可以开始/结束和标注。
重启 demo 后旧网页会根据会话 ID 自动重新加载；旧令牌指令不会自动重试。
指令错误会持续显示，不再被每 250ms 的状态刷新覆盖。
原单相机入口 `start_joint_demo.sh`、`config.local.json` 及其录制目录保留。
不要同时运行两个控制 demo，也不要在录制时打开其他占用相机的程序。

```bash
./start_joint_dual_demo.sh --check  # 只读检查，不移动机械臂
./start_joint_dual_demo.sh --mock   # 两路模拟图像，不连接相机/CAN/机械臂
```

## 两个固定视角

配置文件：`config.joint.dual.local.json`。

| 视角键 | 设备 | 当前格式 |
| --- | --- | --- |
| front | Dabai DC1（随臂主视角） | RGB，640×480，采集 MJPG 30 FPS，保存 JPEG90 / 20 Hz |
| astra | Astra Pro HD Camera（辅助第三视角） | RGB，640×480，采集 MJPG 30 FPS，保存 JPEG80 / 20 Hz |

按 `/dev/v4l/by-id/` 名称选设备，不依赖 video0/video2 编号。
本 demo 仅保存两路彩色 RGB，未加入深度。页面显示的 JPEG 只用于预览，
训练图片从采集端 RGB 单独编码保存，不使用页面预览 JPEG 作为训练数据。
新的双相机录制采用 JPEG，主视角质量 90、保留完整色度；辅助视角质量 80、采用 4:2:0 色度采样。
JPEG 是有损压缩，少量细节会变化；关节状态、时间戳、两路 20 Hz 录制节拍和分辨率不变。
目前先通过压缩减小体积，没有降低辅助视角频率，保持微调和部署的时间对齐方式。
安装位置可以自行选择，但同一数据集、微调和部署时必须保持视角键与相机位置一致。

## 分视角录制格式

```text
data/piper_raw_joint_dual/
  episode_000000/
    manifest.json
    frames.jsonl
    events.jsonl
    images/
      front/000000.jpg ...
      astra/000000.jpg ...
```

每一采样行同时包含两路图片相对路径、各自的接收时间戳，以及同一行的实际关节和夹爪状态。
两路按 20 Hz 录制，帧号对应同一次采样；检查两路图像与机械臂状态的时间差，
仍使用原配置的 100ms 上限，任一相机缺帧/重复/过期会使该条失败，不用空图补齐。
这是软件接收时间同步，不是硬件曝光同步。设备、视角标签和分辨率保存在 manifest 中。
模拟录制存储在单独的 `data/piper_raw_joint_dual_mock/`，不能作为实机训练样本使用。

## 微调导出

至少录制并标为成功两条完整示教后：

```bash
/home/eureka/miniconda3/envs/lerobot/bin/python -m e14.export_dataset \
  --raw-root data/piper_raw_joint_dual \
  --output data/lerobot_joint_dual \
  --repo-id local/piper_joint_dual
```

导出后的独立视觉特征是 `observation.images.front` 和 `observation.images.astra`，
另有 `observation.state` 与 `action`；保持六关节角度（度）和夹爪开口（米）的动作定义。
原训练脚本按导出 schema 自动创建两个视觉输入，双相机配置的 server.camera_map 也已配置两项。
部署时需要同时提供这两个视角。不要将旧的单视角记录直接混入双视角目录导出；
缺失视角会被校验拒绝。原始目录分开，不拼成一张大图。

## 图片压缩与空间

新双相机录制保存为 JPG；旧 PNG 记录不改动，两种记录都可以导出。
`config.joint.dual.local.json` 中 front 使用 `image_format: jpeg`、`jpeg_quality: 90`、
`jpeg_subsampling: 0`；astra 使用质量 80、subsampling 2。
将某个视角的 `image_format` 改为 `png` 可恢复该视角的无损存储。
未配置编码参数的其他 demo 仍按原来的 PNG 保存。

根据 episode_000001 每视角抽样 20 张图片估算，约 51 秒的图片总量由约 788MB 降至约 102MB，
减少约 87%；实际大小随画面复杂度变化，日志文件大小不包含在该图片估算内。
估算报告：`outputs/recording_compression_estimate.json`。
JPEG 编码不会改变关节时间轴、帧数和两个视角的命名，但会损失少量图像细节。

导出器会将原始 JPEG 文件字节直接嵌入最终的 LeRobot parquet，避免最终训练数据又被转成 PNG，
也避免第二次 JPEG 压缩。导出过程中仍会临时生成 LeRobot 的标准中间数据，需要额外临时磁盘空间。
两路仍以 20Hz 对齐，推理输入特征和旧双相机格式一致，无需因为压缩修改模型输入结构。

## 回放

推荐使用终端选择界面，输入编号选择单相机或双相机记录：

```bash
./start_replay_menu.sh
```

也可以继续使用原来的命令：

```bash
./replay_joint_dual_demo.sh --list
./replay_joint_dual_demo.sh --execute
```

复用现有关节平滑回放，默认先低速返回首帧，再按原时间轴执行。
回放仅使用关节/夹爪数据，不需要打开相机。返回路径仍没有自动避障。

## 验证

2026-11-04 已识别两台真实相机；同时采集、分别保存 PNG 并模拟预览压缩，
20 Hz 连续 120 次，两路重复帧均为 0；最大双相机接收时间差约 30.4ms，
最大采样调度延迟约 5.6ms。Dabai 曾在首次打开时超时一次，重试后成功，
故仍保留有总超时的启动检查及现有重试机制。
已验证模拟双画面页面和两视角 LeRobot 导出/重新加载。本次未启动实机机械臂。
测试结果：`outputs/dual_camera_validation.json`；模拟界面截图：`outputs/dual_camera_preview_mock.png`。


合并说明：当前所有入口无需授权 JSON 或续期，使用 `--execute` 开启实机输出。文中同学机器的历史验证结果不代表已在本机复测。双相机配置请从 `config.joint.dual.example.json` 复制为本地配置并填写设备。
