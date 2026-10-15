# Git 管理方式

私有仓库：https://github.com/Lv1321546/piper-smolvla-deploy
默认分支：master（沿用账号创建仓库的默认设置）。

两边是同一个仓库的工作副本，保留全部代码，但运行职责不同：
- E14：/home/eureka/piper_ws/smolvla_deploy，运行 e14/。
- 服务器：/home/lvzx/workspace/piper_smolvla_deploy，运行 server/。

## 日常修改与同步

建议在 E14 修改、测试、提交，然后服务器更新：

```bash
# E14
cd /home/eureka/piper_ws/smolvla_deploy
git status
git diff
git add e14/ server/ shared/ tests/ README.md
git commit -m "Describe the change"
git push

# 服务器
cd /home/lvzx/workspace/piper_smolvla_deploy
git status
git pull --ff-only
```

提交前按实际改动选择 git add 的文件；上面只是示例。
两端不要同时改同一文件后互相覆盖。服务器有未提交修改时，先提交或保存，再更新。
较大的实验可新建 codex/实验名称 分支，验证后再合并。

## 哪些内容不提交

只管理代码、测试、文档与示例配置。模型、采集数据、输出、日志、环境文件和密钥均由 .gitignore 排除。
各机器的真实设备、端口、模型路径写 config.local.json；保留在本机，不提交。
NAS 数据和模型不会因为 git pull 自动复制或删除。

## 检查版本

```bash
git remote -v
git rev-parse --short HEAD
git status --short
```

两端 HEAD 一致且工作区无代码修改，表示代码版本一致；运行日志和本地配置可以不同。

## GitHub 登录

origin 使用 HTTPS，地址中不含 token。此次上传使用已连接的 GitHub 插件授权。
首次在终端 git push / git pull 时，如果 Git 尚未登录，需通过 VS Code 的 GitHub 登录或 Git 凭据管理器完成认证。
GitHub 账户密码不能直接作为 HTTPS Git 密码；不要把 token、密码或私钥写入仓库配置/文档。
无需将 GitHub 凭据上传到 NAS，也不需要复制台式机私钥。
