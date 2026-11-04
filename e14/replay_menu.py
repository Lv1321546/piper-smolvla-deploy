"""Terminal picker for joint recordings. No hardware is opened by the menu.

Only selecting an eligible number launches the existing supervised replay CLI.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import signal
import subprocess
import sys
from zoneinfo import ZoneInfo

from e14.replay_joint import load_episode
from servo_bridge.core import load_config

ROOT=Path(__file__).resolve().parents[1]


@dataclass
class Recording:
    path: Path
    config: Path
    kind: str
    manifest: dict
    duration: float | None
    error: str = ''

    @property
    def eligible(self):
        return (not self.error and self.manifest.get('status')=='complete'
                and self.manifest.get('mock') is False)


def sources(root=None, config=None):
    if root is not None:
        return [(Path(root).resolve(),Path(config or ROOT/'config.joint.dual.local.json').resolve(),'自定义')]
    return [(ROOT/'data/piper_raw_joint_dual',ROOT/'config.joint.dual.local.json','双相机'),
            (ROOT/'data/piper_raw_joint',ROOT/'config.local.json','单相机')]


def scan(source_list):
    result=[]
    for root,config,kind in source_list:
        for path in sorted(root.glob('episode_*')):
            if not path.is_dir():continue
            manifest={};duration=None;error=''
            try:
                manifest=json.loads((path/'manifest.json').read_text())
                with (path/'frames.jsonl').open() as stream:
                    first=next(stream,None);last=first
                    for line in stream:
                        if line.strip():last=line
                if first and last:
                    duration=float(json.loads(last)['captured_monotonic'])-float(json.loads(first)['captured_monotonic'])
            except (OSError,ValueError,KeyError,TypeError) as exc:
                error=str(exc)
            result.append(Recording(path,config,kind,manifest,duration,error))
    return result


def text(value):
    return ''.join(c if c.isprintable() else ' ' for c in str(value))


def show(records, output=print):
    output('\n===== 关节录制回放选择 =====')
    for index,r in enumerate(records,1):
        m=r.manifest
        status='完整' if m.get('status')=='complete' else '失败/不完整'
        outcome={'success':'成功','failure':'任务失败','unknown':'未标注'}.get(m.get('task_outcome'),'未知')
        length='未知时长' if r.duration is None else '%.1f秒' % r.duration
        suffix='' if r.eligible else ' [禁止回放]'
        output('%2d | %s | %s | %s | %s/%s%s' % (index,r.kind,text(r.path.name),length,status,outcome,suffix))
    if not records:output('没有找到录制文件。')


def details(r, output=print):
    m=r.manifest
    output('\n文件：'+str(r.path))
    output('任务：'+text(m.get('task','未知')))
    output('帧数：%s；视角：%s' % (m.get('frames','未知'),', '.join(m.get('cameras',{}))))
    if isinstance(m.get('created_unix_s'),(int,float)):
        output('录制时间：'+datetime.fromtimestamp(m['created_unix_s'],ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
    if m.get('error') or r.error:output('失败原因：'+text(m.get('error') or r.error))


def replay_command(r,profile,*,execute=True):
    if not r.eligible:raise ValueError('这条记录失败、不完整或是模拟数据，禁止实机回放。输入 v 编号查看原因。')
    cfg=load_config(profile)
    # Revalidate files, units, timestamps and limits immediately before launch.
    load_episode(r.path,cfg)
    command=[sys.executable,'-u','-m','e14.replay_joint',str(r.path.resolve()),
             '--root',str(r.path.parent.resolve()),'--config',str(r.config.resolve()),
             '--profile',str(Path(profile).resolve())]
    if execute:command.append('--execute')
    return command


def launch(command):
    # Parent forwards a single stop signal, avoiding duplicate terminal SIGINTs
    # interrupting the child's measured-position hold cleanup.
    process=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('回放进程未确认退出；请查看原终端和现场停止状态，选择界面不会再启动其他回放。') from exc
        return 130


def menu(source_list,profile,*,check_only=False,input_fn=input,output=print,runner=launch):
    while True:
        records=scan(source_list);show(records,output)
        if check_only:
            output('检查模式：输入编号只做离线检查，不连接机械臂。')
        else:
            output('输入编号：立即启动该条实机回放（先低速返回首帧，再平滑原速执行）。')
        output('v 编号：详情；c 编号：离线检查；r：刷新；q：退出。回放前退出采集 demo，保持返回路径空旷。')
        try:choice=input_fn('选择 > ').strip().lower()
        except (EOFError,KeyboardInterrupt):output('\n已退出选择界面。');return
        if choice in ('q','quit','退出'):return
        if choice in ('r','刷新',''):continue
        parts=choice.split()
        try:
            action='play'
            if len(parts)==2 and parts[0] in ('v','c'):
                action,number=parts[0],int(parts[1])
            elif len(parts)==1:number=int(parts[0])
            else:raise ValueError('请输入编号，或 v 编号 / c 编号 / q。')
            if not 1<=number<=len(records):raise ValueError('编号超出列表范围。')
            selected=records[number-1]
            if action=='v':details(selected,output);continue
            command=replay_command(selected,profile,execute=not(check_only or action=='c'))
            output('\n已选择：%s / %s' % (selected.kind,selected.path.name))
            code=runner(command)
            output('已中断回放。' if code==130 else '执行结束，退出码 %s。' % code)
        except (OSError,ValueError,KeyError) as exc:
            output('无法启动：'+str(exc))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,help='只列出这个录制目录')
    parser.add_argument('--config',type=Path,help='自定义录制目录对应的配置')
    parser.add_argument('--profile',type=Path,default=(ROOT/'servo_bridge/config.joint.local.json' if (ROOT/'servo_bridge/config.joint.local.json').is_file() else ROOT/'servo_bridge/config.joint.example.json'))
    parser.add_argument('--list',action='store_true',help='仅列出记录后退出')
    parser.add_argument('--check-only',action='store_true',help='菜单只允许离线检查')
    args=parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt('选择界面收到停止信号')
    for signum in (signal.SIGTERM,signal.SIGHUP):signal.signal(signum,interrupted)
    if args.config and not args.root:parser.error('--config 需要配合 --root')
    source_list=sources(args.root,args.config)
    if args.list:show(scan(source_list));return
    try:menu(source_list,args.profile,check_only=args.check_only)
    except RuntimeError as exc:parser.exit(1,str(exc)+'\n')


if __name__=='__main__':main()
