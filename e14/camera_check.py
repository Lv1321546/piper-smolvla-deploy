"""Bounded camera-only startup check; never import a robot or send CAN commands.

Use the same CameraReader as collection, inside a disposable child so a blocked
V4L2 open/read cannot leave the launch script waiting without an overall limit.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def check_recording_cadence(reader, hz, *, clock=time.monotonic, sleep=time.sleep):
    """One second at collection frequency; a first frame alone is insufficient."""
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz) or hz <= 0:
        raise ValueError('采集频率必须为正数')
    previous = None
    started = clock()
    count = max(2, math.ceil(hz)+1)
    for index in range(count):
        deadline = started+index/hz
        sleep(max(0., deadline-clock()))
        if clock()-deadline > .5/hz:
            raise RuntimeError('相机预检采样调度超时，无法确认录制稳定性')
        stamps = reader.read()['image_timestamps']
        if previous is not None and any(stamps[name] <= previous[name] for name in stamps):
            raise RuntimeError('相机在 %.1f Hz 录制节拍下出现重复帧，请检查缓冲、USB 或其他占用程序' % hz)
        previous = stamps
    return count


def worker_command(config, mock=False):
    command = [sys.executable, '-u', '-m', 'e14.camera_check', '--worker',
               '--config', str(config)]
    if mock:
        command.append('--mock')
    return command


def check_cameras(config, timeout=45., *, mock=False):
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('相机检查总超时必须为正数')
    print('使用采集端 CameraReader 检查相机，总超时 %.1f 秒。' % timeout, flush=True)
    try:
        result = subprocess.run(worker_command(config, mock), timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError('相机检查超过 %.1f 秒，检查子进程已终止；可能卡在设备打开、读取或关闭。'
                           '请确认无其他程序占用相机，再检查 USB 连接。' % timeout) from exc
    if result.returncode:
        raise RuntimeError('相机检查失败（子进程退出码 %d），未启动机械臂。' % result.returncode)


def check_in_worker(config, *, mock=False):
    from e14.cameras import CameraReader
    config = json.loads(Path(config).read_text())
    cameras = config['cameras']
    for name, camera in cameras.items():
        print('检查 %s：%s，%dx%d @ %s FPS，格式 %s' %
              (name, camera.get('device', 'mock'), camera['width'], camera['height'],
               camera['fps'], camera.get('fourcc', '默认')), flush=True)
    with CameraReader(cameras, mock=mock) as reader:
        frame = reader.read()
        for name, image in frame['images'].items():
            print('相机 %s 首帧正常：%s，%s' % (name, image.shape, image.dtype), flush=True)
        hz = config['robot']['control_hz']
        count = check_recording_cadence(reader, hz)
        print('录制节拍检查通过：%.1f Hz，连续 %d 次均为更新帧。' % (hz, count), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=45.)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            check_in_worker(args.config, mock=args.mock)
        else:
            check_cameras(args.config, args.timeout, mock=args.mock)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print('[相机检查失败] %s' % exc, file=sys.stderr, flush=True)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
