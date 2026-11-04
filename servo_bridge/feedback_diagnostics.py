"""Read-only age details; never refresh timestamps or change safety limits."""
import math


def frame_ages(snapshot, now, max_age):
    stamps=(snapshot or {}).get('frame_timestamps',{})
    ages={}
    stale=[]
    for name,stamp in stamps.items():
        age=now-stamp
        if not math.isfinite(age):
            ages[name]=None;stale.append(name)
        else:
            ages[name]=round(age*1000,3)
            if age<0 or age>max_age:stale.append(name)
    return {'frame_age_ms':ages,'stale_frame_ids':stale,'limit_ms':max_age*1000}


def stale_description(details):
    names=details['stale_frame_ids']
    return ', '.join('%s=%sms' % (name,details['frame_age_ms'][name]) for name in names) or 'no per-frame timestamps'


def readiness_description(snapshot):
    """Distinguish finished teaching recording from actual CAN control ownership."""
    mode = snapshot.get('ctrl_mode')
    name = {0:'待机', 1:'CAN控制', 2:'示教控制'}.get(mode, '其他控制模式')
    reasons = []
    if mode != 1:
        reasons.append('尚未切入 CAN 控制（需要 ctrl_mode=1）')
    if snapshot.get('mode_feed') != 1:
        reasons.append('运动模式不是 MOVE_J（需要 mode_feed=1）')
    enabled = snapshot.get('enabled', [])
    disabled = [i+1 for i, value in enumerate(enabled) if not value]
    if len(enabled) != 6 or disabled:
        reasons.append('未使能关节='+str(disabled) if len(enabled)==6 else '电机使能反馈不完整')
    if (snapshot.get('arm_status') != 0 or snapshot.get('err_code') != 0 or
            any(snapshot.get('motor_faults', []))):
        reasons.append('存在机械臂/电机故障状态')
    teach = snapshot.get('teach_status')
    teach_name = {0:'关闭', 1:'示教记录中', 2:'示教记录已结束', 3:'示教轨迹执行中'}.get(teach, '未知')
    return (('；'.join(reasons) or '状态满足 CAN/MOVE_J/使能要求') + '。实际反馈：'
            f'ctrl_mode={mode}（{name}），mode_feed={snapshot.get("mode_feed")}，'
            f'teach_status={teach}（{teach_name}），enabled={enabled}，'
            f'arm_status={snapshot.get("arm_status")}，err_code={snapshot.get("err_code")}，'
            f'motor_faults={snapshot.get("motor_faults")}')
