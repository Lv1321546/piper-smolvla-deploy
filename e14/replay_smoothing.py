"""Offline, NumPy-only replay interpolation. Never modifies the raw episode."""
import math
import numpy as np


def fixed_ticks(duration, period):
    if not math.isfinite(period) or period <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('回放时长和周期必须为正数')
    ticks = np.arange(0., duration, period)
    # Merge a tiny last interval into the preceding one instead of a CAN burst.
    if len(ticks) > 1 and duration-ticks[-1] < period*.5:
        ticks[-1] = duration
    else:
        ticks = np.r_[ticks, duration]
    return ticks


def pchip_coefficients(t, q):
    """Shape-preserving cubic Hermite; coefficients in seconds from each knot."""
    h = np.diff(t)
    delta = np.diff(q, axis=0)/h[:, None]
    d = np.zeros_like(q)
    if len(t) == 2:
        d[:] = delta[0]
    else:
        for i in range(1, len(t)-1):
            same = delta[i-1]*delta[i] > 0
            w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
            d[i, same] = (w1+w2)/(w1/delta[i-1, same]+w2/delta[i, same])
        for k, a, b, ha, hb in [(0, 0, 1, h[0], h[1]), (-1, -1, -2, h[-1], h[-2])]:
            slope = ((2*ha+hb)*delta[a]-ha*delta[b])/(ha+hb)
            slope = np.where(np.sign(slope) != np.sign(delta[a]), 0., slope)
            limit = (np.sign(delta[a]) != np.sign(delta[b])) & (abs(slope) > 3*abs(delta[a]))
            d[k] = np.where(limit, 3*delta[a], slope)
    a = (d[:-1]+d[1:]-2*delta)/h[:, None]**2
    b = (3*delta-2*d[:-1]-d[1:])/h[:, None]
    return a, b, d[:-1], q[:-1]


def evaluate(t, coefficients, ticks):
    idx = np.clip(np.searchsorted(t, ticks, side='right')-1, 0, len(t)-2)
    x = (ticks-t[idx])[:, None]
    a,b,c,d = (v[idx] for v in coefficients)
    return ((a*x+b)*x+c)*x+d


def curve_metrics(t, coefficients, original):
    """Analytic extrema, including deviation from original linear interpolation."""
    a,b,c,d = coefficients
    h = np.diff(t)[:, None]
    velocity = max(float(abs(c).max()), float(abs(3*a*h*h+2*b*h+c).max()))
    x = np.zeros_like(a)
    np.divide(-b, 3*a, out=x, where=abs(a)>1e-15)
    interior = (x>0) & (x<h)
    velocity = max(velocity, float(np.max(np.where(interior, abs(3*a*x*x+2*b*x+c), 0.))))
    acceleration = max(float(abs(2*b).max()), float(abs(6*a*h+2*b).max()))
    delta = np.diff(original, axis=0)/h
    error = max(float(abs(d-original[:-1]).max()),
                float(abs(((a*h+b)*h+c)*h+d-original[1:]).max()))
    # Critical points of (cubic curve - original linear segment).
    for i in range(len(t)-1):
        for j in range(original.shape[1]):
            aa,bb,cc = 3*a[i,j],2*b[i,j],c[i,j]-delta[i,j]
            if abs(aa)<1e-15:
                roots = [] if abs(bb)<1e-15 else [-cc/bb]
            else:
                disc = bb*bb-4*aa*cc
                roots = [] if disc<0 else [(-bb-math.sqrt(disc))/(2*aa),(-bb+math.sqrt(disc))/(2*aa)]
            for s in roots:
                if 0<s<h[i,0]:
                    error = max(error, abs(((a[i,j]*s+b[i,j])*s+cc)*s+d[i,j]-original[i,j]))
    return velocity, acceleration, error


def filtered_joints(t, q, sigma):
    if sigma == 0:
        return q.copy()
    result = q.copy()
    for i, stamp in enumerate(t):
        left = np.searchsorted(t, stamp-3*sigma, side='left')
        right = np.searchsorted(t, stamp+3*sigma, side='right')
        indexes = np.arange(left, right)
        weights = np.exp(-.5*((t[indexes]-stamp)/sigma)**2)
        result[i] = weights@q[indexes]/weights.sum()
    # Anchor the smoothed curve with a smooth endpoint correction. Fading the
    # entire filter back to the raw samples restores endpoint chatter and can
    # introduce a sharp turn at the fade boundary, defeating acceleration checks.
    span = min(.3, t[-1]/2)
    def taper(distance):
        x = np.clip(distance/span, 0., 1.)
        return 1.-x**3*(10.+x*(-15.+6.*x))
    start_error, end_error = result[0]-q[0], result[-1]-q[-1]
    result -= (taper(t)[:, None]*start_error +
               taper(t[-1]-t)[:, None]*end_error)
    result[0], result[-1] = q[0], q[-1]
    return result


def prepare_trajectory(stamps, states, period, cfg=None, *, smoothing='smooth'):
    t, q = np.array(stamps, dtype=float, copy=True), np.array(states, dtype=float, copy=True)
    if (t.ndim != 1 or len(t)<2 or q.shape != (len(t),7) or
            not np.isfinite(t).all() or not np.isfinite(q).all() or t[0]!=0 or np.any(np.diff(t)<=0)):
        raise ValueError('回放轨迹必须为有序时间戳与七维有限状态')
    ticks = fixed_ticks(float(t[-1]), period)
    summary = dict(smoothing=smoothing, send_hz=1/period, joint_max_deviation_deg=0., filter_sigma_s=0.)
    if smoothing == 'linear':
        values = np.column_stack([np.interp(ticks,t,q[:,j]) for j in range(7)])
    elif smoothing == 'smooth':
        if cfg is None:
            raise ValueError('平滑回放必须提供速度、加速度和限位配置')
        vmax = math.degrees(cfg['maximum_joint_velocity_rad_s'])
        amax = math.degrees(cfg['maximum_joint_acceleration_rad_s2'])
        selected = None
        for sigma in (0., .025, *[i/1000 for i in range(50,126,5)]):
            joints = filtered_joints(t,q[:,:6],sigma)
            coefficients = pchip_coefficients(t,joints)
            v,a,error = curve_metrics(t,coefficients,q[:,:6])
            if v<=vmax+1e-8 and a<=amax+1e-8 and error<=.5+1e-8:
                selected = coefficients
                summary.update(filter_sigma_s=sigma,joint_max_deviation_deg=error,
                               peak_joint_velocity_deg_s=v,peak_joint_acceleration_deg_s2=a)
                break
        if selected is None:
            raise ValueError('无法在原时长及 0.5° 偏差内满足平滑速度/加速度限制；未发送运动命令。请检查样本，或用 --smoothing linear 对照原轨迹')
        values = np.column_stack([evaluate(t,selected,ticks),np.interp(ticks,t,q[:,6])])
    else:
        raise ValueError('未知回放平滑模式')
    values[0], values[-1] = q[0], q[-1]
    if cfg is not None:
        from servo_bridge.core import CommandGuard
        bounds = np.vstack([np.degrees(CommandGuard(cfg).limits),cfg['gripper_limits_m']])
        quantized = np.column_stack([np.round(values[:,:6]*1000)/1000,np.round(values[:,6]*1e6)/1e6])
        for data in (values,quantized):
            if np.any(data<bounds[:,0]) or np.any(data>bounds[:,1]):
                raise ValueError('插值轨迹超出关节/夹爪限位')
        speed = abs(np.diff(quantized,axis=0)/np.diff(ticks)[:,None])
        if np.max(speed[:,:6])>math.degrees(cfg['maximum_joint_velocity_rad_s'])+1e-8 or np.max(speed[:,6])>cfg['gripper_speed_m_s']+1e-8:
            raise ValueError('插值量化后的轨迹速度超限')
        if smoothing == 'smooth' and len(ticks)>2:
            dt = np.diff(ticks)
            signed_velocity = np.diff(quantized[:,:6],axis=0)/dt[:,None]
            acceleration = np.diff(signed_velocity,axis=0)/((dt[:-1]+dt[1:])[:,None]/2)
            if np.max(abs(acceleration)) > math.degrees(cfg['maximum_joint_acceleration_rad_s2'])+1e-8:
                raise ValueError('平滑量化后的发送轨迹加速度超限')
    return ticks, values, summary
