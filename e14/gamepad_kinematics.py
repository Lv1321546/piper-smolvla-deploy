"""Bounded local FK/IK using Gamepad_PiPER's pinned URDF (metres, radians).

This is a kinematic solver, not a collision or dynamics planner. Targets are
seeded from measured joints; no branch jumps or unbounded iterative steps.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

URDF = Path(__file__).resolve().parents[1] / 'third_party/gamepad_piper/piper.urdf'


def rotation(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x,y,z = axis
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)


class PiperKinematics:
    def __init__(self, limits_deg, urdf=URDF):
        root = ET.parse(urdf).getroot()
        self.origins, self.axes = [], []
        urdf_limits = []
        parent = 'base_link'
        for i in range(1,7):
            joint = root.find(f"joint[@name='joint{i}']")
            if joint is None or joint.get('type') != 'revolute' or joint.find('parent').get('link') != parent:
                raise ValueError('Expected a six-joint PiPER base_link -> link6 chain')
            parent = joint.find('child').get('link')
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz','0 0 0'), sep=' ')
            rpy = np.fromstring(origin.get('rpy','0 0 0'), sep=' ')
            t = np.eye(4)
            t[:3,3] = xyz
            t[:3,:3] = rotation([0,0,1],rpy[2]) @ rotation([0,1,0],rpy[1]) @ rotation([1,0,0],rpy[0])
            self.origins.append(t)
            self.axes.append(np.fromstring(joint.find('axis').get('xyz'),sep=' '))
            limit = joint.find('limit')
            urdf_limits.append([float(limit.get('lower')), float(limit.get('upper'))])
        bounds = np.radians(np.asarray(limits_deg, float))
        if bounds.shape != (6,2) or not np.isfinite(bounds).all():
            raise ValueError('Six finite joint limit pairs required')
        self.low = np.maximum(bounds[:,0], np.array(urdf_limits)[:,0])
        self.high = np.minimum(bounds[:,1], np.array(urdf_limits)[:,1])
        if np.any(self.low >= self.high):
            raise ValueError('Configured and URDF joint limits do not intersect')

    def forward(self, q, *, jacobian=False):
        q = np.asarray(q,float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Six finite radians required')
        t = np.eye(4)
        positions, axes = [], []
        for origin, axis, angle in zip(self.origins,self.axes,q):
            t = t @ origin
            positions.append(t[:3,3].copy())
            axes.append(t[:3,:3] @ axis)
            move = np.eye(4)
            move[:3,:3] = rotation(axis,angle)
            t = t @ move
        if not jacobian:
            return t
        j = np.zeros((6,6))
        for i,(point,axis) in enumerate(zip(positions,axes)):
            j[:3,i] = np.cross(axis,t[:3,3]-point)
            j[3:,i] = axis
        return t,j

    def inverse(self, position, orientation, seed, max_delta, *, iterations=12):
        seed = np.asarray(seed,float)
        position, orientation = np.asarray(position,float), np.asarray(orientation,float)
        max_delta = np.asarray(max_delta,float)
        if (seed.shape != (6,) or max_delta.shape != (6,) or position.shape != (3,) or orientation.shape != (3,3)
                or not all(np.isfinite(v).all() for v in (seed,position,orientation,max_delta)) or np.any(max_delta<=0)):
            raise ValueError('Invalid bounded IK inputs')
        if not np.allclose(orientation.T@orientation,np.eye(3),atol=1e-6) or abs(np.linalg.det(orientation)-1)>1e-6:
            raise ValueError('IK orientation must be a proper rotation matrix')
        lo,hi = np.maximum(self.low,seed-max_delta), np.minimum(self.high,seed+max_delta)
        if np.any(lo>hi):
            return None
        q = np.clip(seed,lo,hi)
        for _ in range(iterations):
            pose,j = self.forward(q,jacobian=True)
            dr = orientation @ pose[:3,:3].T
            angle = math.acos(np.clip((np.trace(dr)-1)/2,-1,1))
            if angle>.25:
                return None  # Local teleop must never take a large SO(3) branch.
            scale = .5 if angle<1e-6 else angle/(2*math.sin(angle))
            angular = scale*np.array([dr[2,1]-dr[1,2],dr[0,2]-dr[2,0],dr[1,0]-dr[0,1]])
            error = np.r_[position-pose[:3,3],angular]
            # Fine-speed teleop can request less than 80 micrometres per tick.
            # A coarse convergence threshold would discard every such request.
            if np.linalg.norm(error[:3]) < 5e-6 and np.linalg.norm(error[3:]) < 5e-5:
                return q
            weights = np.array([1.,1.,1.,.2,.2,.2])
            a, b = weights[:,None]*j, weights*error
            dq = a.T @ np.linalg.solve(a@a.T+np.eye(6)*1e-6,b)
            dq *= min(1., .02/max(1e-12,np.max(np.abs(dq))))
            q = np.clip(q+dq,lo,hi)
        return None
