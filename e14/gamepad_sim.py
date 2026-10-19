"""Optional MuJoCo kinematic mirror. No physics stepping, CAN or real cameras."""
import xml.etree.ElementTree as ET
import time
import numpy as np

from e14.gamepad_kinematics import URDF, PiperKinematics


class MujocoKinematics(PiperKinematics):
    """Use the displayed model for mock-mode IK, including its joint limits."""
    def __init__(self, model, limits_deg):
        import mujoco
        self.mj, self.model, self.data = mujoco, model, mujoco.MjData(model)
        joints = [model.joint(f'joint{i}') for i in range(1,7)]
        if any(j.type[0] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError('PiPER preview requires six named hinge joints')
        self.addresses = [j.qposadr[0] for j in joints]
        self.dofs = [j.dofadr[0] for j in joints]
        self.body = int(model.jnt_bodyid[joints[-1].id])
        bounds = np.radians(np.asarray(limits_deg,float))
        self.low = np.maximum(bounds[:,0], [j.range[0] for j in joints])
        self.high = np.minimum(bounds[:,1], [j.range[1] for j in joints])
        if np.any(self.low >= self.high):
            raise ValueError('No joint range intersection between preview and config')

    def forward(self, q, *, jacobian=False):
        q = np.asarray(q,float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Six finite radians required')
        self.data.qpos[self.addresses] = q
        self.mj.mj_forward(self.model,self.data)
        pose = np.eye(4)
        pose[:3,3] = self.data.xpos[self.body]
        pose[:3,:3] = self.data.xmat[self.body].reshape(3,3)
        if not jacobian:
            return pose
        jp,jr = np.zeros((3,self.model.nv)),np.zeros((3,self.model.nv))
        self.mj.mj_jacBody(self.model,self.data,jp,jr,self.body)
        return pose,np.vstack([jp[:,self.dofs],jr[:,self.dofs]])


def simplified_scene():
    """Mesh-free geometry with the pinned URDF's exact joint transforms."""
    root = ET.Element('mujoco', model='PiPER gamepad kinematic preview')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', gravity='0 0 0')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -1 2')
    ET.SubElement(world, 'geom', type='plane', size='1 1 .01', rgba='.16 .2 .25 1')
    ET.SubElement(world, 'camera', name='overview', pos='1 -1 .85',
                  xyaxes='.707107 .707107 0 -.331295 .331295 .883452')
    ET.SubElement(world, 'geom', type='box', pos='.4 .15 .002', size='.12 .10 .002', rgba='.1 .35 .6 1')
    ET.SubElement(world, 'geom', type='box', pos='.35 -.05 .02', size='.02 .02 .02', rgba='.03 .03 .03 1')
    base = ET.SubElement(world, 'body', name='base_link')
    ET.SubElement(base, 'geom', type='cylinder', pos='0 0 .04', size='.055 .04', rgba='.6 .65 .7 1')
    source = ET.parse(URDF).getroot()
    bodies = {'base_link': base}
    joints = [j for j in source.findall('joint') if j.find('parent').get('link') != 'base_footprint']
    for joint in joints:
        parent = bodies[joint.find('parent').get('link')]
        origin = joint.find('origin')
        r,p,y = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')/2
        cr,cp,cy = np.cos([r,p,y]); sr,sp,sy = np.sin([r,p,y])
        quat = [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]
        name = joint.find('child').get('link')
        body = ET.SubElement(parent, 'body', name=name, pos=origin.get('xyz', '0 0 0'),
                             quat=' '.join(map(str,quat)))
        bodies[name] = body
        if joint.get('type') != 'fixed':
            limits = joint.find('limit')
            ET.SubElement(body, 'joint', name=joint.get('name'),
                          type='slide' if joint.get('type') == 'prismatic' else 'hinge',
                          axis=joint.find('axis').get('xyz'),
                          range=f"{limits.get('lower')} {limits.get('upper')}")
        if name in {'link7','link8'}:
            ET.SubElement(body, 'geom', type='box', pos='0 -.02 0', size='.009 .03 .006', rgba='.2 .25 .3 1')
        else:
            # Capsules connect adjacent joint origins; spheres show zero-length links.
            child = next((j for j in joints if j.find('parent').get('link') == name), None)
            xyz = np.fromstring(child.find('origin').get('xyz'),sep=' ') if child is not None else np.zeros(3)
            if np.linalg.norm(xyz) > .01:
                ET.SubElement(body, 'geom', type='capsule', fromto='0 0 0 '+' '.join(map(str,xyz)),
                              size='.022', rgba='.75 .8 .85 1')
            else:
                ET.SubElement(body, 'geom', type='sphere', size='.028', rgba='.35 .4 .5 1')
    ET.SubElement(bodies['link6'], 'site', name='flange', size='.006', rgba='1 .3 .1 1')
    return ET.tostring(root, encoding='unicode')


class MujocoMirror:
    def __init__(self, path=None, *, headless=False):
        import mujoco
        self.mj=mujoco
        self.model=(mujoco.MjModel.from_xml_path(str(path.resolve())) if path
                    else mujoco.MjModel.from_xml_string(simplified_scene()))
        self.data=mujoco.MjData(self.model)
        self.addresses=[self.model.joint(f'joint{i}').qposadr[0] for i in range(1,7)]
        self.viewer=None
        if not headless:
            import mujoco.viewer
            self.viewer=mujoco.viewer.launch_passive(self.model,self.data)

    def kinematics(self, limits_deg):
        return MujocoKinematics(self.model,limits_deg)

    def update(self, state):
        if self.viewer and not self.viewer.is_running():
            raise RuntimeError('MuJoCo 窗口已关闭，请重新启动手柄程序')
        def apply():
            self.data.qpos[self.addresses]=np.radians(state[:6])
            for i,sign in [(7,1),(8,-1)]:
                joint=self.model.joint(f'joint{i}')
                self.data.qpos[joint.qposadr[0]]=np.clip(sign*state[6]/2,*joint.range)
            self.data.qvel[:]=0
            self.mj.mj_forward(self.model,self.data)
        if self.viewer:
            with self.viewer.lock():apply()
            self.viewer.sync()
        else:apply()

    def render(self, width=640, height=480):
        with self.mj.Renderer(self.model,height=height,width=width) as renderer:
            renderer.update_scene(self.data,camera='overview')
            return renderer.render().copy()

    def close(self):
        if self.viewer:
            self.viewer.close()
            # launch_passive owns a daemon render thread. Its public running
            # flag becomes false when exit is requested, before GL teardown.
            # Keep the handle/model alive until the current pygame/MuJoCo
            # combination has finished releasing the native viewer.
            sim_ref=getattr(self.viewer,'_sim',None)
            deadline=time.monotonic()+3.
            while callable(sim_ref) and sim_ref() is not None and time.monotonic()<deadline:
                time.sleep(.01)
            self.viewer=None
