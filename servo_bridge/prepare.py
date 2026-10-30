"""Generate ROS parameters from the same pinned URDF used by the old route."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml
from servo_bridge.core import load_config

ROOT = Path(__file__).resolve().parents[1]


def prepare(output, mesh_source, cfg):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'third_party/gamepad_piper/piper.urdf'
    tree = ET.parse(source)
    root = tree.getroot()
    meshdir = output / 'meshes'
    meshdir.mkdir(exist_ok=True)
    manifest = {}
    for mesh in root.findall('.//mesh'):
        name = Path(mesh.get('filename')).name
        src = Path(mesh_source) / name
        data = src.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = json.loads((Path(__file__).with_name('mesh_manifest.json')).read_text())['sha256'][name]
        if digest != expected:
            raise ValueError(f'Mesh hash mismatch: {name}')
        target = meshdir / name
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            shutil.copyfile(src, target)
        mesh.set('filename', target.as_uri())
        manifest[name] = digest
    # The source's velocity=5 rad/s is a mechanical capability, not a teleop
    # commissioning speed. Intersect position limits with the current profile.
    import math
    limits = {}
    for i, bounds in enumerate(cfg['joint_limits_deg'], 1):
        joint = root.find(f"joint[@name='joint{i}']")
        limit = joint.find('limit')
        low, high = max(float(limit.get('lower')), math.radians(bounds[0])), min(float(limit.get('upper')), math.radians(bounds[1]))
        limit.set('lower', str(low))
        limit.set('upper', str(high))
        limit.set('velocity', str(cfg['maximum_joint_velocity_rad_s']))
        limits[f'joint{i}'] = {'has_velocity_limits': True, 'max_velocity': cfg['maximum_joint_velocity_rad_s'],
                              'has_acceleration_limits': True, 'max_acceleration': cfg['maximum_joint_acceleration_rad_s2']}
    urdf = ET.tostring(root, encoding='unicode')
    srdf = Path(__file__).with_name('piper.srdf').read_text()
    servo = yaml.safe_load(Path('/opt/ros/humble/share/moveit_servo/config/panda_simulated_config.yaml').read_text())
    servo.update(command_in_type='speed_units', publish_period=cfg['publish_period_s'],
                 move_group_name='arm', planning_frame='base_link', ee_frame_name='link6',
                 robot_link_command_frame='base_link',
                 incoming_command_timeout=cfg['command_max_age_s'],
                 joint_topic='/piper_servo/joint_states',
                 cartesian_command_in_topic='/piper_servo/twist',
                 joint_command_in_topic='/piper_servo/jog',
                 status_topic='/piper_servo/status',
                 command_out_topic='/piper_servo/candidate_trajectory')
    # Deliberately retain upstream singularity/collision thresholds and checks.
    params = {'robot_description': urdf, 'robot_description_semantic': srdf,
              'robot_description_planning': {'joint_limits': limits}, 'moveit_servo': servo}
    (output / 'piper.urdf').write_text(urdf)
    (output / 'params.yaml').write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}, sort_keys=False))
    (output / 'manifest.json').write_text(json.dumps({
        'urdf_source': str(source), 'urdf_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'urdf_generated_sha256': hashlib.sha256(urdf.encode()).hexdigest(),
        'srdf_sha256': hashlib.sha256(srdf.encode()).hexdigest(), 'mesh_sha256': manifest,
        'planning_frame': 'base_link', 'control_frame': 'link6', 'hardware_output': False,
        'servo_parameters': servo}, indent=2) + '\n')
    return output / 'params.yaml'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--mesh-source', required=True, help='Pinned Gamepad_PiPER/piper/meshes directory')
    parser.add_argument('--output', default='models/piper_servo')
    args = parser.parse_args()
    print(prepare(args.output, args.mesh_source, load_config(args.config)))


if __name__ == '__main__':
    main()
