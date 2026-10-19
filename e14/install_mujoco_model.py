"""Download the pinned AgileX PiPER model assets; never runs upstream code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = 'ac41fcbcdda598f01b51cf6175ed9a24d0dacadc'
SOURCE = 'https://github.com/agilexrobotics/piper_ros'
MODEL_ROOT = Path(__file__).resolve().parents[1] / 'models/piper_official'
DESCRIPTION = 'src/piper_description'
FILES = ['LICENSE', f'{DESCRIPTION}/mujoco_model/piper_description.xml'] + [
    f'{DESCRIPTION}/meshes/{name}.STL' for name in
    ['base_link', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'gripper_base', 'link7', 'link8']]
SCENE = '''<mujoco model="PiPER official gamepad preview">
  <include file="src/piper_description/mujoco_model/piper_description.xml"/>
  <visual><headlight ambient=".45 .45 .45" diffuse=".6 .6 .6"/>
    <global azimuth="135" elevation="-25"/></visual>
  <worldbody>
    <light pos="0 -1 2"/>
    <geom type="plane" size="1 1 .01" rgba=".16 .2 .25 1"/>
    <camera name="overview" pos=".9 -.9 .75" xyaxes=".707107 .707107 0 -.331295 .331295 .883452"/>
    <geom type="box" pos=".4 .15 .002" size=".12 .10 .002" rgba=".1 .35 .6 1"/>
    <geom type="box" pos=".35 -.05 .02" size=".02 .02 .02" rgba=".03 .03 .03 1"/>
  </worldbody>
</mujoco>
'''


def install(root=MODEL_ROOT):
    root = Path(root)
    manifest = root/'source_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['commit'] != COMMIT or set(saved['sha256']) != set(FILES + ['scene.xml']):
            raise ValueError('Existing manifest does not match this pinned model')
        for name, digest in saved['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Model file changed: {name}')
        return root/'scene.xml'

    def download(name):
        url = f'https://raw.githubusercontent.com/agilexrobotics/piper_ros/{COMMIT}/{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read(16*1024*1024+1)
        if not content or len(content) > 16*1024*1024:
            raise ValueError(f'Unexpected asset size: {name}')
        return name, content

    with ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(download, FILES))
    assets.append(('scene.xml', SCENE.encode()))
    for name, content in assets:
        path = root/name
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f'Refusing to overwrite a different model: {path}')
    digests = {}
    for name, content in assets:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digests[name] = hashlib.sha256(content).hexdigest()
    manifest.write_text(json.dumps({'source': SOURCE, 'commit': COMMIT, 'sha256': digests,
                                    'note': 'Upstream XML/STL/LICENSE unchanged; scene.xml is a local wrapper.'}, indent=2)+'\n')
    return root/'scene.xml'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=MODEL_ROOT)
    args=p.parse_args()
    print(install(args.output))


if __name__ == '__main__':
    main()
