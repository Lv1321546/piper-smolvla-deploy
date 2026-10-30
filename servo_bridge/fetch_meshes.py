"""Fetch only the pinned, SHA-256-checked meshes; never execute upstream code."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='models/piper_servo_mesh_source')
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('mesh_manifest.json').read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, digest in manifest['sha256'].items():
        target = output / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            continue
        url = f"https://raw.githubusercontent.com/kehuanjack/Gamepad_PiPER/{manifest['commit']}/piper/meshes/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'SHA-256 mismatch for {name}; file was not installed')
        target.write_bytes(data)
        print(name)
    print('Ready:', output)


if __name__ == '__main__':
    main()
