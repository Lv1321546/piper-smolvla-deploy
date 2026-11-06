"""Forward only inference HTTP via an existing SSH host/jump-host configuration.

Creates a temporary, minimal SSH configuration to avoid inheriting unrelated
RemoteForward entries from the normal server alias. No keys are copied.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def settings(host):
    result = subprocess.run(['ssh', '-G', host], check=True, capture_output=True, text=True)
    values = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition(' ')
        values.setdefault(key, []).append(value)
    if values.get('proxycommand', ['none'])[0] != 'none':
        raise ValueError('This helper supports direct SSH or one ProxyJump alias; use a manual tunnel for ProxyCommand')
    return values


def host_block(alias, values):
    rows = ['Host '+alias]
    for key in ('hostname', 'user', 'port', 'identityfile', 'identitiesonly', 'hostkeyalias'):
        for value in values.get(key, []):
            rows.append('    '+key+' '+json.dumps(value))
    return rows


def tunnel_config(host):
    target = settings(host)
    rows = host_block('inference-target', target)
    jump = target.get('proxyjump', ['none'])[0]
    if jump != 'none':
        if ',' in jump or ':' in jump or '@' in jump:
            raise ValueError('Use a named single ProxyJump host alias with this helper')
        hop = settings(jump)
        if hop.get('proxyjump', ['none'])[0] != 'none':
            raise ValueError('Nested ProxyJump is not supported by this helper')
        rows += ['    ProxyJump inference-jump'] + host_block('inference-jump', hop)
    rows += ['Host *', '    ServerAliveInterval 15', '    ServerAliveCountMax 3']
    return '\n'.join(rows)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', default='Inspur')
    parser.add_argument('--local-port', type=int, default=8081)
    parser.add_argument('--remote-port', type=int, default=8081)
    args = parser.parse_args()
    if not 1 <= args.local_port <= 65535 or not 1 <= args.remote_port <= 65535:
        parser.error('Ports must be 1..65535')
    with tempfile.TemporaryDirectory(prefix='piper-inference-ssh-') as directory:
        file = Path(directory)/'config'
        file.write_text(tunnel_config(args.host))
        file.chmod(0o600)
        print(f'HTTP tunnel: 127.0.0.1:{args.local_port} -> {args.host}:127.0.0.1:{args.remote_port}; Ctrl+C to close',flush=True)
        try:
            return subprocess.call(['ssh','-F',str(file),'-N','-o','ExitOnForwardFailure=yes',
                                    '-L',f'127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}','inference-target'])
        except KeyboardInterrupt:
            return 130


if __name__ == '__main__':
    raise SystemExit(main())
