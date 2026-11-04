"""Atomic, bounded session-local intent mailbox between collector and runtime."""
import json
from pathlib import Path
import time


class ControlMailbox:
    def __init__(self, path, *, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        self.sent = self.received = 0

    def send(self, command):
        if command not in ('reset', 'cancel_reset'):
            raise ValueError('Unsupported reset command')
        sequence = self.sent+1
        packet = dict(version=1, seq=sequence, monotonic=self.clock(), command=command)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(packet)+'\n')
        temporary.replace(self.path)
        self.sent = sequence
        return sequence

    def receive(self):
        try:
            if self.path.stat().st_size > 1024:
                raise ValueError('Oversized reset intent')
            packet = json.loads(self.path.read_text())
        except FileNotFoundError:
            return None
        if not isinstance(packet, dict):
            raise ValueError('Invalid reset intent')
        sequence = packet.get('seq')
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError('Invalid reset sequence')
        if sequence <= self.received:
            return None
        self.received = sequence
        stamp = packet.get('monotonic')
        if (packet.get('version') != 1 or isinstance(stamp, bool) or
                not isinstance(stamp, (int, float)) or not 0 <= self.clock()-stamp <= .5 or
                packet.get('command') not in ('reset', 'cancel_reset')):
            raise ValueError('Expired or invalid reset intent')
        return packet
