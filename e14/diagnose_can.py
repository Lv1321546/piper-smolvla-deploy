"""Passively log CAN frames and PiPER feedback, including stopped/error states.

Does not use SDK control methods, set CAN up, enable motors, select modes,
query firmware, stop or reset. Observe control frames from other bus users
without attributing their origin to this process. SDK sockets using
local_loopback=False cannot be captured here; use a send-side journal too.
Never overwrites a log.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from e14.piper_robot import FeedbackStore, ensure_can
from shared.config import load_config

CONTROL_IDS = {0x150, 0x151, 0x155, 0x156, 0x157, 0x159, 0x471}


def capture(bus, output, *, duration=5., clock=time.monotonic, interface='can0', ready=None, stop_requested=None):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    store = FeedbackStore(clock)
    counts = Counter()
    started = clock()
    last_snapshot = started - .05
    summary = {'event': 'summary', 'status': 'completed', 'read_only': True,
               'actuator_commands_sent': 0, 'error_frames': 0}
    with Path(output).open('x', encoding='utf-8') as log:
        def write(event):
            log.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

        def snapshot(now):
            try:
                value = store.snapshot()
            except RuntimeError as exc:
                return {'event': 'feedback', 'monotonic': now, 'error': str(exc)}
            return {'event': 'feedback', 'monotonic': now, **value,
                    'frame_age_ms': {key: round((now-stamp)*1000, 3)
                                     for key, stamp in value['frame_timestamps'].items()}}

        write({'event': 'metadata', 'format': 'piper_passive_can_v1', 'read_only': True,
               'interface': interface, 'started_utc': datetime.now(timezone.utc).isoformat(),
               'started_monotonic': started, 'duration_s': duration,
               'note': 'Receives only; other senders using local_loopback=False are not observable here. Use a send-side journal too.'})
        log.flush()
        if ready is not None:
            ready()
        try:
            while clock()-started < duration and not (stop_requested and stop_requested()):
                frame = bus.recv(timeout=min(.1, max(0., duration-(clock()-started))))
                now = clock()
                if frame is not None:
                    key = hex(frame.arbitration_id)
                    counts[key] += 1
                    summary['error_frames'] += int(frame.is_error_frame)
                    write({'event': 'frame', 'monotonic': now, 'bus_timestamp': frame.timestamp,
                           'can_id': key, 'data_hex': bytes(frame.data).hex(),
                           'is_error_frame': bool(frame.is_error_frame),
                           'is_remote_frame': bool(frame.is_remote_frame),
                           'is_extended_id': bool(frame.is_extended_id),
                           'observed_control_frame': frame.arbitration_id in CONTROL_IDS})
                    if not (frame.is_error_frame or frame.is_remote_frame or frame.is_extended_id):
                        store.ingest(frame.arbitration_id, frame.data, received_at=now)
                if now-last_snapshot >= .05:
                    write(snapshot(now))
                    log.flush()
                    last_snapshot = now
        except BaseException as exc:
            summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                           error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            summary.update(elapsed_s=clock()-started, counts_by_id=dict(counts),
                           observed_control_counts={key: count for key, count in counts.items()
                                                    if int(key, 16) in CONTROL_IDS},
                           final_feedback=snapshot(clock()))
            write(summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--output', type=Path,
                        default=Path('outputs') / f'can_diagnostic_{time.time_ns()}.jsonl')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive and finite')
    interface = load_config(args.config)['robot']['can_interface']
    ensure_can(interface)  # Inspection only; initialize/restart stay False.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import can
    print(f'Passive CAN log: {args.output.resolve()} (no actuator commands)', flush=True)
    with can.Bus(channel=interface, interface='socketcan', receive_own_messages=False) as bus:
        result = capture(bus, args.output, duration=args.duration, interface=interface)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
