"""Gamepad_PiPER input adapter and read-only controller inspector.

Linux mapping adapted from kehuanjack/Gamepad_PiPER at aa00c80 (MIT).
Copyright (c) 2025 科幻木影; full license: third_party/gamepad_piper/LICENSE.
This module never imports a robot SDK, opens CAN or sends robot commands.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import time

AXES = dict(left_x=0, left_y=1, right_x=3, right_y=4, left_trigger=2, right_trigger=5)
BUTTONS = dict(a=0, b=1, x=2, y=3, lb=4, rb=5, back=6, start=7)


@dataclass
class PadState:
    connected: bool = False
    timestamp: float = 0.
    name: str = "未连接"
    guid: str = ""
    instance: int = -1
    axes: dict = field(default_factory=lambda: {k: 0. for k in AXES})
    buttons: dict = field(default_factory=lambda: {k: False for k in BUTTONS})
    hat: tuple = (0, 0)
    raw_axes: list = field(default_factory=list)
    raw_buttons: list = field(default_factory=list)
    error: str = ""


def deadzone(value, width=.12):
    if not math.isfinite(value) or not 0 <= width < 1:
        raise ValueError("Finite axis value and deadzone in [0,1) required")
    value = max(-1., min(1., value))
    return math.copysign(max(0., (abs(value) - width) / (1 - width)), value)


class PygamePad:
    def __init__(self, settings=None, *, clock=time.monotonic):
        settings = settings or {}
        self.clock = clock
        self.axis_map = {**AXES, **settings.get("axis_map", {})}
        self.button_map = {**BUTTONS, **settings.get("button_map", {})}
        for mapping in (self.axis_map, self.button_map):
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in mapping.values()):
                raise ValueError("Gamepad mappings need non-negative integer indices")
        self.index = settings.get("index", 0)
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("gamepad.index needs a non-negative integer")
        self.guid = settings.get("guid")
        self.zone = settings.get("deadzone", .12)
        self.trigger_min = settings.get("trigger_min", -1.)
        self.trigger_max = settings.get("trigger_max", 1.)
        if (not 0 <= self.zone < 1 or not math.isfinite(self.trigger_min)
                or not math.isfinite(self.trigger_max) or self.trigger_min >= self.trigger_max):
            raise ValueError("Invalid deadzone/trigger range")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError("Install gamepad extra: python -m pip install -r requirements-gamepad.txt") from exc
        self.pg = pygame
        pygame.display.init()
        pygame.display.set_mode((1, 1))
        pygame.joystick.init()
        self.joy = None
        self.selected_guid = self.guid

    def list_devices(self):
        self.pg.event.pump()
        result = []
        for i in range(self.pg.joystick.get_count()):
            j = self.pg.joystick.Joystick(i)
            result.append(dict(index=i, name=j.get_name(), guid=j.get_guid(), axes=j.get_numaxes(),
                               buttons=j.get_numbuttons(), hats=j.get_numhats()))
        return result

    def poll(self):
        raw, raw_buttons = [], []
        try:
            for event in self.pg.event.get():
                if event.type == self.pg.JOYDEVICEREMOVED and self.joy is not None:
                    if event.instance_id == self.joy.get_instance_id():
                        self.joy = None
            if self.joy is not None and not self.joy.get_attached():
                self.joy = None
            if self.joy is None:
                devices = self.list_devices()
                candidates = [d for d in devices if d['guid'] == self.selected_guid] if self.selected_guid else devices[self.index:self.index+1]
                if not candidates:
                    return PadState(timestamp=self.clock())
                self.joy = self.pg.joystick.Joystick(candidates[0]['index'])
                self.joy.init()
                self.selected_guid = self.joy.get_guid()
            j = self.joy
            raw = [float(j.get_axis(i)) for i in range(j.get_numaxes())]
            raw_buttons = [bool(j.get_button(i)) for i in range(j.get_numbuttons())]
            if max(self.axis_map.values()) >= len(raw) or max(self.button_map.values()) >= j.get_numbuttons():
                raise ValueError("手柄轴/按钮数量与映射不符；先运行 e14.gamepad --monitor 并配置映射")
            axes = {}
            for name, index in self.axis_map.items():
                value = raw[index]
                if not math.isfinite(value):
                    raise ValueError("Non-finite gamepad axis")
                if 'trigger' in name:
                    axes[name] = max(0., min(1., (value-self.trigger_min)/(self.trigger_max-self.trigger_min)))
                else:
                    axes[name] = deadzone(value, self.zone)
            return PadState(True, self.clock(), j.get_name(), j.get_guid(), j.get_instance_id(), axes,
                            {k: raw_buttons[i] for k,i in self.button_map.items()},
                            j.get_hat(0) if j.get_numhats() else (0,0), raw, raw_buttons)
        except (self.pg.error, ValueError, OSError) as exc:
            self.joy = None
            return PadState(timestamp=self.clock(), error=str(exc), raw_axes=raw, raw_buttons=raw_buttons)

    def close(self):
        self.pg.joystick.quit()
        self.pg.display.quit()


class DemoPad:
    """Synthetic input, permitted only with a synthetic robot."""
    def __init__(self, *, clock=time.monotonic):
        self.clock, self.started = clock, clock()

    def poll(self):
        t = (self.clock() - self.started) % 12
        state = PadState(True, self.clock(), "模拟手柄", "mock", 1)
        if 1 < t < 5:
            state.buttons['lb'] = True
            state.axes['left_y'] = -.3
        elif 6 < t < 10:
            state.buttons['lb'] = True
            state.axes['left_y'] = .3
            state.axes['left_trigger'] = .2
        return state

    def close(self):
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--duration', type=float, default=15.)
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text()).get('gamepad', {}) if args.config else {}
    pad = PygamePad(settings)
    try:
        print(json.dumps(pad.list_devices(), ensure_ascii=False, indent=2))
        if args.monitor:
            until = time.monotonic() + args.duration
            while time.monotonic() < until:
                print(json.dumps(asdict(pad.poll()), ensure_ascii=False), flush=True)
                time.sleep(.2)
    finally:
        pad.close()


if __name__ == '__main__':
    main()
