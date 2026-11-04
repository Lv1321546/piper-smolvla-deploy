"""Latest-frame RGB camera reader. Importing does not open any camera.

The capture thread owns the device. A failed or stalled camera does not refresh
its timestamp. Timestamps are host receipt times after cap.read(), not hardware
exposure times; USB buffering/sensor latency still requires physical calibration.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np


class CameraReader:
    def __init__(self, config, mock=False, *, clock=time.monotonic, capture_factory=None):
        if not isinstance(config, dict) or not config:
            raise ValueError("At least one named camera is required")
        self.config = {name: dict(value) for name, value in config.items()}
        self.mock = bool(mock)
        self._clock = clock
        self._capture_factory = capture_factory
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frames = {}
        self._errors = {}
        self._captures = {}
        self._threads = []
        self._connected = False
        self._mock_index = 0
        self._validate_config()

    def _validate_config(self):
        for name, config in self.config.items():
            if not isinstance(name, str) or not name or name.startswith("observation."):
                raise ValueError("Camera names must be short nonempty names, e.g. front")
            if self.mock:
                for key, default in (("width", 320), ("height", 240), ("fps", 30), ("max_age_ms", 300)):
                    if config.get(key) is None:
                        config[key] = default
            for key in ("width", "height"):
                if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] <= 0:
                    raise ValueError(f"cameras.{name}.{key} must be a positive integer")
            for key in ("fps", "max_age_ms"):
                value = config.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"cameras.{name}.{key} must be a positive finite number")
            if not self.mock:
                device = config.get("device")
                if isinstance(device, bool) or not isinstance(device, (str, int)) or device == "":
                    raise ValueError(f"cameras.{name}.device is required; prefer /dev/v4l/by-id/... for stable identity")
            for key in ("connect_timeout_s", "join_timeout_s"):
                if key in config and (isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]) or config[key] <= 0):
                    raise ValueError(f"cameras.{name}.{key} must be positive")
            if "connect_retries" in config:
                retries = config["connect_retries"]
                if isinstance(retries, bool) or not isinstance(retries, int) or retries < 1:
                    raise ValueError(f"cameras.{name}.connect_retries must be a positive integer")
            if "buffer_size" in config:
                size = config['buffer_size']
                if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 32:
                    raise ValueError(f"cameras.{name}.buffer_size must be an integer in 1..32")
        devices = [str(c.get("device")) for c in self.config.values()]
        if not self.mock and len(set(devices)) != len(devices):
            raise ValueError("Multiple camera names must not point to the same device")

    @property
    def is_connected(self):
        return self._connected

    def connect(self):
        if self._connected:
            return
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError("Previous camera thread is still blocked; restart the process before reconnecting")
        self._stop.clear()
        self._frames.clear()
        self._errors.clear()
        if self.mock:
            self._connected = True
            return
        last_error = None
        # Local opt-in: default keeps the single-shot contract used by tests.
        retries = max(1, max(int(c.get("connect_retries", 1)) for c in self.config.values()))
        for attempt in range(retries):
            if attempt:
                # A failed cap.read() can block inside the driver. Wait for its
                # owner thread to exit before reopening the same device.
                deadline = time.monotonic() + 20
                while any(t.is_alive() for t in self._threads) and time.monotonic() < deadline:
                    time.sleep(.1)
                self._threads = [t for t in self._threads if t.is_alive()]
                if self._threads:
                    raise RuntimeError("Previous camera thread is still blocked; restart the process before reconnecting")
                self._captures.clear()
                self._stop.clear()
                self._frames.clear()
                self._errors.clear()
            try:
                self._connect_once()
                return
            except (TimeoutError, RuntimeError) as exc:
                last_error = exc
                self.close()
                if attempt < retries - 1:
                    time.sleep(.8)
        raise last_error

    def _connect_once(self):
        if any(t.is_alive() for t in self._threads):
            raise RuntimeError("Previous camera thread is still blocked; restart the process before reconnecting")
        import cv2
        factory = self._capture_factory or cv2.VideoCapture
        try:
            for name, config in self.config.items():
                # Linux V4L2 gives stable /dev/video* behaviour, no network URLs.
                capture = factory(config["device"], cv2.CAP_V4L2)
                if not capture.isOpened():
                    capture.release()
                    raise RuntimeError(f"Could not open camera {name}: {config['device']}")
                self._captures[name] = capture
                if config.get("fourcc"):
                    # Some UVC cameras need MJPG when USB bandwidth is limited.
                    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config["fourcc"]))
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, config["width"])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config["height"])
                capture.set(cv2.CAP_PROP_FPS, config["fps"])
                capture.set(cv2.CAP_PROP_BUFFERSIZE, config.get('buffer_size', 1))  # Best effort.
                reported_fps = capture.get(cv2.CAP_PROP_FPS)
                if reported_fps > 0 and abs(reported_fps - config["fps"]) / config["fps"] > .1:
                    raise RuntimeError(f"Camera {name} reports {reported_fps:g} FPS, requested {config['fps']}")
                thread = threading.Thread(target=self._capture_loop, args=(name, capture), daemon=True,
                                          name=f"camera-{name}")
                self._threads.append(thread)
                thread.start()
            deadline = time.monotonic() + max(c.get("connect_timeout_s", 5) for c in self.config.values())
            with self._condition:
                while len(self._frames) != len(self.config):
                    if self._errors:
                        raise RuntimeError(f"Camera capture failed: {self._errors}")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Timed out waiting for first camera frames")
                    self._condition.wait(min(remaining, .1))
            self._connected = True
            self.read()  # Also check startup frame freshness and geometry.
        except BaseException:
            self.close()
            raise

    def _capture_loop(self, name, capture):
        config = self.config[name]
        try:
            while not self._stop.is_set():
                ok, bgr = capture.read()
                captured_at = self._clock()
                if not ok or bgr is None:
                    raise RuntimeError("cap.read() failed")
                expected = (config["height"], config["width"], 3)
                if bgr.shape != expected or bgr.dtype != np.uint8:
                    raise RuntimeError(f"Expected uint8 BGR {expected}, got {bgr.shape} / {bgr.dtype}")
                rgb = np.ascontiguousarray(bgr[:, :, ::-1])
                with self._condition:
                    self._frames[name] = (rgb, captured_at)
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._errors[name] = str(exc)
                    self._condition.notify_all()
        finally:
            capture.release()

    def read(self):
        if not self._connected:
            raise RuntimeError("Cameras are not connected")
        if self.mock:
            self._mock_index += 1
            images = {}
            timestamps = {}
            for index, (name, config) in enumerate(self.config.items()):
                height, width = config["height"], config["width"]
                image = np.zeros((height, width, 3), dtype=np.uint8)
                image[:, :, 0] = (np.arange(width, dtype=np.uint16)[None, :] + self._mock_index) % 256
                image[:, :, 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
                image[:, :, 2] = (index * 50 + self._mock_index) % 256
                images[name] = image
                timestamps[name] = self._clock()
            return {"images": images, "image_timestamps": timestamps}
        with self._condition:
            if self._errors:
                raise RuntimeError(f"Camera capture failed: {self._errors}")
            snapshots = {name: (image.copy(), stamp) for name, (image, stamp) in self._frames.items()}
        now = self._clock()
        for name, config in self.config.items():
            if name not in snapshots:
                raise RuntimeError(f"Camera {name} has not delivered a frame")
            age = now - snapshots[name][1]
            if age < 0 or age * 1000 > config["max_age_ms"]:
                raise TimeoutError(f"Camera {name} frame stale: {age*1000:.1f} ms")
        return {
            "images": {name: frame[0] for name, frame in snapshots.items()},
            "image_timestamps": {name: frame[1] for name, frame in snapshots.items()},
        }

    def close(self):
        self._connected = False
        self._stop.set()
        deadline = time.monotonic() + max((c.get("join_timeout_s", 2) for c in self.config.values()), default=2)
        for thread in self._threads:
            thread.join(timeout=max(0, deadline - time.monotonic()))
        # cap.read can block in a faulty driver. Do not release a live cap from a
        # second thread (OpenCV backends can crash); its daemon owner releases it.
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        for name, capture in self._captures.items():
            if not any(thread.name == f"camera-{name}" for thread in self._threads):
                capture.release()
        if not self._threads:
            self._captures.clear()
        with self._condition:
            self._frames.clear()
            self._condition.notify_all()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Read cameras and optionally save one RGB observation; no robot control")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", help="Optional image output directory")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    with CameraReader(config["cameras"], mock=args.mock) as reader:
        observation = reader.read()
        for name, image in observation["images"].items():
            print(name, image.shape, image.dtype, observation["image_timestamps"][name])
            if args.output:
                from PIL import Image
                output = Path(args.output)
                output.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(output / f"{name}.png")


if __name__ == "__main__":
    main()
