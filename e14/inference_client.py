"""Bounded HTTP client; local monotonic timestamps never cross clock domains."""
import json
import http.client
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid

from shared.protocol import dumps, encode_images, validate_metadata, validate_request, validate_response


class InferenceClient:
    def __init__(self, config):
        self.config = config
        options = config['client']
        self.url = options['url'].rstrip('/')
        parsed = urlsplit(self.url)
        if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError('Use a localhost SSH tunnel URL')
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.timeout = options['timeout_ms'] / 1000
        if not 0 < self.timeout <= 60:
            raise ValueError('client.timeout_ms must be in (0,60000]')
        self.max_bytes = options.get('max_response_bytes', 2 * 1024 * 1024)
        if not isinstance(self.max_bytes, int) or not 0 < self.max_bytes <= 8 * 1024 * 1024:
            raise ValueError('Invalid max_response_bytes')

    def _request(self, route, payload=None):
        # A socket timeout alone can be defeated by a server dripping bytes.
        # Shutdown the actual transport at an absolute deadline, including headers.
        deadline = time.monotonic() + self.timeout
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        expired = threading.Event()
        timer = None
        try:
            connection.connect()
            transport = connection.sock
            def abort():
                expired.set()
                try:
                    transport.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                transport.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Inference connection deadline exceeded')
            timer = threading.Timer(remaining, abort)
            timer.daemon = True
            timer.start()
            connection.request('POST' if payload is not None else 'GET', route,
                               body=dumps(payload) if payload is not None else None,
                               headers={'Content-Type': 'application/json'})
            response = connection.getresponse()
            raw = response.read(self.max_bytes + 1)
            if expired.is_set() or time.monotonic() > deadline:
                raise TimeoutError('Inference overall request deadline exceeded')
            if len(raw) > self.max_bytes:
                raise ValueError('Inference response too large')
            if response.status != 200:
                raise RuntimeError(f'Inference HTTP {response.status}: {raw[:4096].decode("utf-8", errors="replace")}')
            return json.loads(raw)
        except (OSError, http.client.HTTPException, ValueError) as error:
            if expired.is_set() or time.monotonic() > deadline:
                raise TimeoutError('Inference overall request deadline exceeded') from error
            raise
        finally:
            if timer:
                timer.cancel()
            connection.close()

    def health(self, require_executable=False):
        response = self._request('/health')
        metadata = response.get('metadata', response)
        validate_metadata(metadata, self.config['robot']['control_hz'], require_executable)
        return metadata

    def predict(self, observation, task, request_id=None, require_executable=False):
        request_id = request_id or uuid.uuid4().hex
        captured = float(observation['captured_monotonic'])
        started = time.monotonic()
        max_age = self.config['client']['max_action_age_ms'] / 1000
        if started - captured > max_age or captured > started:
            raise ValueError('Observation already stale or from a different clock')
        payload = {
            'protocol_version': 'piper-v1', 'request_id': request_id, 'robot_type': 'piper',
            'state': list(observation['state']), 'images': encode_images(observation['images']),
            'task': task, 'control_hz': self.config['robot']['control_hz'],
        }
        validate_request(payload)
        response = self._request('/predict', payload)
        received = time.monotonic()
        validate_response(response, request_id, self.config['robot']['control_hz'], require_executable)
        if received - captured > max_age:
            raise TimeoutError('Prediction expired before arrival; discard chunk')
        return {
            'response': response, 'request_id': request_id,
            'captured_monotonic': captured, 'request_started_monotonic': started,
            'received_monotonic': received, 'round_trip_ms': (received - started) * 1000,
        }

    def check_response(self, response, request_id, require_executable=False):
        return validate_response(response, request_id, self.config['robot']['control_hz'], require_executable)
