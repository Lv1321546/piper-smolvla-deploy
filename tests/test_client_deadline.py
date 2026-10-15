import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
import unittest

from e14.inference_client import InferenceClient


class ClientDeadlineTest(unittest.TestCase):
    def test_dripping_headers_and_body_obey_whole_request_deadline(self):
        for stage in ('headers', 'body'):
            with self.subTest(stage=stage):
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args): pass
                    def do_GET(self):
                        try:
                            if stage == 'headers':
                                for byte in b'HTTP/1.0 200 OK\r\n':
                                    self.connection.sendall(bytes([byte]))
                                    time.sleep(.025)
                            else:
                                self.send_response(200)
                                self.send_header('Content-Length', '100')
                                self.end_headers()
                                for _ in range(100):
                                    self.wfile.write(b' ')
                                    self.wfile.flush()
                                    time.sleep(.025)
                        except (OSError, BrokenPipeError): pass
                server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                server.daemon_threads = True
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                config = json.loads((Path(__file__).resolve().parents[1] / 'config.mock.json').read_text())
                config['client'].update(url=f'http://127.0.0.1:{server.server_address[1]}', timeout_ms=100)
                try:
                    start = time.monotonic()
                    with self.assertRaises(TimeoutError):
                        InferenceClient(config)._request('/health')
                    self.assertLess(time.monotonic() - start, .5)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=1)


if __name__ == '__main__': unittest.main()
