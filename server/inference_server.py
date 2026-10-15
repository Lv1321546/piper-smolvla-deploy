"""Single-flight HTTP inference, normally reached through SSH local forwarding.

Run: python -m server.inference_server --config config.json --mock
"""
from __future__ import annotations
import argparse
import json
import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from shared.protocol import PROTOCOL_VERSION, decode_images, validate_request, validate_response
from server.policy_runner import PolicyRunner

LOG = logging.getLogger(__name__)


def create_app(runner: PolicyRunner, config: dict) -> ThreadingHTTPServer:
    options = config.get("server", {})
    maximum = int(options.get("max_request_bytes", 8 * 1024 * 1024))
    max_pixels = int(options.get("max_image_pixels", 2073600))
    if not 1024 <= maximum <= 32 * 1024 * 1024:
        raise ValueError("max_request_bytes must be between 1 KiB and 32 MiB")
    gate = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def setup(self):
            super().setup()
            self.connection.settimeout(float(options.get("read_timeout_s", options.get("request_timeout_s", 10))))

        def log_message(self, format, *args):
            LOG.info("%s %s", self.client_address[0], format % args)

        def reply(self, status, payload):
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if urlsplit(self.path).path != "/health":
                return self.reply(404, {"error": "not_found"})
            return self.reply(200 if runner.loaded else 503,
                              {"status": "ready" if runner.loaded else "not_ready", **runner.metadata})

        def do_POST(self):
            if urlsplit(self.path).path != "/predict":
                return self.reply(404, {"error": "not_found"})
            if not gate.acquire(blocking=False):
                return self.reply(429, {"error": "busy"})
            try:
                if not runner.loaded:
                    return self.reply(503, {"error": "not_ready"})
                if self.headers.get("Transfer-Encoding"):
                    return self.reply(400, {"error": "unsupported_transfer_encoding"})
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if not 0 < length <= maximum:
                    return self.reply(413 if length > maximum else 400, {"error": "invalid_content_length"})
                if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                    return self.reply(415, {"error": "content_type_must_be_json"})
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request body")
                def reject_constant(value):
                    raise ValueError(f"Invalid JSON numeric constant: {value}")
                payload = validate_request(json.loads(body, parse_constant=reject_constant))
                hz = runner.metadata["control_hz"]
                if not math.isclose(float(payload["control_hz"]), float(hz), rel_tol=0, abs_tol=1e-6):
                    raise ValueError("Request frequency differs from model training frequency")
                images = decode_images(payload["images"], max_pixels=max_pixels)
                actions, milliseconds = runner.predict({"state": payload["state"], "images": images}, payload["task"])
                result = {"protocol_version": PROTOCOL_VERSION, "request_id": payload["request_id"],
                          "actions": actions, "metadata": runner.metadata, "inference_ms": milliseconds}
                validate_response(result, request_id=payload["request_id"], expected_hz=hz)
                return self.reply(200, result)
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                return self.reply(400, {"error": "invalid_request", "message": str(error)[:240]})
            except TimeoutError:
                return self.reply(408, {"error": "request_timeout"})
            except Exception:
                LOG.exception("Inference failed")
                return self.reply(503, {"error": "inference_failed", "message": "Inspect server log"})
            finally:
                gate.release()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 4

    host = options.get("bind", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost") and not options.get("allow_network_bind", False):
        raise ValueError("Non-loopback bind requires explicit server.allow_network_bind=true")
    return Server((host, int(options.get("port", 8080))), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--port", type=int, help="Override port; 0 selects a free port")
    args = parser.parse_args(argv)
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    if args.port is not None:
        config.setdefault("server", {})["port"] = args.port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = PolicyRunner(config, mock=args.mock).load()
    server = create_app(runner, config)
    LOG.info("Listening on http://%s:%d backend=%s execution_ready=%s",
             *server.server_address, runner.metadata["backend"], runner.metadata["execution_ready"])
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        LOG.info("Stopping inference server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
