"""Dependency-free local HTTP API for the ABS diagnostic dashboard."""

from __future__ import annotations

import argparse
import json
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from numbers import Integral, Real
from urllib.parse import urlparse

from .live_service import LIVE_SERVICE
from .service import SERVICE, list_scenarios


def _json_safe(value: object) -> object:
    """Convert numpy scalars and non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


class DiagnosticAPIHandler(BaseHTTPRequestHandler):
    server_version = "ABSDiagnosticAPI/1.0"

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._send_json({}, HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                self._send_json(
                    {
                        "status": "ok",
                        "service": "ABS diagnostic API",
                        "live_api_version": 3,
                    }
                )
                return
            if path == "/api/scenarios":
                self._send_json({"scenarios": list_scenarios()})
                return
            if path == "/api/live/state":
                self._send_json(LIVE_SERVICE.state())
                return
            if path == "/api/live/hud":
                self._send_json(LIVE_SERVICE.hud_state())
                return
            if path == "/api/live/config":
                self._send_json(LIVE_SERVICE.driver_config())
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:  # The API boundary returns a readable local error.
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/diagnose":
                source = str(payload.get("source", "faulty"))
                simulation_id = int(payload["simulation_id"])
                self._send_json(SERVICE.diagnose(source, simulation_id))
                return
            if path == "/api/live/start":
                self._send_json(LIVE_SERVICE.start(payload))
                return
            if path == "/api/live/driver-started":
                LIVE_SERVICE.mark_running(payload)
                self._send_json({"status": "ok"})
                return
            if path == "/api/live/driver-config":
                self._send_json({"config": LIVE_SERVICE.update_driver_config(payload)})
                return
            if path == "/api/live/frames":
                frames = payload.get("frames")
                if not isinstance(frames, list):
                    raise ValueError("frames must be a list.")
                self._send_json({"accepted": LIVE_SERVICE.ingest(frames)})
                return
            if path == "/api/live/stop":
                self._send_json(LIVE_SERVICE.stop())
                return
            if path == "/api/live/driver-stopped":
                LIVE_SERVICE.mark_stopped()
                self._send_json({"status": "ok"})
                return
            if path == "/api/live/driver-failed":
                LIVE_SERVICE.mark_failed(str(payload.get("error", "CARLA driver failed.")))
                self._send_json({"status": "ok"})
                return
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[ABS API] {self.address_string()} - {format % args}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    server = ThreadingHTTPServer((arguments.host, arguments.port), DiagnosticAPIHandler)
    print(f"ABS diagnostic API: http://{arguments.host}:{arguments.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
