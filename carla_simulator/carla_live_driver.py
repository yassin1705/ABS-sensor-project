"""CARLA drive and ABS dashboard in combined or separate-window mode."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import urllib.error
import urllib.request
from collections import deque
from typing import Any

try:
    import carla
    import pygame
    from pynput import keyboard
except ImportError as exc:
    raise SystemExit(
        "Live driving requires CARLA, pygame, and pynput. Run "
        "'carla_env\\Scripts\\python.exe -m pip install -r "
        "carla_simulator\\requirements.txt'."
    ) from exc

from carla_abs_simulator import (
    ABSECU,
    ABSPulseSensor,
    ECUConfig,
    FaultConfig,
    FourWheelSpeedEstimator,
    SensorConfig,
    VehicleGeometry,
    WHEELS,
)


def api_request(
    api_url: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Diagnostic API returned HTTP {exc.code} for {path}: {details}"
        ) from exc


class TelemetryPublisher:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self.items: queue.Queue[list[dict[str, Any]] | None] = queue.Queue(maxsize=20)
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def publish(self, frames: list[dict[str, Any]]) -> None:
        if self.error:
            raise RuntimeError(self.error)
        self.items.put(frames, timeout=2.0)

    def close(self) -> None:
        self.items.put(None)
        self.thread.join(timeout=10.0)

    def _run(self) -> None:
        try:
            while True:
                frames = self.items.get()
                if frames is None:
                    return
                api_request(
                    self.api_url,
                    "/api/live/frames",
                    {"frames": frames},
                    timeout=30.0,
                )
        except Exception as exc:  # Main loop reads this before publishing again.
            self.error = f"Telemetry publishing stopped: {exc}"


class DiagnosticStateReader:
    """Poll compact model results without blocking the 100 Hz CARLA loop."""

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state: dict[str, Any] = {}
        self.history = {wheel: deque(maxlen=120) for wheel in WHEELS}
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self) -> tuple[dict[str, Any], str | None]:
        with self._lock:
            snapshot = dict(self.state)
            snapshot["_residual_history"] = {
                wheel: list(values) for wheel, values in self.history.items()
            }
            return snapshot, self.error

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = api_request(self.api_url, "/api/live/hud", timeout=5.0)
                with self._lock:
                    self.state = state
                    diagnostic = state.get("diagnostic") or {}
                    frames = diagnostic.get("frames") or []
                    latest = frames[-1].get("wheels", {}) if frames else {}
                    for wheel in WHEELS:
                        residual = latest.get(wheel, {}).get("residual_mps")
                        if residual is not None:
                            self.history[wheel].append(float(residual))
                    self.error = None
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
            self._stop.wait(0.5)


class CameraBuffer:
    """Keep only the newest CARLA RGB frame for the combined renderer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: tuple[int, int, bytes, int] | None = None
        self._sequence = 0

    def update(self, image: Any) -> None:
        with self._lock:
            self._sequence += 1
            self._frame = (
                int(image.width),
                int(image.height),
                bytes(image.raw_data),
                self._sequence,
            )

    def latest(self) -> tuple[int, int, bytes, int] | None:
        with self._lock:
            return self._frame


class GlobalKeyboardControl:
    """Capture driving keys even while the CARLA server window has focus."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self.quit_requested = False
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()

    @staticmethod
    def _name(key: Any) -> str | None:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        special = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space",
            keyboard.Key.esc: "escape",
        }
        return special.get(key)

    def _on_press(self, key: Any) -> None:
        name = self._name(key)
        if name is None:
            return
        with self._lock:
            self._pressed.add(name)
            if name == "escape":
                self.quit_requested = True

    def _on_release(self, key: Any) -> None:
        name = self._name(key)
        if name is None:
            return
        with self._lock:
            self._pressed.discard(name)

    def control(self, steer_cache: float) -> tuple[Any, float, bool]:
        with self._lock:
            pressed = set(self._pressed)
            quit_requested = self.quit_requested
        throttle = 1.0 if {"w", "up"} & pressed else 0.0
        brake = 1.0 if {"s", "down"} & pressed else 0.0

        if {"a", "left"} & pressed:
            steer_cache -= 0.035
        elif {"d", "right"} & pressed:
            steer_cache += 0.035
        elif steer_cache > 0:
            steer_cache = max(0.0, steer_cache - 0.06)
        else:
            steer_cache = min(0.0, steer_cache + 0.06)
        steer_cache = max(-0.8, min(0.8, steer_cache))

        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer_cache,
            hand_brake="space" in pressed,
        )
        return control, steer_cache, quit_requested

    def close(self) -> None:
        self.listener.stop()
        self.listener.join(timeout=2.0)


FAULT_OPTIONS = (
    ("none", "All sensors healthy"),
    ("FL", "Front-left sensor faulty"),
    ("FR", "Front-right sensor faulty"),
    ("RL", "Rear-left sensor faulty"),
    ("RR", "Rear-right sensor faulty"),
)

COLORS = {
    "background": (7, 16, 25),
    "panel": (13, 30, 44),
    "panel_alt": (17, 39, 54),
    "text": (235, 244, 250),
    "muted": (137, 160, 179),
    "green": (63, 220, 158),
    "amber": (245, 174, 66),
    "red": (239, 82, 82),
    "blue": (82, 166, 245),
}


def dashboard_fonts() -> dict[str, Any]:
    return {
        "title": pygame.font.SysFont("segoeui", 27, bold=True),
        "heading": pygame.font.SysFont("segoeui", 20, bold=True),
        "percent": pygame.font.SysFont("segoeui", 34, bold=True),
        "body": pygame.font.SysFont("consolas", 16),
        "small": pygame.font.SysFont("consolas", 13),
        "tiny": pygame.font.SysFont("consolas", 11),
    }


def choose_fault(display: Any, fonts: dict[str, Any], default: str) -> str:
    selected = next(
        (index for index, option in enumerate(FAULT_OPTIONS) if option[0] == default),
        0,
    )
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt
            if event.key in {pygame.K_UP, pygame.K_w}:
                selected = (selected - 1) % len(FAULT_OPTIONS)
            elif event.key in {pygame.K_DOWN, pygame.K_s}:
                selected = (selected + 1) % len(FAULT_OPTIONS)
            elif pygame.K_1 <= event.key <= pygame.K_5:
                selected = event.key - pygame.K_1
            elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
                return FAULT_OPTIONS[selected][0]

        display.fill(COLORS["background"])
        display.blit(fonts["title"].render("ABS LIVE DIAGNOSTIC", True, COLORS["text"]), (54, 45))
        display.blit(fonts["body"].render("Select the sensor condition before driving", True, COLORS["muted"]), (56, 88))
        panel = pygame.Rect(52, 130, min(680, display.get_width() - 104), 390)
        pygame.draw.rect(display, COLORS["panel"], panel, border_radius=16)
        for index, (_, label) in enumerate(FAULT_OPTIONS):
            row = pygame.Rect(panel.x + 22, panel.y + 22 + index * 67, panel.width - 44, 52)
            active = index == selected
            pygame.draw.rect(display, (27, 73, 91) if active else COLORS["panel_alt"], row, border_radius=10)
            color = COLORS["green"] if active else COLORS["text"]
            display.blit(fonts["body"].render(f"{index + 1}   {label}", True, color), (row.x + 16, row.y + 14))
        display.blit(fonts["small"].render("UP/DOWN or 1-5 select  ·  ENTER start  ·  ESC cancel", True, COLORS["muted"]), (56, panel.bottom + 24))
        pygame.display.flip()
        clock.tick(30)


def decision_color(state: str) -> tuple[int, int, int]:
    if state in {"CONFIRMED_FAULTY", "SUSPECTED_FAULTY"}:
        return COLORS["red"]
    if state in {"WARNING", "CROSS_EFFECT", "AMBIGUOUS", "INSUFFICIENT_SIGNAL"}:
        return COLORS["amber"]
    if state == "HEALTHY":
        return COLORS["green"]
    return COLORS["blue"]


def wheel_card_rects(width: int) -> dict[str, Any]:
    margin, gap = 24, 12
    card_width = (width - margin * 2 - gap * 3) // 4
    return {
        wheel: pygame.Rect(margin + index * (card_width + gap), 112, card_width, 205)
        for index, wheel in enumerate(WHEELS)
    }


def combined_wheel_card_rects(display_width: int, camera_width: int) -> dict[str, Any]:
    panel_width = display_width - camera_width
    margin, gap = 18, 10
    card_width = (panel_width - margin * 2 - gap) // 2
    return {
        wheel: pygame.Rect(
            camera_width + margin + (index % 2) * (card_width + gap),
            98 + (index // 2) * 148,
            card_width,
            136,
        )
        for index, wheel in enumerate(WHEELS)
    }


def render_combined_dashboard(
    display: Any,
    camera_surface: Any | None,
    camera_width: int,
    fonts: dict[str, Any],
    state: dict[str, Any],
    state_error: str | None,
    measurements: dict[str, dict[str, float | bool | str]],
    timestamp_s: float,
    vehicle_speed_mps: float,
    selected_wheel: str,
    configured_fault: str,
) -> None:
    """Render the CARLA camera and diagnostics inside one Pygame window."""
    height = display.get_height()
    display.fill(COLORS["background"])
    if camera_surface is None:
        waiting = fonts["body"].render("Waiting for CARLA camera...", True, COLORS["muted"])
        display.blit(waiting, waiting.get_rect(center=(camera_width // 2, height // 2)))
    else:
        display.blit(camera_surface, (0, 0))

    camera_header = pygame.Surface((camera_width, 82), pygame.SRCALPHA)
    camera_header.fill((4, 13, 21, 210))
    display.blit(camera_header, (0, 0))
    display.blit(fonts["heading"].render("CARLA LIVE DRIVE", True, COLORS["text"]), (20, 14))
    drive_text = f"{vehicle_speed_mps * 3.6:5.1f} km/h  |  {timestamp_s:6.1f} s  |  WASD/arrows drive  SPACE brake  ESC stop"
    display.blit(fonts["small"].render(drive_text, True, COLORS["muted"]), (20, 49))

    panel_x = camera_width
    panel_width = display.get_width() - panel_x
    pygame.draw.rect(display, COLORS["background"], (panel_x, 0, panel_width, height))
    display.blit(fonts["title"].render("ABS DIAGNOSTICS", True, COLORS["text"]), (panel_x + 18, 15))
    fault_label = "ALL SENSORS HEALTHY" if configured_fault == "none" else f"INJECTED FAULT - {configured_fault}"
    display.blit(fonts["small"].render(fault_label, True, COLORS["green"]), (panel_x + 20, 59))

    diagnostic = state.get("diagnostic") or {}
    summaries = diagnostic.get("wheels") or {}
    frames = diagnostic.get("frames") or []
    latest = frames[-1].get("wheels", {}) if frames else {}
    for wheel, rect in combined_wheel_card_rects(display.get_width(), camera_width).items():
        summary = summaries.get(wheel, {})
        decision = summary.get("decision", {})
        state_name = str(decision.get("state", "COLLECTING"))
        color = decision_color(state_name)
        health = float(summary.get("health", {}).get("health_percent", 100.0) or 0.0)
        probability = float(decision.get("independent_probability", 0.0) or 0.0)
        measured = latest.get(wheel, {}).get("measured_mps")
        if measured is None:
            measured = measurements.get(wheel, {}).get("speed_mps", 0.0)
        pygame.draw.rect(display, COLORS["panel_alt"] if wheel == selected_wheel else COLORS["panel"], rect, border_radius=10)
        pygame.draw.rect(display, color, rect, width=3 if wheel == selected_wheel else 1, border_radius=10)
        display.blit(fonts["heading"].render(wheel, True, COLORS["text"]), (rect.x + 12, rect.y + 9))
        health_text = fonts["percent"].render(f"{health:.0f}%", True, color)
        display.blit(health_text, (rect.right - health_text.get_width() - 10, rect.y + 7))
        display.blit(fonts["tiny"].render(state_name.replace("_", " ")[:18], True, color), (rect.x + 12, rect.y + 48))
        display.blit(fonts["tiny"].render(f"Speed  {float(measured or 0.0) * 3.6:5.1f} km/h", True, COLORS["muted"]), (rect.x + 12, rect.y + 76))
        display.blit(fonts["tiny"].render(f"Model  {probability * 100:5.1f}%", True, COLORS["muted"]), (rect.x + 12, rect.y + 101))

    detail = pygame.Rect(panel_x + 18, 407, panel_width - 36, height - 425)
    pygame.draw.rect(display, COLORS["panel"], detail, border_radius=12)
    selected = summaries.get(selected_wheel, {})
    decision = selected.get("decision", {})
    old_spc = selected.get("old_spc", {})
    state_name = str(decision.get("state", "COLLECTING"))
    color = decision_color(state_name)
    display.blit(fonts["heading"].render(f"{selected_wheel} SENSOR DETAIL", True, COLORS["text"]), (detail.x + 14, detail.y + 13))
    display.blit(fonts["tiny"].render("Click a wheel card or press 1-4", True, COLORS["muted"]), (detail.x + 15, detail.y + 43))
    probability = float(decision.get("independent_probability", 0.0) or 0.0)
    lines = (
        f"Decision  {state_name.replace('_', ' ')}",
        f"SPC       {str(old_spc.get('class', 'COLLECTING')).replace('_', ' ')}",
        f"Model     {probability * 100:.2f}%",
        f"Alarms    {old_spc.get('alarm_sample_count', 0)}",
    )
    for index, value in enumerate(lines):
        display.blit(fonts["small"].render(value, True, color if index == 0 else COLORS["text"]), (detail.x + 15, detail.y + 70 + index * 24))

    chart = pygame.Rect(detail.x + 14, detail.y + 174, detail.width - 28, max(62, detail.height - 190))
    pygame.draw.rect(display, (8, 21, 31), chart, border_radius=8)
    pygame.draw.line(display, (42, 66, 82), (chart.x, chart.centery), (chart.right, chart.centery), 1)
    history = state.get("_residual_history", {}).get(selected_wheel, [])
    if len(history) > 1:
        scale = max(0.25, max(abs(value) for value in history))
        points = [
            (
                int(chart.x + index * chart.width / max(1, len(history) - 1)),
                int(chart.centery - (value / scale) * (chart.height * 0.42)),
            )
            for index, value in enumerate(history)
        ]
        pygame.draw.lines(display, color, False, points, 2)
    warning = state_error or state.get("telemetry_warning") or state.get("diagnostic_error")
    if warning:
        display.blit(fonts["tiny"].render(str(warning)[:72], True, COLORS["red"]), (detail.x + 14, detail.bottom - 18))


def render_dashboard(
    display: Any,
    fonts: dict[str, Any],
    state: dict[str, Any],
    state_error: str | None,
    measurements: dict[str, dict[str, float | bool | str]],
    vehicle_speed_mps: float,
    selected_wheel: str,
    configured_fault: str,
) -> None:
    display.fill(COLORS["background"])
    diagnostic = state.get("diagnostic") or {}
    summaries = diagnostic.get("wheels") or {}
    frames = diagnostic.get("frames") or []
    latest = frames[-1].get("wheels", {}) if frames else {}

    display.blit(fonts["title"].render("ABS DIAGNOSTIC PLATFORM", True, COLORS["text"]), (24, 20))
    display.blit(fonts["small"].render("CARLA native 3D view · WASD/arrows drive · SPACE brake · ESC stop", True, COLORS["muted"]), (25, 61))
    speed_text = fonts["heading"].render(f"{vehicle_speed_mps * 3.6:5.1f} km/h", True, COLORS["text"])
    display.blit(speed_text, (display.get_width() - speed_text.get_width() - 26, 22))
    fault_label = "ALL SENSORS HEALTHY" if configured_fault == "none" else f"INJECTED FAULT · {configured_fault}"
    display.blit(fonts["small"].render(fault_label, True, COLORS["green"]), (25, 84))

    for wheel, rect in wheel_card_rects(display.get_width()).items():
        summary = summaries.get(wheel, {})
        decision = summary.get("decision", {})
        state_name = str(decision.get("state", "COLLECTING"))
        color = decision_color(state_name)
        health = float(summary.get("health", {}).get("health_percent", 100.0) or 0.0)
        probability = float(decision.get("independent_probability", 0.0) or 0.0)
        measured = latest.get(wheel, {}).get("measured_mps")
        if measured is None:
            measured = measurements.get(wheel, {}).get("speed_mps", 0.0)
        valid = bool(measurements.get(wheel, {}).get("valid", False))

        pygame.draw.rect(display, COLORS["panel_alt"] if wheel == selected_wheel else COLORS["panel"], rect, border_radius=12)
        pygame.draw.rect(display, color, rect, width=3 if wheel == selected_wheel else 1, border_radius=12)
        display.blit(fonts["heading"].render(wheel, True, COLORS["text"]), (rect.x + 15, rect.y + 12))
        display.blit(fonts["small"].render(state_name.replace("_", " ")[:20], True, color), (rect.x + 15, rect.y + 44))
        display.blit(fonts["percent"].render(f"{health:.0f}%", True, color), (rect.x + 15, rect.y + 70))
        display.blit(fonts["small"].render(f"Speed  {float(measured or 0.0) * 3.6:5.1f} km/h", True, COLORS["muted"]), (rect.x + 15, rect.y + 126))
        display.blit(fonts["small"].render(f"Model  {probability * 100:5.1f}%", True, COLORS["muted"]), (rect.x + 15, rect.y + 151))
        display.blit(fonts["small"].render("VALID" if valid else "PULSE LOSS", True, COLORS["green"] if valid else COLORS["red"]), (rect.x + 15, rect.y + 176))

    detail = pygame.Rect(24, 337, display.get_width() - 48, display.get_height() - 361)
    pygame.draw.rect(display, COLORS["panel"], detail, border_radius=14)
    selected = summaries.get(selected_wheel, {})
    decision = selected.get("decision", {})
    old_spc = selected.get("old_spc", {})
    state_name = str(decision.get("state", "COLLECTING"))
    color = decision_color(state_name)
    display.blit(fonts["heading"].render(f"{selected_wheel} SENSOR DETAIL", True, COLORS["text"]), (detail.x + 18, detail.y + 16))
    display.blit(fonts["small"].render("Click a wheel card to inspect it", True, COLORS["muted"]), (detail.x + 18, detail.y + 48))

    values = (
        ("Final decision", state_name.replace("_", " "), color),
        ("SPC status", str(old_spc.get("class", "COLLECTING")).replace("_", " "), COLORS["text"]),
        ("Independent probability", f"{float(decision.get('independent_probability', 0.0) or 0.0) * 100:.2f}%", COLORS["text"]),
        ("Alarm samples", str(old_spc.get("alarm_sample_count", 0)), COLORS["text"]),
    )
    for index, (label, value, value_color) in enumerate(values):
        x = detail.x + 18 + index * ((detail.width - 36) // 4)
        display.blit(fonts["tiny"].render(label.upper(), True, COLORS["muted"]), (x, detail.y + 82))
        display.blit(fonts["body"].render(value, True, value_color), (x, detail.y + 104))

    chart = pygame.Rect(detail.x + 18, detail.y + 146, detail.width - 36, max(70, detail.height - 174))
    pygame.draw.rect(display, (8, 21, 31), chart, border_radius=8)
    pygame.draw.line(display, (42, 66, 82), (chart.x, chart.centery), (chart.right, chart.centery), 1)
    history = state.get("_residual_history", {}).get(selected_wheel, [])
    if len(history) > 1:
        scale = max(0.25, max(abs(value) for value in history))
        points = []
        for index, value in enumerate(history):
            x = chart.x + index * chart.width / max(1, len(history) - 1)
            y = chart.centery - (value / scale) * (chart.height * 0.42)
            points.append((int(x), int(y)))
        pygame.draw.lines(display, color, False, points, 2)
    display.blit(fonts["tiny"].render("LIVE RESIDUAL HISTORY", True, COLORS["muted"]), (chart.x + 10, chart.y + 8))

    warning = state_error or state.get("telemetry_warning") or state.get("diagnostic_error")
    if warning:
        display.blit(fonts["tiny"].render(str(warning)[:120], True, COLORS["red"]), (detail.x + 18, detail.bottom - 20))


def build_frame(
    vehicle: Any,
    timestamp_s: float,
    control: Any,
    measurements: dict[str, dict[str, float | bool | str]],
) -> dict[str, Any]:
    velocity = vehicle.get_velocity()
    frame: dict[str, Any] = {
        "time_s": timestamp_s,
        "vehicle_speed_mps": (
            velocity.x**2 + velocity.y**2 + velocity.z**2
        ) ** 0.5,
        "throttle": float(control.throttle),
        "brake": float(control.brake),
        "steer": float(control.steer),
    }
    for wheel in WHEELS:
        frame[f"wheel_speed_ecu_{wheel}_mps"] = measurements[wheel]["speed_mps"]
        frame[f"ecu_valid_{wheel}"] = measurements[wheel]["valid"]
    return frame


def run(arguments: argparse.Namespace) -> None:
    driver_config = api_request(arguments.api_url, "/api/live/config")
    config = driver_config["config"]

    pygame.init()
    pygame.font.init()
    combined_view = arguments.view_mode == "combined"
    display_width = max(1360, arguments.width) if combined_view else arguments.width
    display_height = max(760, arguments.height) if combined_view else arguments.height
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption(
        "CARLA drive · ABS diagnostic dashboard"
        if combined_view
        else "ABS diagnostic dashboard · CARLA live"
    )
    fonts = dashboard_fonts()
    try:
        selected_fault = choose_fault(display, fonts, str(config["fault_wheel"]))
    except KeyboardInterrupt:
        pygame.quit()
        api_request(arguments.api_url, "/api/live/driver-stopped", {})
        return
    config = api_request(
        arguments.api_url,
        "/api/live/driver-config",
        {"fault_wheel": selected_fault},
    )["config"]

    geometry = VehicleGeometry()
    sensor_config = SensorConfig()
    fault = FaultConfig(
        wheel=None if config["fault_wheel"] == "none" else config["fault_wheel"],
        start_s=float(config["fault_start_s"]),
        end_s=float(config["fault_start_s"]) + float(config["fault_duration_s"]),
        severity=float(config["fault_severity"]),
    )
    estimator = FourWheelSpeedEstimator(geometry)
    pulse_sensor = ABSPulseSensor(geometry, sensor_config, fault, seed=42)
    ecu = ABSECU(geometry, sensor_config, ECUConfig(sample_time_s=0.01))

    client = carla.Client(arguments.host, arguments.port)
    client.set_timeout(60.0)
    world = client.get_world()
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.01
    settings.substepping = True
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 1
    world.apply_settings(settings)

    vehicle = None
    camera = None
    publisher = TelemetryPublisher(arguments.api_url)
    keyboard_control = GlobalKeyboardControl()
    state_reader: DiagnosticStateReader | None = None
    camera_buffer = CameraBuffer()

    try:
        blueprints = list(world.get_blueprint_library().filter(config["vehicle_filter"]))
        if not blueprints:
            raise RuntimeError(f"No vehicle matches {config['vehicle_filter']!r}.")
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(blueprints[0], spawn_points[0])
        if vehicle is None:
            raise RuntimeError("Could not spawn the CARLA vehicle.")

        camera_width = int(display_width * 0.62)
        if combined_view:
            camera_blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_blueprint.set_attribute("image_size_x", str(camera_width))
            camera_blueprint.set_attribute("image_size_y", str(display_height))
            camera_blueprint.set_attribute("fov", "95")
            camera_blueprint.set_attribute("sensor_tick", "0.033333")
            camera = world.spawn_actor(
                camera_blueprint,
                carla.Transform(
                    carla.Location(x=-6.5, z=3.2),
                    carla.Rotation(pitch=-12.0),
                ),
                attach_to=vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )
            camera.listen(camera_buffer.update)

        api_request(
            arguments.api_url,
            "/api/live/driver-started",
            {"map": world.get_map().name, "vehicle": vehicle.type_id},
        )
        state_reader = DiagnosticStateReader(arguments.api_url)

        running = True
        steer_cache = 0.0
        selected_wheel = "FL"
        sample_index = 0
        batch: list[dict[str, Any]] = []
        measurements: dict[str, dict[str, float | bool | str]] = {
            wheel: {"speed_mps": 0.0, "valid": False} for wheel in WHEELS
        }
        clock = pygame.time.Clock()
        spectator = None if combined_view else world.get_spectator()
        cached_camera_surface = None
        cached_camera_sequence = -1
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    card_rects = (
                        combined_wheel_card_rects(display.get_width(), camera_width)
                        if combined_view
                        else wheel_card_rects(display.get_width())
                    )
                    for wheel, rect in card_rects.items():
                        if rect.collidepoint(event.pos):
                            selected_wheel = wheel
                elif event.type == pygame.KEYDOWN and pygame.K_1 <= event.key <= pygame.K_4:
                    selected_wheel = WHEELS[event.key - pygame.K_1]
            control, steer_cache, escape = keyboard_control.control(steer_cache)
            if escape:
                running = False
            vehicle.apply_control(control)
            world.tick()
            sample_index += 1
            timestamp_s = sample_index * 0.01

            if spectator is not None:
                vehicle_transform = vehicle.get_transform()
                forward = vehicle_transform.get_forward_vector()
                spectator_location = carla.Location(
                    x=vehicle_transform.location.x - forward.x * 6.5,
                    y=vehicle_transform.location.y - forward.y * 6.5,
                    z=vehicle_transform.location.z + 3.2,
                )
                spectator.set_transform(
                    carla.Transform(
                        spectator_location,
                        carla.Rotation(
                            pitch=-14.0,
                            yaw=vehicle_transform.rotation.yaw,
                            roll=0.0,
                        ),
                    )
                )

            estimated, _ = estimator.estimate(vehicle)
            arrivals, _ = pulse_sensor.step(timestamp_s, 0.01, estimated)
            measurements = ecu.step(timestamp_s, arrivals)
            batch.append(build_frame(vehicle, timestamp_s, control, measurements))
            if len(batch) >= 10:
                publisher.publish(batch)
                batch = []

            if sample_index % 25 == 0:
                state = api_request(arguments.api_url, "/api/live/config", timeout=3.0)
                if state.get("stop_requested"):
                    running = False

            hud_state, hud_error = state_reader.snapshot()
            velocity = vehicle.get_velocity()
            vehicle_speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
            if combined_view:
                camera_frame = camera_buffer.latest()
                if camera_frame is not None and camera_frame[3] != cached_camera_sequence:
                    frame_width, frame_height, raw_data, cached_camera_sequence = camera_frame
                    cached_camera_surface = pygame.image.frombuffer(
                        raw_data,
                        (frame_width, frame_height),
                        "BGRA",
                    ).convert()
                render_combined_dashboard(
                    display,
                    cached_camera_surface,
                    camera_width,
                    fonts,
                    hud_state,
                    hud_error,
                    measurements,
                    timestamp_s,
                    vehicle_speed,
                    selected_wheel,
                    str(config["fault_wheel"]),
                )
            else:
                render_dashboard(
                    display,
                    fonts,
                    hud_state,
                    hud_error,
                    measurements,
                    vehicle_speed,
                    selected_wheel,
                    str(config["fault_wheel"]),
                )
            pygame.display.flip()
            clock.tick_busy_loop(100)

        if batch:
            publisher.publish(batch)
    finally:
        try:
            publisher.close()
        finally:
            if state_reader is not None:
                state_reader.close()
            keyboard_control.close()
            if camera is not None:
                camera.stop()
                camera.destroy()
            if vehicle is not None:
                vehicle.destroy()
            world.apply_settings(original_settings)
            pygame.quit()
            try:
                api_request(arguments.api_url, "/api/live/driver-stopped", {})
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument(
        "--view-mode",
        choices=("combined", "separate"),
        default="combined",
        help="Show the CARLA camera inside the dashboard or keep native CARLA separate.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    try:
        run(arguments)
    except Exception as exc:
        try:
            api_request(arguments.api_url, "/api/live/driver-failed", {"error": str(exc)})
        except (OSError, urllib.error.URLError):
            pass
        raise


if __name__ == "__main__":
    main()
