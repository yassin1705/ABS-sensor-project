"""Combined CARLA camera, vehicle controls, and live ABS diagnostic HUD."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
import urllib.request
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
    """Poll the compact diagnostic feed without blocking CARLA world ticks."""

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.state: dict[str, Any] = {}
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def snapshot(self) -> tuple[dict[str, Any], str | None]:
        with self._lock:
            return dict(self.state), self.error

    def close(self) -> None:
        self._stop.set()
        self.thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                state = api_request(
                    self.api_url,
                    "/api/live/hud",
                    timeout=5.0,
                )
                with self._lock:
                    self.state = state
                    self.error = None
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
            self._stop.wait(0.5)


class CameraBuffer:
    """Keep only the newest CARLA RGB frame for the Pygame renderer."""

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

STATE_COLORS = {
    "HEALTHY": (43, 190, 125),
    "COLLECTING": (85, 135, 175),
    "WARNING": (245, 174, 66),
    "CROSS_EFFECT": (245, 174, 66),
    "AMBIGUOUS": (245, 174, 66),
    "INSUFFICIENT_SIGNAL": (245, 174, 66),
    "SUSPECTED_FAULTY": (239, 96, 96),
    "CONFIRMED_FAULTY": (239, 68, 68),
}


def choose_fault(display: Any, fonts: dict[str, Any], default: str) -> str:
    """Show the pre-drive sensor-state selector in the combined window."""
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

        display.fill((7, 16, 25))
        display.blit(
            fonts["title"].render("ABS SENSOR SETUP", True, (235, 244, 250)),
            (54, 48),
        )
        display.blit(
            fonts["body"].render(
                "Choose the sensor state before the simulation begins",
                True,
                (145, 166, 185),
            ),
            (56, 92),
        )
        panel = pygame.Rect(52, 138, min(650, display.get_width() - 104), 390)
        pygame.draw.rect(display, (13, 30, 44), panel, border_radius=16)
        for index, (_, label) in enumerate(FAULT_OPTIONS):
            row = pygame.Rect(panel.x + 22, panel.y + 22 + index * 67, panel.width - 44, 52)
            active = index == selected
            pygame.draw.rect(
                display,
                (28, 76, 94) if active else (18, 40, 55),
                row,
                border_radius=10,
            )
            color = (102, 225, 190) if active else (203, 216, 225)
            display.blit(
                fonts["body"].render(f"{index + 1}   {label}", True, color),
                (row.x + 16, row.y + 14),
            )
        display.blit(
            fonts["small"].render(
                "UP/DOWN or 1-5 to select  ·  ENTER to start  ·  ESC to cancel",
                True,
                (126, 151, 171),
            ),
            (56, panel.bottom + 25),
        )
        pygame.display.flip()
        clock.tick(30)


def _text(surface: Any, font: Any, value: str, color: tuple[int, int, int], x: int, y: int) -> None:
    surface.blit(font.render(value, True, color), (x, y))


def render_hud(
    display: Any,
    camera_surface: Any | None,
    camera_width: int,
    fonts: dict[str, Any],
    state: dict[str, Any],
    state_error: str | None,
    measurements: dict[str, dict[str, float | bool | str]],
    timestamp_s: float,
    vehicle_speed_mps: float,
    configured_fault: str,
) -> None:
    """Render the CARLA view and compact four-wheel diagnostic board."""
    if camera_surface is None:
        display.fill((8, 18, 27))
        message = fonts["body"].render("Waiting for CARLA RGB camera…", True, (147, 169, 187))
        display.blit(message, message.get_rect(center=(camera_width // 2, display.get_height() // 2)))
    else:
        display.blit(camera_surface, (0, 0))

    # Camera overlay keeps the most important driving information readable.
    overlay = pygame.Surface((camera_width, 76), pygame.SRCALPHA)
    overlay.fill((4, 13, 21, 205))
    display.blit(overlay, (0, 0))
    _text(display, fonts["heading"], "CARLA · ABS LIVE", (237, 246, 250), 20, 14)
    _text(
        display,
        fonts["small"],
        f"{vehicle_speed_mps * 3.6:5.1f} km/h   ·   {timestamp_s:6.1f} s   ·   WASD/arrows drive   SPACE brake   ESC stop",
        (174, 195, 210),
        20,
        46,
    )

    panel_x = camera_width
    panel_width = display.get_width() - camera_width
    pygame.draw.rect(display, (7, 17, 26), (panel_x, 0, panel_width, display.get_height()))
    _text(display, fonts["title"], "WHEEL DIAGNOSTICS", (236, 245, 250), panel_x + 22, 20)
    fault_label = "ALL HEALTHY" if configured_fault == "none" else f"INJECTED FAULT · {configured_fault}"
    _text(display, fonts["small"], fault_label, (100, 224, 187), panel_x + 23, 58)

    diagnostic = state.get("diagnostic") or {}
    wheel_summaries = diagnostic.get("wheels") or {}
    latest_frames = diagnostic.get("frames") or []
    latest_wheels = latest_frames[-1].get("wheels", {}) if latest_frames else {}
    sample_count = int(state.get("sample_count", 0))
    _text(display, fonts["small"], f"{sample_count} samples", (131, 154, 173), panel_x + 23, 82)

    card_top = 112
    gap = 12
    card_width = (panel_width - 56) // 2
    card_height = 205
    for index, wheel in enumerate(WHEELS):
        column, row = index % 2, index // 2
        rect = pygame.Rect(
            panel_x + 20 + column * (card_width + gap),
            card_top + row * (card_height + gap),
            card_width,
            card_height,
        )
        summary = wheel_summaries.get(wheel, {})
        decision = summary.get("decision", {})
        decision_state = str(decision.get("state", "COLLECTING"))
        color = STATE_COLORS.get(decision_state, (85, 135, 175))
        pygame.draw.rect(display, (14, 31, 44), rect, border_radius=12)
        pygame.draw.rect(display, color, rect, width=2, border_radius=12)
        _text(display, fonts["heading"], wheel, (238, 246, 250), rect.x + 14, rect.y + 12)
        compact_state = decision_state.replace("_", " ")
        if len(compact_state) > 18:
            compact_state = compact_state.replace("CONFIRMED ", "CONF. ").replace("INSUFFICIENT ", "INSUFF. ")
        _text(display, fonts["tiny"], compact_state, color, rect.x + 14, rect.y + 43)

        current = latest_wheels.get(wheel, {})
        measured = current.get("measured_mps")
        if measured is None:
            measured = measurements.get(wheel, {}).get("speed_mps", 0.0)
        residual = current.get("residual_mps")
        probability = float(decision.get("independent_probability", 0.0) or 0.0)
        health = float(summary.get("health", {}).get("health_percent", 100.0) or 0.0)
        valid = bool(measurements.get(wheel, {}).get("valid", False))
        _text(display, fonts["tiny"], "SPEED", (120, 145, 164), rect.x + 14, rect.y + 77)
        _text(display, fonts["body"], f"{float(measured or 0.0) * 3.6:5.1f} km/h", (223, 235, 242), rect.x + 14, rect.y + 95)
        residual_text = "—" if residual is None else f"{float(residual):+.3f} m/s"
        _text(display, fonts["tiny"], f"Residual  {residual_text}", (153, 174, 190), rect.x + 14, rect.y + 128)
        _text(display, fonts["tiny"], f"Model     {probability * 100:5.1f}%", (153, 174, 190), rect.x + 14, rect.y + 150)
        _text(display, fonts["tiny"], f"Health    {health:5.1f}%   {'VALID' if valid else 'LOSS'}", (153, 174, 190), rect.x + 14, rect.y + 172)

    footer_y = card_top + 2 * (card_height + gap) + 4
    primary = (diagnostic.get("isolation") or {}).get("primary_wheel")
    message = f"Primary isolated wheel: {primary}" if primary else "Primary isolated wheel: none"
    _text(display, fonts["small"], message, (174, 195, 210), panel_x + 22, footer_y)
    warning = state_error or state.get("telemetry_warning")
    if warning:
        _text(display, fonts["tiny"], str(warning)[:58], (239, 96, 96), panel_x + 22, footer_y + 27)


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
    display_width = max(1100, arguments.width)
    display_height = max(650, arguments.height)
    display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA drive · ABS diagnostic board")
    fonts = {
        "title": pygame.font.SysFont("segoeui", 25, bold=True),
        "heading": pygame.font.SysFont("segoeui", 20, bold=True),
        "body": pygame.font.SysFont("consolas", 17),
        "small": pygame.font.SysFont("consolas", 14),
        "tiny": pygame.font.SysFont("consolas", 12),
    }
    try:
        selected_fault = choose_fault(display, fonts, str(config["fault_wheel"]))
    except KeyboardInterrupt:
        pygame.quit()
        try:
            api_request(arguments.api_url, "/api/live/driver-stopped", {})
        except Exception:
            pass
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
    client.set_timeout(20.0)
    world = client.get_world()
    requested_map = str(config["map"])
    current_map = world.get_map().name.rsplit("/", 1)[-1]
    if current_map != requested_map:
        world = client.load_world(requested_map)
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

        camera_width = display_width - 430
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
        sample_index = 0
        batch: list[dict[str, Any]] = []
        clock = pygame.time.Clock()
        cached_camera_surface = None
        cached_camera_sequence = -1
        latest_measurements: dict[str, dict[str, float | bool | str]] = {
            wheel: {"speed_mps": 0.0, "valid": False} for wheel in WHEELS
        }
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            control, steer_cache, escape = keyboard_control.control(steer_cache)
            if escape:
                running = False
            vehicle.apply_control(control)
            world.tick()
            sample_index += 1
            timestamp_s = sample_index * 0.01

            # Keep the actual CARLA server spectator behind the ego vehicle.
            # No RGB images are captured or transferred to Python/browser.
            vehicle_transform = vehicle.get_transform()
            forward = vehicle_transform.get_forward_vector()
            spectator_location = carla.Location(
                x=vehicle_transform.location.x - forward.x * 6.5,
                y=vehicle_transform.location.y - forward.y * 6.5,
                z=vehicle_transform.location.z + 3.2,
            )
            world.get_spectator().set_transform(
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
            latest_measurements = measurements
            batch.append(build_frame(vehicle, timestamp_s, control, measurements))
            if len(batch) >= 10:
                publisher.publish(batch)
                batch = []

            if sample_index % 25 == 0:
                state = api_request(arguments.api_url, "/api/live/config", timeout=3.0)
                if state.get("stop_requested"):
                    running = False

            camera_frame = camera_buffer.latest()
            if camera_frame is not None and camera_frame[3] != cached_camera_sequence:
                frame_width, frame_height, raw_data, cached_camera_sequence = camera_frame
                cached_camera_surface = pygame.image.frombuffer(
                    raw_data,
                    (frame_width, frame_height),
                    "BGRA",
                ).convert()
            hud_state, hud_error = state_reader.snapshot()
            velocity = vehicle.get_velocity()
            vehicle_speed = (velocity.x**2 + velocity.y**2 + velocity.z**2) ** 0.5
            render_hud(
                display,
                cached_camera_surface,
                camera_width,
                fonts,
                hud_state,
                hud_error,
                latest_measurements,
                timestamp_s,
                vehicle_speed,
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
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
