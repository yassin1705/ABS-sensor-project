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


def draw_cockpit_health(world: Any, vehicle: Any, state: dict[str, Any]) -> None:
    """Draw four wheel-health percentages in front of the native cockpit view."""
    diagnostic = state.get("diagnostic") or {}
    wheel_summaries = diagnostic.get("wheels") or {}
    transform = vehicle.get_transform()
    lateral_positions = {"FL": -0.72, "FR": -0.24, "RL": 0.24, "RR": 0.72}

    for wheel in WHEELS:
        summary = wheel_summaries.get(wheel, {})
        health = float(summary.get("health", {}).get("health_percent", 100.0) or 0.0)
        decision = str(summary.get("decision", {}).get("state", "COLLECTING"))
        if decision in {"CONFIRMED_FAULTY", "SUSPECTED_FAULTY"}:
            color = carla.Color(255, 65, 65)
        elif decision in {"WARNING", "CROSS_EFFECT", "AMBIGUOUS"}:
            color = carla.Color(255, 190, 55)
        else:
            color = carla.Color(70, 255, 165)

        location = transform.transform(
            carla.Location(x=2.1, y=lateral_positions[wheel], z=1.08)
        )
        world.debug.draw_string(
            location,
            f"{wheel} {health:3.0f}%",
            draw_shadow=True,
            color=color,
            life_time=0.15,
            persistent_lines=False,
        )


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
    display_width = max(720, arguments.width)
    display_height = max(600, arguments.height)
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
    # Pygame is used only for pre-drive configuration. The actual drive uses
    # CARLA's native window, avoiding the unstable RGB PixelReader path.
    pygame.quit()

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
    publisher = TelemetryPublisher(arguments.api_url)
    keyboard_control = GlobalKeyboardControl()
    state_reader: DiagnosticStateReader | None = None

    try:
        blueprints = list(world.get_blueprint_library().filter(config["vehicle_filter"]))
        if not blueprints:
            raise RuntimeError(f"No vehicle matches {config['vehicle_filter']!r}.")
        spawn_points = world.get_map().get_spawn_points()
        vehicle = world.try_spawn_actor(blueprints[0], spawn_points[0])
        if vehicle is None:
            raise RuntimeError("Could not spawn the CARLA vehicle.")

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
        spectator = world.get_spectator()
        while running:
            loop_started = time.perf_counter()
            control, steer_cache, escape = keyboard_control.control(steer_cache)
            if escape:
                running = False
            vehicle.apply_control(control)
            world.tick()
            sample_index += 1
            timestamp_s = sample_index * 0.01

            # Native spectator cockpit: no RGB sensor and therefore no
            # PixelReader render/copy path that can crash packaged CARLA.
            vehicle_transform = vehicle.get_transform()
            cockpit_location = vehicle_transform.transform(
                carla.Location(x=0.35, y=-0.32, z=1.25)
            )
            spectator.set_transform(
                carla.Transform(
                    cockpit_location,
                    carla.Rotation(
                        pitch=-3.0,
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

            if sample_index % 10 == 0:
                hud_state, _ = state_reader.snapshot()
                draw_cockpit_health(world, vehicle, hud_state)

            remaining = 0.01 - (time.perf_counter() - loop_started)
            if remaining > 0:
                time.sleep(remaining)

        if batch:
            publisher.publish(batch)
    finally:
        try:
            publisher.close()
        finally:
            if state_reader is not None:
                state_reader.close()
            keyboard_control.close()
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
    parser.add_argument("--width", type=int, default=760)
    parser.add_argument("--height", type=int, default=600)
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
