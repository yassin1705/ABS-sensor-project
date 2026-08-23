"""Interactive CARLA driving window that publishes ABS telemetry only."""

from __future__ import annotations

import argparse
import json
import queue
import threading
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
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((arguments.width, arguments.height))
    pygame.display.set_caption("CARLA manual drive · ABS live diagnostics")
    font = pygame.font.SysFont("consolas", 18)
    keyboard_control = GlobalKeyboardControl()

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

        running = True
        steer_cache = 0.0
        sample_index = 0
        batch: list[dict[str, Any]] = []
        clock = pygame.time.Clock()
        spectator = world.get_spectator()
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

            # Keep CARLA's native spectator behind the vehicle. No RGB camera
            # sensor is created and the browser receives measurements only.
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

            display.fill((7, 17, 26))
            display.blit(font.render("CARLA VEHICLE CONTROL", True, (235, 245, 250)), (18, 12))
            display.blit(font.render("GLOBAL KEYS · WASD/arrows drive · SPACE brake · ESC stop", True, (160, 178, 196)), (18, 42))
            fault_text = "all sensors healthy" if fault.wheel is None else f"{fault.wheel} intermittent loss · severity {fault.severity:.2f}"
            display.blit(font.render(fault_text, True, (82, 227, 165)), (18, 72))
            display.blit(font.render("3D chase view stays in the CarlaUE4 window", True, (96, 165, 250)), (18, 102))
            pygame.display.flip()
            clock.tick_busy_loop(100)

        if batch:
            publisher.publish(batch)
    finally:
        try:
            publisher.close()
        finally:
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
    parser.add_argument("--width", type=int, default=620)
    parser.add_argument("--height", type=int, default=140)
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
