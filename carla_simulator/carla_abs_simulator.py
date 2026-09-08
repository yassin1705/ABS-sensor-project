"""CARLA-backed ABS simulation and dataset generator.

This module replaces the active MATLAB simulation chain with:

CARLA vehicle physics -> four-wheel ground-speed estimation -> ABS pulse
generation -> latency/jitter/fault injection -> ECU reconstruction -> CSV.

CARLA remains a separate server process. Run this file with the Python client
environment that contains the matching ``carla`` package.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import carla
except ImportError as exc:  # Give a useful message instead of a cryptic crash.
    raise SystemExit(
        "The CARLA Python client is missing. Activate carla_env and run "
        "'python -m pip install carla==0.9.16'."
    ) from exc


WHEELS = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class VehicleGeometry:
    wheelbase_m: float = 2.65
    track_width_m: float = 1.55
    wheel_radius_m: float = 0.31

    def wheel_positions(self) -> dict[str, tuple[float, float]]:
        half_wheelbase = self.wheelbase_m / 2.0
        half_track = self.track_width_m / 2.0
        # CARLA local coordinates are x-forward and y-right.
        return {
            "FL": (half_wheelbase, -half_track),
            "FR": (half_wheelbase, half_track),
            "RL": (-half_wheelbase, -half_track),
            "RR": (-half_wheelbase, half_track),
        }


@dataclass(frozen=True)
class SensorConfig:
    edges_per_revolution: int = 48
    latency_ms: float = 0.2
    absolute_jitter_us: float = 2.0
    relative_jitter_percent: float = 0.05


@dataclass(frozen=True)
class ECUConfig:
    sample_time_s: float = 0.01
    no_pulse_timeout_s: float = 0.3


@dataclass(frozen=True)
class FaultConfig:
    wheel: str | None = "FL"
    start_s: float = 1.5
    end_s: float = 3.0
    severity: float = 0.8
    base_drop_probability: float = 0.5

    def active(self, wheel: str, timestamp_s: float) -> bool:
        return (
            self.wheel == wheel
            and self.start_s <= timestamp_s <= self.end_s
        )

    @property
    def effective_drop_probability(self) -> float:
        return min(1.0, max(0.0, self.severity * self.base_drop_probability))


@dataclass(frozen=True)
class ScenarioConfig:
    simulation_id: int = 1
    duration_s: float = 5.0
    sample_rate_hz: int = 100
    initial_speed_kmh: float = 70.0
    brake_start_s: float = 0.5
    brake_end_s: float = 4.5
    brake_value: float = 0.65
    steering_value: float = 0.0
    warmup_s: float = 0.5
    random_seed: int = 42

    @property
    def dt_s(self) -> float:
        return 1.0 / self.sample_rate_hz

    @property
    def sample_count(self) -> int:
        return round(self.duration_s * self.sample_rate_hz)

    def control_at(self, timestamp_s: float) -> tuple[float, float, float]:
        braking = self.brake_start_s <= timestamp_s <= self.brake_end_s
        brake = self.brake_value if braking else 0.0
        return 0.0, brake, self.steering_value


class FourWheelSpeedEstimator:
    """Estimate wheel-contact longitudinal speeds from chassis motion."""

    def __init__(self, geometry: VehicleGeometry) -> None:
        self.geometry = geometry
        self.positions = geometry.wheel_positions()

    @staticmethod
    def _dot(vector: Any, axis: Any) -> float:
        return vector.x * axis.x + vector.y * axis.y + vector.z * axis.z

    @staticmethod
    def _steering_angles_rad(vehicle: Any) -> dict[str, float]:
        angles = {wheel: 0.0 for wheel in WHEELS}
        locations = getattr(carla, "VehicleWheelLocation", None)
        if locations is None or not hasattr(vehicle, "get_wheel_steer_angle"):
            return angles

        names = {
            "FL": "FL_Wheel",
            "FR": "FR_Wheel",
            "RL": "BL_Wheel",
            "RR": "BR_Wheel",
        }
        for wheel, enum_name in names.items():
            location = getattr(locations, enum_name, None)
            if location is None:
                continue
            try:
                angles[wheel] = math.radians(
                    float(vehicle.get_wheel_steer_angle(location))
                )
            except RuntimeError:
                # Some vehicle blueprints do not support this getter.
                pass
        return angles

    def estimate(self, vehicle: Any) -> tuple[dict[str, float], dict[str, float]]:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        angular_velocity = vehicle.get_angular_velocity()

        velocity_x = self._dot(velocity, transform.get_forward_vector())
        velocity_y = self._dot(velocity, transform.get_right_vector())
        yaw_rate_radps = math.radians(float(angular_velocity.z))
        steer = self._steering_angles_rad(vehicle)

        speeds: dict[str, float] = {}
        for wheel, (position_x, position_y) in self.positions.items():
            local_x = velocity_x - yaw_rate_radps * position_y
            local_y = velocity_y + yaw_rate_radps * position_x
            wheel_angle = steer[wheel]
            projected = (
                local_x * math.cos(wheel_angle)
                + local_y * math.sin(wheel_angle)
            )
            # ABS sensors report rotational-speed magnitude. Reverse driving is
            # outside the current five-second braking campaign.
            speeds[wheel] = max(0.0, projected)
        return speeds, steer


class ABSPulseSensor:
    """Convert estimated linear wheel speed into timestamped ABS edges."""

    def __init__(
        self,
        geometry: VehicleGeometry,
        sensor: SensorConfig,
        fault: FaultConfig,
        seed: int,
    ) -> None:
        self.sensor = sensor
        self.fault = fault
        self.rng = random.Random(seed)
        self.edge_distance_m = (
            2.0 * math.pi * geometry.wheel_radius_m
            / sensor.edges_per_revolution
        )
        self.distance_remainder_m = {wheel: 0.0 for wheel in WHEELS}

    def step(
        self,
        timestamp_s: float,
        dt_s: float,
        wheel_speeds_mps: dict[str, float],
    ) -> tuple[dict[str, list[float]], dict[str, int]]:
        arrivals = {wheel: [] for wheel in WHEELS}
        dropped = {wheel: 0 for wheel in WHEELS}
        interval_start = timestamp_s - dt_s

        for wheel in WHEELS:
            speed = max(0.0, wheel_speeds_mps[wheel])
            previous_remainder = self.distance_remainder_m[wheel]
            travelled = speed * dt_s
            total = previous_remainder + travelled
            edge_count = int(total / self.edge_distance_m)
            self.distance_remainder_m[wheel] = total % self.edge_distance_m

            if edge_count == 0 or speed <= 1e-9:
                continue

            first_distance = self.edge_distance_m - previous_remainder
            for edge_index in range(edge_count):
                edge_time = interval_start + (
                    first_distance + edge_index * self.edge_distance_m
                ) / speed

                if (
                    self.fault.active(wheel, edge_time)
                    and self.rng.random() < self.fault.effective_drop_probability
                ):
                    dropped[wheel] += 1
                    continue

                nominal_period = self.edge_distance_m / speed
                absolute_jitter = self.rng.gauss(
                    0.0, self.sensor.absolute_jitter_us * 1e-6
                )
                relative_sigma = self.sensor.relative_jitter_percent / 100.0
                relative_jitter = self.rng.gauss(
                    0.0, nominal_period * relative_sigma
                )
                arrival_time = (
                    edge_time
                    + self.sensor.latency_ms * 1e-3
                    + absolute_jitter
                    + relative_jitter
                )
                arrivals[wheel].append(max(interval_start, arrival_time))

        return arrivals, dropped


@dataclass
class _ECUWheelState:
    pending_edges: list[float] = field(default_factory=list)
    previous_edge_s: float | None = None
    last_edge_s: float | None = None
    total_captured: int = 0
    captured_at_last_sample: int = 0


class ABSECU:
    """Reconstruct wheel speed from the periods of received ABS edges."""

    def __init__(
        self,
        geometry: VehicleGeometry,
        sensor: SensorConfig,
        ecu: ECUConfig,
    ) -> None:
        self.geometry = geometry
        self.sensor = sensor
        self.ecu = ecu
        self.state = {wheel: _ECUWheelState() for wheel in WHEELS}

    def step(
        self,
        timestamp_s: float,
        new_edges: dict[str, list[float]],
    ) -> dict[str, dict[str, float | bool | str]]:
        output: dict[str, dict[str, float | bool | str]] = {}
        for wheel in WHEELS:
            state = self.state[wheel]
            state.pending_edges.extend(new_edges[wheel])
            state.pending_edges.sort()

            due_count = 0
            while (
                due_count < len(state.pending_edges)
                and state.pending_edges[due_count] <= timestamp_s + 1e-12
            ):
                edge_time = state.pending_edges[due_count]
                state.previous_edge_s = state.last_edge_s
                state.last_edge_s = edge_time
                state.total_captured += 1
                due_count += 1
            if due_count:
                del state.pending_edges[:due_count]

            speed_mps = math.nan
            valid = False
            status = "initializing"
            edge_period_s = math.nan

            if state.last_edge_s is not None:
                edge_age_s = timestamp_s - state.last_edge_s
                if edge_age_s > self.ecu.no_pulse_timeout_s:
                    speed_mps = 0.0
                    valid = True
                    status = "zero_timeout"
                elif state.previous_edge_s is not None:
                    edge_period_s = state.last_edge_s - state.previous_edge_s
                    if edge_period_s > 0.0:
                        angle_per_edge = (
                            2.0 * math.pi / self.sensor.edges_per_revolution
                        )
                        speed_mps = (
                            self.geometry.wheel_radius_m
                            * angle_per_edge
                            / edge_period_s
                        )
                        valid = True
                        status = (
                            "period"
                            if state.total_captured > state.captured_at_last_sample
                            else "held"
                        )

            state.captured_at_last_sample = state.total_captured
            output[wheel] = {
                "speed_mps": speed_mps,
                "valid": valid,
                "status": status,
                "edge_period_s": edge_period_s,
            }
        return output


class CarlaABSSimulator:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.geometry = VehicleGeometry(
            wheelbase_m=arguments.wheelbase_m,
            track_width_m=arguments.track_width_m,
            wheel_radius_m=arguments.wheel_radius_m,
        )
        self.sensor = SensorConfig()
        self.ecu_config = ECUConfig(sample_time_s=1.0 / arguments.sample_rate_hz)
        self.fault = FaultConfig(
            wheel=None if arguments.fault_wheel == "none" else arguments.fault_wheel,
            start_s=arguments.fault_start_s,
            end_s=arguments.fault_end_s,
            severity=arguments.fault_severity,
        )
        self.scenario = ScenarioConfig(
            simulation_id=arguments.simulation_id,
            duration_s=arguments.duration_s,
            sample_rate_hz=arguments.sample_rate_hz,
            initial_speed_kmh=arguments.initial_speed_kmh,
            brake_start_s=arguments.brake_start_s,
            brake_end_s=arguments.brake_end_s,
            brake_value=arguments.brake_value,
            steering_value=arguments.steering_value,
            warmup_s=arguments.warmup_s,
            random_seed=arguments.seed,
        )
        self.estimator = FourWheelSpeedEstimator(self.geometry)
        self.pulse_sensor = ABSPulseSensor(
            self.geometry, self.sensor, self.fault, self.scenario.random_seed
        )
        self.ecu = ABSECU(self.geometry, self.sensor, self.ecu_config)

    @staticmethod
    def _select_blueprint(world: Any, pattern: str, seed: int) -> Any:
        candidates = list(world.get_blueprint_library().filter(pattern))
        if not candidates:
            raise RuntimeError(f"No CARLA vehicle matches {pattern!r}.")
        return random.Random(seed).choice(candidates)

    @staticmethod
    def _select_spawn_point(world: Any, index: int) -> Any:
        points = world.get_map().get_spawn_points()
        if not points:
            raise RuntimeError("The selected CARLA map has no spawn points.")
        return points[index % len(points)]

    def _configure_world(self, world: Any) -> Any:
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.scenario.dt_s
        settings.no_rendering_mode = self.arguments.no_rendering
        settings.substepping = True
        settings.max_substep_delta_time = min(0.01, self.scenario.dt_s)
        settings.max_substeps = max(
            1,
            math.ceil(
                self.scenario.dt_s / settings.max_substep_delta_time
            ),
        )
        world.apply_settings(settings)
        return original

    def _apply_vehicle_physics(self, vehicle: Any) -> None:
        physics = vehicle.get_physics_control()
        physics.mass = self.arguments.vehicle_mass_kg
        for wheel in physics.wheels:
            wheel.tire_friction = self.arguments.tire_friction
        vehicle.apply_physics_control(physics)

    def _build_row(
        self,
        vehicle: Any,
        sample_index: int,
        timestamp_s: float,
        throttle: float,
        brake: float,
        steering: float,
        estimated: dict[str, float],
        steer_angles: dict[str, float],
        measurements: dict[str, dict[str, float | bool | str]],
        dropped: dict[str, int],
    ) -> dict[str, Any]:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        angular_velocity = vehicle.get_angular_velocity()
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()
        vehicle_speed = math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )

        row: dict[str, Any] = {
            "simulation_id": self.scenario.simulation_id,
            "simulation_source": "carla",
            "carla_version": getattr(carla, "__version__", "unknown"),
            "map_name": vehicle.get_world().get_map().name,
            "vehicle_blueprint": vehicle.type_id,
            "carla_frame": sample_index + 1,
            "simulation_seed": self.scenario.random_seed,
            "requested_phenomenon": "normal_braking",
            "observed_phenomenon": "not_computed",
            "time_s": timestamp_s,
            "throttle": throttle,
            "brake": brake,
            "steering_rad": 0.5 * (steer_angles["FL"] + steer_angles["FR"]),
            "vehicle_speed_mps": vehicle_speed,
            "yaw_rate_radps": math.radians(angular_velocity.z),
            "acceleration_x_mps2": (
                acceleration.x * forward.x
                + acceleration.y * forward.y
                + acceleration.z * forward.z
            ),
            "acceleration_y_mps2": (
                acceleration.x * right.x
                + acceleration.y * right.y
                + acceleration.z * right.z
            ),
        }

        for wheel in WHEELS:
            measurement = measurements[wheel]
            active = self.fault.active(wheel, timestamp_s)
            row[f"wheel_ground_speed_estimated_{wheel}_mps"] = estimated[wheel]
            row[f"wheel_steer_angle_{wheel}_rad"] = steer_angles[wheel]
            row[f"wheel_speed_ecu_{wheel}_mps"] = measurement["speed_mps"]
            row[f"ecu_valid_{wheel}"] = measurement["valid"]
            row[f"ecu_status_{wheel}"] = measurement["status"]
            row[f"dropped_edges_{wheel}"] = dropped[wheel]
            row[f"fault_active_{wheel}"] = active
            row[f"fault_type_{wheel}"] = "intermittent_loss" if active else "none"
            row[f"fault_severity_{wheel}"] = self.fault.severity if active else 0.0
        return row

    def _write_outputs(self, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
        output_directory = self.arguments.output_directory.resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        kind = "faulty" if self.fault.wheel else "healthy"
        stem = f"abs_carla_{kind}_dataset_{timestamp}"
        dataset_path = output_directory / f"{stem}.csv"
        manifest_path = output_directory / f"{stem.replace('_dataset_', '_manifest_')}.csv"

        with dataset_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        total_dropped = sum(
            int(row[f"dropped_edges_{wheel}"])
            for row in rows
            for wheel in WHEELS
        )
        manifest = {
            "simulation_id": self.scenario.simulation_id,
            "source": "carla",
            "requested_phenomenon": "normal_braking",
            "observed_phenomenon": "not_computed",
            "initial_speed_kmh": self.scenario.initial_speed_kmh,
            "maximum_abs_steering_wheel_deg": abs(
                self.scenario.steering_value * 450.0
            ),
            "fault_wheel": self.fault.wheel or "none",
            "fault_type": "intermittent_loss" if self.fault.wheel else "none",
            "fault_start_s": self.fault.start_s if self.fault.wheel else "",
            "fault_end_s": self.fault.end_s if self.fault.wheel else "",
            "fault_severity": self.fault.severity if self.fault.wheel else 0.0,
            "dropped_edge_count": total_dropped,
            "sample_count": len(rows),
            "sample_rate_hz": self.scenario.sample_rate_hz,
            "random_seed": self.scenario.random_seed,
        }
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest))
            writer.writeheader()
            writer.writerow(manifest)
        return dataset_path, manifest_path

    def run(self) -> tuple[Path, Path]:
        client = carla.Client(self.arguments.host, self.arguments.port)
        client.set_timeout(self.arguments.timeout_s)
        world = client.load_world(self.arguments.map)
        original_settings = self._configure_world(world)
        vehicle = None

        try:
            blueprint = self._select_blueprint(
                world, self.arguments.vehicle_filter, self.scenario.random_seed
            )
            spawn = self._select_spawn_point(world, self.arguments.spawn_index)
            vehicle = world.try_spawn_actor(blueprint, spawn)
            if vehicle is None:
                raise RuntimeError(
                    "CARLA could not spawn the vehicle. Try another --spawn-index."
                )
            self._apply_vehicle_physics(vehicle)

            warmup_ticks = round(self.scenario.warmup_s / self.scenario.dt_s)
            vehicle.apply_control(carla.VehicleControl())
            for _ in range(warmup_ticks):
                world.tick()

            initial_speed_mps = self.scenario.initial_speed_kmh / 3.6
            forward = vehicle.get_transform().get_forward_vector()
            vehicle.set_target_velocity(
                carla.Vector3D(
                    x=forward.x * initial_speed_mps,
                    y=forward.y * initial_speed_mps,
                    z=forward.z * initial_speed_mps,
                )
            )
            world.tick()

            rows: list[dict[str, Any]] = []
            for sample_index in range(self.scenario.sample_count):
                timestamp_s = (sample_index + 1) * self.scenario.dt_s
                throttle, brake, steering = self.scenario.control_at(timestamp_s)
                vehicle.apply_control(
                    carla.VehicleControl(
                        throttle=throttle,
                        brake=brake,
                        steer=steering,
                    )
                )
                world.tick()

                estimated, steer_angles = self.estimator.estimate(vehicle)
                arrivals, dropped = self.pulse_sensor.step(
                    timestamp_s, self.scenario.dt_s, estimated
                )
                measurements = self.ecu.step(timestamp_s, arrivals)
                rows.append(
                    self._build_row(
                        vehicle,
                        sample_index,
                        timestamp_s,
                        throttle,
                        brake,
                        steering,
                        estimated,
                        steer_angles,
                        measurements,
                        dropped,
                    )
                )

            return self._write_outputs(rows)
        finally:
            if vehicle is not None:
                vehicle.destroy()
            world.apply_settings(original_settings)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one CARLA ABS braking scenario and compatible CSV."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--map", default="Town04")
    parser.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--simulation-id", type=int, default=1)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--sample-rate-hz", type=int, default=100)
    parser.add_argument("--initial-speed-kmh", type=float, default=70.0)
    parser.add_argument("--warmup-s", type=float, default=0.5)
    parser.add_argument("--brake-start-s", type=float, default=0.5)
    parser.add_argument("--brake-end-s", type=float, default=4.5)
    parser.add_argument("--brake-value", type=float, default=0.65)
    parser.add_argument("--steering-value", type=float, default=0.0)
    parser.add_argument(
        "--fault-wheel", choices=("none", *WHEELS), default="FL"
    )
    parser.add_argument("--fault-start-s", type=float, default=1.5)
    parser.add_argument("--fault-end-s", type=float, default=3.0)
    parser.add_argument("--fault-severity", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wheelbase-m", type=float, default=2.65)
    parser.add_argument("--track-width-m", type=float, default=1.55)
    parser.add_argument("--wheel-radius-m", type=float, default=0.31)
    parser.add_argument("--vehicle-mass-kg", type=float, default=1400.0)
    parser.add_argument("--tire-friction", type=float, default=2.0)
    parser.add_argument("--no-rendering", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_results",
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.duration_s <= 0 or arguments.sample_rate_hz <= 0:
        raise SystemExit("Duration and sample rate must be positive.")
    if not 0.0 <= arguments.brake_value <= 1.0:
        raise SystemExit("--brake-value must be between 0 and 1.")
    if not -1.0 <= arguments.steering_value <= 1.0:
        raise SystemExit("--steering-value must be between -1 and 1.")
    if not 0.0 <= arguments.fault_severity <= 1.0:
        raise SystemExit("--fault-severity must be between 0 and 1.")
    if arguments.fault_start_s >= arguments.fault_end_s:
        raise SystemExit("Fault start must be earlier than fault end.")


def main() -> None:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    dataset, manifest = CarlaABSSimulator(arguments).run()
    print(f"Dataset:  {dataset}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
