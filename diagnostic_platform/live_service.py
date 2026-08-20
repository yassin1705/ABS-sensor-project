"""Live CARLA telemetry session and rolling ABS diagnostics."""

from __future__ import annotations

import math
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_training.data import SPEED_COLUMNS, VALID_COLUMNS, WHEELS
from model_training.realtime_spc_replay import CausalSPCReplayEngine

from .decision_logic import WheelIsolationTracker, classify_wheel
from .service import DiagnosticService, SERVICE, _clean_number


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CARLA_PYTHON = PROJECT_ROOT / "carla_env" / "Scripts" / "python.exe"
CARLA_DRIVER = PROJECT_ROOT / "carla_simulator" / "carla_live_driver.py"
DEFAULT_API_URL = "http://127.0.0.1:8765"


def _default_config() -> dict[str, Any]:
    return {
        "fault_wheel": "none",
        "fault_start_s": 5.0,
        "fault_duration_s": 10.0,
        "fault_severity": 0.8,
        "map": "Town04",
        "vehicle_filter": "vehicle.tesla.model3",
    }


class LiveDiagnosticService:
    """Own live session state while CARLA runs in another Python process."""

    def __init__(self, diagnostic_service: DiagnosticService = SERVICE) -> None:
        self.diagnostic_service = diagnostic_service
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._status = "idle"
        self._message = "Configure the sensors, then start a live drive."
        self._config = _default_config()
        self._engine: CausalSPCReplayEngine | None = None
        self._records: deque[dict[str, Any]] = deque(maxlen=500)
        self._raw_speeds: deque[np.ndarray] = deque(maxlen=500)
        self._raw_valid: deque[np.ndarray] = deque(maxlen=500)
        self._fusion: dict[str, dict[str, Any]] = {}
        self._sample_count = 0
        self._last_fusion_sample = 0
        self._vehicle: dict[str, Any] = {}
        self._last_telemetry_error: str | None = None
        self._isolation_tracker = WheelIsolationTracker()

    @staticmethod
    def _validate_config(payload: dict[str, Any]) -> dict[str, Any]:
        config = _default_config()
        config.update({key: value for key, value in payload.items() if key in config})
        wheel = str(config["fault_wheel"])
        if wheel not in {"none", *WHEELS}:
            raise ValueError("fault_wheel must be none, FL, FR, RL, or RR.")
        severity = float(config["fault_severity"])
        start = float(config["fault_start_s"])
        duration = float(config["fault_duration_s"])
        if not 0.0 <= severity <= 1.0:
            raise ValueError("fault_severity must be between 0 and 1.")
        if start < 0.0 or duration <= 0.0:
            raise ValueError("Fault start must be non-negative and duration positive.")
        config.update(
            fault_wheel=wheel,
            fault_severity=severity,
            fault_start_s=start,
            fault_duration_s=duration,
            map=str(config["map"]),
            vehicle_filter=str(config["vehicle_filter"]),
        )
        return config

    def _reset_diagnostics(self) -> None:
        self.diagnostic_service._ensure_models()
        assert self.diagnostic_service._old_predictor is not None
        assert self.diagnostic_service._limits_artifact is not None
        self._engine = CausalSPCReplayEngine(
            self.diagnostic_service._old_predictor,
            self.diagnostic_service._limits_artifact["limits"],
        )
        self._records.clear()
        self._raw_speeds.clear()
        self._raw_valid.clear()
        self._fusion.clear()
        self._sample_count = 0
        self._last_fusion_sample = 0
        self._vehicle = {}
        self._last_telemetry_error = None
        self._isolation_tracker = WheelIsolationTracker()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise ValueError("A CARLA live driver is already running.")
            if not CARLA_PYTHON.exists():
                raise FileNotFoundError(f"CARLA Python environment not found: {CARLA_PYTHON}")
            if not CARLA_DRIVER.exists():
                raise FileNotFoundError(f"CARLA live driver not found: {CARLA_DRIVER}")
            self._config = self._validate_config(payload)
            self._reset_diagnostics()
            self._status = "starting"
            self._message = "Launching the CARLA driving window…"
            self._process = subprocess.Popen(
                [
                    str(CARLA_PYTHON),
                    str(CARLA_DRIVER),
                    "--api-url",
                    DEFAULT_API_URL,
                ],
                cwd=PROJECT_ROOT,
            )
            return self.state(include_diagnostic=False)

    def driver_config(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "stop_requested": self._status == "stopping",
                "config": dict(self._config),
            }

    def update_driver_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept the Pygame pre-drive sensor selection before telemetry starts."""
        with self._lock:
            if self._status != "starting":
                raise ValueError("Sensor configuration can only change before driving starts.")
            merged = dict(self._config)
            merged.update(payload)
            self._config = self._validate_config(merged)
            return dict(self._config)

    def mark_running(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._status = "running"
            self._message = "Drive in the CARLA window; telemetry is live."
            self._vehicle = dict(payload)

    def mark_failed(self, message: str) -> None:
        with self._lock:
            self._status = "error"
            self._message = message

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._status in {"starting", "running"}:
                self._status = "stopping"
                self._message = "Stopping the CARLA live driver…"
            return self.state(include_diagnostic=False)

    def mark_stopped(self) -> None:
        with self._lock:
            self._status = "stopped"
            self._message = "Live drive stopped. The last measurements remain visible."

    def ingest(self, frames: list[dict[str, Any]]) -> int:
        if not frames:
            return 0
        with self._lock, self.diagnostic_service._inference_lock:
            if self._status not in {"running", "stopping"}:
                raise ValueError("The live session is not running.")
            assert self._engine is not None
            accepted = 0
            for frame in frames:
                try:
                    speeds = np.asarray(
                        [float(frame[column]) for column in SPEED_COLUMNS],
                        dtype=np.float32,
                    )
                    valid = np.asarray(
                        [bool(frame[column]) for column in VALID_COLUMNS],
                        dtype=bool,
                    )
                    record = self._engine.process(float(frame["time_s"]), speeds, valid)
                    self._records.append(record)
                    self._raw_speeds.append(speeds)
                    self._raw_valid.append(valid)
                    self._sample_count += 1
                    accepted += 1
                    self._vehicle = {
                        **self._vehicle,
                        "vehicle_speed_mps": _clean_number(frame.get("vehicle_speed_mps")),
                        "throttle": _clean_number(frame.get("throttle")),
                        "brake": _clean_number(frame.get("brake")),
                        "steer": _clean_number(frame.get("steer")),
                    }
                except Exception as exc:
                    self._last_telemetry_error = (
                        f"Skipped one telemetry sample: {type(exc).__name__}: {exc}"
                    )
            if (
                len(self._raw_speeds) == 500
                and self._sample_count - self._last_fusion_sample >= 50
            ):
                self._run_fusion()
            return accepted

    def _run_fusion(self) -> None:
        assert self.diagnostic_service._fusion_predictor is not None
        speeds = np.stack(self._raw_speeds)
        valid = np.stack(self._raw_valid)
        for index, wheel in enumerate(WHEELS):
            try:
                self._fusion[wheel] = self.diagnostic_service._fusion_predictor.predict(
                    speeds[:, index], valid[:, index]
                )
            except ValueError as exc:
                # A stationary vehicle can have no ABS edges, hence no finite
                # ECU speed during the first rolling window. Keep acquisition
                # alive and retry after another 50 samples.
                self._fusion[wheel] = {
                    "fault_detected": False,
                    "fault_probability": 0.0,
                    "fault_threshold": 0.5,
                    "fault_parameters": None,
                    "state": "insufficient_valid_signal",
                }
                self._last_telemetry_error = f"{wheel} rolling model waiting: {exc}"
        self._isolation_tracker.update(self._fusion)
        self._last_fusion_sample = self._sample_count

    def _old_summary(self, wheel: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        valid = [record for record in records if record[f"valid_{wheel}"]]
        warnings = [record for record in records if record[f"warning_{wheel}"]]
        alarms = [record for record in records if record[f"alarm_{wheel}"]]
        localized = [record for record in records if record["suspected_wheel"] == wheel]
        residuals = [
            abs(float(record[f"residual_{wheel}_mps"]))
            for record in valid
            if math.isfinite(float(record[f"residual_{wheel}_mps"]))
        ]
        summary = {
            "valid_residual_count": len(residuals),
            "warning_count": len(warnings),
            "alarm_sample_count": len(alarms),
            "first_alarm_time_s": float(alarms[0]["time_s"]) if alarms else None,
            "maximum_absolute_residual_mps": max(residuals) if residuals else None,
            "localized_suspect_sample_count": len(localized),
        }
        return {"class": self.diagnostic_service._old_spc_class(summary), **summary}

    def _diagnostic(self, *, include_frames: bool = True) -> dict[str, Any] | None:
        records = list(self._records)
        if not records or self.diagnostic_service._limits_artifact is None:
            return None
        wheels: dict[str, Any] = {}
        configured_fault = self._config["fault_wheel"]
        old_summaries = {
            wheel: self._old_summary(wheel, records) for wheel in WHEELS
        }
        isolation = self._isolation_tracker.snapshot
        for wheel in WHEELS:
            old = old_summaries[wheel]
            fusion = self._fusion.get(
                wheel,
                {
                    "fault_detected": False,
                    "fault_probability": 0.0,
                    "fault_threshold": 0.5,
                    "fault_parameters": None,
                    "state": "collecting_500_samples",
                },
            )
            probability = float(fusion["fault_probability"])
            truth: dict[str, Any] = {"class": "HEALTHY"}
            if configured_fault == wheel:
                truth = {
                    "class": "FAULTY",
                    "fault_type": "intermittent_loss",
                    "fault_start_s": self._config["fault_start_s"],
                    "fault_end_s": (
                        self._config["fault_start_s"]
                        + self._config["fault_duration_s"]
                    ),
                    "fault_severity": self._config["fault_severity"],
                }
            wheels[wheel] = {
                "old_spc": old,
                "model": {
                    "class": "FAULTY" if fusion["fault_detected"] else "HEALTHY",
                    **fusion,
                },
                "health": self.diagnostic_service._health_score(probability, old),
                "decision": classify_wheel(wheel, old, fusion, isolation),
                "ground_truth_evaluation_only": truth,
            }
        diagnostic = {
            "simulation": {
                "source": "carla_live",
                "simulation_engine": "carla",
                "dataset_kind": "live",
                "simulation_id": 0,
                "requested_phenomenon": "manual_drive",
                "observed_phenomenon": "live",
                "duration_s": float(records[-1]["time_s"]),
                "sample_rate_hz": 100,
                "sample_count": self._sample_count,
            },
            "model_context": {
                "training_source": "matlab",
                "spc_calibration_source": "matlab",
                "cross_domain_evaluation": True,
            },
            "score_definition": {
                "label": "Prototype combined health score",
                "formula": "risk = 0.75 × model_fault_probability + 0.25 × MSP_risk",
                "msp_formula": "MSP_risk = 0.75 × alarm_rate + 0.25 × localization_rate",
                "calibrated_probability": False,
            },
            "limits": self.diagnostic_service._limits_artifact["limits"],
            "isolation": isolation.to_dict(),
            "wheels": wheels,
        }
        frame_records = records if include_frames else records[-1:]
        diagnostic["frames"] = self.diagnostic_service._frames(
            pd.DataFrame.from_records(frame_records)
        )
        return diagnostic

    def hud_state(self) -> dict[str, Any]:
        """Return the compact live state used by the local CARLA/Pygame HUD."""
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is not None
                and self._status in {"starting", "running", "stopping"}
            ):
                self._status = "error" if self._process.returncode else "stopped"
                self._message = (
                    "The CARLA live driver exited unexpectedly."
                    if self._process.returncode
                    else "Live drive stopped."
                )
            diagnostic = self._diagnostic(include_frames=False)
            return {
                "status": self._status,
                "message": self._message,
                "sample_count": self._sample_count,
                "telemetry_warning": self._last_telemetry_error,
                "diagnostic": diagnostic,
            }

    def state(self, *, include_diagnostic: bool = True) -> dict[str, Any]:
        with self._lock:
            if (
                self._process is not None
                and self._process.poll() is not None
                and self._status in {"starting", "running", "stopping"}
            ):
                self._status = "error" if self._process.returncode else "stopped"
                self._message = (
                    "The CARLA live driver exited unexpectedly."
                    if self._process.returncode
                    else "Live drive stopped."
                )
            diagnostic = None
            diagnostic_error = None
            if include_diagnostic:
                try:
                    diagnostic = self._diagnostic()
                except Exception as exc:
                    diagnostic_error = f"{type(exc).__name__}: {exc}"
            return {
                "status": self._status,
                "message": self._message,
                "config": dict(self._config),
                "vehicle": dict(self._vehicle),
                "sample_count": self._sample_count,
                "telemetry_warning": self._last_telemetry_error,
                "diagnostic_error": diagnostic_error,
                "diagnostic": diagnostic,
            }


LIVE_SERVICE = LiveDiagnosticService()
