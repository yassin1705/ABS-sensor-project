"""Scenario catalog and inference service for the local ABS dashboard."""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fault_parameter_training.inference import CNNGRUFusionPredictor
from model_training.data import SPEED_COLUMNS, VALID_COLUMNS, WHEELS
from model_training.realtime_spc_replay import (
    DEFAULT_LIMITS,
    FAULT_COLUMNS,
    load_control_limits,
    replay_simulation,
    summarize_replay,
)
from model_training.spc_calibration import (
    CSV_COLUMNS,
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA,
    PhysicalUnitGRUPredictor,
    iter_selected_simulations,
)

from .decision_logic import WheelIsolationTracker, classify_wheel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(
    os.environ.get(
        "PFA_MATLAB_DATA_ROOT",
        PROJECT_ROOT / "ABS_SoH_Simulator" / "simulation_results",
    )
)
CARLA_RESULTS_ROOT = Path(
    os.environ.get(
        "PFA_CARLA_DATA_ROOT",
        PROJECT_ROOT / "carla_simulator" / "simulation_results",
    )
)
FUSION_EXPERIMENTS = Path(
    os.environ.get(
        "PFA_FAULT_EXPERIMENTS",
        PROJECT_ROOT / "fault_parameter_training" / "experiments",
    )
)
FUSION_CACHE = Path(
    os.environ.get(
        "PFA_FAULT_CACHE",
        PROJECT_ROOT / "fault_parameter_training" / "cache" / "fault_parameter_dataset.npz",
    )
)


DATASET_SOURCES: dict[str, dict[str, Any]] = {
    "healthy": {
        "engine": "matlab",
        "kind": "healthy",
        "root": RESULTS_ROOT,
        "pattern": "abs_healthy_braking_dataset_*.csv",
    },
    "faulty": {
        "engine": "matlab",
        "kind": "faulty",
        "root": RESULTS_ROOT,
        "pattern": "abs_faulty_braking_dataset_*.csv",
    },
    "carla_healthy": {
        "engine": "carla",
        "kind": "healthy",
        "root": CARLA_RESULTS_ROOT,
        "pattern": "abs_carla_healthy_dataset_*.csv",
    },
    "carla_faulty": {
        "engine": "carla",
        "kind": "faulty",
        "root": CARLA_RESULTS_ROOT,
        "pattern": "abs_carla_faulty_dataset_*.csv",
    },
}


def _source_spec(source: str) -> dict[str, Any]:
    try:
        return DATASET_SOURCES[source]
    except KeyError as exc:
        choices = ", ".join(DATASET_SOURCES)
        raise ValueError(f"source must be one of: {choices}.") from exc


def _latest_dataset_pair(source: str) -> tuple[Path, Path]:
    spec = _source_spec(source)
    root = Path(spec["root"])
    datasets = sorted(root.glob(str(spec["pattern"])))
    if not datasets:
        raise FileNotFoundError(f"No dataset was found for source {source!r}.")
    dataset = datasets[-1]
    manifest = root / dataset.name.replace("_dataset_", "_manifest_")
    if not manifest.exists():
        raise FileNotFoundError(f"Matching manifest is missing: {manifest}")
    return dataset, manifest


def _clean_number(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scenario_entry(source: str, row: pd.Series) -> dict[str, Any]:
    spec = _source_spec(source)
    is_faulty = spec["kind"] == "faulty"
    engine_label = str(spec["engine"]).upper()
    fault_wheel = str(row.get("fault_wheel", "none"))
    severity = _clean_number(row.get("fault_severity", 0)) or 0.0
    return {
        "key": f"{source}:{int(row['simulation_id'])}",
        "source": source,
        "simulation_engine": str(spec["engine"]),
        "dataset_kind": str(spec["kind"]),
        "simulation_id": int(row["simulation_id"]),
        "label": (
            f"{engine_label} · Simulation {int(row['simulation_id'])} · "
            f"{fault_wheel} · severity {severity:.2f}"
            if is_faulty
            else f"{engine_label} · Simulation {int(row['simulation_id'])} · healthy reference"
        ),
        "requested_phenomenon": str(row.get("requested_phenomenon", "unknown")),
        "observed_phenomenon": str(row.get("observed_phenomenon", "unknown")),
        "initial_speed_kmh": _clean_number(row.get("initial_speed_kmh")),
        "fault_wheel": fault_wheel if is_faulty else None,
        "fault_severity": severity if is_faulty else None,
        "fault_start_s": _clean_number(row.get("fault_start_s")),
        "fault_end_s": _clean_number(row.get("fault_end_s")),
        "dropped_edge_count": int(_clean_number(row.get("dropped_edge_count")) or 0),
    }


def list_scenarios() -> list[dict[str, Any]]:
    """Return a small balanced catalog instead of exposing thousands of rows."""
    _, faulty_manifest_path = _latest_dataset_pair("faulty")
    _, healthy_manifest_path = _latest_dataset_pair("healthy")

    faulty = pd.read_csv(faulty_manifest_path)
    faulty = faulty.loc[faulty["dropped_edge_count"] > 0].copy()
    faulty.sort_values("simulation_id", inplace=True)

    selected_faulty: list[pd.Series] = []
    demo = faulty.loc[faulty["simulation_id"] == 5]
    if not demo.empty:
        selected_faulty.append(demo.iloc[0])
    for _, group in faulty.groupby(["fault_wheel", "fault_severity"], sort=True):
        row = group.iloc[0]
        if not any(int(item["simulation_id"]) == int(row["simulation_id"]) for item in selected_faulty):
            selected_faulty.append(row)

    healthy = pd.read_csv(healthy_manifest_path)
    healthy.sort_values("simulation_id", inplace=True)
    selected_healthy = [group.iloc[0] for _, group in healthy.groupby("requested_phenomenon", sort=True)]

    catalog = [_scenario_entry("faulty", row) for row in selected_faulty]
    catalog.extend(_scenario_entry("healthy", row) for row in selected_healthy)

    # A CARLA run writes its own dataset/manifest pair. Add every scenario in
    # the latest pair when present, without making CARLA mandatory for startup.
    for source in ("carla_faulty", "carla_healthy"):
        try:
            _, manifest_path = _latest_dataset_pair(source)
        except FileNotFoundError:
            continue
        manifest = pd.read_csv(manifest_path)
        if _source_spec(source)["kind"] == "faulty":
            manifest = manifest.loc[manifest["dropped_edge_count"] > 0]
        manifest.sort_values("simulation_id", inplace=True)
        catalog.extend(
            _scenario_entry(source, row) for _, row in manifest.iterrows()
        )
    return catalog


def _as_boolean(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"U", "S", "O"}:
        return np.isin(np.char.lower(values.astype(str)), ["1", "true"])
    return values.astype(bool)


class DiagnosticService:
    """Lazily loads the trained models and diagnoses one five-second scenario."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._old_predictor: PhysicalUnitGRUPredictor | None = None
        self._fusion_predictor: CNNGRUFusionPredictor | None = None
        self._limits_artifact: dict[str, Any] | None = None

    def _ensure_models(self) -> None:
        if self._old_predictor is not None and self._fusion_predictor is not None:
            return
        with self._load_lock:
            if self._old_predictor is None:
                self._old_predictor = PhysicalUnitGRUPredictor.load(
                    DEFAULT_CHECKPOINT,
                    DEFAULT_METADATA,
                    device="cpu",
                )
            if self._fusion_predictor is None:
                self._fusion_predictor = CNNGRUFusionPredictor(
                    FUSION_EXPERIMENTS,
                    FUSION_CACHE,
                    device="cpu",
                )
            if self._limits_artifact is None:
                self._limits_artifact = load_control_limits(DEFAULT_LIMITS)

    @staticmethod
    def _old_spc_class(summary: dict[str, Any]) -> str:
        if int(summary["localized_suspect_sample_count"]) > 0:
            return "SUSPECTED_FAULTY"
        if int(summary["alarm_sample_count"]) > 0:
            return "CROSS_ALARM"
        return "HEALTHY"

    @staticmethod
    def _truth(simulation: pd.DataFrame, wheel: str) -> dict[str, Any]:
        active_column = f"fault_active_{wheel}"
        if active_column not in simulation:
            return {"class": "HEALTHY"}
        active = _as_boolean(simulation[active_column].to_numpy())
        if not active.any():
            return {"class": "HEALTHY"}
        rows = simulation.loc[active]
        return {
            "class": "FAULTY",
            "fault_type": str(rows[f"fault_type_{wheel}"].iloc[0]),
            "fault_start_s": float(rows["time_s"].min()),
            "fault_end_s": float(rows["time_s"].max()),
            "fault_severity": float(rows[f"fault_severity_{wheel}"].max()),
            "dropped_edge_count": int(simulation[f"dropped_edges_{wheel}"].sum()),
        }

    @staticmethod
    def _health_score(
        fault_probability: float,
        old_summary: dict[str, Any],
    ) -> dict[str, float | str]:
        valid = max(1, int(old_summary["valid_residual_count"]))
        alarm_rate = int(old_summary["alarm_sample_count"]) / valid
        localized_rate = int(old_summary["localized_suspect_sample_count"]) / valid
        msp_risk = min(1.0, 0.75 * alarm_rate + 0.25 * localized_rate)
        combined_risk = min(1.0, 0.75 * fault_probability + 0.25 * msp_risk)
        return {
            "health_percent": round(100.0 * (1.0 - combined_risk), 1),
            "model_health_percent": round(100.0 * (1.0 - fault_probability), 1),
            "msp_health_percent": round(100.0 * (1.0 - msp_risk), 1),
            "alarm_rate": round(alarm_rate, 4),
            "localized_rate": round(localized_rate, 4),
            "score_kind": "prototype_heuristic_not_calibrated_probability",
        }

    @staticmethod
    def _frames(results: pd.DataFrame) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for row in results.to_dict(orient="records"):
            wheels: dict[str, Any] = {}
            for wheel in WHEELS:
                wheels[wheel] = {
                    "measured_mps": _clean_number(row[f"measured_{wheel}_mps"]),
                    "predicted_mps": _clean_number(row[f"predicted_{wheel}_mps"]),
                    "residual_mps": _clean_number(row[f"residual_{wheel}_mps"]),
                    "status": str(row[f"status_{wheel}"]),
                    "warning": bool(row[f"warning_{wheel}"]),
                    "alarm": bool(row[f"alarm_{wheel}"]),
                    "valid": bool(row[f"valid_{wheel}"]),
                    "local_disruption_mps": _clean_number(row[f"local_disruption_{wheel}_mps"]),
                }
            frames.append(
                {
                    "time_s": float(row["time_s"]),
                    "suspected_wheel": (
                        str(row["suspected_wheel"])
                        if row.get("suspected_wheel") in WHEELS
                        else None
                    ),
                    "wheels": wheels,
                }
            )
        return frames

    def diagnose(self, source: str, simulation_id: int) -> dict[str, Any]:
        spec = _source_spec(source)
        self._ensure_models()
        assert self._old_predictor is not None
        assert self._fusion_predictor is not None
        assert self._limits_artifact is not None

        dataset, _ = _latest_dataset_pair(source)
        columns = (
            (*CSV_COLUMNS, *FAULT_COLUMNS)
            if spec["kind"] == "faulty"
            else CSV_COLUMNS
        )
        try:
            simulation = next(
                iter_selected_simulations(dataset, [simulation_id], columns=columns)
            )
        except StopIteration as exc:
            raise ValueError(f"Simulation {simulation_id} was not found in {source} data.") from exc

        with self._inference_lock:
            old_results, _ = replay_simulation(
                simulation,
                self._old_predictor,
                self._limits_artifact["limits"],
                live=False,
                playback_speed=0,
            )
            old_wheels = summarize_replay(old_results)

            fusion_results: dict[str, dict[str, Any]] = {}
            for wheel, speed_column, valid_column in zip(
                WHEELS, SPEED_COLUMNS, VALID_COLUMNS, strict=True
            ):
                speed = simulation[speed_column].to_numpy(dtype=np.float32)
                valid = _as_boolean(simulation[valid_column].to_numpy())
                fusion_results[wheel] = self._fusion_predictor.predict(speed, valid)

            isolation_tracker = WheelIsolationTracker(
                persistence_window=1,
                votes_required=1,
            )
            isolation = isolation_tracker.update(fusion_results)
            wheel_results: dict[str, Any] = {}
            for wheel in WHEELS:
                fusion = fusion_results[wheel]
                old = old_wheels[wheel]
                probability = float(fusion["fault_probability"])
                wheel_results[wheel] = {
                    "old_spc": {
                        "class": self._old_spc_class(old),
                        **old,
                    },
                    "model": {
                        "class": "FAULTY" if fusion["fault_detected"] else "HEALTHY",
                        **fusion,
                    },
                    "health": self._health_score(probability, old),
                    "decision": classify_wheel(wheel, old, fusion, isolation),
                    "ground_truth_evaluation_only": self._truth(simulation, wheel),
                }

        return {
            "simulation": {
                "source": source,
                "simulation_engine": str(spec["engine"]),
                "dataset_kind": str(spec["kind"]),
                "simulation_id": simulation_id,
                "requested_phenomenon": str(simulation["requested_phenomenon"].iloc[0]),
                "observed_phenomenon": str(simulation["observed_phenomenon"].iloc[0]),
                "duration_s": float(simulation["time_s"].max()),
                "sample_rate_hz": 100,
                "sample_count": len(simulation),
            },
            "model_context": {
                "training_source": "matlab",
                "spc_calibration_source": "matlab",
                "cross_domain_evaluation": spec["engine"] == "carla",
            },
            "score_definition": {
                "label": "Prototype combined health score",
                "formula": "risk = 0.75 × model_fault_probability + 0.25 × MSP_risk",
                "msp_formula": "MSP_risk = 0.75 × alarm_rate + 0.25 × localization_rate",
                "calibrated_probability": False,
            },
            "limits": self._limits_artifact["limits"],
            "isolation": isolation.to_dict(),
            "wheels": wheel_results,
            "frames": self._frames(old_results),
        }


SERVICE = DiagnosticService()
