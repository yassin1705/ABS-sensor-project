"""Rejeu temps reel d'une serie ABS dans les cartes MSP du GRU."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import SPEED_COLUMNS, VALID_COLUMNS, WHEELS
from .spc_calibration import (
    CSV_COLUMNS,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET_CSV,
    DEFAULT_METADATA,
    DEFAULT_SPLIT_CSV,
    PROJECT_ROOT,
    PhysicalUnitGRUPredictor,
    iter_selected_simulations,
)


DEFAULT_LIMITS = Path(
    os.environ.get(
        "PFA_SPC_LIMITS",
        PROJECT_ROOT
        / "model_training"
        / "spc_calibration_poc"
        / "control_limits.json",
    )
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "model_training" / "spc_replay"
FAULT_COLUMNS = tuple(
    column
    for wheel in WHEELS
    for column in (
        f"fault_active_{wheel}",
        f"fault_type_{wheel}",
        f"fault_severity_{wheel}",
        f"dropped_edges_{wheel}",
    )
)


def load_control_limits(path: str | Path) -> dict:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("residual_definition") != "measured_minus_predicted":
        raise ValueError(
            "Le rejeu exige des limites mesure moins prediction."
        )
    if int(artifact.get("forecast_horizon", -1)) != 1:
        raise ValueError("Le rejeu exige des limites calibrees a t+1.")
    missing = set(WHEELS).difference(artifact["limits"])
    if missing:
        raise ValueError(f"Limites absentes pour : {sorted(missing)}")
    return artifact


def select_replay_simulation_id(
    split_csv: str | Path,
    limits_artifact: dict,
    *,
    split: str = "test",
    requested_id: int | None = None,
    preferred_phenomenon: str | None = "normal_braking",
) -> int:
    if requested_id is not None and split == "any":
        return requested_id
    manifest = pd.read_csv(
        split_csv,
        usecols=[
            "simulation_id",
            "dataset_split",
            "observed_phenomenon",
        ],
    )
    if requested_id is not None:
        row = manifest.loc[manifest["simulation_id"] == requested_id]
        if row.empty:
            raise ValueError(
                f"Simulation {requested_id} absente du manifeste."
            )
        actual_split = str(row["dataset_split"].iloc[0])
        if actual_split != split:
            raise ValueError(
                f"Simulation {requested_id} appartient a {actual_split}, "
                f"pas a {split}."
            )
        return requested_id

    calibration_ids = set(
        int(value) for value in limits_artifact.get("simulation_ids", [])
    )
    candidates = manifest.loc[
        manifest["dataset_split"] == split
    ].copy()
    candidates = candidates[
        ~candidates["simulation_id"].astype(int).isin(calibration_ids)
    ]
    if preferred_phenomenon is not None:
        preferred = candidates[
            candidates["observed_phenomenon"].astype(str)
            == preferred_phenomenon
        ]
        if not preferred.empty:
            candidates = preferred
    if candidates.empty:
        raise ValueError(
            f"Aucune simulation {split} hors calibrage n'est disponible."
        )
    return int(candidates["simulation_id"].iloc[0])


def select_faulty_demo_simulation_id(manifest_csv: str | Path) -> int:
    """Choisit un cas fautif demonstratif ayant reellement perdu des fronts."""
    manifest = pd.read_csv(
        manifest_csv,
        usecols=[
            "simulation_id",
            "observed_phenomenon",
            "fault_severity",
            "dropped_edge_count",
        ],
    )
    candidates = manifest.loc[manifest["dropped_edge_count"] > 0].copy()
    if candidates.empty:
        raise ValueError("Le manifeste ne contient aucun front effectivement perdu.")
    normal = candidates[
        candidates["observed_phenomenon"].astype(str) == "normal_braking"
    ]
    if not normal.empty:
        candidates = normal
    medium = candidates[np.isclose(candidates["fault_severity"], 0.50)]
    if not medium.empty:
        candidates = medium
    return int(candidates.sort_values("simulation_id").iloc[0]["simulation_id"])


def _display_bounds(
    wheel_limits: dict[str, float],
    *,
    padding_fraction: float = 0.10,
) -> tuple[float, float]:
    """Garde les limites lisibles sans laisser un pic ecraser l'echelle."""
    lower = float(wheel_limits["lcl_mps"])
    upper = float(wheel_limits["ucl_mps"])
    span = max(upper - lower, 1e-6)
    padding = padding_fraction * span
    return lower - padding, upper + padding


class CausalSPCReplayEngine:
    """Aligne causalement une prediction t+1 avec la mesure suivante."""

    def __init__(
        self,
        predictor,
        limits: dict[str, dict[str, float]],
        *,
        history_length: int = 20,
        alarm_window: int = 5,
        alarm_required: int = 3,
        localization_absolute_threshold_mps: float = 0.50,
        localization_ratio: float = 4.0,
        localization_votes_required: int = 2,
    ) -> None:
        if not 1 <= alarm_required <= alarm_window:
            raise ValueError(
                "alarm_required doit etre compris dans alarm_window."
            )
        self.predictor = predictor
        self.limits = limits
        self.history_length = history_length
        self.alarm_window = alarm_window
        self.alarm_required = alarm_required
        self.localization_absolute_threshold_mps = (
            localization_absolute_threshold_mps
        )
        self.localization_ratio = localization_ratio
        self.localization_votes_required = localization_votes_required
        self.speed_history: deque[np.ndarray] = deque(
            maxlen=history_length
        )
        self.valid_history: deque[np.ndarray] = deque(
            maxlen=history_length
        )
        self.pending_prediction: np.ndarray | None = None
        self.previous_speeds: np.ndarray | None = None
        self.previous_valid: np.ndarray | None = None
        self.localization_votes: deque[str | None] = deque(maxlen=5)
        self.exceedance_history = {
            wheel: deque(maxlen=alarm_window) for wheel in WHEELS
        }

    def process(
        self,
        timestamp: float,
        speeds_mps: np.ndarray,
        valid: np.ndarray,
    ) -> dict:
        speeds = np.asarray(speeds_mps, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(speeds)
        if speeds.shape != (4,) or valid.shape != (4,):
            raise ValueError("Quatre vitesses et quatre validites sont requises.")

        prediction = (
            self.pending_prediction.copy()
            if self.pending_prediction is not None
            else np.full(4, np.nan, dtype=np.float32)
        )
        residual = np.full(4, np.nan, dtype=np.float32)
        statuses: list[str] = []
        warnings = np.zeros(4, dtype=bool)
        alarms = np.zeros(4, dtype=bool)
        local_disruption = np.full(4, np.nan, dtype=np.float32)

        localization_candidate: str | None = None
        if self.previous_speeds is not None and self.previous_valid is not None:
            delta_usable = valid & self.previous_valid
            if int(delta_usable.sum()) >= 3:
                delta = speeds - self.previous_speeds
                center = float(np.median(delta[delta_usable]))
                local_disruption[delta_usable] = np.abs(
                    delta[delta_usable] - center
                )
                ranked = np.sort(local_disruption[delta_usable])
                maximum = float(ranked[-1])
                second = float(ranked[-2]) if len(ranked) > 1 else 0.0
                if (
                    maximum >= self.localization_absolute_threshold_mps
                    and maximum
                    >= self.localization_ratio * max(second, 0.05)
                ):
                    candidate_index = int(np.nanargmax(local_disruption))
                    localization_candidate = WHEELS[candidate_index]
        self.localization_votes.append(localization_candidate)
        vote_counts = {
            wheel: sum(value == wheel for value in self.localization_votes)
            for wheel in WHEELS
        }
        suspected_wheel = max(vote_counts, key=vote_counts.get)
        if vote_counts[suspected_wheel] < self.localization_votes_required:
            suspected_wheel = None

        if self.pending_prediction is None:
            for values in self.exceedance_history.values():
                values.clear()

        for index, wheel in enumerate(WHEELS):
            if not valid[index]:
                statuses.append("invalid")
                continue
            if self.pending_prediction is None:
                statuses.append("warmup")
                continue

            residual[index] = speeds[index] - prediction[index]
            wheel_limits = self.limits[wheel]
            outside = bool(
                residual[index] < wheel_limits["lcl_mps"]
                or residual[index] > wheel_limits["ucl_mps"]
            )
            self.exceedance_history[wheel].append(outside)
            alarm = (
                len(self.exceedance_history[wheel]) == self.alarm_window
                and sum(self.exceedance_history[wheel])
                >= self.alarm_required
            )
            warnings[index] = outside
            alarms[index] = alarm
            if alarm:
                statuses.append("alarm")
            elif outside:
                statuses.append("warning")
            else:
                statuses.append("normal")

        self.speed_history.append(speeds.copy())
        self.valid_history.append(valid.copy())
        self.previous_speeds = speeds.copy()
        self.previous_valid = valid.copy()
        history_is_usable = (
            len(self.speed_history) == self.history_length
            and np.stack(self.valid_history).all()
        )
        if history_is_usable:
            history = np.stack(self.speed_history)[None, :, :]
            self.pending_prediction = self.predictor.predict(history)[0, 0, :]
        else:
            self.pending_prediction = None

        record: dict[str, object] = {
            "time_s": float(timestamp),
            "localization_candidate": localization_candidate,
            "suspected_wheel": suspected_wheel,
        }
        for index, wheel in enumerate(WHEELS):
            record[f"measured_{wheel}_mps"] = float(speeds[index])
            record[f"predicted_{wheel}_mps"] = float(prediction[index])
            record[f"residual_{wheel}_mps"] = float(residual[index])
            record[f"status_{wheel}"] = statuses[index]
            record[f"warning_{wheel}"] = bool(warnings[index])
            record[f"alarm_{wheel}"] = bool(alarms[index])
            record[f"valid_{wheel}"] = bool(valid[index])
            record[f"local_disruption_{wheel}_mps"] = float(
                local_disruption[index]
            )
        return record


class LiveSPCDashboard:
    """Affiche les quatre cartes de controle sur une fenetre glissante."""

    def __init__(
        self,
        limits: dict[str, dict[str, float]],
        *,
        visible_seconds: float = 2.0,
    ) -> None:
        self.limits = limits
        self.visible_seconds = visible_seconds
        plt.ion()
        self.figure, self.axes = plt.subplots(
            4, 1, figsize=(14, 10), sharex=True
        )
        self.lines = {}
        self.warning_points = {}
        self.alarm_points = {}
        self.invalid_points = {}
        self.status_text = {}
        self.fault_spans = {wheel: None for wheel in WHEELS}

        for axis, wheel in zip(self.axes, WHEELS):
            wheel_limits = limits[wheel]
            self.lines[wheel], = axis.plot(
                [], [], color="#1f77b4", linewidth=1
            )
            self.warning_points[wheel] = axis.scatter(
                [], [], s=28, color="#ff7f0e", label="Avertissement"
            )
            self.alarm_points[wheel] = axis.scatter(
                [], [], s=38, color="#d62728", label="Alarme", zorder=4
            )
            self.invalid_points[wheel] = axis.scatter(
                [], [], s=24, color="#7f7f7f", marker="x", label="Invalide"
            )
            axis.axhline(
                wheel_limits["mean_mps"],
                color="#2ca02c",
                linewidth=1,
                label="Moyenne",
            )
            axis.axhline(
                wheel_limits["ucl_mps"],
                color="#d62728",
                linestyle="--",
                linewidth=1,
                label="UCL/LCL",
            )
            axis.axhline(
                wheel_limits["lcl_mps"],
                color="#d62728",
                linestyle="--",
                linewidth=1,
            )
            display_lower, display_upper = _display_bounds(wheel_limits)
            axis.set_ylim(display_lower, display_upper)
            self.status_text[wheel] = axis.text(
                0.01,
                0.88,
                "WARMUP",
                transform=axis.transAxes,
                fontsize=10,
                fontweight="bold",
            )
            axis.set_ylabel(f"{wheel}\nresidu (m/s)")
            axis.grid(True, alpha=0.25)

        self.axes[0].legend(loc="upper right", ncol=5, fontsize=8)
        self.axes[-1].set_xlabel("Temps (s)")
        self.figure.suptitle("Surveillance MSP temps reel — GRU t+1")
        self.figure.tight_layout()
        self.figure.show()

    @staticmethod
    def _offsets(x: np.ndarray, y: np.ndarray, selected: np.ndarray) -> np.ndarray:
        if not selected.any():
            return np.empty((0, 2))
        return np.column_stack([x[selected], y[selected]])

    def update(self, records: list[dict]) -> None:
        if not records or not plt.fignum_exists(self.figure.number):
            return
        end_time = float(records[-1]["time_s"])
        start_time = max(float(records[0]["time_s"]), end_time - self.visible_seconds)
        visible = [
            record
            for record in records
            if float(record["time_s"]) >= start_time
        ]
        x = np.asarray([record["time_s"] for record in visible], dtype=float)

        for axis, wheel in zip(self.axes, WHEELS):
            y = np.asarray(
                [record[f"residual_{wheel}_mps"] for record in visible],
                dtype=float,
            )
            warning = np.asarray(
                [record[f"warning_{wheel}"] for record in visible], dtype=bool
            )
            alarm = np.asarray(
                [record[f"alarm_{wheel}"] for record in visible], dtype=bool
            )
            localized = np.asarray(
                [record.get("suspected_wheel") == wheel for record in visible],
                dtype=bool,
            )
            invalid = np.asarray(
                [record[f"status_{wheel}"] == "invalid" for record in visible],
                dtype=bool,
            )
            fault_active = np.asarray(
                [
                    bool(record.get(f"fault_active_{wheel}", False))
                    for record in visible
                ],
                dtype=bool,
            )
            finite = np.isfinite(y)
            display_lower, display_upper = _display_bounds(
                self.limits[wheel]
            )
            displayed_y = np.clip(y, display_lower, display_upper)
            self.lines[wheel].set_data(x[finite], displayed_y[finite])
            self.warning_points[wheel].set_offsets(
                self._offsets(
                    x,
                    displayed_y,
                    finite & ((warning & ~alarm) | (alarm & ~localized)),
                )
            )
            self.alarm_points[wheel].set_offsets(
                self._offsets(x, displayed_y, alarm & localized & finite)
            )
            invalid_y = np.full(len(x), self.limits[wheel]["mean_mps"])
            self.invalid_points[wheel].set_offsets(
                self._offsets(x, invalid_y, invalid)
            )

            previous_span = self.fault_spans[wheel]
            if previous_span is not None:
                previous_span.remove()
                self.fault_spans[wheel] = None
            if fault_active.any():
                active_times = x[fault_active]
                self.fault_spans[wheel] = axis.axvspan(
                    active_times.min(),
                    active_times.max(),
                    color="#d62728",
                    alpha=0.08,
                    zorder=0,
                )

            status = str(visible[-1][f"status_{wheel}"]).upper()
            if status == "ALARM":
                status = (
                    "SUSPECTED"
                    if visible[-1].get("suspected_wheel") == wheel
                    else "CROSS-ALARM"
                )
            color = {
                "NORMAL": "#2ca02c",
                "WARNING": "#ff7f0e",
                "ALARM": "#d62728",
                "SUSPECTED": "#d62728",
                "CROSS-ALARM": "#ff7f0e",
                "INVALID": "#7f7f7f",
                "WARMUP": "#1f77b4",
            }[status]
            self.status_text[wheel].set_text(status)
            self.status_text[wheel].set_color(color)

            axis.set_ylim(display_lower, display_upper)

        if len(x) == 1:
            self.axes[-1].set_xlim(x[0] - 0.01, x[0] + self.visible_seconds)
        else:
            self.axes[-1].set_xlim(start_time, max(end_time, start_time + 0.01))
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=160)

    def hold(self) -> None:
        plt.ioff()
        plt.show()


def plot_static_replay(
    results: pd.DataFrame,
    limits: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    x = results["time_s"].to_numpy(dtype=float)
    for axis, wheel in zip(axes, WHEELS):
        y = results[f"residual_{wheel}_mps"].to_numpy(dtype=float)
        warning = results[f"warning_{wheel}"].to_numpy(dtype=bool)
        alarm = results[f"alarm_{wheel}"].to_numpy(dtype=bool)
        localized = results["suspected_wheel"].eq(wheel).to_numpy(
            dtype=bool
        )
        invalid = results[f"status_{wheel}"].eq("invalid").to_numpy()
        fault_column = f"fault_active_{wheel}"
        fault_active = (
            results[fault_column].astype(bool).to_numpy()
            if fault_column in results
            else np.zeros(len(results), dtype=bool)
        )
        finite = np.isfinite(y)
        display_lower, display_upper = _display_bounds(limits[wheel])
        displayed_y = np.clip(y, display_lower, display_upper)
        axis.plot(
            x[finite],
            displayed_y[finite],
            linewidth=0.9,
            color="#1f77b4",
        )
        axis.scatter(
            x[finite & ((warning & ~alarm) | (alarm & ~localized))],
            displayed_y[
                finite & ((warning & ~alarm) | (alarm & ~localized))
            ],
            s=25,
            color="#ff7f0e",
            label="Alerte residuelle / effet croise",
        )
        axis.scatter(
            x[alarm & localized & finite],
            displayed_y[alarm & localized & finite],
            s=35,
            color="#d62728",
            label="Capteur suspecte",
        )
        axis.scatter(
            x[invalid],
            np.full(invalid.sum(), limits[wheel]["mean_mps"]),
            s=25,
            color="#7f7f7f",
            marker="x",
            label="Invalide",
        )
        axis.axhline(limits[wheel]["mean_mps"], color="#2ca02c")
        axis.axhline(
            limits[wheel]["ucl_mps"], color="#d62728", linestyle="--"
        )
        axis.axhline(
            limits[wheel]["lcl_mps"], color="#d62728", linestyle="--"
        )
        if fault_active.any():
            active_times = x[fault_active]
            axis.axvspan(
                active_times.min(),
                active_times.max(),
                color="#d62728",
                alpha=0.08,
                label="Defaut injecte",
                zorder=0,
            )
        axis.set_ylim(display_lower, display_upper)
        axis.set_ylabel(f"{wheel}\nresidu (m/s)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4)
    axes[-1].set_xlabel("Temps (s)")
    figure.suptitle(
        "Rejeu MSP — GRU t+1 "
        "(pics hors echelle affiches sur le bord)"
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def replay_simulation(
    simulation: pd.DataFrame,
    predictor,
    limits: dict[str, dict[str, float]],
    *,
    live: bool = True,
    playback_speed: float = 1.0,
    render_every: int = 5,
    visible_seconds: float = 2.0,
) -> tuple[pd.DataFrame, LiveSPCDashboard | None]:
    if playback_speed < 0:
        raise ValueError("playback_speed doit etre positif ou nul.")
    simulation = simulation.sort_values("time_s", kind="stable")
    engine = CausalSPCReplayEngine(
        predictor,
        limits,
        history_length=20,
        alarm_window=5,
        alarm_required=3,
    )
    dashboard = (
        LiveSPCDashboard(limits, visible_seconds=visible_seconds)
        if live
        else None
    )
    records: list[dict] = []
    previous_timestamp: float | None = None

    for row_number, (_, row) in enumerate(simulation.iterrows(), start=1):
        timestamp = float(row["time_s"])
        started = time.perf_counter()
        speeds = row[list(SPEED_COLUMNS)].to_numpy(dtype=np.float32)
        valid_values = row[list(VALID_COLUMNS)].to_numpy()
        if valid_values.dtype.kind in {"U", "S", "O"}:
            valid = np.isin(
                np.char.lower(valid_values.astype(str)), ["1", "true"]
            )
        else:
            valid = valid_values.astype(bool)
        record = engine.process(timestamp, speeds, valid)
        record["simulation_id"] = int(row["simulation_id"])
        record["requested_phenomenon"] = str(row["requested_phenomenon"])
        record["observed_phenomenon"] = str(row["observed_phenomenon"])
        for wheel in WHEELS:
            active_column = f"fault_active_{wheel}"
            type_column = f"fault_type_{wheel}"
            severity_column = f"fault_severity_{wheel}"
            dropped_column = f"dropped_edges_{wheel}"
            if active_column in simulation.columns:
                record[active_column] = bool(row[active_column])
                record[type_column] = str(row[type_column])
                record[severity_column] = float(row[severity_column])
                record[dropped_column] = int(row[dropped_column])
        records.append(record)

        if dashboard is not None and (
            row_number % render_every == 0 or row_number == len(simulation)
        ):
            dashboard.update(records)

        if playback_speed > 0 and previous_timestamp is not None:
            target_delay = (timestamp - previous_timestamp) / playback_speed
            elapsed = time.perf_counter() - started
            if target_delay > elapsed:
                time.sleep(target_delay - elapsed)
        previous_timestamp = timestamp

    return pd.DataFrame(records), dashboard


def summarize_replay(results: pd.DataFrame) -> dict:
    summary: dict[str, dict[str, float | int | None]] = {}
    for wheel in WHEELS:
        warning = results[f"warning_{wheel}"].astype(bool)
        alarm = results[f"alarm_{wheel}"].astype(bool)
        residual = results[f"residual_{wheel}_mps"]
        first_alarm = results.loc[alarm, "time_s"]
        wheel_summary: dict[str, float | int | str | None] = {
            "valid_residual_count": int(residual.notna().sum()),
            "warning_count": int(warning.sum()),
            "alarm_sample_count": int(alarm.sum()),
            "first_alarm_time_s": (
                float(first_alarm.iloc[0]) if not first_alarm.empty else None
            ),
            "maximum_absolute_residual_mps": (
                float(residual.abs().max()) if residual.notna().any() else None
            ),
            "localized_suspect_sample_count": int(
                results["suspected_wheel"].eq(wheel).sum()
            ),
        }
        fault_column = f"fault_active_{wheel}"
        if fault_column in results:
            fault_active = results[fault_column].astype(bool)
            if fault_active.any():
                fault_rows = results.loc[fault_active]
                fault_start = float(fault_rows["time_s"].min())
                fault_end = float(fault_rows["time_s"].max())
                alarms_after_fault = results.loc[
                    alarm & (results["time_s"] >= fault_start), "time_s"
                ]
                first_detection = (
                    float(alarms_after_fault.iloc[0])
                    if not alarms_after_fault.empty
                    else None
                )
                wheel_summary.update(
                    {
                        "fault_type": str(
                            fault_rows[f"fault_type_{wheel}"].iloc[0]
                        ),
                        "fault_start_time_s": fault_start,
                        "fault_end_time_s": fault_end,
                        "fault_severity": float(
                            fault_rows[f"fault_severity_{wheel}"].max()
                        ),
                        "dropped_edge_count": int(
                            results[f"dropped_edges_{wheel}"].sum()
                        ),
                        "first_detection_time_s": first_detection,
                        "detection_delay_s": (
                            first_detection - fault_start
                            if first_detection is not None
                            else None
                        ),
                    }
                )
        summary[wheel] = wheel_summary
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rejoue une simulation ABS dans quatre cartes MSP."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--simulation-id", type=int)
    parser.add_argument(
        "--fault-manifest",
        type=Path,
        help="Manifeste utilise pour choisir automatiquement un cas fautif utile.",
    )
    parser.add_argument(
        "--phenomenon",
        default="normal_braking",
        help=(
            "Phenomenon observe prefere pour la demonstration automatique; "
            "utiliser 'any' pour ne pas filtrer."
        ),
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--render-every", type=int, default=5)
    parser.add_argument("--visible-seconds", type=float, default=2.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument(
        "--fault-aware",
        action="store_true",
        help="Charge les labels de defaut uniquement pour affichage/evaluation.",
    )
    parser.add_argument("--close-on-finish", action="store_true")
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    artifact = load_control_limits(arguments.limits)
    if arguments.simulation_id is None and arguments.fault_manifest is not None:
        simulation_id = select_faulty_demo_simulation_id(
            arguments.fault_manifest
        )
    else:
        simulation_id = select_replay_simulation_id(
            arguments.split_csv,
            artifact,
            split=arguments.split,
            requested_id=arguments.simulation_id,
            preferred_phenomenon=(
                None if arguments.phenomenon == "any" else arguments.phenomenon
            ),
        )
    replay_columns = (
        (*CSV_COLUMNS, *FAULT_COLUMNS)
        if arguments.fault_aware
        else CSV_COLUMNS
    )
    simulation = next(
        iter_selected_simulations(
            arguments.dataset_csv,
            [simulation_id],
            columns=replay_columns,
        )
    )
    predictor = PhysicalUnitGRUPredictor.load(
        arguments.checkpoint,
        arguments.metadata,
        device=arguments.device,
    )
    results, dashboard = replay_simulation(
        simulation,
        predictor,
        artifact["limits"],
        live=not arguments.no_live,
        playback_speed=arguments.playback_speed,
        render_every=arguments.render_every,
        visible_seconds=arguments.visible_seconds,
    )

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    prefix = "faulty_" if arguments.fault_aware else ""
    stem = f"{prefix}simulation_{simulation_id}_spc_replay"
    result_path = arguments.output_directory / f"{stem}.csv"
    summary_path = arguments.output_directory / f"{stem}.summary.json"
    chart_path = arguments.output_directory / f"{stem}.png"
    results.to_csv(result_path, index=False)
    summary = {
        "simulation_id": simulation_id,
        "dataset_split": arguments.split,
        "requested_phenomenon": str(
            simulation["requested_phenomenon"].iloc[0]
        ),
        "observed_phenomenon": str(
            simulation["observed_phenomenon"].iloc[0]
        ),
        "alarm_rule": "3 of 5 valid residuals outside LCL/UCL",
        "control_limits": str(arguments.limits.resolve()),
        "fault_labels_used_by_diagnostic": False,
        "wheels": summarize_replay(results),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if dashboard is not None:
        dashboard.save(chart_path)
    else:
        plot_static_replay(results, artifact["limits"], chart_path)
    print(f"Simulation : {simulation_id}")
    print(json.dumps(summary["wheels"], indent=2))
    print(f"Resultats  : {result_path}")
    print(f"Resume     : {summary_path}")
    print(f"Figure     : {chart_path}")

    if (
        dashboard is not None
        and not arguments.close_on_finish
        and plt.fignum_exists(dashboard.figure.number)
    ):
        dashboard.hold()


if __name__ == "__main__":
    main()
