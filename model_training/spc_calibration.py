"""Calibrage MSP des residus t+1 du predicteur GRU ABS."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import SPEED_COLUMNS, VALID_COLUMNS, WHEELS
from .models import GRUForecaster


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get(
        "PFA_MATLAB_DATA_ROOT",
        PROJECT_ROOT / "ABS_SoH_Simulator" / "simulation_results",
    )
)
_healthy_datasets = sorted(DEFAULT_DATASET_ROOT.glob("abs_healthy_braking_dataset_*.csv"))
DEFAULT_DATASET_CSV = Path(os.environ["PFA_HEALTHY_DATASET"]) if os.environ.get(
    "PFA_HEALTHY_DATASET"
) else (
    _healthy_datasets[-1]
    if _healthy_datasets
    else DEFAULT_DATASET_ROOT / "abs_healthy_braking_dataset.csv"
)
DEFAULT_SPLIT_CSV = Path(
    os.environ.get(
        "PFA_SPLIT_CSV",
        PROJECT_ROOT / "model_training" / "cache" / "simulation_splits.csv",
    )
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "PFA_FORECAST_CHECKPOINT",
        PROJECT_ROOT / "model_training" / "experiments" / "gru" / "best_model.pt",
    )
)
DEFAULT_METADATA = Path(
    os.environ.get(
        "PFA_FORECAST_METADATA",
        DEFAULT_CHECKPOINT.with_name("run_metadata.json"),
    )
)
DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT / "model_training" / "spc_calibration"
)

CSV_COLUMNS = (
    "simulation_id",
    "requested_phenomenon",
    "observed_phenomenon",
    "time_s",
    "brake",
    "vehicle_speed_mps",
    *SPEED_COLUMNS,
    *VALID_COLUMNS,
)


@dataclass
class PhysicalUnitGRUPredictor:
    """Charge le GRU et predit directement les vitesses en m/s."""

    model: GRUForecaster
    mean: np.ndarray
    std: np.ndarray
    device: torch.device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        metadata_path: str | Path,
        *,
        device: str = "cpu",
    ) -> "PhysicalUnitGRUPredictor":
        metadata = json.loads(
            Path(metadata_path).read_text(encoding="utf-8")
        )
        model = GRUForecaster(
            history_length=int(metadata["history_length"]),
            horizon=int(metadata["horizon"]),
        )
        torch_device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch_device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(torch_device).eval()
        return cls(
            model=model,
            mean=np.asarray(
                metadata["normalization_mean"], dtype=np.float32
            ),
            std=np.asarray(
                metadata["normalization_std"], dtype=np.float32
            ),
            device=torch_device,
        )

    @torch.no_grad()
    def predict(self, histories_mps: np.ndarray) -> np.ndarray:
        histories = np.asarray(histories_mps, dtype=np.float32)
        normalized = (histories - self.mean) / self.std
        prediction = self.model(
            torch.from_numpy(normalized).to(self.device)
        )
        return prediction.cpu().numpy() * self.std + self.mean


def select_simulation_ids(
    split_csv: str | Path,
    *,
    split: str = "validation",
    simulation_count: int = 100,
    seed: int = 42,
    selection: str = "random",
) -> list[int]:
    """Selectionne des simulations completes d'une partition existante."""
    manifest = pd.read_csv(
        split_csv, usecols=["simulation_id", "dataset_split"]
    )
    candidates = (
        manifest.loc[
            manifest["dataset_split"] == split, "simulation_id"
        ]
        .astype(int)
        .to_numpy()
    )
    if simulation_count <= 0:
        raise ValueError("simulation_count doit etre strictement positif.")
    if simulation_count > len(candidates):
        raise ValueError(
            f"{simulation_count} simulations demandees, mais seulement "
            f"{len(candidates)} sont disponibles dans {split}."
        )
    if selection == "first":
        selected = np.sort(candidates)[:simulation_count]
    elif selection == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(
            candidates, size=simulation_count, replace=False
        )
    else:
        raise ValueError("selection doit etre 'random' ou 'first'.")
    return sorted(int(value) for value in selected)


def iter_selected_simulations(
    dataset_csv: str | Path,
    simulation_ids: Iterable[int],
    *,
    chunk_rows: int = 250_000,
    columns: Iterable[str] = CSV_COLUMNS,
) -> Iterator[pd.DataFrame]:
    """Lit le gros CSV par blocs et retourne uniquement les simulations cible."""
    remaining = set(int(value) for value in simulation_ids)
    if not remaining:
        return

    selected_columns = tuple(columns)
    pending = pd.DataFrame(columns=selected_columns)
    for chunk in pd.read_csv(
        dataset_csv,
        usecols=list(selected_columns),
        chunksize=chunk_rows,
    ):
        combined = pd.concat([pending, chunk], ignore_index=True)
        last_id = int(combined["simulation_id"].iloc[-1])
        complete = combined[combined["simulation_id"] != last_id]
        pending = combined[combined["simulation_id"] == last_id].copy()

        for simulation_id, simulation in complete.groupby(
            "simulation_id", sort=False
        ):
            simulation_id = int(simulation_id)
            if simulation_id in remaining:
                remaining.remove(simulation_id)
                yield simulation.sort_values("time_s", kind="stable")
        if not remaining:
            return

    if not pending.empty:
        simulation_id = int(pending["simulation_id"].iloc[0])
        if simulation_id in remaining:
            remaining.remove(simulation_id)
            yield pending.sort_values("time_s", kind="stable")

    if remaining:
        raise ValueError(
            "Simulations absentes du CSV : "
            + ", ".join(str(value) for value in sorted(remaining))
        )


def _boolean_matrix(
    frame: pd.DataFrame, columns: Iterable[str]
) -> np.ndarray:
    values = frame[list(columns)].to_numpy()
    if values.dtype.kind in {"U", "S", "O"}:
        normalized = np.char.lower(values.astype(str))
        return np.isin(normalized, ["1", "true"])
    return values.astype(bool)


def extract_t_plus_one_residuals(
    simulation: pd.DataFrame,
    predictor,
    *,
    history_length: int = 20,
    inference_batch_size: int = 2048,
) -> tuple[pd.DataFrame, int]:
    """Calcule mesure moins prediction t+1 pour une simulation saine."""
    simulation = simulation.sort_values("time_s", kind="stable")
    speeds = simulation[list(SPEED_COLUMNS)].to_numpy(dtype=np.float32)
    valid = _boolean_matrix(simulation, VALID_COLUMNS)
    usable = valid & np.isfinite(speeds)

    histories: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_rows: list[int] = []
    skipped = 0
    for target_row in range(history_length, len(simulation)):
        start = target_row - history_length
        if not usable[start : target_row + 1].all():
            skipped += 1
            continue
        histories.append(speeds[start:target_row])
        targets.append(speeds[target_row])
        target_rows.append(target_row)

    if not histories:
        return pd.DataFrame(), skipped

    history_array = np.stack(histories)
    forecasts = []
    for start in range(0, len(history_array), inference_batch_size):
        batch = history_array[start : start + inference_batch_size]
        forecasts.append(predictor.predict(batch)[:, 0, :])
    t_plus_one = np.concatenate(forecasts, axis=0)
    target_array = np.stack(targets)
    residuals = target_array - t_plus_one

    metadata = simulation.iloc[target_rows]
    result = pd.DataFrame(
        {
            "simulation_id": metadata["simulation_id"].astype(int).to_numpy(),
            "time_s": metadata["time_s"].to_numpy(dtype=float),
            "requested_phenomenon": metadata[
                "requested_phenomenon"
            ].astype(str).to_numpy(),
            "observed_phenomenon": metadata[
                "observed_phenomenon"
            ].astype(str).to_numpy(),
            "brake": metadata["brake"].to_numpy(dtype=float),
            "vehicle_speed_mps": metadata[
                "vehicle_speed_mps"
            ].to_numpy(dtype=float),
        }
    )
    for wheel_index, wheel in enumerate(WHEELS):
        result[f"measured_{wheel}_mps"] = target_array[:, wheel_index]
        result[f"predicted_{wheel}_mps"] = t_plus_one[:, wheel_index]
        result[f"residual_{wheel}_mps"] = residuals[:, wheel_index]
    return result, skipped


def calculate_control_limits(
    residuals: pd.DataFrame,
    *,
    sigma_multiplier: float = 3.0,
) -> dict[str, dict[str, float]]:
    """Calcule les limites Shewhart classiques pour chaque roue."""
    if residuals.empty:
        raise ValueError("Aucun residu valide pour le calibrage.")
    limits: dict[str, dict[str, float]] = {}
    for wheel in WHEELS:
        values = residuals[f"residual_{wheel}_mps"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        lower = mean - sigma_multiplier * standard_deviation
        upper = mean + sigma_multiplier * standard_deviation
        outside = (values < lower) | (values > upper)
        quantiles = np.quantile(values, [0.001, 0.01, 0.5, 0.99, 0.999])
        limits[wheel] = {
            "sample_count": int(len(values)),
            "mean_mps": mean,
            "std_mps": standard_deviation,
            "lcl_mps": float(lower),
            "ucl_mps": float(upper),
            "healthy_outside_rate": float(np.mean(outside)),
            "q_0_1_percent_mps": float(quantiles[0]),
            "q_1_percent_mps": float(quantiles[1]),
            "median_mps": float(quantiles[2]),
            "q_99_percent_mps": float(quantiles[3]),
            "q_99_9_percent_mps": float(quantiles[4]),
        }
    return limits


def plot_control_charts(
    residuals: pd.DataFrame,
    limits: dict[str, dict[str, float]],
    output_path: str | Path,
    *,
    maximum_points: int = 5000,
) -> None:
    """Trace un apercu des quatre cartes de controle saines."""
    plotted = residuals.iloc[:maximum_points]
    figure, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    x = np.arange(len(plotted))
    for axis, wheel in zip(axes, WHEELS):
        values = plotted[f"residual_{wheel}_mps"].to_numpy(dtype=float)
        wheel_limits = limits[wheel]
        outside = (values < wheel_limits["lcl_mps"]) | (
            values > wheel_limits["ucl_mps"]
        )
        axis.plot(x, values, linewidth=0.7, color="#1f77b4")
        axis.scatter(
            x[outside],
            values[outside],
            s=12,
            color="#d62728",
            label="Hors limites",
            zorder=3,
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
        axis.set_ylabel(f"{wheel}\nresidu (m/s)")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("Echantillons valides, simulations concatenees")
    figure.suptitle("Cartes de controle saines — residu GRU t+1")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def calibrate(
    *,
    dataset_csv: str | Path,
    split_csv: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    output_directory: str | Path,
    split: str = "validation",
    simulation_count: int = 100,
    selection_seed: int = 42,
    selection: str = "random",
    chunk_rows: int = 250_000,
    device: str = "cpu",
    sigma_multiplier: float = 3.0,
    maximum_chart_points: int = 5000,
) -> dict:
    """Execute le calibrage MSP et ecrit les artefacts du POC."""
    simulation_ids = select_simulation_ids(
        split_csv,
        split=split,
        simulation_count=simulation_count,
        seed=selection_seed,
        selection=selection,
    )
    predictor = PhysicalUnitGRUPredictor.load(
        checkpoint_path, metadata_path, device=device
    )

    frames = []
    skipped_windows = 0
    for index, simulation in enumerate(
        iter_selected_simulations(
            dataset_csv, simulation_ids, chunk_rows=chunk_rows
        ),
        start=1,
    ):
        residuals, skipped = extract_t_plus_one_residuals(
            simulation,
            predictor,
            history_length=predictor.model.history_length,
        )
        frames.append(residuals)
        skipped_windows += skipped
        print(
            f"Simulation calibree {index}/{len(simulation_ids)} : "
            f"{int(simulation['simulation_id'].iloc[0])}",
            flush=True,
        )

    all_residuals = pd.concat(frames, ignore_index=True)
    limits = calculate_control_limits(
        all_residuals, sigma_multiplier=sigma_multiplier
    )
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    residual_path = output_directory / "healthy_t_plus_one_residuals.csv"
    limits_path = output_directory / "control_limits.json"
    chart_path = output_directory / "healthy_control_charts.png"
    all_residuals.to_csv(residual_path, index=False)

    artifact = {
        "version": 1,
        "model": "gru",
        "residual_definition": "measured_minus_predicted",
        "forecast_horizon": 1,
        "history_length": predictor.model.history_length,
        "sample_time_seconds": 0.01,
        "wheel_order": list(WHEELS),
        "source_dataset_csv": str(Path(dataset_csv).resolve()),
        "source_split_csv": str(Path(split_csv).resolve()),
        "source_checkpoint": str(Path(checkpoint_path).resolve()),
        "calibration_split": split,
        "selection": selection,
        "selection_seed": selection_seed,
        "simulation_ids": simulation_ids,
        "valid_window_count": int(len(all_residuals)),
        "skipped_invalid_window_count": int(skipped_windows),
        "sigma_multiplier": sigma_multiplier,
        "limits": limits,
    }
    limits_path.write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    plot_control_charts(
        all_residuals,
        limits,
        chart_path,
        maximum_points=maximum_chart_points,
    )
    print(f"Limites : {limits_path}")
    print(f"Residus : {residual_path}")
    print(f"Cartes  : {chart_path}")
    return artifact


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibre quatre cartes MSP avec les residus GRU t+1."
    )
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--simulation-count", type=int, default=100)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument(
        "--selection", choices=("random", "first"), default="random"
    )
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sigma-multiplier", type=float, default=3.0)
    parser.add_argument("--maximum-chart-points", type=int, default=5000)
    return parser


def main() -> None:
    arguments = build_argument_parser().parse_args()
    calibrate(
        dataset_csv=arguments.dataset_csv,
        split_csv=arguments.split_csv,
        checkpoint_path=arguments.checkpoint,
        metadata_path=arguments.metadata,
        output_directory=arguments.output_directory,
        split=arguments.split,
        simulation_count=arguments.simulation_count,
        selection_seed=arguments.selection_seed,
        selection=arguments.selection,
        chunk_rows=arguments.chunk_rows,
        device=arguments.device,
        sigma_multiplier=arguments.sigma_multiplier,
        maximum_chart_points=arguments.maximum_chart_points,
    )


if __name__ == "__main__":
    main()
