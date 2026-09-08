"""Preparation et entrainement du second modele de diagnostic ABS.

Le modele recoit la serie temporelle d'un seul capteur fautif et estime le
debut, la duree et la severite du defaut intermittent injecte.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .models import FaultParameterGRU


WHEELS = ("FL", "FR", "RL", "RR")
FAULT_TYPE = "intermittent_loss"
BASE_DROPOUT_PROBABILITY = 0.50
SIMULATION_DURATION_S = 5.0
EXPECTED_SAMPLE_COUNT = 500
TARGET_NAMES = ("start_fraction", "duration_fraction", "severity")


def targets_from_manifest(row: pd.Series) -> np.ndarray:
    start = float(row["fault_start_s"])
    end = float(row["fault_end_s"])
    severity = float(row["fault_severity"])
    if not (0 <= start < end <= SIMULATION_DURATION_S + 1e-6):
        raise ValueError(f"Intervalle de defaut invalide pour simulation {row['simulation_id']}.")
    if not (0 < severity <= 1):
        raise ValueError(f"Severite invalide pour simulation {row['simulation_id']}.")
    return np.asarray(
        [start / SIMULATION_DURATION_S, (end - start) / SIMULATION_DURATION_S, severity],
        dtype=np.float32,
    )


def physical_parameters(normalized: np.ndarray) -> dict[str, float | str]:
    values = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
    start_s = float(values[0] * SIMULATION_DURATION_S)
    duration_s = float(values[1] * SIMULATION_DURATION_S)
    severity = float(values[2])
    return {
        "fault_type": FAULT_TYPE,
        "fault_start_s": start_s,
        "fault_duration_s": duration_s,
        "fault_end_s": min(SIMULATION_DURATION_S, start_s + duration_s),
        "fault_severity": severity,
        "dropout_probability": BASE_DROPOUT_PROBABILITY,
        "effective_dropout_probability": severity * BASE_DROPOUT_PROBABILITY,
    }


def _stratified_split(manifest: pd.DataFrame, seed: int) -> np.ndarray:
    """Partitionne par simulation, en equilibrant severite et phenomene."""
    split = np.full(len(manifest), "train", dtype="<U10")
    rng = np.random.default_rng(seed)
    strata = manifest["requested_phenomenon"].astype(str) + "|" + manifest[
        "fault_severity"
    ].astype(str)
    for indices in strata.groupby(strata).groups.values():
        indices = np.asarray(list(indices), dtype=np.int64)
        rng.shuffle(indices)
        n_test = max(1, int(round(0.15 * len(indices)))) if len(indices) >= 3 else 0
        n_validation = (
            max(1, int(round(0.15 * len(indices)))) if len(indices) >= 3 else 0
        )
        if n_test + n_validation >= len(indices):
            n_validation = max(0, len(indices) - n_test - 1)
        split[indices[:n_test]] = "test"
        split[indices[n_test : n_test + n_validation]] = "validation"

    # Reequilibrage global utile pour les petits jeux de smoke test, ou les
    # strates rares. Les simulations restent mutuellement exclusives.
    target_count = max(1, int(round(0.15 * len(manifest))))
    for target_split in ("test", "validation"):
        current = np.flatnonzero(split == target_split)
        if len(current) > target_count:
            rng.shuffle(current)
            split[current[target_count:]] = "train"
        elif len(current) < target_count:
            candidates = np.flatnonzero(split == "train")
            rng.shuffle(candidates)
            split[candidates[: target_count - len(current)]] = target_split
    return split


def select_manifest(
    manifest_csv: str | Path,
    *,
    max_simulations: int | None = None,
    split_seed: int = 42,
) -> pd.DataFrame:
    required = {
        "simulation_id",
        "requested_phenomenon",
        "fault_wheel",
        "fault_type",
        "fault_start_s",
        "fault_end_s",
        "fault_severity",
        "dropout_probability",
        "dropped_edge_count",
    }
    manifest = pd.read_csv(manifest_csv)
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du manifeste : {sorted(missing)}")
    manifest = manifest.loc[
        manifest["fault_wheel"].isin(WHEELS)
        & manifest["fault_type"].eq(FAULT_TYPE)
        & manifest["dropped_edge_count"].gt(0)
    ].copy()
    manifest.sort_values("simulation_id", inplace=True)
    if max_simulations is not None:
        # Echantillon reproductible, puis tri pour permettre la lecture streaming.
        manifest = manifest.sample(
            n=min(max_simulations, len(manifest)), random_state=split_seed
        ).sort_values("simulation_id")
    manifest.reset_index(drop=True, inplace=True)
    if len(manifest) < 3:
        raise ValueError("Au moins trois simulations observables sont necessaires.")
    manifest["dataset_split"] = _stratified_split(manifest, split_seed)
    return manifest


def _sensor_sequence(group: pd.DataFrame, wheel: str) -> np.ndarray:
    group = group.sort_values("time_s")
    speed = pd.to_numeric(
        group[f"wheel_speed_ecu_{wheel}_mps"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    valid = pd.to_numeric(
        group[f"ecu_valid_{wheel}"], errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float32)
    valid = ((valid > 0) & np.isfinite(speed)).astype(np.float32)
    if len(speed) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Simulation {group['simulation_id'].iloc[0]} : {len(speed)} echantillons, "
            f"{EXPECTED_SAMPLE_COUNT} attendus."
        )
    finite_indices = np.flatnonzero(np.isfinite(speed))
    if len(finite_indices) == 0:
        raise ValueError("Serie capteur entierement invalide.")
    # Imputation causale : remplissage avant, puis valeur initiale pour le demarrage.
    speed_series = pd.Series(speed).ffill().fillna(float(speed[finite_indices[0]]))
    return np.column_stack((speed_series.to_numpy(np.float32), valid)).astype(np.float32)


def prepare_dataset(
    dataset_csv: str | Path,
    manifest_csv: str | Path,
    output_npz: str | Path,
    *,
    max_simulations: int | None = None,
    split_seed: int = 42,
    chunksize: int = 100_000,
) -> Path:
    manifest = select_manifest(
        manifest_csv, max_simulations=max_simulations, split_seed=split_seed
    )
    selected = manifest.set_index("simulation_id")
    selected_ids = set(int(value) for value in selected.index)
    usecols = ["simulation_id", "time_s"]
    for wheel in WHEELS:
        usecols.extend((f"wheel_speed_ecu_{wheel}_mps", f"ecu_valid_{wheel}"))

    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    simulation_ids: list[int] = []
    wheels: list[int] = []
    splits: list[str] = []
    carry = pd.DataFrame(columns=usecols)

    def consume(frame: pd.DataFrame) -> None:
        for simulation_id, group in frame.groupby("simulation_id", sort=False):
            simulation_id = int(simulation_id)
            if simulation_id not in selected_ids:
                continue
            row = selected.loc[simulation_id]
            wheel = str(row["fault_wheel"])
            sequences.append(_sensor_sequence(group, wheel))
            targets.append(targets_from_manifest(row))
            simulation_ids.append(simulation_id)
            wheels.append(WHEELS.index(wheel))
            splits.append(str(row["dataset_split"]))

    for chunk in pd.read_csv(dataset_csv, usecols=usecols, chunksize=chunksize):
        combined = pd.concat((carry, chunk), ignore_index=True)
        last_id = combined["simulation_id"].iloc[-1]
        consume(combined.loc[combined["simulation_id"] != last_id])
        carry = combined.loc[combined["simulation_id"] == last_id].copy()
    if not carry.empty:
        consume(carry)

    if len(sequences) != len(manifest):
        found = set(simulation_ids)
        missing = sorted(selected_ids.difference(found))
        raise RuntimeError(
            f"{len(sequences)}/{len(manifest)} simulations extraites; absentes : {missing[:10]}"
        )

    order = np.argsort(np.asarray(simulation_ids))
    x = np.stack(sequences)[order]
    y = np.stack(targets)[order]
    simulation_id_array = np.asarray(simulation_ids, dtype=np.int32)[order]
    wheel_array = np.asarray(wheels, dtype=np.uint8)[order]
    split_array = np.asarray(splits)[order]

    train_speed = x[split_array == "train", :, 0]
    speed_mean = float(train_speed.mean())
    speed_std = float(train_speed.std())
    speed_std = max(speed_std, 1e-6)
    x[:, :, 0] = (x[:, :, 0] - speed_mean) / speed_std

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x=x,
        y=y,
        simulation_id=simulation_id_array,
        wheel=wheel_array,
        split=split_array,
        speed_mean=np.float32(speed_mean),
        speed_std=np.float32(speed_std),
    )
    manifest.sort_values("simulation_id").to_csv(
        output_npz.with_suffix(".manifest.csv"), index=False
    )
    print(
        f"Dataset parametres : {len(x)} series | "
        f"train={(split_array == 'train').sum()}, "
        f"validation={(split_array == 'validation').sum()}, "
        f"test={(split_array == 'test').sum()}"
    )
    return output_npz


class FaultParameterDataset(Dataset):
    def __init__(self, cache: dict[str, np.ndarray], split: str) -> None:
        mask = cache["split"].astype(str) == split
        self.x = torch.from_numpy(cache["x"][mask].astype(np.float32, copy=False))
        self.y = torch.from_numpy(cache["y"][mask].astype(np.float32, copy=False))
        self.simulation_id = torch.from_numpy(cache["simulation_id"][mask])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "simulation_id": self.simulation_id[index],
        }


def _evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_sum = 0.0
    loss_function = nn.SmoothL1Loss(beta=0.05)
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            prediction = model(x)
            loss_sum += loss_function(prediction, y).item() * len(x)
            predictions.append(prediction.cpu().numpy())
            targets.append(y.cpu().numpy())
    return (
        loss_sum / len(loader.dataset),
        np.concatenate(predictions),
        np.concatenate(targets),
    )


def train_model(
    dataset_npz: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 8,
    device: str | None = None,
) -> Path:
    with np.load(dataset_npz) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    datasets = {
        split: FaultParameterDataset(cache, split)
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Les partitions train, validation et test doivent etre non vides.")
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = FaultParameterGRU().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_function = nn.SmoothL1Loss(beta=0.05)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_path = output_directory / "best_model.pt"
    history: list[dict[str, float]] = []
    best_validation = float("inf")
    stale_epochs = 0
    metadata = {
        "model": "fault_parameter_gru",
        "input_channels": ["normalized_wheel_speed_ecu_mps", "ecu_valid"],
        "target_names": list(TARGET_NAMES),
        "simulation_duration_s": SIMULATION_DURATION_S,
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "fault_type": FAULT_TYPE,
        "dropout_probability": BASE_DROPOUT_PROBABILITY,
        "speed_mean": float(cache["speed_mean"]),
        "speed_std": float(cache["speed_std"]),
    }

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            x = batch["x"].to(torch_device)
            y = batch["y"].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        train_loss = total_loss / len(datasets["train"])
        validation_loss, validation_prediction, validation_target = _evaluate(
            model, loaders["validation"], torch_device
        )
        validation_mae = np.mean(
            np.abs(validation_prediction - validation_target), axis=0
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_start_mae_s": validation_mae[0] * SIMULATION_DURATION_S,
            "validation_duration_mae_s": validation_mae[1] * SIMULATION_DURATION_S,
            "validation_severity_mae": validation_mae[2],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_directory / "history.csv", index=False)
        print(
            f"epoch {epoch:03d} | train={train_loss:.5f} | val={validation_loss:.5f} | "
            f"MAE debut={record['validation_start_mae_s']:.3f}s, "
            f"duree={record['validation_duration_mae_s']:.3f}s, "
            f"severite={record['validation_severity_mae']:.3f}"
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "metadata": metadata, "epoch": epoch},
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(best_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_prediction, test_target = _evaluate(
        model, loaders["test"], torch_device
    )
    test_mae = np.mean(np.abs(test_prediction - test_target), axis=0)
    metrics = {
        "test_loss": float(test_loss),
        "test_start_mae_s": float(test_mae[0] * SIMULATION_DURATION_S),
        "test_duration_mae_s": float(test_mae[1] * SIMULATION_DURATION_S),
        "test_severity_mae": float(test_mae[2]),
        "test_effective_dropout_probability_mae": float(
            test_mae[2] * BASE_DROPOUT_PROBABILITY
        ),
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.zeros(1, EXPECTED_SAMPLE_COUNT, 2),
        output_directory / "fault_parameter_gru.onnx",
        input_names=["sensor_series"],
        output_names=["normalized_fault_parameters"],
        opset_version=17,
        dynamo=False,
    )
    print(json.dumps(metrics, indent=2))
    return best_path


def _latest_pair(project_root: Path) -> tuple[Path, Path]:
    results = project_root / "ABS_SoH_Simulator" / "simulation_results"
    datasets = sorted(results.glob("abs_faulty_braking_dataset_*.csv"))
    if not datasets:
        raise FileNotFoundError("Aucun dataset fautif trouve.")
    dataset = datasets[-1]
    manifest = results / dataset.name.replace("_dataset_", "_manifest_")
    if not manifest.exists():
        raise FileNotFoundError(f"Manifeste correspondant absent : {manifest}")
    return dataset, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "all"), nargs="?", default="all")
    parser.add_argument("--dataset-csv", type=Path)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("model_training/cache/fault_parameter_dataset.npz"),
    )
    parser.add_argument(
        "--output-directory", type=Path,
        default=Path("model_training/experiments/fault_parameter_gru"),
    )
    parser.add_argument("--max-simulations", type=int)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    dataset_csv, manifest_csv = _latest_pair(project_root)
    dataset_csv = args.dataset_csv or dataset_csv
    manifest_csv = args.manifest_csv or manifest_csv
    if args.command in ("prepare", "all"):
        prepare_dataset(
            dataset_csv,
            manifest_csv,
            args.cache,
            max_simulations=args.max_simulations,
        )
    if args.command in ("train", "all"):
        train_model(
            args.cache,
            args.output_directory,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            device=args.device,
        )


if __name__ == "__main__":
    main()
