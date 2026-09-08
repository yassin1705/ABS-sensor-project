"""Preparation streaming d'un dataset mixte sain/fautif mono-capteur."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    EXPECTED_SAMPLE_COUNT,
    FAULT_TYPE,
    SIMULATION_DURATION_S,
    WHEELS,
)


def encode_faulty_targets(row: pd.Series) -> np.ndarray:
    """Encode [fault, start, duration, severity] dans [0, 1]."""
    start = float(row["fault_start_s"])
    end = float(row["fault_end_s"])
    severity = float(row["fault_severity"])
    if not 0 <= start < end <= SIMULATION_DURATION_S + 1e-6:
        raise ValueError(f"Intervalle invalide pour simulation {row['simulation_id']}.")
    if not 0 < severity <= 1:
        raise ValueError(f"Severite invalide pour simulation {row['simulation_id']}.")
    return np.asarray(
        [
            1.0,
            start / SIMULATION_DURATION_S,
            (end - start) / SIMULATION_DURATION_S,
            severity,
        ],
        dtype=np.float32,
    )


def encode_healthy_targets() -> np.ndarray:
    """Les parametres n'ont pas de sens lorsque fault=0 et seront masques."""
    return np.zeros(4, dtype=np.float32)


# Alias temporaire pour les anciens imports du POC.
encode_targets = encode_faulty_targets


def _allocate_splits(manifest: pd.DataFrame, strata: pd.Series, seed: int) -> np.ndarray:
    """Construit des partitions 70/15/15 sans fuite de simulation."""
    result = np.full(len(manifest), "train", dtype="<U10")
    rng = np.random.default_rng(seed)
    for indices in strata.groupby(strata).groups.values():
        indices = np.asarray(list(indices), dtype=np.int64)
        rng.shuffle(indices)
        test_count = max(1, round(0.15 * len(indices))) if len(indices) >= 3 else 0
        validation_count = (
            max(1, round(0.15 * len(indices))) if len(indices) >= 3 else 0
        )
        if test_count + validation_count >= len(indices):
            validation_count = max(0, len(indices) - test_count - 1)
        result[indices[:test_count]] = "test"
        result[indices[test_count : test_count + validation_count]] = "validation"

    target_count = max(1, round(0.15 * len(manifest)))
    for target_split in ("test", "validation"):
        current = np.flatnonzero(result == target_split)
        if len(current) > target_count:
            rng.shuffle(current)
            result[current[target_count:]] = "train"
        elif len(current) < target_count:
            candidates = np.flatnonzero(result == "train")
            rng.shuffle(candidates)
            result[candidates[: target_count - len(current)]] = target_split
    return result


def select_faulty_manifest(
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
        raise ValueError(f"Colonnes absentes du manifeste fautif : {sorted(missing)}")
    manifest = manifest.loc[
        manifest["fault_wheel"].isin(WHEELS)
        & manifest["fault_type"].eq(FAULT_TYPE)
        & manifest["dropped_edge_count"].gt(0)
    ].copy()
    if not np.allclose(manifest["dropout_probability"], 0.50):
        raise ValueError("La probabilite de dropout doit etre constante a 0.50.")
    manifest.sort_values("simulation_id", inplace=True)
    if max_simulations is not None:
        manifest = manifest.head(min(max_simulations, len(manifest))).copy()
    manifest.reset_index(drop=True, inplace=True)
    if len(manifest) < 3:
        raise ValueError("Au moins trois simulations fautives observables sont necessaires.")
    strata = (
        manifest["requested_phenomenon"].astype(str)
        + "|"
        + manifest["fault_severity"].astype(str)
    )
    manifest["dataset_split"] = _allocate_splits(manifest, strata, split_seed)
    return manifest


def select_healthy_manifest(
    manifest_csv: str | Path,
    *,
    max_simulations: int | None = None,
    split_seed: int = 42,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    required = {"simulation_id", "requested_phenomenon"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du manifeste sain : {sorted(missing)}")
    manifest.sort_values("simulation_id", inplace=True)
    if max_simulations is not None:
        manifest = manifest.head(min(max_simulations, len(manifest))).copy()
    manifest.reset_index(drop=True, inplace=True)
    if len(manifest) < 3:
        raise ValueError("Au moins trois simulations saines sont necessaires.")
    manifest["dataset_split"] = _allocate_splits(
        manifest,
        manifest["requested_phenomenon"].astype(str),
        split_seed + 1,
    )
    # Une roue saine deterministe par scenario evite de submerger les positifs.
    manifest["selected_wheel"] = [WHEELS[index % 4] for index in range(len(manifest))]
    return manifest


def _extract_sensor_series(simulation: pd.DataFrame, wheel: str) -> np.ndarray:
    simulation = simulation.sort_values("time_s")
    speed = pd.to_numeric(
        simulation[f"wheel_speed_ecu_{wheel}_mps"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    raw_valid = pd.to_numeric(
        simulation[f"ecu_valid_{wheel}"], errors="coerce"
    ).fillna(0).to_numpy(dtype=np.float32)
    valid = ((raw_valid > 0) & np.isfinite(speed)).astype(np.float32)
    if len(speed) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Simulation {simulation['simulation_id'].iloc[0]} : "
            f"{len(speed)} echantillons au lieu de {EXPECTED_SAMPLE_COUNT}."
        )
    finite = np.flatnonzero(np.isfinite(speed))
    if len(finite) == 0:
        raise ValueError("Serie capteur entierement invalide.")
    filled = pd.Series(speed).ffill().fillna(float(speed[finite[0]])).to_numpy(np.float32)
    return np.column_stack((filled, valid)).astype(np.float32)


def _read_source_records(
    dataset_csv: str | Path,
    manifest: pd.DataFrame,
    *,
    source_code: int,
    faulty_source: bool,
    csv_chunk_rows: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray, int, str]]:
    selected = manifest.set_index("simulation_id")
    selected_ids = set(int(value) for value in selected.index)
    usecols = ["simulation_id", "time_s"]
    for wheel in WHEELS:
        usecols += [f"wheel_speed_ecu_{wheel}_mps", f"ecu_valid_{wheel}"]
    records: list[tuple[int, int, np.ndarray, np.ndarray, int, str]] = []
    completed_simulations: set[int] = set()
    pending = pd.DataFrame(columns=usecols)

    def consume(frame: pd.DataFrame) -> None:
        for simulation_id, simulation in frame.groupby("simulation_id", sort=False):
            simulation_id = int(simulation_id)
            if simulation_id not in selected_ids:
                continue
            row = selected.loc[simulation_id]
            split = str(row["dataset_split"])
            if faulty_source:
                fault_wheel = str(row["fault_wheel"])
                positive_target = encode_faulty_targets(row)
                for wheel_index, wheel in enumerate(WHEELS):
                    target = positive_target if wheel == fault_wheel else encode_healthy_targets()
                    records.append(
                        (
                            source_code,
                            simulation_id,
                            _extract_sensor_series(simulation, wheel),
                            target,
                            wheel_index,
                            split,
                        )
                    )
            else:
                wheel = str(row["selected_wheel"])
                records.append(
                    (
                        source_code,
                        simulation_id,
                        _extract_sensor_series(simulation, wheel),
                        encode_healthy_targets(),
                        WHEELS.index(wheel),
                        split,
                    )
                )
            completed_simulations.add(simulation_id)

    for chunk in pd.read_csv(dataset_csv, usecols=usecols, chunksize=csv_chunk_rows):
        combined = pd.concat((pending, chunk), ignore_index=True)
        last_id = combined["simulation_id"].iloc[-1]
        consume(combined.loc[combined["simulation_id"] != last_id])
        pending = combined.loc[combined["simulation_id"] == last_id].copy()
        if len(completed_simulations) == len(manifest):
            pending = pending.iloc[0:0]
            break
    if not pending.empty:
        consume(pending)
    missing = sorted(selected_ids.difference(completed_simulations))
    if missing:
        raise RuntimeError(f"Simulations absentes du CSV : {missing[:10]}")
    return records


def prepare_npz_dataset(
    faulty_dataset_csv: str | Path,
    faulty_manifest_csv: str | Path,
    healthy_dataset_csv: str | Path,
    healthy_manifest_csv: str | Path,
    output_npz: str | Path,
    *,
    max_simulations: int | None = None,
    split_seed: int = 42,
    csv_chunk_rows: int = 100_000,
) -> dict[str, object]:
    """Cree un dataset avec positifs fautifs et plusieurs sources saines.

    Chaque scenario fautif fournit quatre enregistrements : la roue injectee
    est positive, les trois autres sont saines. Chaque scenario sain fournit
    une roue saine equilibree. Tous les enregistrements d'un meme scenario
    restent dans la meme partition.
    """
    faulty_manifest = select_faulty_manifest(
        faulty_manifest_csv,
        max_simulations=max_simulations,
        split_seed=split_seed,
    )
    healthy_manifest = select_healthy_manifest(
        healthy_manifest_csv,
        max_simulations=max_simulations,
        split_seed=split_seed,
    )
    records = _read_source_records(
        faulty_dataset_csv,
        faulty_manifest,
        source_code=1,
        faulty_source=True,
        csv_chunk_rows=csv_chunk_rows,
    )
    records += _read_source_records(
        healthy_dataset_csv,
        healthy_manifest,
        source_code=0,
        faulty_source=False,
        csv_chunk_rows=csv_chunk_rows,
    )
    records.sort(key=lambda record: (record[0], record[1], record[4]))
    source = np.asarray([record[0] for record in records], dtype=np.uint8)
    simulation_ids = np.asarray([record[1] for record in records], dtype=np.int32)
    x = np.stack([record[2] for record in records])
    y = np.stack([record[3] for record in records])
    wheels = np.asarray([record[4] for record in records], dtype=np.uint8)
    splits = np.asarray([record[5] for record in records])
    train_speed = x[splits == "train", :, 0]
    speed_mean = float(train_speed.mean())
    speed_std = max(float(train_speed.std()), 1e-6)
    x[:, :, 0] = (x[:, :, 0] - speed_mean) / speed_std

    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        x=x,
        y=y,
        source=source,
        simulation_id=simulation_ids,
        wheel=wheels,
        split=splits,
        speed_mean=np.float32(speed_mean),
        speed_std=np.float32(speed_std),
    )
    faulty_manifest.to_csv(output_npz.with_suffix(".faulty_manifest.csv"), index=False)
    healthy_manifest.to_csv(output_npz.with_suffix(".healthy_manifest.csv"), index=False)
    summary = {
        "cache": str(output_npz),
        "records": len(x),
        "faulty_records": int(y[:, 0].sum()),
        "healthy_records": int((y[:, 0] == 0).sum()),
        "train": int((splits == "train").sum()),
        "validation": int((splits == "validation").sum()),
        "test": int((splits == "test").sum()),
        "speed_mean": speed_mean,
        "speed_std": speed_std,
    }
    print(summary)
    return summary


class FaultParameterDataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Partition inconnue : {split}")
        with np.load(npz_path) as cache:
            mask = cache["split"].astype(str) == split
            self.x = torch.from_numpy(cache["x"][mask].astype(np.float32, copy=False))
            self.y = torch.from_numpy(cache["y"][mask].astype(np.float32, copy=False))
            self.sources = torch.from_numpy(cache["source"][mask].astype(np.int64))
            self.simulation_ids = torch.from_numpy(cache["simulation_id"][mask])
            self.wheels = torch.from_numpy(cache["wheel"][mask].astype(np.int64))
            self.speed_mean = float(cache["speed_mean"])
            self.speed_std = float(cache["speed_std"])

    def __len__(self) -> int:
        return len(self.x)

    @property
    def positive_count(self) -> int:
        return int(self.y[:, 0].sum().item())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "source": self.sources[index],
            "simulation_id": self.simulation_ids[index],
            "wheel": self.wheels[index],
        }


def create_dataloaders(
    npz_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = {
        split: FaultParameterDataset(npz_path, split)
        for split in ("train", "validation", "test")
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Les trois partitions doivent etre non vides.")
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        DataLoader(datasets["validation"], shuffle=False, **common),
        DataLoader(datasets["test"], shuffle=False, **common),
    )
