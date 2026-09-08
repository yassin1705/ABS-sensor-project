"""Preparation commune du CSV MATLAB et chargement des fenetres temporelles."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


WHEELS = ("FL", "FR", "RL", "RR")
SPEED_COLUMNS = tuple(f"wheel_speed_ecu_{wheel}_mps" for wheel in WHEELS)
VALID_COLUMNS = tuple(f"ecu_valid_{wheel}" for wheel in WHEELS)
REQUIRED_COLUMNS = (
    "simulation_id",
    "requested_phenomenon",
    *SPEED_COLUMNS,
    *VALID_COLUMNS,
)
REGIME_TO_CODE = {
    "normal_braking": 0,
    "sliding": 1,
    "wheel_lock": 2,
    "random_braking": 3,
}
CODE_TO_REGIME = {value: key for key, value in REGIME_TO_CODE.items()}


def _allocate_stratified_splits(
    manifest: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Les proportions train/validation/test doivent sommer a 1.")

    required = {
        "simulation_id",
        "requested_phenomenon",
        "initial_speed_kmh",
        "maximum_abs_steering_wheel_deg",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du manifeste : {sorted(missing)}")

    result = manifest.copy()
    result["speed_bin"] = pd.cut(
        result["initial_speed_kmh"],
        bins=[39.999, 60, 80, 100.001],
        labels=["40_60", "60_80", "80_100"],
        include_lowest=True,
    ).astype("string")
    result["steering_bin"] = pd.cut(
        result["maximum_abs_steering_wheel_deg"],
        bins=[0, 20, 40, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype("string")
    result["stratum"] = (
        result["requested_phenomenon"].astype("string")
        + "|"
        + result["speed_bin"]
        + "|"
        + result["steering_bin"]
    )
    result["dataset_split"] = ""

    rng = np.random.default_rng(seed)
    for _, indices in result.groupby("stratum", dropna=False).groups.items():
        shuffled = np.asarray(list(indices), dtype=np.int64)
        rng.shuffle(shuffled)
        count = len(shuffled)

        test_count = int(round(test_ratio * count))
        validation_count = int(round(validation_ratio * count))
        if count >= 3:
            test_count = max(1, test_count)
            validation_count = max(1, validation_count)
        if test_count + validation_count >= count:
            overflow = test_count + validation_count - count + 1
            validation_count = max(0, validation_count - overflow)

        result.loc[shuffled[:test_count], "dataset_split"] = "test"
        result.loc[
            shuffled[test_count : test_count + validation_count],
            "dataset_split",
        ] = "validation"
        result.loc[
            shuffled[test_count + validation_count :],
            "dataset_split",
        ] = "train"

    total_count = len(result)
    target_counts = {
        "test": int(round(test_ratio * total_count)),
        "validation": int(round(validation_ratio * total_count)),
    }
    target_counts["train"] = (
        total_count
        - target_counts["test"]
        - target_counts["validation"]
    )

    # Les strates contenant une seule simulation sont initialement placees
    # dans train. Ce reequilibrage garantit les proportions globales, meme
    # pour un petit dataset de validation du pipeline.
    for split in ("test", "validation"):
        current_indices = result.index[
            result["dataset_split"] == split
        ].to_numpy(copy=True)
        excess = len(current_indices) - target_counts[split]
        if excess > 0:
            rng.shuffle(current_indices)
            result.loc[current_indices[:excess], "dataset_split"] = "train"

    for split in ("test", "validation"):
        current_count = int((result["dataset_split"] == split).sum())
        deficit = target_counts[split] - current_count
        if deficit > 0:
            candidates = result.index[
                result["dataset_split"] == "train"
            ].to_numpy(copy=True)
            rng.shuffle(candidates)
            result.loc[candidates[:deficit], "dataset_split"] = split

    if (result["dataset_split"] == "").any():
        raise RuntimeError("Certaines simulations n'ont pas recu de partition.")
    actual_counts = result["dataset_split"].value_counts().to_dict()
    if any(
        actual_counts.get(split, 0) != count
        for split, count in target_counts.items()
    ):
        raise RuntimeError("Le reequilibrage global des partitions a echoue.")
    return result


def _as_boolean_matrix(frame: pd.DataFrame, columns: Iterable[str]) -> np.ndarray:
    values = frame[list(columns)].to_numpy()
    if values.dtype.kind in {"U", "S", "O"}:
        normalized = np.char.lower(values.astype(str))
        return np.isin(normalized, ["1", "true"])
    return values.astype(bool)


def _create_hdf5_layout(
    handle: h5py.File,
    history_length: int,
    horizon: int,
) -> None:
    for split in ("train", "validation", "test"):
        group = handle.create_group(split)
        group.create_dataset(
            "x",
            shape=(0, history_length, 4),
            maxshape=(None, history_length, 4),
            dtype="float32",
            chunks=(1024, history_length, 4),
            compression="lzf",
        )
        group.create_dataset(
            "y",
            shape=(0, horizon, 4),
            maxshape=(None, horizon, 4),
            dtype="float32",
            chunks=(1024, horizon, 4),
            compression="lzf",
        )
        for name, dtype in (("regime", "uint8"), ("simulation_id", "int32")):
            group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=dtype,
                chunks=(8192,),
                compression="lzf",
            )


def _append_to_hdf5(
    handle: h5py.File,
    split: str,
    x: np.ndarray,
    y: np.ndarray,
    regime: np.ndarray,
    simulation_id: np.ndarray,
) -> None:
    if len(x) == 0:
        return
    group = handle[split]
    old_size = len(group["x"])
    new_size = old_size + len(x)
    for name, values in (
        ("x", x),
        ("y", y),
        ("regime", regime),
        ("simulation_id", simulation_id),
    ):
        dataset = group[name]
        dataset.resize(new_size, axis=0)
        dataset[old_size:new_size] = values


def _make_windows(
    simulation: pd.DataFrame,
    history_length: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    speeds = simulation[list(SPEED_COLUMNS)].to_numpy(dtype=np.float32)
    valid = _as_boolean_matrix(simulation, VALID_COLUMNS)
    usable = valid & np.isfinite(speeds)

    last_start = len(simulation) - history_length - horizon
    if last_start < 0:
        empty_x = np.empty((0, history_length, 4), dtype=np.float32)
        empty_y = np.empty((0, horizon, 4), dtype=np.float32)
        return empty_x, empty_y

    starts = np.arange(0, last_start + 1, stride)
    keep = np.fromiter(
        (
            usable[start : start + history_length + horizon].all()
            for start in starts
        ),
        dtype=bool,
        count=len(starts),
    )
    starts = starts[keep]
    if len(starts) == 0:
        empty_x = np.empty((0, history_length, 4), dtype=np.float32)
        empty_y = np.empty((0, horizon, 4), dtype=np.float32)
        return empty_x, empty_y

    x = np.stack(
        [speeds[start : start + history_length] for start in starts]
    ).astype(np.float32, copy=False)
    y = np.stack(
        [
            speeds[
                start + history_length : start + history_length + horizon
            ]
            for start in starts
        ]
    ).astype(np.float32, copy=False)
    return x, y


def prepare_hdf5_dataset(
    dataset_csv: str | Path,
    manifest_csv: str | Path,
    output_hdf5: str | Path,
    split_csv: str | Path,
    *,
    history_length: int = 20,
    horizon: int = 5,
    stride: int = 2,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    csv_chunk_rows: int = 250_000,
    hdf5_flush_windows: int = 50_000,
) -> dict:
    """Transforme le CSV brut en cache HDF5 commun aux trois modeles."""
    dataset_csv = Path(dataset_csv)
    manifest_csv = Path(manifest_csv)
    output_hdf5 = Path(output_hdf5)
    split_csv = Path(split_csv)
    if output_hdf5.exists():
        raise FileExistsError(
            f"Le cache existe deja : {output_hdf5}. Supprimez-le explicitement."
        )
    if history_length <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("history_length, horizon et stride doivent etre positifs.")

    manifest = pd.read_csv(manifest_csv)
    split_manifest = _allocate_stratified_splits(
        manifest,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(split_csv, index=False)
    split_by_simulation = dict(
        zip(
            split_manifest["simulation_id"].astype(int),
            split_manifest["dataset_split"].astype(str),
        )
    )

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    buffers: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list)
        for split in ("train", "validation", "test")
    }
    buffered_counts = {split: 0 for split in buffers}
    train_sum = np.zeros(4, dtype=np.float64)
    train_square_sum = np.zeros(4, dtype=np.float64)
    train_count = np.zeros(4, dtype=np.int64)
    processed_simulations = 0

    def flush(handle: h5py.File, split: str) -> None:
        if buffered_counts[split] == 0:
            return
        values = buffers[split]
        _append_to_hdf5(
            handle,
            split,
            np.concatenate(values["x"], axis=0),
            np.concatenate(values["y"], axis=0),
            np.concatenate(values["regime"], axis=0),
            np.concatenate(values["simulation_id"], axis=0),
        )
        buffers[split] = defaultdict(list)
        buffered_counts[split] = 0

    def process_simulation(handle: h5py.File, frame: pd.DataFrame) -> None:
        nonlocal processed_simulations, train_sum, train_square_sum, train_count
        simulation_id = int(frame["simulation_id"].iloc[0])
        if simulation_id not in split_by_simulation:
            raise KeyError(f"Simulation {simulation_id} absente du manifeste.")
        split = split_by_simulation[simulation_id]
        frame = frame.sort_values("time_s", kind="stable")

        if split == "train":
            speeds = frame[list(SPEED_COLUMNS)].to_numpy(dtype=np.float64)
            valid = _as_boolean_matrix(frame, VALID_COLUMNS)
            usable = valid & np.isfinite(speeds)
            safe_speeds = np.where(usable, speeds, 0.0)
            train_sum += safe_speeds.sum(axis=0)
            train_square_sum += np.square(safe_speeds).sum(axis=0)
            train_count += usable.sum(axis=0)

        x, y = _make_windows(frame, history_length, horizon, stride)
        if len(x):
            requested = str(frame["requested_phenomenon"].iloc[0])
            regime_code = REGIME_TO_CODE.get(requested, 255)
            buffers[split]["x"].append(x)
            buffers[split]["y"].append(y)
            buffers[split]["regime"].append(
                np.full(len(x), regime_code, dtype=np.uint8)
            )
            buffers[split]["simulation_id"].append(
                np.full(len(x), simulation_id, dtype=np.int32)
            )
            buffered_counts[split] += len(x)
            if buffered_counts[split] >= hdf5_flush_windows:
                flush(handle, split)

        processed_simulations += 1
        if processed_simulations % 500 == 0:
            print(f"Simulations preparees : {processed_simulations}")

    use_columns = list(dict.fromkeys((*REQUIRED_COLUMNS, "time_s")))
    with h5py.File(output_hdf5, "w") as handle:
        _create_hdf5_layout(handle, history_length, horizon)
        pending = pd.DataFrame(columns=use_columns)

        for chunk in pd.read_csv(
            dataset_csv,
            usecols=use_columns,
            chunksize=csv_chunk_rows,
        ):
            combined = pd.concat([pending, chunk], ignore_index=True)
            last_simulation_id = combined["simulation_id"].iloc[-1]
            complete = combined[
                combined["simulation_id"] != last_simulation_id
            ]
            pending = combined[
                combined["simulation_id"] == last_simulation_id
            ].copy()

            for _, simulation in complete.groupby(
                "simulation_id", sort=False
            ):
                process_simulation(handle, simulation)

        if not pending.empty:
            process_simulation(handle, pending)
        for split in buffers:
            flush(handle, split)

        if np.any(train_count == 0):
            raise RuntimeError("Aucune valeur valide pour normaliser le train.")
        mean = train_sum / train_count
        variance = train_square_sum / train_count - np.square(mean)
        standard_deviation = np.sqrt(np.maximum(variance, 1e-12))

        handle.attrs["history_length"] = history_length
        handle.attrs["horizon"] = horizon
        handle.attrs["stride"] = stride
        handle.attrs["wheel_order"] = json.dumps(WHEELS)
        handle.attrs["normalization_mean"] = mean
        handle.attrs["normalization_std"] = standard_deviation
        handle.attrs["regime_mapping"] = json.dumps(REGIME_TO_CODE)
        counts = {
            split: int(len(handle[split]["x"]))
            for split in ("train", "validation", "test")
        }

    summary = {
        "dataset_csv": str(dataset_csv),
        "manifest_csv": str(manifest_csv),
        "output_hdf5": str(output_hdf5),
        "split_csv": str(split_csv),
        "processed_simulations": processed_simulations,
        "window_counts": counts,
        "normalization_mean": mean.tolist(),
        "normalization_std": standard_deviation.tolist(),
    }
    summary_path = output_hdf5.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


class HDF5WindowDataset(Dataset):
    """Dataset PyTorch ouvrant le cache HDF5 a la demande par worker."""

    def __init__(self, hdf5_path: str | Path, split: str) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Partition inconnue : {split}")
        self.hdf5_path = str(Path(hdf5_path))
        self.split = split
        self._handle: h5py.File | None = None
        with h5py.File(self.hdf5_path, "r") as handle:
            self._length = len(handle[split]["x"])
            self.mean = np.asarray(
                handle.attrs["normalization_mean"], dtype=np.float32
            )
            self.std = np.asarray(
                handle.attrs["normalization_std"], dtype=np.float32
            )

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.hdf5_path, "r")
        return self._handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        group = self._file()[self.split]
        x = (group["x"][index] - self.mean) / self.std
        y = (group["y"][index] - self.mean) / self.std
        return {
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "regime": torch.tensor(
                int(group["regime"][index]), dtype=torch.long
            ),
            "simulation_id": torch.tensor(
                int(group["simulation_id"][index]), dtype=torch.long
            ),
        }

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        """Charge un batch entier avec une seule lecture HDF5 contigue."""
        if not indices:
            raise IndexError("Un batch vide ne peut pas etre charge.")
        index_array = np.asarray(indices, dtype=np.int64)
        if len(index_array) > 1 and not np.all(np.diff(index_array) == 1):
            raise ValueError(
                "HDF5WindowDataset exige des indices contigus par batch."
            )

        selector = slice(int(index_array[0]), int(index_array[-1]) + 1)
        group = self._file()[self.split]
        x = (group["x"][selector] - self.mean) / self.std
        y = (group["y"][selector] - self.mean) / self.std
        regimes = group["regime"][selector].astype(np.int64, copy=False)
        simulation_ids = group["simulation_id"][selector].astype(
            np.int64, copy=False
        )
        return {
            "x": torch.from_numpy(x.astype(np.float32, copy=False)),
            "y": torch.from_numpy(y.astype(np.float32, copy=False)),
            "regime": torch.from_numpy(regimes),
            "simulation_id": torch.from_numpy(simulation_ids),
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self) -> None:
        self.close()


class BlockShuffleBatchSampler(Sampler[list[int]]):
    """Melange les blocs sans rendre les lectures HDF5 aleatoires."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int = 42,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size doit etre positif ou nul.")
        if batch_size <= 0:
            raise ValueError("batch_size doit etre strictement positif.")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        starts = np.arange(0, self.dataset_size, self.batch_size)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(starts)
            self.epoch += 1
        for start in starts:
            end = min(int(start) + self.batch_size, self.dataset_size)
            yield list(range(int(start), end))

    def __len__(self) -> int:
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def _identity_collate(batch):
    """Le dataset HDF5 retourne deja le batch tensoriel."""
    return batch


def create_dataloaders(
    hdf5_path: str | Path,
    *,
    batch_size: int = 1024,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    shuffle_seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cree des DataLoaders avec lectures HDF5 contigues par batch."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    datasets = {
        split: HDF5WindowDataset(hdf5_path, split)
        for split in ("train", "validation", "test")
    }
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "collate_fn": _identity_collate,
    }
    train_loader = DataLoader(
        datasets["train"],
        batch_sampler=BlockShuffleBatchSampler(
            len(datasets["train"]),
            batch_size,
            shuffle=True,
            seed=shuffle_seed,
        ),
        **common,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_sampler=BlockShuffleBatchSampler(
            len(datasets["validation"]),
            batch_size,
            shuffle=False,
        ),
        **common,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_sampler=BlockShuffleBatchSampler(
            len(datasets["test"]),
            batch_size,
            shuffle=False,
        ),
        **common,
    )
    return train_loader, validation_loader, test_loader
