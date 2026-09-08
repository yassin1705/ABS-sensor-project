"""Tests cibles du pipeline d'estimation des parametres de defaut."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model_training.fault_parameter_estimation import (
    EXPECTED_SAMPLE_COUNT,
    physical_parameters,
    prepare_dataset,
    select_manifest,
    targets_from_manifest,
)
from model_training.models import FaultParameterGRU


class FaultParameterEstimationTest(unittest.TestCase):
    def test_target_conversion_and_physical_parameters(self) -> None:
        row = pd.Series(
            {
                "simulation_id": 7,
                "fault_start_s": 1.5,
                "fault_end_s": 3.5,
                "fault_severity": 0.75,
            }
        )
        target = targets_from_manifest(row)
        np.testing.assert_allclose(target, [0.3, 0.4, 0.75])
        decoded = physical_parameters(target)
        self.assertAlmostEqual(decoded["fault_start_s"], 1.5, places=6)
        self.assertAlmostEqual(decoded["fault_end_s"], 3.5, places=6)
        self.assertAlmostEqual(
            decoded["effective_dropout_probability"], 0.375, places=6
        )

    def test_model_contract(self) -> None:
        model = FaultParameterGRU(hidden_size=8, num_layers=1)
        output = model(torch.zeros(2, EXPECTED_SAMPLE_COUNT, 2))
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))
        with self.assertRaises(ValueError):
            model(torch.zeros(2, EXPECTED_SAMPLE_COUNT, 1))

    def test_preparation_uses_only_the_faulty_sensor_and_disjoint_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_rows = []
            signal_rows = []
            for simulation_id in range(1, 9):
                wheel = ("FL", "FR", "RL", "RR")[(simulation_id - 1) % 4]
                manifest_rows.append(
                    {
                        "simulation_id": simulation_id,
                        "requested_phenomenon": "normal_braking",
                        "fault_wheel": wheel,
                        "fault_type": "intermittent_loss",
                        "fault_start_s": 1.5,
                        "fault_end_s": 3.0,
                        "fault_severity": 0.25 if simulation_id <= 4 else 0.5,
                        "dropout_probability": 0.5,
                        "dropped_edge_count": 10,
                    }
                )
                for sample in range(EXPECTED_SAMPLE_COUNT):
                    row = {
                        "simulation_id": simulation_id,
                        "time_s": (sample + 1) / 100,
                    }
                    for candidate_index, candidate in enumerate(("FL", "FR", "RL", "RR")):
                        row[f"wheel_speed_ecu_{candidate}_mps"] = (
                            10 * simulation_id + candidate_index
                        )
                        row[f"ecu_valid_{candidate}"] = 1
                    signal_rows.append(row)
            manifest_path = root / "manifest.csv"
            dataset_path = root / "dataset.csv"
            cache_path = root / "cache.npz"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            pd.DataFrame(signal_rows).to_csv(dataset_path, index=False)

            selected = select_manifest(manifest_path, split_seed=4)
            self.assertEqual(set(selected["dataset_split"]), {"train", "validation", "test"})
            prepare_dataset(
                dataset_path, manifest_path, cache_path, split_seed=4, chunksize=733
            )
            with np.load(cache_path) as cache:
                self.assertEqual(cache["x"].shape, (8, EXPECTED_SAMPLE_COUNT, 2))
                self.assertEqual(len(np.unique(cache["simulation_id"])), 8)
                self.assertTrue(np.all(cache["x"][:, :, 1] == 1))


if __name__ == "__main__":
    unittest.main()
