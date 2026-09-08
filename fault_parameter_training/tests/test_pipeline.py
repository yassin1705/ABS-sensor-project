"""Tests cibles du benchmark mixte sain/fautif."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fault_parameter_training.config import EXPECTED_SAMPLE_COUNT
from fault_parameter_training.data import (
    FaultParameterDataset,
    encode_faulty_targets,
    prepare_npz_dataset,
)
from fault_parameter_training.evaluation import DetectionParameterLoss
from fault_parameter_training.models import (
    CNNFaultParameterEstimator,
    CNNGRUFaultParameterEstimator,
    CNNGRUFusionEstimator,
    GRUFaultParameterEstimator,
    LSTMFaultParameterEstimator,
)


class FaultParameterPipelineTest(unittest.TestCase):
    def test_all_models_detect_and_estimate_parameters(self) -> None:
        x = torch.zeros(2, EXPECTED_SAMPLE_COUNT, 2)
        for model in (
            CNNFaultParameterEstimator(),
            CNNGRUFaultParameterEstimator(hidden_size=8, num_layers=1),
            GRUFaultParameterEstimator(hidden_size=8, num_layers=1),
            LSTMFaultParameterEstimator(hidden_size=8, num_layers=1),
        ):
            output = model(x)
            self.assertEqual(tuple(output.shape), (2, 4))
            self.assertTrue(torch.all((output >= 0) & (output <= 1)))

    def test_target_encoding(self) -> None:
        target = encode_faulty_targets(
            pd.Series(
                {
                    "simulation_id": 12,
                    "fault_start_s": 1.75,
                    "fault_end_s": 3.50,
                    "fault_severity": 0.75,
                }
            )
        )
        np.testing.assert_allclose(target, [1.0, 0.35, 0.35, 0.75])

    def test_fusion_uses_cnn_timing_and_gru_severity(self) -> None:
        class FixedEstimator(torch.nn.Module):
            def __init__(self, values) -> None:
                super().__init__()
                self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

            def forward(self, x):
                return self.values.unsqueeze(0).repeat(len(x), 1)

        cnn = FixedEstimator([0.8, 0.30, 0.40, 0.60])
        gru = FixedEstimator([0.6, 0.35, 0.45, 0.75])
        fusion = CNNGRUFusionEstimator(cnn, gru)
        output = fusion(torch.zeros(2, EXPECTED_SAMPLE_COUNT, 2))
        expected = torch.tensor([0.7, 0.30, 0.40, 0.75]).repeat(2, 1)
        torch.testing.assert_close(output, expected)

    def test_healthy_parameter_targets_are_masked(self) -> None:
        loss = DetectionParameterLoss(positive_weight=2.0)
        prediction = torch.tensor([[0.2, 0.1, 0.2, 0.3]])
        healthy_a = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        healthy_b = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        self.assertAlmostEqual(
            loss(prediction, healthy_a).item(),
            loss(prediction, healthy_b).item(),
        )

    def test_mixed_preparation_and_split_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            faulty_manifest_rows = []
            healthy_manifest_rows = []
            dataset_rows = []
            wheel_order = ("FL", "FR", "RL", "RR")
            for simulation_id in range(1, 9):
                fault_wheel = wheel_order[(simulation_id - 1) % 4]
                faulty_manifest_rows.append(
                    {
                        "simulation_id": simulation_id,
                        "requested_phenomenon": "normal_braking",
                        "fault_wheel": fault_wheel,
                        "fault_type": "intermittent_loss",
                        "fault_start_s": 1.5,
                        "fault_end_s": 3.25,
                        "fault_severity": 0.25 if simulation_id <= 4 else 0.5,
                        "dropout_probability": 0.5,
                        "dropped_edge_count": 20,
                    }
                )
                healthy_manifest_rows.append(
                    {
                        "simulation_id": simulation_id,
                        "requested_phenomenon": "normal_braking",
                    }
                )
                for sample in range(EXPECTED_SAMPLE_COUNT):
                    row = {"simulation_id": simulation_id, "time_s": (sample + 1) / 100}
                    for wheel_index, wheel in enumerate(wheel_order):
                        row[f"wheel_speed_ecu_{wheel}_mps"] = simulation_id + wheel_index
                        row[f"ecu_valid_{wheel}"] = 1
                    dataset_rows.append(row)
            faulty_manifest_path = root / "faulty_manifest.csv"
            healthy_manifest_path = root / "healthy_manifest.csv"
            faulty_dataset_path = root / "faulty_dataset.csv"
            healthy_dataset_path = root / "healthy_dataset.csv"
            cache_path = root / "dataset.npz"
            pd.DataFrame(faulty_manifest_rows).to_csv(faulty_manifest_path, index=False)
            pd.DataFrame(healthy_manifest_rows).to_csv(healthy_manifest_path, index=False)
            frame = pd.DataFrame(dataset_rows)
            frame.to_csv(faulty_dataset_path, index=False)
            frame.to_csv(healthy_dataset_path, index=False)
            summary = prepare_npz_dataset(
                faulty_dataset_path,
                faulty_manifest_path,
                healthy_dataset_path,
                healthy_manifest_path,
                cache_path,
                split_seed=9,
                csv_chunk_rows=731,
            )
            self.assertEqual(summary["records"], 40)
            self.assertEqual(summary["faulty_records"], 8)
            self.assertEqual(summary["healthy_records"], 32)
            split_keys = {}
            for split in ("train", "validation", "test"):
                dataset = FaultParameterDataset(cache_path, split)
                split_keys[split] = set(
                    zip(dataset.sources.tolist(), dataset.simulation_ids.tolist())
                )
            self.assertFalse(split_keys["train"] & split_keys["validation"])
            self.assertFalse(split_keys["train"] & split_keys["test"])
            self.assertFalse(split_keys["validation"] & split_keys["test"])


if __name__ == "__main__":
    unittest.main()
