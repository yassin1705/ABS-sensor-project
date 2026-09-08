"""Tests cibles du calibrage MSP sans lancer l'entrainement."""

import unittest

import numpy as np
import pandas as pd

from model_training.data import SPEED_COLUMNS, VALID_COLUMNS
from model_training.spc_calibration import (
    calculate_control_limits,
    extract_t_plus_one_residuals,
)


class LastValuePredictor:
    def predict(self, histories_mps: np.ndarray) -> np.ndarray:
        result = np.zeros((len(histories_mps), 5, 4), dtype=np.float32)
        result[:, 0, :] = histories_mps[:, -1, :]
        return result


class SPCCalibrationTest(unittest.TestCase):
    def make_simulation(self) -> pd.DataFrame:
        rows = 25
        frame = pd.DataFrame(
            {
                "simulation_id": np.ones(rows, dtype=int),
                "requested_phenomenon": ["normal_braking"] * rows,
                "observed_phenomenon": ["normal_braking"] * rows,
                "time_s": np.arange(1, rows + 1) * 0.01,
                "brake": np.full(rows, 0.2),
                "vehicle_speed_mps": np.linspace(20, 19, rows),
            }
        )
        for wheel_index, column in enumerate(SPEED_COLUMNS):
            frame[column] = np.arange(rows) + wheel_index
        for column in VALID_COLUMNS:
            frame[column] = True
        return frame

    def test_t_plus_one_alignment_and_sign(self) -> None:
        residuals, skipped = extract_t_plus_one_residuals(
            self.make_simulation(), LastValuePredictor()
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(len(residuals), 5)
        for wheel in ("FL", "FR", "RL", "RR"):
            np.testing.assert_allclose(
                residuals[f"residual_{wheel}_mps"], 1.0
            )
        self.assertAlmostEqual(residuals["time_s"].iloc[0], 0.21)

    def test_invalid_history_is_skipped(self) -> None:
        frame = self.make_simulation()
        frame.loc[10, "ecu_valid_FL"] = False
        residuals, skipped = extract_t_plus_one_residuals(
            frame, LastValuePredictor()
        )
        self.assertEqual(len(residuals), 0)
        self.assertEqual(skipped, 5)

    def test_classical_control_limits(self) -> None:
        residuals = pd.DataFrame(
            {
                f"residual_{wheel}_mps": [-1.0, 0.0, 1.0]
                for wheel in ("FL", "FR", "RL", "RR")
            }
        )
        limits = calculate_control_limits(residuals)
        self.assertAlmostEqual(limits["FL"]["mean_mps"], 0.0)
        self.assertAlmostEqual(limits["FL"]["std_mps"], 1.0)
        self.assertAlmostEqual(limits["FL"]["lcl_mps"], -3.0)
        self.assertAlmostEqual(limits["FL"]["ucl_mps"], 3.0)


if __name__ == "__main__":
    unittest.main()
