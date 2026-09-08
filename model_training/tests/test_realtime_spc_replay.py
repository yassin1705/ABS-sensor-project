"""Tests du rejeu causal MSP sans interface graphique."""

import unittest

import numpy as np

from model_training.realtime_spc_replay import CausalSPCReplayEngine


class LastValuePredictor:
    def predict(self, histories_mps: np.ndarray) -> np.ndarray:
        result = np.zeros((len(histories_mps), 5, 4), dtype=np.float32)
        result[:, 0, :] = histories_mps[:, -1, :]
        return result


def limits(lower: float = -0.5, upper: float = 0.5) -> dict:
    return {
        wheel: {
            "mean_mps": 0.0,
            "lcl_mps": lower,
            "ucl_mps": upper,
        }
        for wheel in ("FL", "FR", "RL", "RR")
    }


class CausalSPCReplayTest(unittest.TestCase):
    def test_first_residual_uses_the_previous_twenty_samples(self) -> None:
        engine = CausalSPCReplayEngine(LastValuePredictor(), limits())
        records = []
        for index in range(21):
            records.append(
                engine.process(
                    (index + 1) * 0.01,
                    np.full(4, index, dtype=np.float32),
                    np.ones(4, dtype=bool),
                )
            )
        self.assertEqual(records[19]["status_FL"], "warmup")
        self.assertEqual(records[20]["status_FL"], "warning")
        self.assertAlmostEqual(records[20]["predicted_FL_mps"], 19.0)
        self.assertAlmostEqual(records[20]["residual_FL_mps"], 1.0)

    def test_three_of_five_rule_confirms_alarm(self) -> None:
        engine = CausalSPCReplayEngine(LastValuePredictor(), limits())
        records = []
        for index in range(25):
            records.append(
                engine.process(
                    index * 0.01,
                    np.full(4, index, dtype=np.float32),
                    np.ones(4, dtype=bool),
                )
            )
        self.assertFalse(records[20]["alarm_FL"])
        self.assertFalse(records[21]["alarm_FL"])
        self.assertTrue(records[24]["alarm_FL"])

    def test_invalid_sample_suppresses_residual_and_future_prediction(self) -> None:
        engine = CausalSPCReplayEngine(LastValuePredictor(), limits())
        for index in range(20):
            engine.process(
                index * 0.01,
                np.full(4, index, dtype=np.float32),
                np.ones(4, dtype=bool),
            )
        invalid = engine.process(
            0.20,
            np.full(4, 20, dtype=np.float32),
            np.zeros(4, dtype=bool),
        )
        after_invalid = engine.process(
            0.21,
            np.full(4, 21, dtype=np.float32),
            np.ones(4, dtype=bool),
        )
        self.assertEqual(invalid["status_FL"], "invalid")
        self.assertTrue(np.isnan(invalid["residual_FL_mps"]))
        self.assertEqual(after_invalid["status_FL"], "warmup")

    def test_local_jump_isolates_one_suspected_wheel(self) -> None:
        engine = CausalSPCReplayEngine(LastValuePredictor(), limits())
        speeds = np.zeros(4, dtype=np.float32)
        for index in range(20):
            speeds += 1
            engine.process(
                index * 0.01, speeds.copy(), np.ones(4, dtype=bool)
            )
        speeds += np.asarray([5, 1, 1, 1], dtype=np.float32)
        first = engine.process(0.20, speeds.copy(), np.ones(4, dtype=bool))
        speeds += np.asarray([5, 1, 1, 1], dtype=np.float32)
        second = engine.process(0.21, speeds.copy(), np.ones(4, dtype=bool))
        self.assertIsNone(first["suspected_wheel"])
        self.assertEqual(second["suspected_wheel"], "FL")
        self.assertGreater(
            second["local_disruption_FL_mps"],
            second["local_disruption_FR_mps"],
        )


if __name__ == "__main__":
    unittest.main()
