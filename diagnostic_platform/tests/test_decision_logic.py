from __future__ import annotations

import unittest

from diagnostic_platform.decision_logic import WheelIsolationTracker, classify_wheel


def result(probability: float) -> dict[str, object]:
    return {"fault_probability": probability, "fault_detected": probability >= 0.5}


def spc(*, alarm: bool = False, localized: bool = False) -> dict[str, int]:
    return {
        "alarm_sample_count": int(alarm),
        "localized_suspect_sample_count": int(localized),
    }


class DecisionLogicTests(unittest.TestCase):
    def test_primary_fault_turns_other_spc_alarms_into_cross_effects(self) -> None:
        tracker = WheelIsolationTracker()
        models = {
            "FL": result(0.91),
            "FR": result(0.08),
            "RL": result(0.06),
            "RR": result(0.04),
        }
        tracker.update(models)
        snapshot = tracker.update(models)

        self.assertEqual(
            classify_wheel("FL", spc(alarm=True, localized=True), models["FL"], snapshot)["state"],
            "CONFIRMED_FAULTY",
        )
        self.assertEqual(
            classify_wheel("FR", spc(alarm=True), models["FR"], snapshot)["state"],
            "CROSS_EFFECT",
        )

    def test_close_independent_probabilities_are_ambiguous(self) -> None:
        tracker = WheelIsolationTracker()
        models = {
            "FL": result(0.78),
            "FR": result(0.73),
            "RL": result(0.04),
            "RR": result(0.03),
        }
        snapshot = tracker.update(models)
        self.assertEqual(snapshot.ambiguous_wheels, ("FL", "FR"))
        self.assertEqual(
            classify_wheel("FL", spc(alarm=True), models["FL"], snapshot)["state"],
            "AMBIGUOUS",
        )

    def test_spc_without_independent_primary_remains_warning(self) -> None:
        tracker = WheelIsolationTracker()
        models = {wheel: result(0.05) for wheel in ("FL", "FR", "RL", "RR")}
        snapshot = tracker.update(models)
        self.assertEqual(
            classify_wheel("RL", spc(alarm=True), models["RL"], snapshot)["state"],
            "WARNING",
        )


if __name__ == "__main__":
    unittest.main()
