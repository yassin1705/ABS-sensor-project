"""Decision-level fusion for wheel fault isolation.

The multivariate GRU/SPC remains the global anomaly detector. The independent
CNN/GRU result for each wheel decides whether an SPC alarm is a primary sensor
fault or only a cross-wheel effect.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from model_training.data import WHEELS


@dataclass(frozen=True)
class IsolationSnapshot:
    candidate_wheel: str | None
    primary_wheel: str | None
    ambiguous_wheels: tuple[str, ...]
    probabilities: dict[str, float]
    probability_margin: float | None
    candidate_votes: int
    votes_required: int
    threshold: float
    margin_required: float
    model_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WheelIsolationTracker:
    """Stabilize the independent-wheel winner across rolling evaluations."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        margin_required: float = 0.10,
        persistence_window: int = 3,
        votes_required: int = 2,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if margin_required < 0.0:
            raise ValueError("margin_required must be non-negative.")
        if not 1 <= votes_required <= persistence_window:
            raise ValueError("votes_required must fit inside persistence_window.")
        self.threshold = threshold
        self.margin_required = margin_required
        self.votes_required = votes_required
        self.history: deque[str | None] = deque(maxlen=persistence_window)
        self.snapshot = IsolationSnapshot(
            candidate_wheel=None,
            primary_wheel=None,
            ambiguous_wheels=(),
            probabilities={},
            probability_margin=None,
            candidate_votes=0,
            votes_required=votes_required,
            threshold=threshold,
            margin_required=margin_required,
            model_ready=False,
        )

    @staticmethod
    def _usable(result: Mapping[str, Any] | None) -> bool:
        return bool(result) and not result.get("state")

    def update(
        self,
        model_results: Mapping[str, Mapping[str, Any]],
    ) -> IsolationSnapshot:
        usable = {
            wheel: result
            for wheel, result in model_results.items()
            if wheel in WHEELS and self._usable(result)
        }
        probabilities = {
            wheel: float(result["fault_probability"])
            for wheel, result in usable.items()
        }
        model_ready = len(probabilities) == len(WHEELS)
        candidate: str | None = None
        ambiguous: tuple[str, ...] = ()
        margin: float | None = None

        if model_ready:
            ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            top_wheel, top_probability = ranked[0]
            second_probability = ranked[1][1]
            margin = top_probability - second_probability
            above_threshold = tuple(
                wheel for wheel, probability in ranked if probability >= self.threshold
            )
            if top_probability >= self.threshold:
                if len(above_threshold) > 1 and margin < self.margin_required:
                    ambiguous = above_threshold
                else:
                    candidate = top_wheel

        self.history.append(candidate)
        votes = Counter(self.history)
        candidate_votes = votes[candidate] if candidate is not None else 0
        primary = (
            candidate
            if candidate is not None and candidate_votes >= self.votes_required
            else None
        )
        self.snapshot = IsolationSnapshot(
            candidate_wheel=candidate,
            primary_wheel=primary,
            ambiguous_wheels=ambiguous,
            probabilities=probabilities,
            probability_margin=margin,
            candidate_votes=candidate_votes,
            votes_required=self.votes_required,
            threshold=self.threshold,
            margin_required=self.margin_required,
            model_ready=model_ready,
        )
        return self.snapshot


def classify_wheel(
    wheel: str,
    spc_summary: Mapping[str, Any],
    model_result: Mapping[str, Any],
    isolation: IsolationSnapshot,
) -> dict[str, Any]:
    """Return an explainable final state for one wheel."""
    probability = float(model_result.get("fault_probability", 0.0))
    model_state = model_result.get("state")
    spc_alarm = int(spc_summary.get("alarm_sample_count", 0)) > 0
    localized = int(spc_summary.get("localized_suspect_sample_count", 0)) > 0

    if model_state == "insufficient_valid_signal":
        state = "INSUFFICIENT_SIGNAL"
        reason = "The independent model has no usable five-second wheel signal."
    elif model_state:
        state = "COLLECTING"
        reason = "The independent model is collecting its first 500 samples."
    elif wheel in isolation.ambiguous_wheels:
        state = "AMBIGUOUS"
        reason = "Multiple independent wheel probabilities are high without enough margin."
    elif isolation.candidate_wheel == wheel:
        if isolation.primary_wheel == wheel and (spc_alarm or localized):
            state = "CONFIRMED_FAULTY"
            reason = "Independent probability is persistent and SPC provides supporting evidence."
        else:
            state = "SUSPECTED_FAULTY"
            reason = "Independent probability is highest; persistence or SPC confirmation is pending."
    elif probability >= isolation.threshold:
        state = "AMBIGUOUS"
        reason = "This wheel is independently suspicious but is not the clear primary candidate."
    elif spc_alarm and isolation.primary_wheel is not None:
        state = "CROSS_EFFECT"
        reason = f"SPC alarm is explained by the primary {isolation.primary_wheel} sensor fault."
    elif spc_alarm or localized:
        state = "WARNING"
        reason = "SPC detected an anomaly, but no independent primary fault is established."
    else:
        state = "HEALTHY"
        reason = "Neither SPC nor the independent model indicates a sensor fault."

    return {
        "state": state,
        "reason": reason,
        "primary_wheel": isolation.primary_wheel,
        "candidate_wheel": isolation.candidate_wheel,
        "independent_probability": probability,
        "probability_margin": isolation.probability_margin,
        "candidate_votes": isolation.candidate_votes,
        "votes_required": isolation.votes_required,
        "spc_alarm_support": spc_alarm,
        "localization_support": localized,
    }
