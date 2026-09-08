"""Evaluation multi-tache : detection puis parametres conditionnels."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import BASE_DROPOUT_PROBABILITY, SIMULATION_DURATION_S


class DetectionParameterLoss(nn.Module):
    """BCE ponderee + regression masquee sur les seuls capteurs fautifs."""

    def __init__(self, positive_weight: float = 1.0, regression_weight: float = 1.0) -> None:
        super().__init__()
        self.positive_weight = float(positive_weight)
        self.regression_weight = float(regression_weight)
        self.parameter_loss = nn.SmoothL1Loss(beta=0.05)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = prediction[:, 0].clamp(1e-6, 1 - 1e-6)
        label = target[:, 0]
        classification = -(
            self.positive_weight * label * torch.log(probability)
            + (1 - label) * torch.log(1 - probability)
        ).mean()
        positive = label > 0.5
        if positive.any():
            regression = self.parameter_loss(prediction[positive, 1:], target[positive, 1:])
        else:
            regression = prediction[:, 1:].sum() * 0.0
        return classification + self.regression_weight * regression


def diagnostic_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    probability = prediction[:, 0]
    true_fault = target[:, 0] >= 0.5
    predicted_fault = probability >= 0.5
    true_positive = int(np.sum(predicted_fault & true_fault))
    false_positive = int(np.sum(predicted_fault & ~true_fault))
    true_negative = int(np.sum(~predicted_fault & ~true_fault))
    false_negative = int(np.sum(~predicted_fault & true_fault))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    metrics = {
        "accuracy": (true_positive + true_negative) / len(target),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
        "false_positive_rate": false_positive / max(1, false_positive + true_negative),
        "fault_probability_mae": float(np.abs(probability - target[:, 0]).mean()),
        "faulty_samples": float(true_fault.sum()),
        "healthy_samples": float((~true_fault).sum()),
    }
    if true_fault.any():
        error = np.abs(prediction[true_fault, 1:] - target[true_fault, 1:])
        metrics.update(
            {
                "start_mae_s": float(error[:, 0].mean() * SIMULATION_DURATION_S),
                "duration_mae_s": float(error[:, 1].mean() * SIMULATION_DURATION_S),
                "severity_mae": float(error[:, 2].mean()),
                "effective_dropout_mae": float(
                    error[:, 2].mean() * BASE_DROPOUT_PROBABILITY
                ),
            }
        )
    else:
        metrics.update(
            {
                "start_mae_s": float("nan"),
                "duration_mae_s": float("nan"),
                "severity_mae": float("nan"),
                "effective_dropout_mae": float("nan"),
            }
        )
    return metrics


class FaultParameterEvaluator:
    def __init__(self, loss_function: nn.Module, device: torch.device) -> None:
        self.loss_function = loss_function
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: Iterable,
        *,
        return_predictions: bool = False,
    ) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
        model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        simulation_ids: list[np.ndarray] = []
        wheels: list[np.ndarray] = []
        sources: list[np.ndarray] = []
        loss_sum = 0.0
        sample_count = 0
        for batch in dataloader:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            prediction = model(x)
            loss_sum += self.loss_function(prediction, y).item() * len(x)
            sample_count += len(x)
            predictions.append(prediction.cpu().numpy())
            targets.append(y.cpu().numpy())
            simulation_ids.append(batch["simulation_id"].numpy())
            wheels.append(batch["wheel"].numpy())
            sources.append(batch["source"].numpy())
        prediction_array = np.concatenate(predictions)
        target_array = np.concatenate(targets)
        metrics = {
            "loss": loss_sum / sample_count,
            "samples": float(sample_count),
            **diagnostic_metrics(prediction_array, target_array),
        }
        if not return_predictions:
            return metrics
        frame = pd.DataFrame(
            {
                "source_code": np.concatenate(sources),
                "simulation_id": np.concatenate(simulation_ids),
                "wheel_code": np.concatenate(wheels),
                "true_fault": target_array[:, 0],
                "predicted_fault_probability": prediction_array[:, 0],
                "predicted_fault": prediction_array[:, 0] >= 0.5,
                "true_start_s": target_array[:, 1] * SIMULATION_DURATION_S,
                "predicted_start_s": prediction_array[:, 1] * SIMULATION_DURATION_S,
                "true_duration_s": target_array[:, 2] * SIMULATION_DURATION_S,
                "predicted_duration_s": prediction_array[:, 2] * SIMULATION_DURATION_S,
                "true_severity": target_array[:, 3],
                "predicted_severity": prediction_array[:, 3],
            }
        )
        frame["true_end_s"] = frame["true_start_s"] + frame["true_duration_s"]
        frame["predicted_end_s"] = (
            frame["predicted_start_s"] + frame["predicted_duration_s"]
        ).clip(upper=SIMULATION_DURATION_S)
        return metrics, frame
