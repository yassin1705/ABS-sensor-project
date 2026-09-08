"""Evaluation centralisee des modeles predictifs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn

from .metrics import RegressionMetricAccumulator


class ForecastEvaluator:
    """Calcule la meme loss et les memes metriques pour chaque modele."""

    def __init__(
        self,
        loss_function: nn.Module,
        normalization_mean: np.ndarray,
        normalization_std: np.ndarray,
        horizon: int,
        device: torch.device,
    ) -> None:
        self.loss_function = loss_function
        self.mean = normalization_mean
        self.std = normalization_std
        self.horizon = horizon
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: Iterable,
        *,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        model.eval()
        accumulator = RegressionMetricAccumulator(
            self.mean, self.std, self.horizon
        )
        loss_sum = 0.0
        sample_count = 0

        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            prediction = model(x)
            loss = self.loss_function(prediction, y)
            batch_size = x.shape[0]
            loss_sum += loss.item() * batch_size
            sample_count += batch_size
            accumulator.update(prediction, y, batch["regime"])

        metrics = accumulator.compute()
        metrics["loss"] = (
            loss_sum / sample_count if sample_count else float("nan")
        )
        metrics["samples"] = float(sample_count)
        return metrics
