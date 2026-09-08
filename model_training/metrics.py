"""Metriques communes a toutes les architectures."""

from __future__ import annotations

import numpy as np
import torch

from .data import CODE_TO_REGIME, WHEELS


class RegressionMetricAccumulator:
    """Accumule MAE/RMSE globales, par roue, horizon et regime."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, horizon: int) -> None:
        self.mean = torch.as_tensor(mean, dtype=torch.float64).reshape(1, 1, 4)
        self.std = torch.as_tensor(std, dtype=torch.float64).reshape(1, 1, 4)
        self.horizon = horizon
        self.absolute_sum = 0.0
        self.square_sum = 0.0
        self.count = 0
        self.wheel_absolute_sum = torch.zeros(4, dtype=torch.float64)
        self.wheel_square_sum = torch.zeros(4, dtype=torch.float64)
        self.wheel_count = torch.zeros(4, dtype=torch.float64)
        self.horizon_absolute_sum = torch.zeros(horizon, dtype=torch.float64)
        self.horizon_square_sum = torch.zeros(horizon, dtype=torch.float64)
        self.horizon_count = torch.zeros(horizon, dtype=torch.float64)
        self.regime_sums: dict[int, list[float]] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        regimes: torch.Tensor,
    ) -> None:
        prediction = prediction.detach().cpu().to(torch.float64)
        target = target.detach().cpu().to(torch.float64)
        regimes = regimes.detach().cpu()
        error = (prediction - target) * self.std
        absolute = error.abs()
        squared = error.square()

        self.absolute_sum += absolute.sum().item()
        self.square_sum += squared.sum().item()
        self.count += absolute.numel()
        self.wheel_absolute_sum += absolute.sum(dim=(0, 1))
        self.wheel_square_sum += squared.sum(dim=(0, 1))
        self.wheel_count += prediction.shape[0] * prediction.shape[1]
        self.horizon_absolute_sum += absolute.sum(dim=(0, 2))
        self.horizon_square_sum += squared.sum(dim=(0, 2))
        self.horizon_count += prediction.shape[0] * prediction.shape[2]

        for regime in regimes.unique():
            code = int(regime.item())
            selected = regimes == regime
            regime_absolute = absolute[selected]
            regime_squared = squared[selected]
            values = self.regime_sums.setdefault(code, [0.0, 0.0, 0.0])
            values[0] += regime_absolute.sum().item()
            values[1] += regime_squared.sum().item()
            values[2] += regime_absolute.numel()

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            return {"mae": float("nan"), "rmse": float("nan")}

        result = {
            "mae": self.absolute_sum / self.count,
            "rmse": float(np.sqrt(self.square_sum / self.count)),
        }
        for index, wheel in enumerate(WHEELS):
            result[f"mae_{wheel}"] = (
                self.wheel_absolute_sum[index] / self.wheel_count[index]
            ).item()
            result[f"rmse_{wheel}"] = torch.sqrt(
                self.wheel_square_sum[index] / self.wheel_count[index]
            ).item()

        for index in range(self.horizon):
            result[f"mae_t+{index + 1}"] = (
                self.horizon_absolute_sum[index] / self.horizon_count[index]
            ).item()
            result[f"rmse_t+{index + 1}"] = torch.sqrt(
                self.horizon_square_sum[index] / self.horizon_count[index]
            ).item()

        for code, (absolute_sum, square_sum, count) in self.regime_sums.items():
            name = CODE_TO_REGIME.get(code, f"unknown_{code}")
            result[f"mae_regime_{name}"] = absolute_sum / count
            result[f"rmse_regime_{name}"] = float(
                np.sqrt(square_sum / count)
            )
        return result
