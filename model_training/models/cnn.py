"""CNN 1D temporel pour la prediction multiroue."""

import torch
from torch import nn

from .base import BaseWheelSpeedForecaster


class CNNForecaster(BaseWheelSpeedForecaster):
    """CNN compact dont l'architecture est definie dans la classe."""

    def __init__(
        self,
        history_length: int = 20,
        horizon: int = 5,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(history_length=history_length, horizon=horizon)

        self.encoder = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * history_length, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, horizon * self.wheel_count),
        )

    @property
    def model_name(self) -> str:
        return "cnn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        # Le DataLoader utilise [B, temps, roues], Conv1D [B, canaux, temps].
        encoded = self.encoder(x.transpose(1, 2))
        return self.reshape_output(self.regression_head(encoded))
