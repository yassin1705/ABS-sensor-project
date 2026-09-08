"""GRU multivariee pour la prediction des quatre vitesses ECU."""

import torch
from torch import nn

from .base import BaseWheelSpeedForecaster


class GRUForecaster(BaseWheelSpeedForecaster):
    """Encode l'historique par une GRU puis predit tout l'horizon."""

    def __init__(
        self,
        history_length: int = 20,
        horizon: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__(history_length=history_length, horizon=horizon)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=self.wheel_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, horizon * self.wheel_count),
        )

    @property
    def model_name(self) -> str:
        return "gru"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        sequence, _ = self.encoder(x)
        return self.reshape_output(self.regression_head(sequence[:, -1, :]))
