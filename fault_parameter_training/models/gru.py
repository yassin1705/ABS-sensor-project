"""GRU bidirectionnelle pour l'estimation des parametres de defaut ABS."""

import torch
from torch import nn

from .base import BaseFaultParameterEstimator


class GRUFaultParameterEstimator(BaseFaultParameterEstimator):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_count),
            nn.Sigmoid(),
        )

    @property
    def model_name(self) -> str:
        return "gru"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        _, hidden = self.encoder(x)
        return self.head(torch.cat((hidden[-2], hidden[-1]), dim=1))
