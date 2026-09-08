"""Modele hybride : extraction locale CNN puis contexte temporel GRU."""

import torch
from torch import nn

from .base import BaseFaultParameterEstimator


class CNNGRUFaultParameterEstimator(BaseFaultParameterEstimator):
    """Combine les signatures locales de perte de fronts et leur chronologie."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.local_features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=96,
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
        return "cnn_gru"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        local_sequence = self.local_features(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.temporal_encoder(local_sequence)
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(encoded)
