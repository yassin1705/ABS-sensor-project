"""CNN 1D pour l'estimation des parametres de defaut ABS."""

import torch
from torch import nn

from .base import BaseFaultParameterEstimator


class CNNFaultParameterEstimator(BaseFaultParameterEstimator):
    def __init__(self, dropout: float = 0.15) -> None:
        super().__init__()
        self.features = nn.Sequential(
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
            nn.AdaptiveAvgPool1d(25),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 25, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.output_count),
            nn.Sigmoid(),
        )

    @property
    def model_name(self) -> str:
        return "cnn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.head(self.features(x.transpose(1, 2)))
