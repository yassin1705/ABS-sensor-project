"""Contrat commun aux modeles de prediction de vitesse de roue."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseWheelSpeedForecaster(nn.Module, ABC):
    """Base des modeles qui predisent les quatre roues sur plusieurs pas."""

    wheel_count = 4

    def __init__(self, history_length: int = 20, horizon: int = 5) -> None:
        super().__init__()
        if history_length <= 0:
            raise ValueError("history_length doit etre strictement positif.")
        if horizon <= 0:
            raise ValueError("horizon doit etre strictement positif.")
        self.history_length = history_length
        self.horizon = horizon

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nom stable utilise pour les sorties d'experience."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne un tenseur [batch, horizon, 4]."""

    def validate_input(self, x: torch.Tensor) -> None:
        expected = (self.history_length, self.wheel_count)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "Entree attendue [batch, "
                f"{self.history_length}, {self.wheel_count}], "
                f"forme recue {tuple(x.shape)}."
            )

    def reshape_output(self, values: torch.Tensor) -> torch.Tensor:
        return values.reshape(-1, self.horizon, self.wheel_count)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
