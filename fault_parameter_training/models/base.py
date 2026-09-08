"""Contrat commun des estimateurs de parametres de defaut."""

from __future__ import annotations

import torch
from torch import nn


class BaseFaultParameterEstimator(nn.Module):
    input_channels = 2
    output_count = 4
    output_names = (
        "fault_probability",
        "start_fraction",
        "duration_fraction",
        "severity",
    )

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def validate_input(self, x: torch.Tensor) -> None:
        if torch.jit.is_tracing():
            return
        if x.ndim != 3 or x.shape[-1] != self.input_channels:
            raise ValueError(
                "Entree attendue : [batch, temps, 2] "
                "(vitesse ECU normalisee, validite)."
            )

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
