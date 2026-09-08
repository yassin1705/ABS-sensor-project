"""GRU hors ligne pour estimer les parametres d'un defaut sur un capteur ABS."""

from __future__ import annotations

import torch
from torch import nn


class FaultParameterGRU(nn.Module):
    """Convertit [vitesse ECU, validite] sur 5 s en trois parametres normalises.

    Les sorties sigmoid sont, dans l'ordre : debut / duree de simulation,
    duree du defaut / duree de simulation, et severite.
    """

    output_names = ("start_fraction", "duration_fraction", "severity")

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != 2:
            raise ValueError(
                "FaultParameterGRU attend [batch, temps, 2] "
                "(vitesse normalisee, validite)."
            )
        _, hidden = self.encoder(x)
        # Derniere couche, directions avant et arriere.
        encoded = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(encoded)
