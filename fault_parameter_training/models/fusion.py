"""Fusion tardive des estimateurs CNN et GRU deja entraines."""

from __future__ import annotations

from pathlib import Path

import torch

from .base import BaseFaultParameterEstimator
from .cnn import CNNFaultParameterEstimator
from .gru import GRUFaultParameterEstimator


class CNNGRUFusionEstimator(BaseFaultParameterEstimator):
    """Combine les sorties selon les forces observees en validation.

    Par defaut, la probabilite de defaut est la moyenne CNN/GRU, le debut et
    la duree viennent du CNN, et la severite vient de la GRU. Les poids restent
    configurables sans reentrainer les deux modeles sources.
    """

    def __init__(
        self,
        cnn: CNNFaultParameterEstimator,
        gru: GRUFaultParameterEstimator,
        *,
        cnn_probability_weight: float = 0.5,
        cnn_timing_weight: float = 1.0,
        gru_severity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("cnn_probability_weight", cnn_probability_weight),
            ("cnn_timing_weight", cnn_timing_weight),
            ("gru_severity_weight", gru_severity_weight),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} doit appartenir a [0, 1].")
        self.cnn = cnn
        self.gru = gru
        self.cnn_probability_weight = float(cnn_probability_weight)
        self.cnn_timing_weight = float(cnn_timing_weight)
        self.gru_severity_weight = float(gru_severity_weight)

    @property
    def model_name(self) -> str:
        return "cnn_gru_fusion"

    def forward_components(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate_input(x)
        cnn_output = self.cnn(x)
        gru_output = self.gru(x)
        fault_probability = (
            self.cnn_probability_weight * cnn_output[:, 0]
            + (1 - self.cnn_probability_weight) * gru_output[:, 0]
        )
        start = (
            self.cnn_timing_weight * cnn_output[:, 1]
            + (1 - self.cnn_timing_weight) * gru_output[:, 1]
        )
        duration = (
            self.cnn_timing_weight * cnn_output[:, 2]
            + (1 - self.cnn_timing_weight) * gru_output[:, 2]
        )
        severity = (
            self.gru_severity_weight * gru_output[:, 3]
            + (1 - self.gru_severity_weight) * cnn_output[:, 3]
        )
        fused = torch.stack((fault_probability, start, duration, severity), dim=1)
        return fused, cnn_output, gru_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused, _, _ = self.forward_components(x)
        return fused

    @classmethod
    def from_experiment_directory(
        cls,
        experiment_directory: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        **fusion_weights: float,
    ) -> "CNNGRUFusionEstimator":
        experiment_directory = Path(experiment_directory)
        cnn = CNNFaultParameterEstimator()
        gru = GRUFaultParameterEstimator()
        for model, name in ((cnn, "cnn"), (gru, "gru")):
            checkpoint_path = experiment_directory / name / "best_model.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint absent : {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
        cnn.eval()
        gru.eval()
        return cls(cnn, gru, **fusion_weights)
