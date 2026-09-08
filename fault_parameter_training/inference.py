"""API haut niveau pour diagnostiquer une serie brute avec la fusion CNN/GRU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import (
    BASE_DROPOUT_PROBABILITY,
    EXPECTED_SAMPLE_COUNT,
    FAULT_TYPE,
    SIMULATION_DURATION_S,
)
from .models import CNNGRUFusionEstimator


class CNNGRUFusionPredictor:
    """Charge CNN+GRU et convertit une serie ABS brute en diagnostic physique."""

    def __init__(
        self,
        experiment_directory: str | Path,
        cache_path: str | Path,
        *,
        device: str | None = None,
        fault_threshold: float = 0.5,
        cnn_probability_weight: float = 0.5,
        cnn_timing_weight: float = 1.0,
        gru_severity_weight: float = 1.0,
    ) -> None:
        if not 0 < fault_threshold < 1:
            raise ValueError("fault_threshold doit appartenir a ]0, 1[.")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fault_threshold = float(fault_threshold)
        with np.load(cache_path) as cache:
            self.speed_mean = float(cache["speed_mean"])
            self.speed_std = float(cache["speed_std"])
        if self.speed_std <= 0:
            raise ValueError("Ecart-type de normalisation invalide.")
        self.model = CNNGRUFusionEstimator.from_experiment_directory(
            experiment_directory,
            map_location="cpu",
            cnn_probability_weight=cnn_probability_weight,
            cnn_timing_weight=cnn_timing_weight,
            gru_severity_weight=gru_severity_weight,
        ).to(self.device)
        self.model.eval()

    def _prepare_input(
        self,
        speed_mps: np.ndarray | list[float],
        valid: np.ndarray | list[bool] | None,
    ) -> torch.Tensor:
        speed = np.asarray(speed_mps, dtype=np.float32).reshape(-1)
        if len(speed) != EXPECTED_SAMPLE_COUNT:
            raise ValueError(
                f"{EXPECTED_SAMPLE_COUNT} vitesses sont attendues, {len(speed)} recues."
            )
        if valid is None:
            usable = np.isfinite(speed)
        else:
            valid_array = np.asarray(valid).reshape(-1)
            if len(valid_array) != EXPECTED_SAMPLE_COUNT:
                raise ValueError(
                    f"{EXPECTED_SAMPLE_COUNT} indicateurs de validite sont attendus."
                )
            usable = valid_array.astype(bool) & np.isfinite(speed)
        finite_indices = np.flatnonzero(np.isfinite(speed))
        if len(finite_indices) == 0:
            raise ValueError("La serie ne contient aucune vitesse finie.")

        filled = speed.copy()
        first_value = float(speed[finite_indices[0]])
        last_value = first_value
        for index in range(len(filled)):
            if np.isfinite(filled[index]):
                last_value = float(filled[index])
            else:
                filled[index] = last_value
        filled[: finite_indices[0]] = first_value
        normalized = (filled - self.speed_mean) / self.speed_std
        model_input = np.column_stack((normalized, usable.astype(np.float32)))
        return torch.from_numpy(model_input).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        speed_mps: np.ndarray | list[float],
        valid: np.ndarray | list[bool] | None = None,
    ) -> dict[str, object]:
        x = self._prepare_input(speed_mps, valid)
        fused, cnn_output, gru_output = self.model.forward_components(x)
        fused_values = fused[0].cpu().numpy()
        cnn_values = cnn_output[0].cpu().numpy()
        gru_values = gru_output[0].cpu().numpy()
        fault_probability = float(fused_values[0])
        fault_detected = fault_probability >= self.fault_threshold
        start_s = float(fused_values[1] * SIMULATION_DURATION_S)
        duration_s = float(fused_values[2] * SIMULATION_DURATION_S)
        severity = float(fused_values[3])
        parameters = None
        if fault_detected:
            parameters = {
                "fault_type": FAULT_TYPE,
                "fault_start_s": start_s,
                "fault_duration_s": duration_s,
                "fault_end_s": min(SIMULATION_DURATION_S, start_s + duration_s),
                "fault_severity": severity,
                "dropout_probability": BASE_DROPOUT_PROBABILITY,
                "effective_dropout_probability": (
                    severity * BASE_DROPOUT_PROBABILITY
                ),
            }
        return {
            "fault_detected": fault_detected,
            "fault_probability": fault_probability,
            "fault_threshold": self.fault_threshold,
            "fault_parameters": parameters,
            "normalized_fused_output": fused_values.tolist(),
            "component_outputs": {
                "cnn": cnn_values.tolist(),
                "gru": gru_values.tolist(),
            },
        }
