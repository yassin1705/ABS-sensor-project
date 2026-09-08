"""Architectures predictives partageant le contrat [B, L, 4] -> [B, H, 4]."""

from .base import BaseWheelSpeedForecaster
from .cnn import CNNForecaster
from .fault_parameter_gru import FaultParameterGRU
from .gru import GRUForecaster
from .lstm import LSTMForecaster

__all__ = [
    "BaseWheelSpeedForecaster",
    "CNNForecaster",
    "FaultParameterGRU",
    "GRUForecaster",
    "LSTMForecaster",
]
