"""Constantes partagees par le pipeline de parametres de defaut."""

WHEELS = ("FL", "FR", "RL", "RR")
FAULT_TYPE = "intermittent_loss"
BASE_DROPOUT_PROBABILITY = 0.50
SIMULATION_DURATION_S = 5.0
EXPECTED_SAMPLE_COUNT = 500
TARGET_NAMES = (
    "fault_probability",
    "start_fraction",
    "duration_fraction",
    "severity",
)
