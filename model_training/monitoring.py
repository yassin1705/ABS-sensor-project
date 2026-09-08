"""Visualisations partagees des historiques d'entrainement."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_histories(histories: Mapping[str, pd.DataFrame]) -> None:
    """Affiche loss et MAE validation/test pour plusieurs modeles."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name, history in histories.items():
        axes[0].plot(
            history["epoch"], history["validation_loss"], label=f"{name} val"
        )
        axes[0].plot(
            history["epoch"],
            history["test_loss"],
            linestyle="--",
            label=f"{name} test",
        )
        axes[1].plot(
            history["epoch"], history["validation_mae"], label=f"{name} val"
        )
        axes[1].plot(
            history["epoch"],
            history["test_mae"],
            linestyle="--",
            label=f"{name} test",
        )

    axes[0].set_title("Loss Huber normalisee")
    axes[0].set_xlabel("Epoque")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("MAE en m/s")
    axes[1].set_xlabel("Epoque")
    axes[1].set_ylabel("MAE")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
