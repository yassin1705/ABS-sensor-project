"""Graphiques du benchmark mixte sain/fautif."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd


def plot_histories(histories: Mapping[str, pd.DataFrame]):
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    for name, history in histories.items():
        axes[0, 0].plot(history["epoch"], history["train_loss"], "--", label=f"{name} train")
        axes[0, 0].plot(history["epoch"], history["validation_loss"], label=f"{name} val")
        axes[0, 1].plot(history["epoch"], history["validation_f1"], label=f"{name} F1")
        axes[0, 1].plot(
            history["epoch"],
            history["validation_false_positive_rate"],
            "--",
            label=f"{name} FPR",
        )
        axes[1, 0].plot(history["epoch"], history["validation_start_mae_s"], label=f"{name} debut")
        axes[1, 0].plot(history["epoch"], history["validation_duration_mae_s"], "--", label=f"{name} duree")
        axes[1, 1].plot(history["epoch"], history["validation_severity_mae"], label=name)
    axes[0, 0].set_title("Loss train / validation")
    axes[0, 0].set_ylabel("loss multi-tache")
    axes[0, 1].set_title("Detection validation")
    axes[0, 1].set_ylabel("score")
    axes[1, 0].set_title("Parametres temporels (fautifs uniquement)")
    axes[1, 0].set_ylabel("MAE (s)")
    axes[1, 1].set_title("Severite (fautifs uniquement)")
    axes[1, 1].set_ylabel("MAE")
    for axis in axes.flat:
        axis.set_xlabel("epoque")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    return figure


def plot_benchmark(benchmark: pd.DataFrame):
    ordered = benchmark.sort_values("validation_loss")
    figure, axes = plt.subplots(1, 3, figsize=(17, 4))
    ordered.plot.bar(
        x="model",
        y=["test_f1", "test_false_positive_rate"],
        ax=axes[0],
    )
    ordered.plot.bar(
        x="model",
        y=["test_start_mae_s", "test_duration_mae_s"],
        ax=axes[1],
    )
    ordered.plot.bar(x="model", y="test_severity_mae", ax=axes[2], color="#2ca02c")
    axes[0].set_title("Detection sur test")
    axes[0].set_ylabel("score")
    axes[1].set_title("Erreur temporelle sur fautifs test")
    axes[1].set_ylabel("MAE (s)")
    axes[2].set_title("Erreur severite sur fautifs test")
    axes[2].set_ylabel("MAE")
    for axis in axes:
        axis.grid(True, axis="y", alpha=0.3)
        axis.set_xlabel("")
    figure.tight_layout()
    return figure
