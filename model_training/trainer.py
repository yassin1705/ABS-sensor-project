"""Boucle d'entrainement partagee par CNN, GRU et LSTM."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import ForecastEvaluator
from .models import BaseWheelSpeedForecaster

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # TensorBoard reste optionnel.
    SummaryWriter = None


class Trainer:
    """Entraine n'importe quel modele respectant le contrat commun."""

    def __init__(
        self,
        train_loader,
        validation_loader,
        test_loader,
        *,
        output_directory: str | Path = "experiments",
        device: str | None = None,
        loss_function: nn.Module | None = None,
    ) -> None:
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.loss_function = loss_function or nn.SmoothL1Loss(beta=1.0)

        train_dataset = train_loader.dataset
        self.normalization_mean = np.asarray(
            train_dataset.mean, dtype=np.float32
        )
        self.normalization_std = np.asarray(
            train_dataset.std, dtype=np.float32
        )

    def train(
        self,
        model: BaseWheelSpeedForecaster,
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 8,
        run_name: str | None = None,
        test_every: int = 1,
        max_test_batches: int | None = None,
        progress_every: int = 100,
    ) -> pd.DataFrame:
        """Entraine le modele et monitore validation et test a chaque epoque.

        Le checkpoint et l'early stopping utilisent exclusivement la loss de
        validation. Le test est affiche pour le suivi demande, mais n'influence
        aucune decision d'entrainement.
        """
        if run_name is None:
            run_name = model.model_name
        run_directory = self.output_directory / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
        )
        evaluator = ForecastEvaluator(
            self.loss_function,
            self.normalization_mean,
            self.normalization_std,
            model.horizon,
            self.device,
        )
        writer = (
            SummaryWriter(log_dir=run_directory / "tensorboard")
            if SummaryWriter is not None
            else None
        )

        history: list[dict[str, float]] = []
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        best_checkpoint = run_directory / "best_model.pt"

        metadata = {
            "model": model.model_name,
            "parameters": model.count_parameters(),
            "history_length": model.history_length,
            "horizon": model.horizon,
            "device": str(self.device),
            "loss": type(self.loss_function).__name__,
            "normalization_mean": self.normalization_mean.tolist(),
            "normalization_std": self.normalization_std.tolist(),
        }
        (run_directory / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        for epoch in range(1, epochs + 1):
            start_time = time.perf_counter()
            train_loss = self._train_epoch(
                model,
                optimizer,
                epoch=epoch,
                progress_every=progress_every,
            )
            validation = evaluator.evaluate(model, self.validation_loader)

            if test_every > 0 and epoch % test_every == 0:
                test = evaluator.evaluate(
                    model,
                    self.test_loader,
                    max_batches=max_test_batches,
                )
            else:
                test = {"loss": float("nan"), "mae": float("nan"), "rmse": float("nan")}

            scheduler.step(validation["loss"])
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **{f"validation_{key}": value for key, value in validation.items()},
                **{f"test_{key}": value for key, value in test.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - start_time,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(
                run_directory / "history.csv", index=False
            )

            if writer is not None:
                for key, value in record.items():
                    if key != "epoch" and np.isfinite(value):
                        writer.add_scalar(key, value, epoch)

            print(
                f"[{model.model_name}] epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"val={validation['loss']:.6f}, "
                f"MAE={validation['mae']:.4f} m/s | "
                f"test={test['loss']:.6f}, "
                f"MAE={test['mae']:.4f} m/s"
            )

            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "validation_loss": best_validation_loss,
                        "metadata": metadata,
                    },
                    best_checkpoint,
                )
            else:
                epochs_without_improvement += 1

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation["loss"],
                    "metadata": metadata,
                },
                run_directory / "last_model.pt",
            )
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping apres {epoch} epoques "
                    "(critere validation uniquement)."
                )
                break

        if writer is not None:
            writer.close()
        self.load_best(model, run_name)
        return pd.DataFrame(history)

    def _train_epoch(
        self,
        model: BaseWheelSpeedForecaster,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        progress_every: int,
    ) -> float:
        model.train()
        loss_sum = 0.0
        sample_count = 0
        total_batches = len(self.train_loader)
        for batch_index, batch in enumerate(self.train_loader, start=1):
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = self.loss_function(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += loss.item() * x.shape[0]
            sample_count += x.shape[0]
            should_report = (
                progress_every > 0
                and (
                    batch_index == 1
                    or batch_index % progress_every == 0
                    or batch_index == total_batches
                )
            )
            if should_report:
                print(
                    f"  epoch {epoch:03d} | "
                    f"batch {batch_index:05d}/{total_batches:05d} | "
                    f"train loss={loss_sum / sample_count:.6f}",
                    flush=True,
                )
        return loss_sum / sample_count

    def load_best(
        self,
        model: BaseWheelSpeedForecaster,
        run_name: str | None = None,
    ) -> dict:
        if run_name is None:
            run_name = model.model_name
        checkpoint_path = (
            self.output_directory / run_name / "best_model.pt"
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        return checkpoint

    def evaluate_test(
        self,
        model: BaseWheelSpeedForecaster,
        *,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        evaluator = ForecastEvaluator(
            self.loss_function,
            self.normalization_mean,
            self.normalization_std,
            model.horizon,
            self.device,
        )
        return evaluator.evaluate(
            model.to(self.device),
            self.test_loader,
            max_batches=max_batches,
        )
