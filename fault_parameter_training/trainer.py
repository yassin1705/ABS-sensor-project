"""Entrainement partage avec monitoring train/validation et test final."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .config import (
    BASE_DROPOUT_PROBABILITY,
    EXPECTED_SAMPLE_COUNT,
    FAULT_TYPE,
    SIMULATION_DURATION_S,
    TARGET_NAMES,
)
from .evaluation import DetectionParameterLoss, FaultParameterEvaluator
from .models import BaseFaultParameterEstimator

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class FaultParameterTrainer:
    def __init__(
        self,
        train_loader,
        validation_loader,
        test_loader,
        *,
        output_directory: str | Path = "fault_parameter_training/experiments",
        device: str | None = None,
        loss_function: nn.Module | None = None,
    ) -> None:
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        train_dataset = train_loader.dataset
        positive_count = int((train_dataset.y[:, 0] > 0.5).sum().item())
        negative_count = len(train_dataset) - positive_count
        if positive_count == 0 or negative_count == 0:
            raise ValueError("Le train doit contenir des capteurs sains et fautifs.")
        self.positive_weight = negative_count / positive_count
        self.loss_function = loss_function or DetectionParameterLoss(
            positive_weight=self.positive_weight
        )

    def train(
        self,
        model: BaseFaultParameterEstimator,
        *,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 8,
    ) -> pd.DataFrame:
        run_directory = self.output_directory / model.model_name
        run_directory.mkdir(parents=True, exist_ok=True)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
        )
        evaluator = FaultParameterEvaluator(self.loss_function, self.device)
        writer = (
            SummaryWriter(run_directory / "tensorboard")
            if SummaryWriter is not None
            else None
        )
        metadata = {
            "model": model.model_name,
            "parameters": model.count_parameters(),
            "input_shape": [EXPECTED_SAMPLE_COUNT, 2],
            "input_channels": ["normalized_wheel_speed_ecu_mps", "ecu_valid"],
            "target_names": list(TARGET_NAMES),
            "simulation_duration_s": SIMULATION_DURATION_S,
            "fault_type": FAULT_TYPE,
            "dropout_probability": BASE_DROPOUT_PROBABILITY,
            "device": str(self.device),
            "loss": type(self.loss_function).__name__,
            "classification_positive_weight": self.positive_weight,
        }
        (run_directory / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        history: list[dict[str, float]] = []
        best_validation_loss = float("inf")
        stale_epochs = 0

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for batch in self.train_loader:
                x = batch["x"].to(self.device, non_blocking=True)
                y = batch["y"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x)
                loss = self.loss_function(prediction, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += loss.item() * len(x)
                sample_count += len(x)
            train_loss = loss_sum / sample_count
            validation = evaluator.evaluate(model, self.validation_loader)
            scheduler.step(validation["loss"])
            record = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                **{f"validation_{key}": value for key, value in validation.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - started,
            }
            history.append(record)
            pd.DataFrame(history).to_csv(run_directory / "history.csv", index=False)
            if writer is not None:
                for key, value in record.items():
                    if key != "epoch":
                        writer.add_scalar(key, value, epoch)
            print(
                f"[{model.model_name}] epoch {epoch:03d} | "
                f"train={train_loss:.5f} | val={validation['loss']:.5f} | "
                f"F1={validation['f1']:.3f} | FPR={validation['false_positive_rate']:.3f} | "
                f"debut={validation['start_mae_s']:.3f}s | "
                f"duree={validation['duration_mae_s']:.3f}s | "
                f"severite={validation['severity_mae']:.3f}"
            )
            checkpoint = {
                "model_state": model.state_dict(),
                "metadata": metadata,
                "epoch": epoch,
                "validation_loss": validation["loss"],
            }
            torch.save(checkpoint, run_directory / "last_model.pt")
            if validation["loss"] < best_validation_loss:
                best_validation_loss = validation["loss"]
                stale_epochs = 0
                torch.save(checkpoint, run_directory / "best_model.pt")
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping apres {epoch} epoques.")
                    break
        if writer is not None:
            writer.close()
        self.load_best(model)
        return pd.DataFrame(history)

    def load_best(self, model: BaseFaultParameterEstimator) -> dict:
        checkpoint = torch.load(
            self.output_directory / model.model_name / "best_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        return checkpoint

    def evaluate_test(
        self, model: BaseFaultParameterEstimator
    ) -> tuple[dict[str, float], pd.DataFrame]:
        self.load_best(model)
        evaluator = FaultParameterEvaluator(self.loss_function, self.device)
        metrics, predictions = evaluator.evaluate(
            model, self.test_loader, return_predictions=True
        )
        run_directory = self.output_directory / model.model_name
        (run_directory / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        predictions.to_csv(run_directory / "test_predictions.csv", index=False)
        return metrics, predictions

    def export_onnx(self, model: BaseFaultParameterEstimator) -> Path:
        self.load_best(model)
        model = model.to("cpu").eval()
        output = self.output_directory / model.model_name / f"{model.model_name}.onnx"
        torch.onnx.export(
            model,
            torch.zeros(1, EXPECTED_SAMPLE_COUNT, 2),
            output,
            input_names=["sensor_series"],
            output_names=["normalized_fault_parameters"],
            opset_version=17,
            dynamo=False,
        )
        return output
