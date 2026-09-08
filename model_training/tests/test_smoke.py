"""Smoke test des formes et de la boucle commune sur des donnees synthetiques."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model_training.models import CNNForecaster, GRUForecaster, LSTMForecaster
from model_training.trainer import Trainer


class TinyDataset(Dataset):
    def __init__(self, size: int = 32) -> None:
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randn(size, 20, 4, generator=generator)
        last = self.x[:, -1:, :]
        self.y = last.repeat(1, 5, 1)
        self.regime = torch.arange(size) % 4
        self.mean = np.zeros(4, dtype=np.float32)
        self.std = np.ones(4, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "y": self.y[index],
            "regime": self.regime[index],
            "simulation_id": torch.tensor(index),
        }


def run_smoke_test(output_directory: Path) -> None:
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    trainer = Trainer(
        loader,
        loader,
        loader,
        output_directory=output_directory,
        device="cpu",
    )

    for model in (
        CNNForecaster(),
        GRUForecaster(),
        LSTMForecaster(),
    ):
        output = model(torch.zeros(2, 20, 4))
        assert output.shape == (2, 5, 4)
        history = trainer.train(
            model,
            epochs=1,
            patience=1,
            run_name=f"smoke_{model.model_name}",
            max_test_batches=1,
        )
        assert len(history) == 1
        assert np.isfinite(history["validation_mae"].iloc[0])


if __name__ == "__main__":
    run_smoke_test(Path("smoke_experiments"))
    print("Smoke test model_training reussi.")
