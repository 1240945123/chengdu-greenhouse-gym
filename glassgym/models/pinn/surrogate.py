from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class DynamicsSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 44,
        output_dim: int = 28,
        hidden_layers: list[int] | None = None,
        activation: str = "silu",
    ):
        super().__init__()
        hidden_layers = hidden_layers or [256, 256, 256]
        activations = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        activation_cls = activations[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(activation_cls())
            prev = width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        values = np.asarray(values, dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


def save_normalizers(path: str, input_normalizer: Normalizer, target_normalizer: Normalizer):
    np.savez(
        path,
        input_mean=input_normalizer.mean,
        input_std=input_normalizer.std,
        target_mean=target_normalizer.mean,
        target_std=target_normalizer.std,
    )


def load_normalizers(path: str) -> tuple[Normalizer, Normalizer]:
    data = np.load(path)
    input_normalizer = Normalizer(mean=data["input_mean"], std=data["input_std"])
    target_normalizer = Normalizer(mean=data["target_mean"], std=data["target_std"])
    return input_normalizer, target_normalizer


def physics_loss(
    pred_next_state: torch.Tensor,
    previous_state: torch.Tensor,
    nonnegative_indices: list[int],
    temperature_indices: list[int],
    lambda_nonnegative: float,
    lambda_temperature: float,
    lambda_delta: float,
) -> torch.Tensor:
    device = pred_next_state.device
    loss = torch.zeros((), dtype=pred_next_state.dtype, device=device)

    if nonnegative_indices:
        nonnegative_values = pred_next_state[:, nonnegative_indices]
        loss = loss + lambda_nonnegative * torch.relu(-nonnegative_values).pow(2).mean()

    if temperature_indices:
        temp_values = pred_next_state[:, temperature_indices]
        low_pen = torch.relu(-20.0 - temp_values).pow(2)
        high_pen = torch.relu(temp_values - 80.0).pow(2)
        loss = loss + lambda_temperature * (low_pen.mean() + high_pen.mean())

    delta = pred_next_state - previous_state
    loss = loss + lambda_delta * torch.relu(torch.abs(delta) - 1e6).pow(2).mean()
    return loss


def train_one_epoch(
    model: DynamicsSurrogate,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    previous_states: torch.Tensor,
    batch_size: int,
    loss_weights: dict[str, float],
) -> float:
    model.train()
    n = inputs.shape[0]
    order = torch.randperm(n, device=inputs.device)
    total_loss = 0.0
    total_seen = 0

    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch_inputs = inputs[idx]
        batch_targets = targets[idx]
        batch_previous = previous_states[idx]

        optimizer.zero_grad()
        pred = model(batch_inputs)
        data_loss = torch.mean((pred - batch_targets) ** 2)
        phys = physics_loss(
            pred,
            batch_previous,
            nonnegative_indices=[22, 23, 24, 25],
            temperature_indices=[2, 3, 4, 5, 6, 7, 8, 9],
            lambda_nonnegative=loss_weights["lambda_nonnegative"],
            lambda_temperature=loss_weights["lambda_temperature"],
            lambda_delta=loss_weights["lambda_delta"],
        )
        loss = data_loss + phys
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(idx)
        total_seen += len(idx)

    return total_loss / max(total_seen, 1)
