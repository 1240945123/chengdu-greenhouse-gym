from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from experiments.reports.chengdu_residual_correction import (
    build_residual_features,
    validate_residual_features,
)


@dataclass(frozen=True)
class TemporalResidualSequences:
    inputs: np.ndarray
    targets: np.ndarray
    target_row_indices: np.ndarray


@dataclass(frozen=True)
class ArrayStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ArrayStandardizer":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim < 2 or len(array) == 0:
            raise ValueError("Standardizer requires a non-empty array with features")
        mean = np.mean(array, axis=tuple(range(array.ndim - 1)), dtype=np.float64)
        scale = np.std(array, axis=tuple(range(array.ndim - 1)), dtype=np.float64)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, values: dict) -> "ArrayStandardizer":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            scale=np.asarray(values["scale"], dtype=np.float32),
        )


def build_temporal_residual_sequences(
    predictions: pd.DataFrame,
    *,
    feature_columns: list[str],
    lookback: int,
) -> TemporalResidualSequences:
    if int(lookback) < 1:
        raise ValueError("lookback must be at least one")
    if "timestamp" not in predictions.columns:
        raise ValueError("Temporal residual data requires timestamp")
    validate_residual_features(feature_columns)
    features = build_residual_features(predictions, feature_columns).to_numpy(dtype=np.float32)
    targets = np.column_stack(
        [
            predictions["next_x_air_temperature"].to_numpy(dtype=np.float32)
            - predictions["pred_air_temperature"].to_numpy(dtype=np.float32),
            predictions["next_x_relative_humidity"].to_numpy(dtype=np.float32)
            - predictions["pred_relative_humidity"].to_numpy(dtype=np.float32),
        ]
    )
    timestamps = pd.to_datetime(predictions["timestamp"], errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Temporal residual rows must be chronological")

    windows: list[np.ndarray] = []
    window_targets: list[np.ndarray] = []
    target_indices: list[int] = []
    width = int(lookback)
    for end in range(width - 1, len(predictions)):
        start = end - width + 1
        if width > 1:
            differences = timestamps.iloc[start : end + 1].diff().iloc[1:]
            if not (differences == pd.Timedelta(hours=1)).all():
                continue
        windows.append(features[start : end + 1])
        window_targets.append(targets[end])
        target_indices.append(end)

    return TemporalResidualSequences(
        inputs=np.asarray(windows, dtype=np.float32).reshape(-1, width, len(feature_columns)),
        targets=np.asarray(window_targets, dtype=np.float32).reshape(-1, 2),
        target_row_indices=np.asarray(target_indices, dtype=np.int64),
    )


class LSTMResidualModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 2):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.encoder = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True)
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _state = self.encoder(inputs)
        return self.output_head(encoded[:, -1, :])


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.conv(F.pad(inputs, (self.left_padding, 0)))
        return self.activation(inputs + transformed)


class TCNResidualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.input_projection = nn.Conv1d(self.input_dim, self.hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            _CausalResidualBlock(self.hidden_dim, kernel_size, dilation=1),
            _CausalResidualBlock(self.hidden_dim, kernel_size, dilation=2),
        )
        self.output_head = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.input_projection(inputs.transpose(1, 2))
        encoded = self.blocks(encoded)
        return self.output_head(encoded[:, :, -1])
