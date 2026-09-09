from __future__ import annotations

import torch
from torch import nn


class TransformerDynamicsSurrogate(nn.Module):
    def __init__(
        self,
        input_dim: int = 44,
        output_dim: int = 28,
        seq_len: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.seq_len = int(seq_len)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TransformerDynamicsSurrogate expects [batch, seq_len, input_dim]")
        if x.shape[1] != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {x.shape[1]}")
        h = self.input_projection(x) + self.position_embedding
        encoded = self.encoder(h)
        return self.output_head(encoded[:, -1, :])


def train_one_epoch(
    model: TransformerDynamicsSurrogate,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
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

        optimizer.zero_grad()
        pred = model(batch_inputs)
        loss = torch.mean((pred - batch_targets) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu()) * len(idx)
        total_seen += len(idx)

    return total_loss / max(total_seen, 1)
