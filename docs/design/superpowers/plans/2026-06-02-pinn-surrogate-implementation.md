# PINN Surrogate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trainable physics-informed neural network surrogate that predicts `x_{t+1}` from `[x_t, u_t, d_t]`, saves model artifacts, and reports one-step/multi-step prediction metrics.

**Architecture:** Add a small `glassgym.models.pinn` package containing a PyTorch MLP surrogate, normalization utilities, weak physics loss, and rollout dataset generation. Add train/evaluate scripts that generate mixed random/baseline/PID transition data, train the surrogate, save artifacts, and write metrics JSON.

**Tech Stack:** Python 3.12, PyTorch from existing `[train]` dependencies, NumPy, pandas-free in-memory rollout data, unittest.

---

## Files

- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\__init__.py`
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\surrogate.py`
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\dataset.py`
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\pinn.yml`
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\train_pinn.py`
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_pinn.py`
- Create: `E:\school\final paper\GreenLight-Gym2\tests\pinn_surrogate.py`
- Output: `E:\school\final paper\GreenLight-Gym2\train_data\pinn\pinn_model.pt`
- Output: `E:\school\final paper\GreenLight-Gym2\train_data\pinn\normalizer.npz`
- Output: `E:\school\final paper\GreenLight-Gym2\results\pinn\pinn_metrics.json`

The workspace is not a git repository because it was downloaded as a zip snapshot, so commit steps are replaced with file/status verification commands.

---

### Task 1: PINN Surrogate Tests

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\tests\pinn_surrogate.py`

- [ ] **Step 1: Write failing tests**

Create `tests/pinn_surrogate.py` with:

```python
import unittest

import numpy as np
import torch

from glassgym.models.pinn.dataset import generate_mixed_transitions
from glassgym.models.pinn.surrogate import (
    DynamicsSurrogate,
    Normalizer,
    physics_loss,
    train_one_epoch,
)
from RL.utils import build_env_kwargs, load_env_params


class TestPINNSurrogate(unittest.TestCase):
    def test_model_maps_input_to_next_state(self):
        model = DynamicsSurrogate(input_dim=44, output_dim=28, hidden_layers=[16, 16])
        x = torch.zeros((3, 44), dtype=torch.float32)
        y = model(x)
        self.assertEqual(tuple(y.shape), (3, 28))

    def test_normalizer_round_trip(self):
        values = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32)
        normalizer = Normalizer.fit(values)
        reconstructed = normalizer.inverse_transform(normalizer.transform(values))
        np.testing.assert_allclose(reconstructed, values, atol=1e-5)

    def test_physics_loss_is_finite_and_nonnegative(self):
        pred = torch.zeros((4, 28), dtype=torch.float32)
        prev = torch.zeros((4, 28), dtype=torch.float32)
        loss = physics_loss(
            pred,
            prev,
            nonnegative_indices=[22, 23, 24, 25],
            temperature_indices=[2, 3, 4, 5, 6, 7, 8, 9],
            lambda_nonnegative=1.0,
            lambda_temperature=1.0,
            lambda_delta=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)

    def test_generate_mixed_transitions_small(self):
        env_kwargs = load_env_params("GreenLightEnv", "glassgym/configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        transitions = generate_mixed_transitions(
            env_kwargs=env_kwargs,
            source_counts={"random": 1, "baseline": 1, "pid": 1},
            max_steps_per_episode=2,
            seed=123,
        )
        self.assertEqual(transitions["inputs"].shape[1], 44)
        self.assertEqual(transitions["targets"].shape[1], 28)
        self.assertEqual(len(transitions["sources"]), 6)

    def test_tiny_training_loop_updates_without_crashing(self):
        model = DynamicsSurrogate(input_dim=44, output_dim=28, hidden_layers=[16])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        inputs = torch.randn((8, 44), dtype=torch.float32)
        targets = torch.randn((8, 28), dtype=torch.float32)
        previous_states = torch.randn((8, 28), dtype=torch.float32)
        loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            previous_states=previous_states,
            batch_size=4,
            loss_weights={
                "lambda_nonnegative": 0.1,
                "lambda_temperature": 0.1,
                "lambda_delta": 0.01,
            },
        )
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail because PINN package does not exist**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pinn_surrogate
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'glassgym.models.pinn'`.

---

### Task 2: Surrogate Model Utilities

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\__init__.py`
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\surrogate.py`

- [ ] **Step 1: Implement model, normalizer, losses, and train helper**

Create `glassgym/models/pinn/__init__.py`:

```python
"""PINN surrogate model utilities."""
```

Create `glassgym/models/pinn/surrogate.py`:

```python
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
```

- [ ] **Step 2: Run tests and capture missing dataset failure**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pinn_surrogate
```

Expected: ERROR because `glassgym.models.pinn.dataset` does not exist yet.

---

### Task 3: Mixed Rollout Dataset Generation

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\models\pinn\dataset.py`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\pinn_surrogate.py`

- [ ] **Step 1: Implement transition generation**

Create `glassgym/models/pinn/dataset.py`:

```python
from __future__ import annotations

from typing import Callable

import numpy as np

from glassgym.components.pid import PIDController
from glassgym.components.rule_based import RuleBasedController
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_model_hyperparams


def build_step_context(env: GreenLightEnv) -> StepContext:
    return StepContext(
        t=env.timestep,
        dt=env.dt,
        Np=env.Np,
        x_prev=env.x_prev,
        x=env.x,
        u=env.u,
        p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day,
        day_of_year=env.day_of_year,
    )


def _controller_for_source(source: str):
    if source == "baseline":
        return RuleBasedController(**load_model_hyperparams("rule_based", "GreenLightEnv"))
    if source == "pid":
        return PIDController(**load_model_hyperparams("pid", "GreenLightEnv"))
    return None


def _action_for_source(env: GreenLightEnv, source: str, rng: np.random.Generator, controller):
    if source == "random":
        return rng.uniform(0.0, 1.0, size=env.nu).astype(np.float32)
    ctx = build_step_context(env)
    return controller.predict(ctx).astype(np.float32)


def generate_mixed_transitions(
    env_kwargs: dict,
    source_counts: dict[str, int],
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    env_kwargs = env_kwargs.copy()
    env_kwargs["normalize_actions"] = False
    env = GreenLightEnv(**env_kwargs)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    previous_states: list[np.ndarray] = []
    sources: list[str] = []

    for source, episodes in source_counts.items():
        controller = _controller_for_source(source)
        for episode in range(int(episodes)):
            env.reset(seed=seed + len(sources) + episode)
            if hasattr(controller, "reset"):
                controller.reset()
            for _ in range(max_steps_per_episode):
                x_t = env.x.copy()
                d_t = env.weather_data[env.timestep].copy()
                u_t = _action_for_source(env, source, rng, controller)
                _obs, _reward, terminated, truncated, _info = env.step(u_t)
                x_next = env.x.copy()

                inputs.append(np.concatenate([x_t, u_t, d_t]).astype(np.float32))
                targets.append(x_next.astype(np.float32))
                previous_states.append(x_t.astype(np.float32))
                sources.append(source)

                if terminated or truncated:
                    break

    env.close()
    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "previous_states": np.asarray(previous_states, dtype=np.float32),
        "sources": np.asarray(sources),
    }
```

- [ ] **Step 2: Run PINN tests**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pinn_surrogate
```

Expected: PASS for all PINN tests.

---

### Task 4: PINN Config

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\pinn.yml`

- [ ] **Step 1: Add training defaults**

Create `configs/agents/pinn.yml`:

```yaml
GreenLightEnv:
  seed: 42
  source_counts:
    random: 2
    baseline: 2
    pid: 2
  max_steps_per_episode: 256
  test_fraction: 0.2

  model:
    input_dim: 44
    output_dim: 28
    hidden_layers: [256, 256, 256]
    activation: silu

  training:
    epochs: 20
    batch_size: 128
    learning_rate: 0.001
    device: cpu

  loss_weights:
    lambda_nonnegative: 0.01
    lambda_temperature: 0.01
    lambda_delta: 0.000001

  evaluation:
    source_counts:
      random: 1
      baseline: 1
      pid: 1
    max_steps_per_episode: 128
    multi_step_rollout_steps: 32
```

- [ ] **Step 2: Verify YAML loads**

Run:

```powershell
@'
from RL.utils import load_model_hyperparams
params = load_model_hyperparams("pinn", "GreenLightEnv")
print(params["model"]["input_dim"], params["model"]["output_dim"])
'@ | .\.venv\Scripts\python.exe -
```

Expected: prints `44 28`.

---

### Task 5: Training Script

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\train_pinn.py`
- Output: `E:\school\final paper\GreenLight-Gym2\train_data\pinn\pinn_model.pt`
- Output: `E:\school\final paper\GreenLight-Gym2\train_data\pinn\normalizer.npz`

- [ ] **Step 1: Implement training script**

Create `experiments/train_pinn.py`:

```python
import argparse
import json
import os

import numpy as np
import torch

from glassgym.models.pinn.dataset import generate_mixed_transitions
from glassgym.models.pinn.surrogate import (
    DynamicsSurrogate,
    Normalizer,
    save_normalizers,
    train_one_epoch,
)
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Train a PINN dynamics surrogate.")
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="glassgym/configs/envs/")
    parser.add_argument("--pinn_config", default="configs/agents/")
    parser.add_argument("--save_dir", default="train_data/pinn/")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    params = load_model_hyperparams("pinn", args.env_id)
    seed = int(params["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    transitions = generate_mixed_transitions(
        env_kwargs=env_kwargs,
        source_counts=params["source_counts"],
        max_steps_per_episode=int(params["max_steps_per_episode"]),
        seed=seed,
    )

    inputs = transitions["inputs"]
    targets = transitions["targets"]
    previous_states = transitions["previous_states"]

    n = len(inputs)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    test_size = max(1, int(n * float(params["test_fraction"])))
    test_idx = order[:test_size]
    train_idx = order[test_size:]

    input_norm = Normalizer.fit(inputs[train_idx])
    target_norm = Normalizer.fit(targets[train_idx])

    train_inputs = torch.tensor(input_norm.transform(inputs[train_idx]), dtype=torch.float32)
    train_targets = torch.tensor(target_norm.transform(targets[train_idx]), dtype=torch.float32)
    train_prev = torch.tensor(target_norm.transform(previous_states[train_idx]), dtype=torch.float32)
    test_inputs = torch.tensor(input_norm.transform(inputs[test_idx]), dtype=torch.float32)
    test_targets = torch.tensor(target_norm.transform(targets[test_idx]), dtype=torch.float32)

    model = DynamicsSurrogate(**params["model"])
    training = params["training"]
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            inputs=train_inputs,
            targets=train_targets,
            previous_states=train_prev,
            batch_size=int(training["batch_size"]),
            loss_weights=params["loss_weights"],
        )
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = torch.mean((model(test_inputs) - test_targets) ** 2).item()
            print(f"epoch={epoch} train_loss={train_loss:.6f} test_mse_norm={test_loss:.6f}")

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "pinn_model.pt")
    normalizer_path = os.path.join(args.save_dir, "normalizer.npz")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": params["model"],
        },
        model_path,
    )
    save_normalizers(normalizer_path, input_norm, target_norm)
    print(json.dumps({"model_path": model_path, "normalizer_path": normalizer_path, "transitions": int(n)}))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a short training smoke test**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m experiments.train_pinn --epochs 2 --save_dir train_data/pinn/
```

Expected: exit code 0, prints training losses, creates `pinn_model.pt` and `normalizer.npz`.

---

### Task 6: Evaluation Script

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_pinn.py`
- Output: `E:\school\final paper\GreenLight-Gym2\results\pinn\pinn_metrics.json`

- [ ] **Step 1: Implement evaluation script**

Create `experiments/evaluate_pinn.py`:

```python
import argparse
import json
import os

import numpy as np
import torch

from glassgym.models.pinn.dataset import generate_mixed_transitions
from glassgym.models.pinn.surrogate import DynamicsSurrogate, load_normalizers
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Evaluate a PINN dynamics surrogate.")
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="glassgym/configs/envs/")
    parser.add_argument("--pinn_config", default="configs/agents/")
    parser.add_argument("--model_path", default="train_data/pinn/pinn_model.pt")
    parser.add_argument("--normalizer_path", default="train_data/pinn/normalizer.npz")
    parser.add_argument("--save_dir", default="results/pinn/")
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    params = load_model_hyperparams("pinn", args.env_id)
    eval_params = params["evaluation"]

    checkpoint = torch.load(args.model_path, map_location="cpu")
    model = DynamicsSurrogate(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_norm, target_norm = load_normalizers(args.normalizer_path)

    transitions = generate_mixed_transitions(
        env_kwargs=env_kwargs,
        source_counts=eval_params["source_counts"],
        max_steps_per_episode=int(eval_params["max_steps_per_episode"]),
        seed=int(params["seed"]) + 999,
    )
    inputs = transitions["inputs"]
    targets = transitions["targets"]

    with torch.no_grad():
        pred_norm = model(torch.tensor(input_norm.transform(inputs), dtype=torch.float32)).numpy()
    preds = target_norm.inverse_transform(pred_norm)
    errors = preds - targets

    one_step_mse = float(np.mean(errors ** 2))
    one_step_mae = float(np.mean(np.abs(errors)))

    rollout_steps = min(int(eval_params["multi_step_rollout_steps"]), len(inputs))
    current_state = inputs[0, :28].copy()
    multi_errors = []
    for i in range(rollout_steps):
        model_input = inputs[i].copy()
        model_input[:28] = current_state
        with torch.no_grad():
            pred_norm_i = model(torch.tensor(input_norm.transform(model_input[None, :]), dtype=torch.float32)).numpy()
        pred_state = target_norm.inverse_transform(pred_norm_i)[0]
        target_state = targets[i]
        multi_errors.append(pred_state - target_state)
        current_state = pred_state.astype(np.float32)

    multi_errors = np.asarray(multi_errors, dtype=np.float32)
    metrics = {
        "one_step_mse": one_step_mse,
        "one_step_mae": one_step_mae,
        "multi_step_mse": float(np.mean(multi_errors ** 2)),
        "multi_step_mae": float(np.mean(np.abs(multi_errors))),
        "num_eval_transitions": int(len(inputs)),
        "multi_step_rollout_steps": int(rollout_steps),
    }

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "pinn_metrics.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run evaluation**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m experiments.evaluate_pinn `
  --model_path train_data/pinn/pinn_model.pt `
  --normalizer_path train_data/pinn/normalizer.npz `
  --save_dir results/pinn/
```

Expected: exit code 0 and creates `results/pinn/pinn_metrics.json`.

---

### Task 7: Final Verification

**Files:**
- Read all created files and outputs.

- [ ] **Step 1: Run tests**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pinn_surrogate
```

Expected: all tests pass.

- [ ] **Step 2: Verify output artifacts**

Run:

```powershell
Get-ChildItem -File -LiteralPath `
  'glassgym\models\pinn\surrogate.py',`
  'glassgym\models\pinn\dataset.py',`
  'configs\agents\pinn.yml',`
  'experiments\train_pinn.py',`
  'experiments\evaluate_pinn.py',`
  'train_data\pinn\pinn_model.pt',`
  'train_data\pinn\normalizer.npz',`
  'results\pinn\pinn_metrics.json' |
  Select-Object FullName,Length,LastWriteTime
```

Expected: all files exist.

- [ ] **Step 3: Print metrics**

Run:

```powershell
Get-Content -LiteralPath 'results\pinn\pinn_metrics.json'
```

Expected: JSON includes one-step and multi-step MSE/MAE metrics.

---

## Completion Criteria

PINN surrogate implementation is complete when:

- `python -m unittest tests.pinn_surrogate` passes.
- `python -m experiments.train_pinn --epochs 2 --save_dir train_data/pinn/` exits with code 0.
- `python -m experiments.evaluate_pinn ...` exits with code 0.
- `train_data/pinn/pinn_model.pt` exists.
- `train_data/pinn/normalizer.npz` exists.
- `results/pinn/pinn_metrics.json` exists and contains finite metrics.
