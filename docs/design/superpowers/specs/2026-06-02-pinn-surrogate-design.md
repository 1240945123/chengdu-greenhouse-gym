# PINN Surrogate Design

**Goal:** Add a physics-informed neural network surrogate model that predicts the next GreenLight state from the current state, control input, and weather disturbance.

**Scope:** This design covers the PINN prediction model itself. It does not implement PINN-MPC or any controller that uses the trained PINN.

## Prediction Task

The surrogate learns one-step GreenLight dynamics:

```text
[x_t, u_t, d_t] -> x_{t+1}
```

Dimensions:

- `x_t`: 28 GreenLight states
- `u_t`: 6 greenhouse controls
- `d_t`: 10 weather disturbance values
- model input: 44 values
- `x_{t+1}`: 28 GreenLight states

The model is evaluated by:

- one-step MSE
- one-step MAE
- multi-step rollout MSE
- multi-step rollout MAE

## Data Design

Training data should be generated from mixed rollouts:

1. `random`
   - Samples actions from the environment action space.
   - Provides broad action/state coverage.

2. `baseline`
   - Uses `RuleBasedController`.
   - Provides trajectories from the original rule-based greenhouse control logic.

3. `pid`
   - Uses the newly added `PIDController`.
   - Provides traditional feedback-control trajectories.

Each transition stores:

```text
x_t, u_t, d_t, x_next, source
```

The first version should generate data programmatically from `GreenLightEnv` rather than requiring existing CSV files. The data generator should accept small episode/step counts for quick tests and larger counts for training.

## Model Design

Use a PyTorch MLP dynamics surrogate:

```text
input_dim = 44
hidden_layers = [256, 256, 256]
output_dim = 28
activation = SiLU
```

Inputs and targets should be standardized:

```text
input_norm = (input - input_mean) / input_std
target_norm = (target - target_mean) / target_std
```

Saved artifacts:

- `train_data/pinn/pinn_model.pt`
- `train_data/pinn/normalizer.npz`

## Physics-Informed Loss

The first version uses practical weak physics constraints rather than full GreenLight ODE residuals.

Training objective:

```text
loss = data_mse
     + lambda_nonnegative * nonnegative_state_penalty
     + lambda_temperature * temperature_range_penalty
     + lambda_delta * excessive_delta_penalty
```

Constraint meanings:

- biomass/dry-matter-like state values should not become negative
- temperature-like states should stay within a reasonable range
- predicted state changes should not be implausibly large

This is intentionally lighter than a full ODE-residual PINN because direct GreenLight residual training would be substantially heavier and harder to stabilize. The project can later upgrade this surrogate to include explicit ODE residual terms.

## Training Design

Add `experiments/train_pinn.py`.

Default behavior:

```bash
python -m experiments.train_pinn \
  --env_id GreenLightEnv \
  --env_config glassgym/configs/envs/ \
  --pinn_config configs/agents/pinn.yml \
  --save_dir train_data/pinn/
```

The script should:

1. Load environment config.
2. Generate mixed rollout transitions.
3. Split transitions into train/test sets.
4. Fit normalizer statistics on train data.
5. Train the MLP with data and weak physics losses.
6. Save model weights and normalizer.
7. Print training and test losses.

## Evaluation Design

Add `experiments/evaluate_pinn.py`.

Default behavior:

```bash
python -m experiments.evaluate_pinn \
  --env_id GreenLightEnv \
  --env_config glassgym/configs/envs/ \
  --pinn_config configs/agents/pinn.yml \
  --model_path train_data/pinn/pinn_model.pt \
  --normalizer_path train_data/pinn/normalizer.npz \
  --save_dir results/pinn/
```

The script should output:

```text
results/pinn/pinn_metrics.json
```

Metrics:

- `one_step_mse`
- `one_step_mae`
- `multi_step_mse`
- `multi_step_mae`
- number of evaluation transitions
- rollout length used for multi-step evaluation

## File Structure

Create:

- `glassgym/models/pinn/__init__.py`
- `glassgym/models/pinn/surrogate.py`
- `glassgym/models/pinn/dataset.py`
- `configs/agents/pinn.yml`
- `experiments/train_pinn.py`
- `experiments/evaluate_pinn.py`
- `tests/pinn_surrogate.py`

Outputs:

- `train_data/pinn/pinn_model.pt`
- `train_data/pinn/normalizer.npz`
- `results/pinn/pinn_metrics.json`

## Testing Design

Tests should stay small and fast:

1. The model maps a `(batch, 44)` tensor to `(batch, 28)`.
2. Normalizer round-trip reconstructs input values.
3. Dataset generation can create a small mixed transition set.
4. Physics loss returns a finite nonnegative scalar.
5. A tiny training loop updates the model without crashing.

Full training is a verification step, not a unit test.

## Expected Result

First-version success means:

- mixed random/baseline/PID data can be generated
- PINN surrogate can train and save artifacts
- evaluation writes one-step and multi-step metrics
- metrics can be used in the thesis as model-prediction evidence

The first version does not need to beat any controller reward because it is not yet a controller.

## Non-Goals

- No PINN-MPC controller in this task.
- No full GreenLight ODE residual loss in the first version.
- No changes to PPO/SAC/PID/MPC evaluation paths.
- No dependency installation beyond existing `[train]` dependencies, which already include PyTorch.

## Open Decisions Resolved

- PNNs means PINNs: Physics-informed Neural Networks.
- The first implementation is a prediction surrogate, not a control policy.
- Training data uses mixed random, rule-based baseline, and PID rollouts.
- Physics-informed loss uses weak practical constraints for the first version.
