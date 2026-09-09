# Transformer Surrogate Design

## Goal

Build a Transformer-based dynamics prediction model for GreenLight-Gym2 that predicts the next greenhouse state from a short history of state, action, and weather vectors.

## Scope

This feature implements the prediction model itself, not a controller. It mirrors the existing PINN surrogate workflow so the result can be compared using one-step and multi-step prediction errors.

## Data Flow

The model consumes sequences shaped `[batch, seq_len, 44]`, where each step is `[state(28), action(10), weather(6)]`. The target is the next state vector shaped `[batch, 28]`.

Training data is generated from the same mixed sources used by PINN: random actions, the rule-based baseline, and PID. Sequence windows are built from transition records, keeping a sliding history of configurable length.

## Architecture

The model projects 44-dimensional inputs into `d_model`, adds learned positional embeddings, applies a Transformer encoder, and maps the final token representation to the 28-dimensional next state prediction.

The implementation uses PyTorch only. Normalizers are reused from the PINN surrogate module to keep raw-space metrics comparable.

## Evaluation

Evaluation reports:

- `one_step_mse`
- `one_step_mae`
- `multi_step_mse`
- `multi_step_mae`
- `num_eval_sequences`
- `multi_step_rollout_steps`

Multi-step evaluation rolls the predicted state forward while reusing the recorded action and weather sequence.

## Testing

Unit tests cover model output shape, sequence window construction, normalizer save/load compatibility, and one training step reducing through a valid differentiable path.
