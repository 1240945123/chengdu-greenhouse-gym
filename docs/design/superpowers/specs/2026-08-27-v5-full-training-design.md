# V5 Full Training and Multi-Season Evaluation Design

## Objective

Train and compare baseline, PID, MPC, PPO, SAC, and PID-guided Residual-PPO
against the Chengdu V5 hybrid greenhouse environment using substantially more
weather coverage than the one-day optimization runs.

## Evidence roles

- Training: all valid 14-day windows from the 2023 and 2024 spring/autumn
  120-day ERA5 seasons. These four seasons precede model selection and testing.
- Validation: fixed non-overlapping 14-day windows in 2025 spring. Checkpoints
  and seeds may be selected only from these windows.
- Final temporal holdout: the complete 120-day 2025 autumn season. It is used
  after controller settings and selected checkpoints are frozen.
- Descriptive robustness: complete 120-day evaluation over all six 2023-2025
  seasons. Training-season rows are labelled descriptive, not generalization.

The V5 residual model was fitted from target greenhouse observations ending
before 2026-07-12. ERA5 weather seasons are exogenous scenario data; overlapping
14-day starts increase weather-state coverage but are not claimed as independent
replicates.

## Controllers

- Fixed baseline, PID, and MPC retain the frozen reward and action constraints.
- PPO and SAC are trained without classical-controller assistance.
- Residual-PPO learns bounded corrections around PID and is always reported as
  a distinct PID-guided method, never as pure PPO.
- Heating and CO2 remain disabled. The learned action space contains roof
  ventilation and fan control only; every evaluation verifies disabled controls.

## Training budget

Use seeds 0, 1, and 2 with cumulative checkpoints at 20,480, 102,400, 307,200,
and 614,400 environment steps. These values align exactly with PPO's 1,024-step
rollout; the first is an early recovery point rather than a paper candidate.
The 307,200-step checkpoint is the minimum full run. Continue
automatically to 614,400 when the validation curve has not plateaued.
Selection first applies numerical, crop-state, completion, and active-safety
gates, then maximizes mean validation reward. Classical algorithms are never
degraded to force the requested ranking.

## Performance prerequisite

The current V5 postprocessor builds a pandas DataFrame for every 15-minute
transition. Replace this hot path with prevalidated NumPy feature construction
and Ridge inference. Tests must compare the fast path with the authoritative
DataFrame implementation across representative and randomized inputs to tight
floating-point tolerance before long training starts.

## Outputs

The run directory contains an immutable protocol manifest, resumable checkpoints,
per-window validation trajectories and metrics, frozen selections, complete
120-day trajectories, aggregated reward/climate/safety/actuator/crop metrics,
learning-curve data, timing data, and a Markdown reasonableness report.

`fruit_state_change` and regional transferred fresh yield are simulated crop
indicators. They must not be described as target-site harvest accuracy because
the repository still has no dated Pidu harvest mass series.
