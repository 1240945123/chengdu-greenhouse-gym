# V5 Controller Smoke Benchmark Design

## Objective

Verify that Baseline, PID, MPC, PPO, and SAC can run end to end in the promoted V5
hybrid environment and produce a leakage-safe preliminary comparison.

## Baseline policy

The baseline is fixed before any validation or test run: roof ventilation target
zero and fan target zero, with the shared safety layer still active. It represents
the no-intelligent-climate-control lower reference. Reward weights, weather, safety,
and metrics are identical for all algorithms. The benchmark may expect this baseline
to be weakest, but must not edit rewards, discard episodes, or tune on test outcomes
to force that ranking.

## Data roles

- Training weather: 2023-2024 Chengdu ERA5 seasons.
- Validation weather: a fixed 2025 spring day, used for PID/MPC candidate selection
  and PPO/SAC checkpoint selection.
- Test weather: a fixed 2025 autumn day, opened once after all selections freeze.

The V5 residual and physics parameter artifacts remain frozen. No target-site rows
on or after 2026-07-12 enter any fitting step.

## Controllers

- Baseline: fixed low target `[0, 0]`.
- PID: bounded PI climate controller. Roof ventilation responds to temperature and
  excess RH; fan responds to excess temperature and is reduced when RH is high.
- MPC: enumerates supported roof/fan levels, rolls a local V5-audit-informed climate
  model over a short horizon, and minimizes comfort, effort, and action-change cost.
- PPO and SAC: Stable-Baselines3 policies trained on flattened V5 hybrid observations
  with two normalized actions. Smoke training uses compact networks and fixed seeds.

All controllers submit incremental actions through the same execution and safety
layer. Metrics use executed controls.

## Selection and promotion

Classical candidates and RL checkpoints are selected by validation total reward,
subject to no numerical failure, no residual fallback, finite states, and crop carbon
nonnegativity within `1e-6 mg/m2`. An algorithm is smoke-promoted only when its
validation reward exceeds the frozen baseline. Failure to do so is reported rather
than hidden.

## Metrics

Report total reward; temperature and RH band MAE; joint comfort; time above/below
limits; safety intervention rate; mean and variation of executed roof/fan actions;
residual fallback; crop carbon minimum; fruit-state change; inference time; and
episode completion. Yield claims remain prohibited for one-day smoke episodes.

## Scope

This is an implementation and directional smoke benchmark, not the final paper
experiment. Passing it permits multi-day, multi-season, multi-seed training under a
separately frozen protocol.
