# Six-season controller evaluation design

## Goal

Evaluate the frozen Baseline, PID, MPC, PPO, and SAC controllers on six independent
120-day Chengdu tomato seasons spanning 2023-2025. Produce paper-ready climate,
safety, actuation, runtime, and simulated crop-yield comparisons without selecting
models on test results.

## Evaluation protocol

- Seasons: spring and autumn in 2023, 2024, and 2025, using the canonical start
  days already defined by the Chengdu crop simulation.
- Horizon and resolution: 120 days at the benchmark environment's native 900 s
  control interval, or 11,520 steps per complete episode.
- Controllers: one deterministic run per season for Baseline, PID, and MPC; all
  five frozen training seeds per season for PPO and SAC.
- Parameter policy: load the selected PID/MPC validation parameters and saved
  PPO/SAC models. No tuning, retraining, or best-seed selection uses test seasons.
- Disabled controls: heating and CO2 remain exactly zero.

## Architecture

Add optional growth-year overrides to the existing classical and RL evaluation
functions while preserving their current call signatures. A separate six-season
runner owns scenario expansion, artifact naming, resumable status, validation,
episode summaries, and aggregate tables. Existing single-scenario benchmark files
remain unchanged.

Each unit of work is identified by `(algorithm, seed, season_id)`. A trajectory is
written atomically before its metrics row is marked complete. Resume mode accepts
an existing unit only after checking scenario identity, expected step count,
termination state, finite numeric values, and disabled controls.

## Outputs

The runner writes under
`results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d_guarded_v2/`:

- `resolved_protocol.json`: immutable season, model, timestep, and provenance data.
- `evaluation_manifest.json`: per-unit running, complete, or failed state.
- `trajectories/<algorithm>_seed_<seed>_<season_id>.csv`: full 15-minute traces.
- `episode_metrics.csv`: one row per algorithm, seed, and season.
- `comparison_by_algorithm.csv`: aggregate across seasons and RL seeds.
- `comparison_by_season.csv`: season-specific algorithm means.

Yield columns are explicitly named and documented as GreenLight simulated yield.
They are suitable for controller comparison within this simulator, not claims of
validated prediction accuracy against Chengdu harvest observations.

## Validation and failure handling

A smoke mode runs a small bounded number of steps for model-loading and numerical
checks but does not mark episodes complete. Full episodes must contain 11,520 rows,
end by termination rather than truncation, contain only finite numeric values, and
keep `uBoil` and `uCO2` at zero. Failed units record the exception and remain
resumable. Aggregate tables are generated only from validated complete units.

## Statistical reporting

The primary table uses all six seasons and all frozen RL seeds. It reports mean,
standard deviation, median, and bootstrap 95% confidence intervals for cumulative
reward, plus mean and standard deviation for climate, comfort, actuator, safety,
runtime, and crop metrics. The season table exposes weather sensitivity. Any
best-seed view is secondary and must be labelled exploratory.

## Test strategy

Tests first establish that both evaluator types honor explicit year/start-day
overrides, that unit keys do not collide across seasons, that valid completed
artifacts resume without rerunning, and that incomplete or mismatched artifacts are
rejected. Synthetic episode metrics verify season and algorithm aggregation before
the expensive physical simulations are started.
