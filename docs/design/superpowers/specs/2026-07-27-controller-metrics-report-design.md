# Controller Metrics Report Design

## Goal

Extend the Chengdu greenhouse controller benchmark so the paper compares normal control-performance indicators, not only cumulative reward.

## Scope

The report is regenerated from the existing fixed test trajectories. No controller retraining or test-window change is required. Heating and CO2 remain disabled and are not reported as controllable resources.

## Metrics

- Tracking: temperature MAE/RMSE and relative-humidity MAE/RMSE.
- Comfort: separate temperature and humidity comfort fractions, joint comfort fraction, and violation hours.
- Control: mean actuator effort, total action variation, switching count, lamp-use hours, and per-actuator mean commands.
- Runtime: mean controller wall time in milliseconds per environment step.
- Robustness: mean and sample standard deviation across PPO/SAC training seeds. Deterministic controllers retain one test run and therefore have zero sample standard deviation.

Comfort is evaluated against the dynamic reward bounds stored in each trajectory. Tracking errors use the corresponding dynamic targets. All algorithms use the same 96-hour Chengdu test interval.

## Outputs

- Preserve `comparison.csv` and `comparison.json` for complete statistics.
- Add `paper_metrics.csv` with concise numeric columns and matching standard deviations.
- Add `paper_metrics.md` with publication-readable mean +/- standard deviation values.
- Add separate tracking-error, comfort/violation, control-usage, and runtime figures. Preserve the reward and trajectory figures.

## Validation

Unit tests verify formulas, aggregation dispersion, concise report artifacts, and non-empty figures. The report is regenerated from all 13 existing trajectories and checked against the saved episode metrics.

