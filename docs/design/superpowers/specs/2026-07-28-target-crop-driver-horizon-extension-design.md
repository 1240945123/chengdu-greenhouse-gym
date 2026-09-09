# Target Crop Driver Horizon Extension Design

## Purpose

Extend the current Pidu diagnostic GreenLight crop trajectory from 2026-05-19
to the final available observed climate/control timestamp, 2026-07-20 09:00.
This prepares a greater-than-90-day fruit-state driver horizon for future
measured-harvest alignment without promoting the failed preliminary crop model.

## Approach

A focused command reruns one forced crop trajectory from the existing
2026-04-01 initialization. It loads the recorded best fruit, leaf, and stem
allocation multipliers from `organ_allocation_calibration.json`, applies them
to the default GreenLight parameter vector, and forces the crop with observed
indoor climate, outdoor weather, and recorded control states through a declared
end timestamp.

The main standing-crop validation period remains unchanged. Sensitivity and
model-selection scenarios are not extended. The new trajectory is explicitly
an extrapolation of a diagnostic parameterization that failed independent
standing-crop acceptance.

## Interface And Outputs

The command is `python -m experiments.crop.extend_chengdu_target_crop_trajectory`.
It accepts crop observations, climate, controls, environment configuration,
harvest configuration, organ-calibration JSON, start/end timestamps, and an
output directory.

It writes:

- `target_crop_trajectory_driver_horizon.csv` with hourly fruit-state and flux
  diagnostics;
- `target_crop_trajectory_driver_horizon_audit.json` with source paths,
  parameter multipliers, horizon, rows, floor corrections, and mass-balance
  residuals.

The audit always declares `driver_model_status=diagnostic_not_final`,
`season_complete=false`, `target_harvest_validated=false`, and
`use=driver_preparation_only_not_calibration_evidence`.

## Driver Integration

The Chengdu harvest-driver manifest points to the extended trajectory. The
builder can therefore produce a driver span above 90 days. Management protocol,
season-completion evidence, and target crop-model acceptance remain independent
blocking gates and are not inferred from the longer trajectory.

## Verification

A short-horizon integration test runs the real forced crop model against a
small observed-data slice and verifies the audit labels and timestamps. The
production command then runs through 2026-07-20, followed by driver rebuilding,
mass-balance checks, focused tests, and full regression.
