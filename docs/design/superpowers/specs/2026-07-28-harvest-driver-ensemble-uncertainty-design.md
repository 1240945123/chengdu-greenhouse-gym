# Harvest Driver Ensemble Uncertainty Design

## Goal

Extend tomato-harvest prediction intervals beyond one fixed GreenLight driver
trajectory by propagating traceable greenhouse-climate and/or outdoor-weather
driver scenarios together with harvest-parameter and residual uncertainty.

## Input Contract

The point driver remains authoritative for fitting and held-out point metrics.
An optional scenario CSV contains the same driver columns plus:

- `driver_scenario_id`: non-empty scenario identifier;
- `driver_uncertainty_source`: one of
  `greenhouse_climate_model_parameter_uncertainty` or
  `outdoor_weather_scenario_uncertainty`.

Every scenario must contain exactly the held-out test-season keys from the
point driver. Timestamp, `dt_hours`, target identity, planting identity,
`pick`, `pick_source`, completion status, model status, and provenance columns
must match the point driver. Physical forecast columns such as indoor air
temperature and net fruit dry-matter change may differ. At least two distinct
scenario IDs are required when the option is used.

## Joint Sampling

`joint_harvest_prediction_intervals` accepts optional scenario drivers. It
simulates every harvest-parameter draw against every supplied driver scenario,
then each predictive bootstrap sample uniformly selects one joint forecast and
adds a season-contained moving residual block. Batch and cumulative quantiles
are computed from this ensemble; cumulative mass resets for every forecast
season as before.

The supplied scenario ensemble excludes the point trajectory unless the user
explicitly includes it as a scenario. This avoids silently changing declared
scenario probabilities. Scenario IDs are sampled uniformly and parameter
draws are sampled uniformly.

## Production Integration

The production CLI adds `--uncertainty-driver-scenarios`. The calibration
function validates and filters these rows to the untouched test season before
calling the joint interval function. The canonical scenario CSV is copied to
`processed/uncertainty_driver_scenarios.csv`. Scenario count, IDs, sources, and
sampling policy are recorded in the calibration JSON.

When no ensemble is supplied, existing fixed-driver conditional uncertainty is
preserved. When supplied, only the declared driver sources move from
`excluded_sources` to `included_sources`; site-transfer and cohort-structure
uncertainty remain excluded. Readiness requires every known uncertainty source
to appear in exactly one of the included/excluded lists.

## Failure Rules

Reject missing/duplicate scenario keys, fewer than two scenarios, unknown
source labels, invalid numeric values, non-identical time axes, altered target
identity, altered management picks, unaccepted crop models, or incomplete
seasons. Validation occurs before parameter bootstrap fitting.

## Verification

Synthetic tests use low/high net-fruit-growth scenarios and zero residuals to
prove that driver scenarios widen final cumulative intervals, results are seed
reproducible, scenarios do not cross season boundaries, and point-only behavior
is unchanged. Production tests prove invalid scenarios stop fitting and audit
metadata reaches the final result.

## Current Evidence Boundary

This capability does not claim that the current Pidu driver uncertainty has
been quantified. The target remains fixed-driver conditional until accepted,
traceable GreenLight scenarios are generated from target data.
