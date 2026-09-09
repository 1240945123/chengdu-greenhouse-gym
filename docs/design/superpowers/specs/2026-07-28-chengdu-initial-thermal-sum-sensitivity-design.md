# Chengdu Initial Thermal Sum Sensitivity Design

## Purpose

Quantify how the unknown pre-baseline GreenLight canopy thermal sum affects
the held-out 2026-05-19 standing-fruit prediction without fitting any parameter
to that held-out mass.

## Evidence Boundary

The prior uses only the three 2026-03-30 flower-truss observations and the
first seven days of measured indoor air temperature beginning 2026-04-01.
The De Koning temperature-rate equation converts flower-truss count into an
equivalent developmental duration and thermal sum. Temperature quantiles
produce low, median, and high scenarios. The 2026-04-27 standing-fruit mean may
fit `rgFruit` separately within each scenario. The 2026-05-19 observation is
used only after scenario construction and calibration are complete.

## Outputs

`derive_initial_thermal_sum_scenarios` returns auditable scenario inputs,
formula metadata, observation counts, temperature window, and limitations.
The target validation runner writes one row per scenario to
`initial_thermal_sum_sensitivity.csv` and a structured report to
`initial_thermal_sum_sensitivity.json`. Each row reports initial thermal sum,
selected `rgFruit` multiplier, calibration prediction/error, held-out
prediction/error, and whether the scenario envelope contains the observed
held-out mean.

## Interpretation

The scenario range is structural sensitivity, not a confidence or predictive
interval. Scenario parameters remain diagnostic and cannot become the final
target parameterization. If all scenarios retain a large held-out error, the
missing thermal sum is not sufficient to explain the late biomass deficit.

## Validation

Unit tests cover deterministic scenario construction, positive finite inputs,
strict pre-holdout evidence fields, and chronological calibration. Existing
readiness gates remain unchanged: no sensitivity scenario can promote the
system to target harvest validation.
