# V5 Actuator Identifiability Experiment Design

## Objective

Determine which non-heating, non-CO2 greenhouse actuators are sufficiently
represented in the valid tomato cultivation window and produce credible local
climate responses before they are exposed to PID, MPC, PPO, or SAC.

## Alternatives considered

1. Train controllers immediately in the existing benchmark. Rejected because that
   benchmark still uses the older ChengduPhysics backend and cannot test V5.
2. Replace GreenLight climate states immediately with V5 predictions. Deferred
   because unsupported actions could be exploited and crop-state conservation must
   first be checked.
3. Run an actuator-support and counterfactual-response audit first. Selected because
   it is leakage-safe, inexpensive, and establishes the defensible action space for
   the hybrid digital twin.

## Data boundary

The audit uses only the V5 training split derived from immutable V4 observations.
Every retained transition satisfies `next_timestamp < 2026-07-12 00:00`. Validation
and test rows do not set support thresholds or action eligibility.

## Actuators

The candidates are `uRoofVent`, `uFan`, `uPad`, `uLamp`, `uThScr`, and `uBlScr`.
`uBoil` and `uCO2` remain disabled by project scope.

## Analyses

For every actuator, report row count, active count and fraction, rounded unique
levels, switching count, and observed quantiles. An actuator has basic empirical
support when it has at least 30 active rows, at least 20 switches, and at least
three rounded levels.

For supported cooling actuators, evaluate one-step low-to-high perturbations on
training rows where indoor temperature exceeds outdoor temperature by at least
2 C. Reinitialize the fixed ChengduPhysicsV4 plus selected V5 residual predictor
for every row so responses remain local. Report median and interquartile changes
in temperature and RH. Roof ventilation and fan response pass when the median
temperature change is negative and at least 60% of sampled responses are cooling.

The counterfactual response is a model sanity check, not a causal estimate from
observational data. Sparse or confounded actuators are marked unsupported rather
than assigned an optimistic response.

## Outputs

Write machine-readable `actuator_support.csv`, `counterfactual_responses.csv`, and
`audit.json`, plus a concise `audit.md`. The audit must include input and model
SHA-256 hashes, cutoff metadata, thresholds, and the resulting recommended control
action space.

## Promotion rule

Only empirically supported actuators with a passing directional check enter the
first hybrid digital-twin controller experiment. Screens may remain fixed schedule
disturbances because their thermal effect is context dependent. Sparse wet-pad or
lamp observations cannot support unconstrained controller optimization.

## Verification

Unit tests cover support classification, cultivation cutoff rejection, local
perturbation isolation, and deterministic artifact generation. The existing full
test suite must remain green.
