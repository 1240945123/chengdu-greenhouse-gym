# V5 Material Safety Metrics Design

## Problem

The day-119 MPC trajectory reports a 32.29% safety-intervention fraction, but an
actuator-level audit finds only one of 96 steps with a roof-vent or fan modification
larger than `1e-8`. The action scheme and safety projector both enforce a 0.1 slew
limit. Float32/float64 differences are compared with `atol=1e-12`, so numerically
negligible differences are counted as controller safety interventions.

This is an evaluation-definition defect, not evidence that the MPC policy materially
violates safety constraints. Controller parameters, rewards, selections, and stored
day-119 trajectories remain frozen.

## Metric design

Retain the existing `safety_intervened` value as a legacy strict diagnostic. Add:

- `material_safety_intervened`: any full-control proposal/execution difference
  greater than `1e-6`;
- `active_safety_intervened`: any difference greater than `1e-6` on the controller's
  owned V5 channels, roof vent index 3 and independent fan index 6;
- material and active projection magnitudes for audit;
- serialized intervention reasons so hard weather/emergency actions remain visible.

The controller-comparison gate uses active material intervention fraction because
all compared algorithms control only roof vent and fan. Full material intervention,
legacy strict intervention, fallback count, invalid proposals, and critical safety
reasons remain reported and cannot be hidden.

## Result migration

Recompute active material fractions from the immutable stored trajectory columns
`proposed_uRoofVent`, `proposed_uFan`, `executed_uRoofVent`, and `executed_uFan`.
The full material metric and reason breakdown require a fresh rollout, but do not
change candidate selection. Re-run only the already frozen day-119 controllers on
the same scenario after instrumentation; do not introduce another holdout date.

Regenerate confirmation metrics and promotion decisions using the active material
gate. Preserve the original decision JSON as a legacy artifact and explain the
definition correction in the scientific report.

## Validation

Tests cover the threshold boundary, strict-versus-material separation, active-channel
selection, summary fractions, and promotion-gate behavior. The complete repository
suite must pass after the migration. A consistency check must confirm identical
candidate IDs, rewards, and climate metrics before and after safety re-evaluation.
