# Chengdu Assimilation Capacity Sensitivity Design

## Goal

Test whether plausible changes in GreenLight leaf electron-transport capacity
can explain the late Pidu biomass deficit after leaf, stem, and fruit sink
coefficients are re-calibrated exclusively on 2026-04-27.

## Registered Scenarios

The `j25LeafMax` parameter at index 129 uses fixed multipliers 0.75, 1.00,
1.25, 1.50, and 2.00. No multiplier is selected or refined from the
2026-05-19 outcome. Within each scenario, `rgFruit`, `rgLeaf`, and `rgStem`
start from the prior three-organ solution and undergo three proportional
calibration evaluations using only the 2026-04-27 inferred dry organ masses.
Updates are bounded and auditable.

## Evaluation

Each calibrated scenario is replayed through 2026-05-19. The report includes
calibration organ NRMSE, held-out organ NRMSE, held-out fresh fruit mass, organ
relative errors, and the envelope across all registered scenarios. The output
must not contain a selected scenario or final target parameterization.

## Interpretation

This is a structural sensitivity analysis, not predictive uncertainty. If
capacity changes can improve the holdout only while degrading the calibration
date, or if re-calibration collapses the holdout predictions back to the same
range, a stage-dependent sink or missing state transition is implicated rather
than a constant photosynthetic-capacity correction.
