# Chengdu Stage-Dependent Sink Design

## Purpose

Evaluate whether a phenology-driven increase in organ sink capacity can
represent the observed late growth acceleration that constant GreenLight sink
coefficients and constant photosynthetic-capacity scenarios cannot reproduce.

## Scientific Status

The hypothesis was formed after inspecting the 2026-05-19 discrepancy. The
experiment is therefore post-hoc model development, not independent held-out
validation. Existing constant-model held-out metrics remain unchanged. No
stage scenario may promote readiness or become a final target parameterization
without a later independent target date or season.

## Registered Structures

The reference thermal sum is the fixed multi-organ model state at 2026-04-27.
Before that state, all sink multipliers equal one, preserving calibration-date
behavior. After it, scenarios use `max(1, (tCanSum/reference)^exponent)` capped
at two. Registered scenarios are constant, square-root all-organ, linear
all-organ, square-root fruit-only, and linear fruit-only. Exponents and organ
sets are not fitted to 2026-05-19.

## Outputs

Trajectories expose leaf, stem, and fruit stage multipliers. The runner writes
`stage_sink_structure_comparison.csv` and `.json` with calibration and
development-date organ errors, fresh fruit predictions, and non-selective
envelopes. Reports carry `posthoc_structure_development=true` and
`independent_validation=false`.
