# Chengdu Crop Flux Diagnostics Design

## Goal

Expose an auditable carbohydrate budget for the measured-climate crop replay
and compare the model's 2026-04-27 to 2026-05-19 biomass increment with the
Pidu leaf, stem, and fruit observations.

## Model Boundary

The forced replay integrates GreenLight photosynthesis, organ allocation,
growth respiration, organ maintenance, leaf pruning, and native fruit harvest
alongside the crop states. Fluxes retain GreenLight's carbohydrate-equivalent
mass semantics and are reported in kg/m2 per interval. They are not relabeled
as measured dry matter.

Each interval must satisfy the crop-state mass balance after accounting for
the explicit nonnegative floor correction. A diagnostic period summarizer
uses the fixed multi-organ trajectory because it was fitted on all three
organs on 2026-04-27. It compares model and inferred observed organ increments
through 2026-05-19, while preserving the transferred Xindu dry-matter-fraction
limitation.

## Outputs

The organ-calibrated trajectory gains interval flux columns and a mass-balance
residual. `late_biomass_flux_diagnostics.json` reports gross photosynthesis,
allocation, respiration, pruning, state changes, observed changes, model
deficit, and start/end modeled LAI. The target validation report embeds the
summary with `diagnostic_not_parameter_calibration` status.

## Acceptance

Unit tests prove interval mass balance and chronological period aggregation.
The formal artifact must have a negligible integrated mass-balance residual.
No photosynthesis or allocation parameter may be selected from the 2026-05-19
comparison, and readiness remains false without real harvest events.
