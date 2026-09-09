# Joint Harvest Parameter Uncertainty Design

## Goal

Propagate uncertainty in the fitted maturity thermal time and fruit dry-matter
fraction into held-out batch-harvest and cumulative-yield intervals, while
preserving the existing within-season temporal residual dependence.

## Method

Fit the point model on training seasons only. Construct residual blocks from
the aligned training batches without crossing season boundaries. For each
parameter-bootstrap replicate, draw one moving-block residual trajectory per
training season, add it to the fitted training batches, clip at zero, and refit
both harvest parameters using the unchanged training drivers and declared
bounds. Failed or degenerate refits are rejected and counted; the experiment
must retain a declared minimum number of valid parameter draws.

For held-out prediction, simulate the test drivers once for every accepted
parameter draw. Each predictive replicate selects a parameter draw and adds a
new season-stratified moving-block residual trajectory. Batch predictions are
clipped at zero and cumulative yield is reset at each forecast-season boundary.
The existing 2.5%, 10%, 50%, 90%, and 97.5% quantiles and interval diagnostics
remain the public output contract.

## Evidence And Scope

The experiment writes a parameter-draw table containing maturity thermal time,
dry-matter fraction, objective value, and replicate ID. The uncertainty audit
reports requested, accepted, rejected, and unique parameter-draw counts plus
parameter quantiles. `harvest_parameter_estimation_uncertainty` moves from the
excluded-source list to the included-source list.

The intervals remain conditional on accepted GreenLight drivers, declared
management, and the cohort-model structure. Weather scenarios, greenhouse
climate-model parameters, site transfer, and structural uncertainty remain
explicitly excluded.

## Failure Rules

- Fewer than two bootstrap replicates is invalid.
- Training and forecast timestamps and masses must be finite.
- Residual blocks never cross training-season boundaries.
- Parameter draws must remain inside the declared calibration bounds.
- If fewer than two valid parameter draws remain, the experiment fails rather
  than silently reverting to point-parameter intervals.
- Held-out observations are used only for scoring and date alignment, never for
  fitting parameters or constructing residual distributions.

## Testing

Synthetic tests verify reproducibility, bounded and non-degenerate parameter
draws, interval monotonicity, cumulative resets by forecast season, and the
reported inclusion/exclusion sources. Existing residual-only helpers remain
available for focused diagnostics, but the production seasonal pipeline uses
the joint method.
