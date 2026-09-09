# Season-Aware Harvest Uncertainty Design

## Purpose

Strengthen conditional harvest uncertainty so moving residual blocks never
cross biological season boundaries and uncertainty is reported for both daily
batch fresh mass and cumulative yield.

## Resampling

Training residuals are partitioned by `season_id` when present. Each block
first chooses a training season, then chooses a circular start within that
season; no block may concatenate the end of one season with the beginning of
another. Forecast cumulative trajectories reset at each forecast season. The
single-season interface remains backward compatible.

## Outputs And Metrics

Bootstrap output contains batch and cumulative 2.5%, 10%, 50%, 90%, and 97.5%
quantiles. Evaluation reports 80% and 95% coverage, mean width, normalized mean
width, and Winkler score for each target. Batch width is normalized by mean
observed daily harvest; cumulative width remains normalized by final observed
yield. Existing cumulative acceptance thresholds remain unchanged.

## Scope

The intervals capture temporal dependence in training batch residuals,
stratified by season. They remain conditional on supplied GreenLight drivers,
crop-model structure, management protocol, and fitted parameters. Parameter,
weather, climate-model, site-transfer, and structural uncertainty remain
explicitly excluded.

## Validation

Tests prove an individual residual block contains values from only one season,
batch quantiles are finite and ordered, cumulative forecasts reset by season,
and batch interval metrics are reported. The synthetic held-out pipeline must
emit both interval families and the complete suite must pass.
