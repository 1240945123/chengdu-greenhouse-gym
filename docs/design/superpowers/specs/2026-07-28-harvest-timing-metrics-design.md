# Harvest Timing Metrics Design

## Goal

Evaluate the full temporal distribution of tomato harvest, not only the first
harvest date, and use those metrics in held-out target-season acceptance.

## Metrics

For each season, daily observed and predicted fresh harvest are normalized to
unit mass distributions on their common calendar-day grid.

- `harvest_timing_wasserstein_days`: the one-dimensional Wasserstein distance,
  computed as the sum of absolute differences between normalized cumulative
  mass distributions. It measures how many days the harvested mass is shifted
  overall and remains stable when one physical picking is split into batches.
- `median_harvest_date_error_days`: predicted minus observed date when
  cumulative harvested mass first reaches 50% of its season total.
- `p90_harvest_date_error_days`: the corresponding error at 90%, capturing
  late-season tail timing.

The existing first-harvest error remains. If predicted total harvest is zero,
all new timing metrics are non-finite and acceptance fails.

For multiple seasons, signed quantile-date errors are summarized as mean
absolute error and maximum absolute error. Wasserstein distances are averaged
across seasons and also report the maximum season value. Calendar gaps between
seasons are never included.

## Acceptance

Held-out acceptance requires mean Wasserstein distance no greater than 7 days,
median-harvest-date absolute error no greater than 7 days, and p90 date absolute
error no greater than 10 days. Thresholds are explicit in the Chengdu harvest
configuration and the returned acceptance audit.

## Testing

Tests use shifted synthetic harvest distributions with analytically known
timing errors, multi-season gaps, zero predicted yield, and acceptance cases
that pass mass metrics but fail temporal-distribution gates.
