# Measured Harvest Trajectories Design

## Goal

Preserve physical picking records while producing the daily greenhouse batch
and cumulative-yield trajectory actually used by the harvest model evaluation.

## Outputs

For every loaded and season-bound harvest dataset, produce:

- `measured_harvest_event_trajectory.csv`: one row per physical field record,
  sorted by season, timestamp, and event ID, with within-season event sequence,
  cumulative fresh kg, and cumulative fresh kg/m2 on the fixed 192 m2 basis.
- `measured_daily_harvest_trajectory.csv`: one row per greenhouse calendar day
  with the number of physical records, summed fresh kg, summed fresh kg/m2, and
  season-local cumulative values.

The model's public `batch_*` metrics are explicitly defined as greenhouse
calendar-day picking totals. Multiple crates, grades, or weighing records on
the same day are preserved in the event artifact and summed only in the daily
evaluation artifact.

## Summary

The readiness report records per-season event count, picking-day count, first
and last harvest timestamp, event span, measured total fresh kg, final measured
yield kg/m2, completion evidence, and completeness counts for batch IDs and
source-record IDs. Ongoing seasons remain visible but ineligible for final
calibration.

## Validation

Masses must be finite and non-negative, timestamps valid, event IDs unique, and
season identity present. Cumulative mass must be monotonic and reset exactly at
season boundaries. Final event and daily totals must reconcile to numerical
tolerance for every season.
