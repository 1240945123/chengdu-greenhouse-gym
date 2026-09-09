# Harvest Field Preflight Design

## Goal

Allow greenhouse staff to validate a partially or fully populated harvest
workbook without requiring SQL extraction, crop-state simulation, or driver
files.

## Command

`python -m experiments.crop.validate_chengdu_harvest_workbook --harvest-workbook <file.xlsx>`
loads the target identity and fixed production area from repository
configuration, then uses the same event loader, season loader, binding rules,
and measured-trajectory builder as the production calibration pipeline.

## Report

The command writes a UTF-8 JSON report plus canonical event and daily
trajectory CSV files. It reports per-season physical event count, picking-day
count, event span, total fresh kg, final kg/m2, completion evidence, batch/source
record completeness, and whether the season satisfies at least 8 events across
30 days.

Global flags distinguish:

- `measurement_calibration_coverage_ready`: at least one complete qualifying
  target season.
- `measurement_independent_validation_coverage_ready`: at least two complete
  qualifying target seasons.

These flags certify measurement coverage only. They never imply an accepted
GreenLight driver model, fitted harvest parameters, or final target validation.

## Failure And Warnings

Structural errors such as invalid mass, area, identity, timestamp, duplicate
records, or invalid season evidence fail immediately. Missing optional batch or
source IDs are counted as traceability warnings. Ongoing seasons remain visible
but cannot satisfy complete-season coverage.

## Testing

End-to-end XLSX tests cover two qualifying complete seasons, a complete plus an
ongoing season, output artifact reconciliation, and strict distinction between
measurement readiness and model validation.
