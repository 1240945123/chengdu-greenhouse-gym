# Production Harvest Protocol Gate Design

## Goal

Prevent production harvest calibration from trusting an unverified
`pick_source` label in a driver CSV. Every calibrated season must have a
traceable, confirmed `HarvestProtocol` record that continuously covers its
complete driver horizon.

## Chosen Approach

The production calibration function receives canonical protocol rows as a
required keyword argument and passes them to the existing shared package
assessor. The production runner resolves protocol input in one of two ways:

- `--harvest-workbook`: the same XLSX/XLSM supplies `HarvestEvents`,
  `HarvestSeasons`, and `HarvestProtocol`;
- separate files: `--harvest-events`, `--season-manifest`, and the new
  `--harvest-protocol` are supplied together.

This is preferred to a CLI-only check because direct Python callers must pass
the same gate. Embedding only a protocol hash in the driver is deferred: a hash
can prove file identity but cannot replace semantic coverage validation.

## Input Rules

- `--drivers` requires harvest events, a season manifest, and a protocol file.
- `--harvest-workbook` is mutually exclusive with all three separate harvest
  input arguments.
- Measurement-only runs remain allowed without a protocol and do not fit
  parameters.
- Missing workbook `HarvestProtocol`, provisional periods, identity mismatch,
  uncovered driver intervals, and protocol-versus-driver `pick` mismatches stop
  calibration before bootstrap fitting.

## Data Flow

`run_pipeline` loads canonical protocol rows with
`load_harvest_protocols`, writes them to
`processed/harvest_protocols.csv`, and calls
`calibrate_and_validate_harvest_seasons`. That function calls
`assess_harvest_calibration_package(..., protocols=protocols)` and requires an
independent validation package before any model parameter estimation. The
returned `calibration_package_assessment` preserves protocol coverage by
season in the final JSON artifact. Coverage begins at the first driver
timestamp minus its declared `dt_hours`. The assessor also recomputes each
interval's expected `pick` from the covering protocol period and records a
`management_protocol_pick_mismatch` blocker if the CSV value differs.

## Verification

Tests prove that input resolution selects the workbook protocol sheet or the
explicit protocol file, rejects missing/mixed protocol sources, blocks direct
calibration with provisional or incomplete coverage before fitting, and allows
synthetic confirmed full-horizon periods. A separate test proves that a forged
`pick_source` label with inconsistent `pick` values is rejected. The full
repository suite must pass.

## Current Target Status

This gate does not create site records or promote the current Pidu diagnostic
trajectory. The current target remains unvalidated until real harvest events,
confirmed management protocols, complete seasons, and accepted crop drivers
are available.
