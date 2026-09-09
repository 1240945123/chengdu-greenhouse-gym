# Chengdu Harvest Driver Builder Design

## Purpose

Convert timestamped GreenLight crop-state trajectories into the canonical,
auditable driver table required by target harvest calibration. This removes the
current manual driver-template step while preserving the boundary between
measured harvest outcomes and exogenous model inputs.

## Input And Accounting

Each state trajectory contains timestamp, standing fruit dry matter, native
GreenLight harvest removed during the preceding interval, and indoor air
temperature. For each adjacent state pair, net fruit dry-matter change is
`current - previous + native_harvest`. The first state becomes the initial
fruit inventory; it is not emitted as a zero-duration driver row. Timestamps
must be unique and strictly chronological, intervals may not exceed 24 hours,
and all masses and temperatures must be finite.

## Picking Protocol

A declared protocol supplies local weekdays and hour. Pick permissions are
generated only by testing whether a scheduled event falls in each state
interval; measured harvest timestamps are never read. If the protocol is
missing, the builder emits no permissions, marks
`pick_source=management_protocol_missing`, and declares the bundle ineligible
for calibration.

## Evidence Boundary

Every row carries `driver_model_status`. Harvest calibration accepts only
`accepted_target_crop_model`; diagnostic and post-hoc crop trajectories are
rejected. The current Pidu partial trajectory is therefore exported as
`diagnostic_not_final`, `season_complete=false`, and protocol missing. It is a
pipeline integration artifact, not harvest calibration evidence.

## Outputs

A YAML manifest describes season identity, trajectory path, crop-model status,
initial maturity fraction, temperature source, completeness, and management
protocol. The command writes one canonical CSV and one JSON audit containing
state/driver row counts, date span, maximum interval, pick permission count,
initial inventory, native harvest added back, net fruit change, and explicit
calibration eligibility and blockers.

## Validation

Unit tests cover mass accounting, interval scheduling, missing protocol,
invalid chronology and coarse intervals. Pipeline tests prove diagnostic driver
models are rejected. A formal partial-season artifact is built from the current
target organ-calibrated trajectory and the complete test suite must pass.
