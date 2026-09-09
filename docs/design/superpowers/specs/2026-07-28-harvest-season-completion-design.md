# Harvest Season Completion Evidence Design

## Purpose

Prevent partial or still-running crops from being treated as complete seasons
for final cumulative-yield calibration and validation.

## Manifest

A separate CSV stores one row per season: season ID, greenhouse identity,
planting code, cultivar, start datetime, optional end datetime, completion
status (`ongoing` or `complete`), completion source, and source record ID. A
complete season requires a valid end after its start and a non-empty source.
An ongoing season cannot declare an end.

## Event Binding

Every harvest event must map to exactly one manifest season with matching
greenhouse, planting, and cultivar identity. Events must occur on or after the
season start and, for complete seasons, no later than the end. Binding adds
season bounds, completion status, source, and `season_complete_evidence` to the
canonical event table.

## Pipeline Gate

The production pipeline requires `--season-manifest` whenever
`--harvest-events` is supplied. Ongoing-season events remain auditable but are
not eligible for calibration or independent validation. Direct synthetic unit
interfaces remain usable, while production artifacts carry explicit completion
evidence and audit counts.

## Validation

Tests cover complete and ongoing seasons, inconsistent identities, events
outside bounds, missing completion source, and pipeline argument requirements.
Current readiness remains false because no target harvest events or season
completion records exist.
