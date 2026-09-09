# Harvest Management Protocol Workbook Design

## Purpose

Make harvest timing an explicit, target-site management input rather than a
YAML assumption or a value inferred from observed harvest outcomes. Add
traceable, season-aware picking schedules to the Chengdu field workbook and
use them to build GreenLight harvest-driver pick permissions.

## Workbook Schema

Add a `HarvestProtocol` sheet. Each row is one effective schedule period with:

- `protocol_id`, `season_id`, target greenhouse and planting identity;
- `timezone`, `effective_start_datetime`, optional
  `effective_end_datetime`;
- `protocol_status`: `confirmed` or `provisional`;
- `pick_weekdays`: unique comma-separated integers where Monday is 0 and
  Sunday is 6;
- `pick_local_hour`: integer 0 through 23;
- `confirmation_source`, `source_record_id`, and optional `notes`.

Multiple periods per season are allowed so actual schedule changes are not
collapsed into one rule. Periods must be chronological and non-overlapping.
Blank rows are ignored. Partially filled rows and duplicate protocol IDs are
errors. Confirmed records require a non-empty confirmation source and source
record ID. Provisional records remain traceable but cannot support calibration.

## Driver Semantics

The driver builder accepts an optional protocol workbook. Workbook periods
override the legacy global manifest protocol for matching seasons. Each driver
interval is matched against protocol effective periods without using measured
harvest timestamps.

A season receives `pick_source=management_protocol` only when confirmed
periods cover its full driver horizon. Provisional coverage uses
`provisional_management_protocol`; missing or gapped coverage uses
`management_protocol_missing`. Only the first value is calibration eligible.
The audit records coverage fraction, uncovered intervals, protocol IDs,
confirmation sources, and generated pick count.

## Preflight Integration

Workbook-only preflight reports protocol records and coverage through the last
currently measured harvest date. Joint package preflight uses the complete
driver horizon and reports protocol coverage per season. Measurement coverage,
driver readiness, protocol readiness, parameter fitting, and final target
validation remain separate flags.

## Backward Compatibility

Existing CSV event/season inputs and manifests continue to work. The legacy
global manifest status `declared` retains its current behavior. Old workbooks
without `HarvestProtocol` remain readable for measured harvest, but preflight
reports `management_protocol_sheet_missing` and cannot claim protocol readiness.

## Testing And Artifact

Tests cover XLSX parsing, multiple periods, overlap and identity errors,
confirmed/provisional/missing driver semantics, protocol coverage, and the
prohibition on deriving picks from harvest events. The field template is
regenerated with five sheets, inspected for headers and validations, scanned
for formula errors, and visually rendered before export.
