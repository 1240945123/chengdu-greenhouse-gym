# Chengdu Harvest Field Workbook Design

## Goal

Provide one practical Excel workbook for Chengdu greenhouse staff to record
measured picking events and independent crop-season completion evidence, and
allow the production harvest pipeline to ingest it without manual CSV
conversion.

## Workbook

The workbook contains four sheets:

- `FieldGuide`: Chinese instructions, target identity, fixed 192 m2 production
  area, units, collection rules, and the production command.
- `HarvestEvents`: canonical event columns accepted by the existing measured
  harvest loader. Each row is one physical picking batch; total, marketable,
  rejected mass, actual covered area, batch ID, and source record are visible.
- `HarvestSeasons`: one row per planting season with start, optional end,
  `ongoing`/`complete` status, completion source, and source record ID.
- `DataDictionary`: field name, Chinese label, required status, type/unit,
  allowed values, and validation meaning.

Input ranges are formatted as Excel tables with frozen headers, filters, date
and numeric formatting, restrained colors, and data validation for categorical
fields. Blank template rows are ignored by the loaders; partially filled rows
remain errors rather than being silently discarded.

## Ingestion

`load_harvest_events` accepts CSV or XLSX. XLSX input reads `HarvestEvents`.
`load_harvest_seasons` accepts CSV or XLSX and reads `HarvestSeasons`. All
existing identity, mass, area, duplicate, timestamp, and completion checks run
after reading, so Excel receives no weaker validation path.

The production CLI adds `--harvest-workbook`. It is mutually exclusive with
the separate `--harvest-events` and `--season-manifest` inputs and internally
uses the same workbook for both loaders. `--drivers` still requires measured
harvest input and complete-season evidence.

## Audit And Testing

Tests cover XLSX event and season ingestion, blank-row removal, partially filled
row rejection, sheet-name errors, CLI mutual exclusion, and end-to-end event to
season binding. The workbook is inspected for required sheets and headers,
formula errors, and rendered visually before export.
