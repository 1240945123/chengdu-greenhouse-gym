# Chengdu Harvest Field Workbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a field-ready Excel harvest ledger and direct production ingestion.

**Architecture:** Reuse the canonical CSV validation after a format-specific table reader. Add one CLI workbook path that resolves to both event and season sheets, while preserving existing CSV inputs.

**Tech Stack:** Python, pandas, pytest, `@oai/artifact-tool`, XLSX.

---

### Task 1: XLSX canonical readers

**Files:**
- Modify: `processing/chengdu_harvest_events.py`
- Modify: `processing/chengdu_harvest_seasons.py`
- Modify: `tests/test_chengdu_harvest_events.py`
- Modify: `tests/test_chengdu_harvest_seasons.py`

- [x] Add failing XLSX ingestion, blank-row, partial-row, and missing-sheet tests.
- [x] Implement CSV/XLSX readers and strict empty-row handling.
- [x] Run both focused loader test modules.

### Task 2: Production CLI workbook contract

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add failing input-contract tests for workbook resolution and mutual exclusion.
- [x] Implement `--harvest-workbook` and pass it to both canonical loaders.
- [x] Run the focused pipeline tests.

### Task 3: Field workbook artifact

**Files:**
- Create: `outputs/chengdu-harvest-intake/chengdu_harvest_field_template.xlsx`
- Modify: `data/README.md`

- [x] Build the four-sheet workbook with canonical headers and validation.
- [x] Inspect key ranges, scan formula errors, and visually render every sheet.
- [x] Document field use and the production command.

### Task 4: Verification

- [x] Run the complete test suite with temporary files on drive E.
- [x] Rebuild current readiness and confirm missing real events remains blocking.
- [x] Keep the overall goal active until real completed target seasons exist.

No Git metadata is present in this workspace; tests and artifact inspection
replace commit checkpoints.
