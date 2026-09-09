# Harvest Management Protocol Workbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add traceable, season-aware harvest schedules to the field workbook and driver pipeline without deriving timing from harvest outcomes.

**Architecture:** A canonical protocol loader validates workbook rows and effective periods. Driver construction resolves confirmed/provisional/missing schedule coverage per season, while preflight reports coverage separately from model validation.

**Tech Stack:** Python, pandas, pytest, YAML, JSON, `@oai/artifact-tool`, XLSX.

---

### Task 1: Canonical protocol loader

**Files:**
- Create: `processing/chengdu_harvest_protocols.py`
- Create: `tests/test_chengdu_harvest_protocols.py`

- [x] Write failing tests for confirmed and provisional XLSX rows, blank rows, weekday parsing, target identity, duplicate IDs, and overlapping effective periods.
- [x] Implement canonical loading, timezone conversion, strict status/hour/source validation, and per-season ordering.
- [x] Run the focused loader tests.

### Task 2: Period-aware driver schedules

**Files:**
- Modify: `processing/chengdu_harvest_drivers.py`
- Modify: `experiments/crop/build_chengdu_harvest_drivers.py`
- Modify: `tests/test_chengdu_harvest_drivers.py`

- [x] Add failing tests for confirmed full coverage, provisional periods, uncovered intervals, and mid-season schedule changes.
- [x] Add optional protocol-workbook input and period resolution while preserving the legacy manifest protocol.
- [x] Record protocol coverage and provenance in driver audits; only full confirmed coverage is eligible.
- [x] Run driver and calibration-gate tests.

### Task 3: Preflight integration

**Files:**
- Modify: `experiments/crop/validate_chengdu_harvest_workbook.py`
- Modify: `experiments/crop/validate_chengdu_harvest_package.py`
- Modify: `experiments/crop/harvest_package_validation.py`
- Modify: `tests/test_chengdu_harvest_preflight.py`
- Modify: `tests/test_chengdu_harvest_package_preflight.py`

- [x] Add failing protocol presence and driver-horizon coverage tests.
- [x] Report protocol readiness, missing-sheet warnings, provisional status, and per-season coverage without promoting target validation.
- [x] Run all preflight tests.

### Task 4: Five-sheet field workbook

**Files:**
- Modify: `outputs/chengdu-harvest-intake/chengdu_harvest_field_template.xlsx`
- Modify: `data/README.md`

- [x] Add `HarvestProtocol`, field-guide instructions, and dictionary rows using `@oai/artifact-tool` while preserving existing style.
- [x] Inspect workbook values/validations, scan formula errors, and render all five sheets.
- [x] Document confirmed versus provisional schedule semantics and CLI usage.

### Task 5: Verification

- [x] Run focused harvest loader, driver, preflight, and pipeline tests on drive E: 28 passed.
- [x] Rebuild the current diagnostic driver and confirm protocol remains missing until site data are entered.
- [x] Run the full suite and record the final count: 244 passed, 4 subtests passed.

No Git metadata is present; test results and generated workbook/JSON audits
replace commit checkpoints.
