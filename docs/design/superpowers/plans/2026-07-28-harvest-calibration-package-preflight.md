# Harvest Calibration Package Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast joint preflight for measured harvest and GreenLight driver packages before calibration.

**Architecture:** Put season-level package assessment in a focused module and call it from both the standalone CLI and the production calibration entry. Canonical workbook loading remains in the CLI; the shared assessor consumes canonical bound events and driver rows, classifies readiness without fitting, and returns a serializable audit.

**Tech Stack:** Python, pandas, YAML, JSON, XLSX, CSV, pytest.

---

### Task 1: Shared package assessment

**Files:**
- Create: `experiments/crop/harvest_package_validation.py`
- Create: `tests/test_chengdu_harvest_package_preflight.py`

- [x] Write a failing test with three complete 100-day seasons and eight measured events per season.
- [x] Assert the assessment returns three qualifying seasons and the chronological `s1` train, `s2` validation, `s3` test split.
- [x] Implement structural validation, season metrics, blockers, readiness flags, and split construction without fitting parameters.
- [x] Run the focused test and confirm it passes.

### Task 2: Business blockers and production shared gate

**Files:**
- Modify: `tests/test_chengdu_harvest_package_preflight.py`
- Modify: `experiments/crop/harvest_package_validation.py`
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`

- [x] Add a failing test where the second season is ongoing and its driver model is `diagnostic_not_final`.
- [x] Assert both blockers are retained, only the first season qualifies, and independent validation is false.
- [x] Add `require_independent_validation_package(...)` and invoke it from production calibration before fitting.
- [x] Run package and existing production-pipeline tests.

### Task 3: Standalone CLI and artifacts

**Files:**
- Create: `experiments/crop/validate_chengdu_harvest_package.py`
- Modify: `tests/test_chengdu_harvest_package_preflight.py`
- Modify: `data/README.md`

- [x] Add a failing real-XLSX/CSV CLI-facing API test.
- [x] Load repository target identity, canonicalize and bind workbook events, load drivers, run shared assessment, and write `harvest_calibration_package_preflight.json`.
- [x] Keep `harvest_parameters_fitted` and `target_harvest_validated` false in every preflight report.
- [x] Document command usage and the distinction between package readiness and accepted held-out accuracy.

### Task 4: Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-28-harvest-calibration-package-preflight.md`

- [x] Run focused harvest-package and production calibration tests with temporary files on drive E.
- [x] Execute the CLI against a synthetic populated package and inspect the JSON.
- [x] Run the complete test suite and record the final count: 226 passed, 4 subtests passed.

No Git metadata is present; test evidence and generated reports replace commit
checkpoints.
