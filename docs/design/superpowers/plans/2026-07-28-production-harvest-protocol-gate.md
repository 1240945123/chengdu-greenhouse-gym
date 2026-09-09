# Production Harvest Protocol Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require verified target-site protocol coverage before production tomato-harvest parameter fitting.

**Architecture:** Extend production input resolution with a protocol source, load it through the canonical protocol loader, and require the shared package assessor to validate it before fitting. Preserve measurement-only workflows and include protocol coverage in existing calibration artifacts.

**Tech Stack:** Python, pandas, pytest, XLSX/CSV canonical loaders.

---

### Task 1: Production input contract

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add failing tests that workbook mode resolves one path for events,
  seasons, and protocol; separate mode requires `--harvest-protocol` whenever
  drivers are supplied; and mixed workbook/separate inputs fail.
- [x] Extend `validate_harvest_pipeline_inputs` to return the three resolved
  harvest paths without restricting measurement-only runs.
- [x] Run the focused input-contract tests.

### Task 2: Fit-before-gate prevention

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_package_preflight.py`

- [x] Add canonical confirmed protocol fixtures for successful synthetic
  calibration tests.
- [x] Add a failing test that provisional or incomplete protocol coverage
  raises before `bootstrap_harvest_parameter_estimates` is called.
- [x] Make `protocols` required by `calibrate_and_validate_harvest_seasons` and
  pass it to `assess_harvest_calibration_package`.
- [x] Recompute every driver interval's `pick` permission from the covering
  protocol and block mismatches, including the first `dt_hours` interval.
- [x] Run all harvest pipeline and package-gate tests: 26 passed.

### Task 3: Runner, artifact, and CLI integration

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `data/README.md`

- [x] Add `harvest_protocol_path` to `run_pipeline` and
  `--harvest-protocol` to the CLI.
- [x] Load canonical protocol rows only when fitting is requested, write
  `processed/harvest_protocols.csv`, and pass rows into calibration.
- [x] Document workbook and separate-file commands and the no-fitting behavior
  when protocol evidence is absent.

### Task 4: Verification

- [x] Run focused protocol, package, and pipeline tests on drive E: 49 passed.
- [x] Run the full suite and record the exact count: 247 passed, 4 subtests passed.
- [x] Confirm the current diagnostic target remains unvalidated and no real
  protocol rows were invented.

No Git metadata is present; test output and generated audits replace commit
checkpoints.
