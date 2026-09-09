# Harvest Driver Scenario Bundle Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build validated uncertainty-driver scenario CSVs directly from traceable GreenLight state trajectories.

**Architecture:** Reuse the canonical protocol loader and state-to-driver builder for every manifest scenario, then run the uncertainty scenario validator against one accepted point driver. Emit the combined CSV only with a complete JSON audit.

**Tech Stack:** Python, pandas, PyYAML, JSON, pytest, XLSX/CSV.

---

### Task 1: Scenario manifest and successful bundle

**Files:**
- Create: `experiments/crop/build_chengdu_harvest_driver_scenarios.py`
- Create: `tests/test_chengdu_harvest_driver_scenarios.py`

- [x] Add a failing test with two complete synthetic state trajectories, a
  confirmed protocol workbook, and an accepted point driver.
- [x] Parse strict manifest/season/scenario fields, build each scenario through
  `build_harvest_drivers_from_state_trajectory`, and add scenario provenance.
- [x] Validate the combined bundle against the point driver and write CSV/JSON.

### Task 2: Failure and provenance gates

**Files:**
- Modify: `tests/test_chengdu_harvest_driver_scenarios.py`
- Modify: `experiments/crop/build_chengdu_harvest_driver_scenarios.py`

- [x] Add failing tests for duplicate/unknown scenarios, timestamp mismatch,
  incomplete season, unaccepted model, and provisional protocol coverage.
- [x] Produce explicit errors before output files are written.
- [x] Record resolved source paths and protocol confirmation sources in audit.

### Task 3: CLI and documentation

**Files:**
- Modify: `data/README.md`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`

- [x] Add a CLI with manifest, protocol workbook, output CSV, and audit paths.
- [x] Document the manifest schema and the two-step build/calibrate workflow.
- [x] State that diagnostic target trajectories remain ineligible.

### Task 4: Verification

- [x] Run scenario-builder, driver, protocol, uncertainty, and production tests:
  `81 passed`.
- [x] Run the full suite: `263 passed, 4 subtests passed`.
- [x] Confirm no current target production scenario artifact was created.

No Git metadata is present; test output and generated audits replace commit
checkpoints.
