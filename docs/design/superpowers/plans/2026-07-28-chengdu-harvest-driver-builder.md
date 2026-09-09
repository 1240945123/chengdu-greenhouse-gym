# Chengdu Harvest Driver Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build canonical harvest drivers directly from GreenLight state trajectories with auditable protocol and crop-model evidence gates.

**Architecture:** A focused processing module performs pure state-to-driver conversion and manifest orchestration. A small experiment CLI writes CSV/JSON artifacts. The existing harvest pipeline rejects any driver bundle whose crop model has not been independently accepted.

**Tech Stack:** Python, pandas, NumPy, PyYAML, pytest.

---

### Task 1: Pure trajectory conversion

**Files:**
- Create: `processing/chengdu_harvest_drivers.py`
- Create: `tests/test_chengdu_harvest_drivers.py`

- [x] Add a failing test with three crop states that verifies adjacent-state mass accounting, initial inventory, interval durations, protocol-derived picks, provenance columns, and audit totals.
- [x] Add failing tests for missing protocol, duplicate/nonchronological timestamps, nonzero first-row native harvest, and intervals longer than 24 hours.
- [x] Implement `build_harvest_drivers_from_state_trajectory` and run the focused tests.

### Task 2: Manifest command and current partial-season artifact

**Files:**
- Create: `experiments/crop/build_chengdu_harvest_drivers.py`
- Create: `configs/crops/chengdu_harvest_driver_manifest.yml`
- Modify: `tests/test_chengdu_harvest_drivers.py`
- Generate: `results/chengdu_agri_greenhouse_001/harvest_model/drivers/target_partial_harvest_drivers.csv`
- Generate: `results/chengdu_agri_greenhouse_001/harvest_model/drivers/target_partial_harvest_driver_audit.json`

- [x] Add a failing manifest test that combines two temporary trajectory files and preserves chronological season identity.
- [x] Implement YAML loading, per-season conversion, overlap checks, CSV/JSON writing, and the module CLI.
- [x] Build the current Pidu partial-season artifact with `diagnostic_not_final`, missing protocol, and `season_complete=false`.

### Task 3: Calibration evidence gate

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `data/raw/chengdu_agri/greenhouse_001/crop/harvest_drivers_template.csv`

- [x] Add a failing test proving `diagnostic_not_final` drivers cannot enter calibration.
- [x] Require `driver_model_status=accepted_target_crop_model` for every calibration and validation driver row.
- [x] Update existing valid synthetic drivers and the template, then run focused tests.

### Task 4: Documentation and audit

**Files:**
- Modify: `data/README.md`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`

- [x] Document the builder command, manifest fields, and current artifact blockers.
- [x] Verify the generated audit marks the current bundle ineligible and readiness remains false.
- [x] Run the complete test suite and artifact consistency checks.

No Git metadata is present in this workspace; generated artifacts and full-suite checks replace commit checkpoints.
