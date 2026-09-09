# Target Crop Driver Horizon Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a diagnostic Pidu GreenLight fruit-state trajectory through the complete observed 2026 climate/control horizon and rebuild harvest drivers from it.

**Architecture:** Add one focused runner that reconstructs the existing diagnostic multi-organ parameterization, performs a single forced trajectory, and emits conservative audit labels. Keep standing-crop validation and driver construction as separate existing modules.

**Tech Stack:** Python, pandas, NumPy, CasADi, YAML, JSON, pytest.

---

### Task 1: Diagnostic extension runner

**Files:**
- Create: `experiments/crop/extend_chengdu_target_crop_trajectory.py`
- Create: `tests/test_chengdu_target_crop_extension.py`

- [x] Write a failing short-horizon integration test using observed climate and controls.
- [x] Assert exact horizon timestamps, diagnostic-only labels, parameter provenance, finite trajectory values, and output files.
- [x] Implement input loading, target initialization, parameter reconstruction, exact partial-day weather loading, audited canopy-gap proxying, forced simulation, audit generation, and CLI.
- [x] Run the focused test and confirm it passes.

### Task 2: Production horizon and driver integration

**Files:**
- Modify: `configs/crops/chengdu_harvest_driver_manifest.yml`
- Generate: `results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation/target_crop_trajectory_driver_horizon.csv`
- Generate: `results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation/target_crop_trajectory_driver_horizon_audit.json`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/drivers/target_partial_harvest_drivers.csv`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/drivers/target_partial_harvest_driver_audit.json`

- [x] Run the extension through `2026-07-20 09:00:00` with 300-second substeps.
- [x] Point the driver manifest to the extended trajectory without changing incomplete/diagnostic/protocol states.
- [x] Rebuild drivers and verify 110.375-day span and zero driver mass-accounting residual.

### Task 3: Documentation and verification

**Files:**
- Modify: `data/README.md`
- Modify: `docs/superpowers/plans/2026-07-28-target-crop-driver-horizon-extension.md`

- [x] Document extrapolation scope and retained blockers.
- [x] Run focused crop/driver/package tests: 53 passed.
- [x] Run the full suite and record the final count: 229 passed, 4 subtests passed.

No Git metadata is present; generated audits and test evidence replace commit
checkpoints.
