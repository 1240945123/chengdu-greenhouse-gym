# Joint Harvest Parameter Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate fitted harvest-parameter uncertainty and season-aware residual uncertainty into held-out batch and cumulative yield intervals.

**Architecture:** Add a focused residual-refit bootstrap and a joint forecast ensemble to the harvest calibration module. The seasonal production pipeline retains training-only fitting, exports parameter draws, and updates its uncertainty evidence contract.

**Tech Stack:** Python, pandas, NumPy, SciPy, pytest.

---

### Task 1: Residual-refit parameter bootstrap

**Files:**
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_harvest_calibration.py`

- [x] Add a failing synthetic test calling `bootstrap_harvest_parameter_estimates` twice with the same seed and asserting identical, bounded parameter draws.
- [x] Add a failing validation test for fewer than two requested or accepted draws.
- [x] Implement season-separated residual block sampling, pseudo-observation construction, parameter refitting, rejection counts, and parameter quantiles.
- [x] Run `python -m pytest tests/test_harvest_calibration.py -q` and require all tests to pass.

### Task 2: Joint parameter and residual forecast intervals

**Files:**
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_harvest_calibration.py`

- [x] Add a failing test for `joint_harvest_prediction_intervals` asserting ordered batch/cumulative quantiles and cumulative reset by forecast season.
- [x] Implement per-draw forecast simulation, daily alignment, residual addition, non-negative clipping, and season-local cumulative sums.
- [x] Verify reproducibility and run the focused calibration tests.

### Task 3: Production pipeline and evidence

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`

- [x] Update the synthetic seasonal pipeline test to require parameter draws and `harvest_parameter_estimation_uncertainty` in included sources.
- [x] Replace production residual-only intervals with the joint method and export `parameter_bootstrap_draws.csv` plus a serializable audit.
- [x] Remove parameter uncertainty from excluded sources while preserving weather, climate-parameter, site-transfer, and structural exclusions.
- [x] Document the conditional scope and computational controls.

### Task 4: Verification and current-state audit

**Files:**
- Modify: `docs/superpowers/plans/2026-07-28-joint-harvest-parameter-uncertainty.md`

- [x] Run the complete test suite with temporary files on drive E.
- [x] Rebuild the current Chengdu readiness report and confirm zero real harvest events still prevents calibration and validation.
- [x] Record all completed checkboxes; do not mark the full user goal complete without real target seasons.

No Git metadata is present in this workspace; test output and generated audit artifacts replace commit checkpoints.
