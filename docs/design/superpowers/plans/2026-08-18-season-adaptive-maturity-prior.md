# Season-Adaptive Maturity Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace cross-season pooled maturity sampling with season-matched thermal-time intervals in the existing 120-day regional yield projection.

**Architecture:** Calibration writes per-season bounds while retaining a global fallback. Projection loading validates the mapping, the existing Latin-hypercube sampler selects the matching interval, and a prefix-aware audit writes v2 results beside untouched v1 artifacts.

**Tech Stack:** Python, pandas, NumPy, SciPy, pytest, YAML/JSON artifacts

---

### Task 1: Calibrate season-specific maturity bounds

**Files:**
- Modify: `experiments/crop/calibrate_chengdu_regional_yield_transfer.py`
- Test: `tests/test_chengdu_regional_yield_transfer.py`

- [ ] Add a failing test asserting each season exposes ordered `low_deg_day`, `central_deg_day`, and `high_deg_day` values.
- [ ] Run `pytest tests/test_chengdu_regional_yield_transfer.py -q` and confirm the new assertion fails because the mapping is absent.
- [ ] Add the validated season mapping to `derive_maturity_thermal_time_prior` while retaining global bounds.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Use matching bounds during scenario sampling

**Files:**
- Modify: `experiments/crop/project_controller_cohort_yield.py`
- Modify: `experiments/crop/simulate_chengdu_greenlight_harvest_priors.py`
- Test: `tests/test_controller_cohort_yield_projection.py`

- [ ] Add failing tests for loading seasonal bounds and sampling distinct spring and autumn maturity ranges.
- [ ] Run the focused tests and confirm failure is caused by global-only sampling.
- [ ] Validate and expose `maturity_thermal_time_deg_day_by_season` in the regional prior loader.
- [ ] Select the season-specific interval in `sample_harvest_scenarios`, with the global interval as a backward-compatible fallback.
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Add prefix-aware timing audit

**Files:**
- Modify: `experiments/crop/audit_regional_controller_yield.py`
- Test: `tests/test_chengdu_regional_yield_transfer.py`

- [ ] Add a failing test for 65-75 day first-harvest coverage.
- [ ] Add timing-window coverage to season and algorithm summaries.
- [ ] Add an artifact prefix argument so v2 audit files are written without replacing v1.
- [ ] Run focused tests and confirm they pass.

### Task 4: Regenerate and verify v2 artifacts

**Files:**
- Modify: `configs/crops/chengdu_regional_yield_transfer.yml`
- Regenerate: `data/processed/external/chengdu_regional_tomato/regional_transfer_priors.json`
- Create: `results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d_guarded_v2/regional_v2_*`

- [ ] Run regional calibration and verify all six season intervals are ordered.
- [ ] Run the 78-episode projection with 16 draws and `regional_v2_` prefix.
- [ ] Run the prefix-aware audit and compare v2 timing and yield metrics with v1.
- [ ] Run the full test suite and record the final pass count.

