# Chengdu Crop Flux Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mass-balanced GreenLight crop flux diagnostics and quantify the late Pidu biomass deficit without fitting the holdout.

**Architecture:** Extend the existing forced replay to integrate a vector of GreenLight crop fluxes at each RK4 substep. Add a pure period summarizer that compares model state changes with externally supplied observed organ changes, then integrate it into the target crop report.

**Tech Stack:** Python, NumPy, pandas, CasADi, pytest.

---

### Task 1: Interval flux integration

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Extend the existing forced-replay test with expected flux columns and a near-zero interval mass-balance residual.
- [x] Run the focused test and verify failure because the columns do not exist.
- [x] Integrate photosynthesis, allocation, respiration, pruning, and harvest fluxes using the same RK4 stage weights; write kg/m2 interval values and a mass-balance residual.
- [x] Re-run the focused tests and require zero failures.

### Task 2: Late-period diagnostic summary

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add a failing test for `summarize_biomass_flux_period` using a two-boundary synthetic trajectory and observed organ masses.
- [x] Implement strict boundary matching, flux aggregation, modeled and observed organ increments, total deficit, LAI, and mass-balance checks.
- [x] Run the focused tests and require zero failures.

### Task 3: Formal evidence

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation/*`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/readiness_report.json`

- [x] Embed the 2026-04-27 to 2026-05-19 summary from the fixed multi-organ trajectory and write `late_biomass_flux_diagnostics.json`.
- [x] Run the formal target validation and inspect carbon balance and observed-model deficit.
- [x] Update configuration and documentation with measured results and limitations.
- [x] Rebuild readiness, run the full test suite, and audit CSV/JSON consistency.

This workspace has no `.git` metadata, so the verification artifacts replace branch commit checkpoints.
