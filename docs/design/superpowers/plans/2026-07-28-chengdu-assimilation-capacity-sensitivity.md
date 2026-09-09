# Chengdu Assimilation Capacity Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify leakage-free photosynthetic-capacity sensitivity after calibration-date-only organ sink re-fitting.

**Architecture:** Add a proportional bounded organ calibration helper that accepts a base GreenLight parameter vector. Run five pre-registered `j25LeafMax` scenarios, calibrate each only through 2026-04-27, then evaluate all on 2026-05-19 and save a non-selective structural envelope.

**Tech Stack:** Python, NumPy, pandas, CasADi, pytest.

---

### Task 1: Proportional organ calibration

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add a failing synthetic test asserting that three bounded proportional evaluations reduce calibration organ NRMSE and preserve the supplied base parameter vector.
- [x] Implement chronological climate truncation, update history, finite bounds, and calibration-NRMSE selection across the evaluated history.
- [x] Run focused tests and require zero failures.

### Task 2: Capacity scenario summary

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add a failing test for a non-selective scenario envelope with calibration and validation metrics.
- [x] Implement strict scenario identity, envelopes, best-case error reporting without returning `selected_scenario`, and leakage metadata.
- [x] Run focused tests and require zero failures.

### Task 3: Formal target experiment

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation/*`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/readiness_report.json`

- [x] Run all five registered scenarios with three calibration evaluations and one full held-out replay each.
- [x] Write `assimilation_capacity_sensitivity.csv` and `.json`, and embed the summary in the target report.
- [x] Update config and documentation from generated values.
- [x] Rebuild readiness and run the complete test and artifact audit.

The workspace has no `.git` metadata; generated evidence and full-suite checks replace commit checkpoints.
