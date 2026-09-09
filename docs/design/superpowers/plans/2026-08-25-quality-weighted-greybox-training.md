# Quality-Weighted Greybox Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a leakage-safe grey-box residual model that uses V4 transition quality weights without changing validation or test metrics.

**Architecture:** Extend the existing closed-form Ridge residual fitter with optional sample weights. Keep chronological fit/calibration/validation roles unchanged, select only on the validation split, and write the weighted model to a new result directory so the frozen unweighted artifact remains intact. Re-evaluate the already-opened test split only as a diagnostic comparison and stratify it by quality and heat regime.

**Tech Stack:** Python, NumPy, pandas, pytest, CasADi-based ChengduPhysicsV4 evaluator.

---

### Task 1: Weighted Ridge core

**Files:**
- Modify: `experiments/reports/chengdu_residual_correction.py`
- Modify: `tests/test_chengdu_greybox_residual.py`

- [ ] Add a failing test in which a zero-weight outlier cannot move the fitted residual.
- [ ] Add validation for finite, nonnegative weights with positive total weight.
- [ ] Compute weighted feature means, variances, normal equations, and model audit metadata.
- [ ] Run `pytest tests/test_chengdu_greybox_residual.py -q`.

### Task 2: Leakage-safe selector integration

**Files:**
- Modify: `experiments/reports/select_chengdu_v4_greybox_residual.py`
- Modify: `tests/test_chengdu_greybox_selection.py`
- Create: `configs/models/chengdu_v4_greybox_residual_quality_weighted.yml`

- [ ] Add a failing selector test proving the configured weight column is required and recorded.
- [ ] Pass `transition_quality_weight` only to fit-stage Ridge estimation.
- [ ] Point the new config to V4 train and validation splits; do not add a test path.
- [ ] Run focused selector tests.

### Task 3: Train, freeze, and diagnose

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/physics_v4_greybox_residual_quality_weighted/*`
- Create: `results/chengdu_agri_greenhouse_001/data_quality_v4/test_quality_weighted/*`

- [ ] Run the weighted selector on train/calibration/validation only.
- [ ] Evaluate the selected model on validation and the previously opened test split.
- [ ] Generate aggregate and quality/heat-stratified metrics.
- [ ] Compare weighted and unweighted frozen rollouts without selecting on test.

### Task 4: Verification

**Files:**
- Verify all files above.

- [ ] Run focused tests for weighted Ridge, selection, V4 data, and quality audit.
- [ ] Run `pytest -q` and report the exact result.
- [ ] Record limitations: only two extreme-heat training transitions and no fresh independent holdout.
