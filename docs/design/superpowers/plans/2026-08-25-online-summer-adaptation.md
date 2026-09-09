# Online Summer Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a fixed-hyperparameter residual model on all observations available before July 13 and evaluate it forward from July 14.

**Architecture:** Add a focused adaptation script that enforces chronological roles, generates physical one-step predictions, fits the existing weighted Ridge model, calibrates uncertainty on one day, and writes an evaluation split and audit manifest. Reuse existing rollout and quality audit tools.

**Tech Stack:** Python, pandas, NumPy, pytest, CasADi ChengduPhysicsV4.

---

### Task 1: Chronological adaptation split

**Files:**
- Create: `experiments/reports/adapt_chengdu_summer_residual.py`
- Create: `tests/test_chengdu_summer_adaptation.py`

- [ ] Write failing tests for target-time boundaries and non-overlap.
- [ ] Implement fit, calibration, and evaluation split construction.
- [ ] Reject empty roles and nonchronological input.

### Task 2: Fixed model fitting

**Files:**
- Modify: `experiments/reports/adapt_chengdu_summer_residual.py`
- Modify: `tests/test_chengdu_summer_adaptation.py`

- [ ] Test the fixed hyperparameter and heat-weight contract.
- [ ] Fit `thermal_dynamics` Ridge with alpha 0.1 and capped heat weights.
- [ ] Calibrate uncertainty without refitting coefficients.
- [ ] Write the adapted model, evaluation CSV, and manifest.

### Task 3: Forward replay

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/physics_v4_summer_online_adaptation/*`

- [ ] Build the adaptation artifact from chronological data.
- [ ] Evaluate adapted and frozen models on the same July 14-20 rows.
- [ ] Generate quality and heat-regime reports.
- [ ] Freeze the retrospective adaptation decision.

### Task 4: Verification

**Files:**
- Verify all modified and generated files.

- [ ] Run focused adaptation, residual, and audit tests.
- [ ] Run the complete test suite.
- [ ] Report exact gains and the remaining independent-holdout limitation.
