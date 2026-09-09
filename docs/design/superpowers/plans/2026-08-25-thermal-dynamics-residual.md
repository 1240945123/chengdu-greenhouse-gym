# Thermal-Dynamics Residual Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add physically interpretable high-heat residual features and accept them only through leakage-safe dual-regime validation.

**Architecture:** Extend the existing residual feature builder and selector without changing the physics backend or canonical data. Train in a separate artifact directory, compare against the frozen primary model on validation, and use the opened test split only after the promotion decision.

**Tech Stack:** Python, NumPy, pandas, pytest, CasADi ChengduPhysicsV4.

---

### Task 1: Thermal feature derivation

**Files:**
- Modify: `experiments/reports/chengdu_residual_correction.py`
- Modify: `tests/test_chengdu_greybox_residual.py`

- [ ] Write failing tests for deterministic thermal interactions and action clipping.
- [ ] Add the `thermal_dynamics` feature set and derived feature calculations.
- [ ] Verify finite outputs and leakage feature restrictions.

### Task 2: Dual-regime validation metrics

**Files:**
- Modify: `experiments/reports/audit_chengdu_data_quality_v4.py`
- Modify: `tests/test_chengdu_data_quality_audit.py`

- [ ] Write failing tests for aggregate, normal, and high-temperature candidate gates.
- [ ] Implement a deterministic comparison against frozen reference rollouts.
- [ ] Require aggregate and high-temperature improvements with at most 2% normal degradation.

### Task 3: Candidate training

**Files:**
- Create: `configs/models/chengdu_v4_greybox_residual_thermal.yml`
- Create: `results/chengdu_agri_greenhouse_001/physics_v4_greybox_residual_thermal/*`

- [ ] Select candidates from train/calibration/validation only.
- [ ] Evaluate the selected candidate and frozen reference with identical validation starts.
- [ ] Apply the dual-regime gate and freeze its decision.
- [ ] Run the opened test split only as a diagnostic after the decision.

### Task 4: Verification

**Files:**
- Verify all modified and generated files.

- [ ] Run focused feature, selection, quality, and rollout tests.
- [ ] Run the complete test suite.
- [ ] Report the exact metrics, promotion result, and remaining data gap.
