# V5 Material Safety Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct numerically inflated V5 safety-intervention comparisons without changing controllers, rewards, selections, or holdout scenarios.

**Architecture:** Instrument step records with strict, full-material, and active-material safety signals. Aggregate all three, use active-material fraction in controller promotion, and regenerate the frozen day-119 comparison while preserving legacy decisions.

**Tech Stack:** Python 3.12, NumPy, pandas, pytest.

---

### Task 1: Step-level material intervention metrics

**Files:**
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Modify: `tests/test_v5_controller_smoke.py`

- [ ] Write failing tests for strict noise below `1e-6`, active modifications above
  `1e-6`, full-control-only modifications, and reason serialization.
- [ ] Run focused tests and verify expected failure.
- [ ] Implement pure `classify_safety_intervention` logic and record its outputs.
- [ ] Aggregate legacy, material, and active fractions and projection magnitudes.
- [ ] Run focused tests and verify pass.

### Task 2: Promotion gate correction

**Files:**
- Modify: `experiments/controllers/v5_controller_optimization.py`
- Modify: `tests/test_v5_controller_optimization.py`

- [ ] Write a failing test proving legacy strict noise cannot fail promotion when
  active material intervention is within the PID tolerance.
- [ ] Change `assess_confirmation` to prefer active material fraction and retain an
  explicit metric-name field in the decision.
- [ ] Run optimization tests and verify pass.

### Task 3: Frozen-result re-evaluation

**Files:**
- Modify: `experiments/controllers/run_v5_residual_ppo_optimization.py`
- Create outputs under: `results/chengdu_agri_greenhouse_001/controller_benchmark/v5_residual_ppo_optimization/`

- [ ] Preserve the original promotion JSON as `promotion_decisions_legacy_strict.json`.
- [ ] Delete only resumable day-119 metric/trajectory cache entries, then rerun the
  frozen four-controller confirmation on day 119.
- [ ] Verify candidate IDs, reward, temperature MAE, and RH MAE match the prior run
  within numerical tolerance.
- [ ] Regenerate the report with all safety definitions and corrected decisions.

### Task 4: Verification

**Files:**
- Verify all modified code and result artifacts.

- [ ] Run focused V5 tests.
- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Check result-role counts, frozen candidate IDs, day 119 role, and legacy artifact
  preservation.
