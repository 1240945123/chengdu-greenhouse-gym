# Chengdu Stage-Dependent Sink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable thermal-stage sink schedules and compare registered structures without claiming independent validation.

**Architecture:** A pure schedule helper calculates organ-specific multipliers. The forced crop replay applies them to GreenLight parameters 154-156 at each RK4 stage and records them in trajectory rows. The target runner compares five fixed structures using the existing calibration-date organ parameters.

**Tech Stack:** Python, NumPy, pandas, CasADi, pytest.

---

### Task 1: Stage schedule and dynamic replay

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add failing tests for constant, square-root, linear, organ-specific, and capped multipliers.
- [x] Add a failing replay test proving schedule columns are emitted and calibration behavior is unchanged below the reference stage.
- [x] Implement schedule validation and RK4-stage parameter adjustment.
- [x] Run focused tests.

### Task 2: Non-selective structure comparison

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add a failing summary test requiring post-hoc and non-independent-validation labels and prohibiting selected-scenario output.
- [x] Run constant, square-root and linear all-organ and fruit-only structures from the existing calibration solution.
- [x] Write comparison CSV/JSON without changing accepted target parameters.

### Task 3: Formal audit

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Regenerate target validation and readiness artifacts.

- [x] Document generated values and epistemic status.
- [x] Verify readiness ignores post-hoc candidate performance and remains false.
- [x] Run full tests and independent artifact checks.

No Git metadata is present in this workspace; generated evidence replaces commit checkpoints.
