# Chengdu Target Crop-State Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize GreenLight from measured Pidu crop biomass and score simulated standing fruit at subsequent target observation dates.

**Architecture:** Keep target state estimation, environment state injection, and trajectory evaluation as separate tested units. Replay target weather and controls only after the initialization and scoring contracts are verified.

**Tech Stack:** Python, NumPy, pandas, SciPy, PyYAML, pytest, CasADi.

---

### Task 1: Target crop-state estimator

**Files:**
- Create: `processing/chengdu_crop_initialization.py`
- Create: `tests/test_chengdu_crop_initialization.py`

- [ ] Write a failing test with Xindu paired fresh/dry masses and three Pidu baseline replicates.
- [ ] Verify it fails because `build_target_crop_initial_state` is absent.
- [ ] Implement validated source dry-fraction estimation, target replicate aggregation, unit conversion, and provenance audit.
- [ ] Verify zero observed target fruit remains zero and ambiguous target secondary mass is never used.

### Task 2: Reset-time crop state injection

**Files:**
- Modify: `glassgym/environments/greenlight_env.py`
- Modify: `tests/env_test.py`

- [ ] Write failing tests for a valid local override and rejection of negative or unknown states.
- [ ] Add reset-time `initial_crop_state_mg_m2` handling after default state construction.
- [ ] Verify ordinary reset still uses the existing default crop state.

### Task 3: Standing-fruit evaluation

**Files:**
- Create: `common/standing_crop_evaluation.py`
- Create: `tests/test_standing_crop_evaluation.py`

- [ ] Write failing tests for replicate statistics, prediction alignment, MAE, RMSE, bias, WMAPE, R2, and undefined metrics.
- [ ] Implement strict identity/date/unit validation and transparent metric semantics.
- [ ] Verify the evaluator rejects harvest-event rows and mismatched greenhouse identity.

### Task 4: Target replay artifact

**Files:**
- Create: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Create: `tests/test_chengdu_target_crop_validation.py`

- [ ] Write tests for observed-control expansion and provenance fields.
- [ ] Replay 2026 target weather and controls from 2026-04-01 through 2026-05-19 using the measured initialization.
- [ ] Write trajectory, aligned observations, metrics, and uncertainty-scope artifacts.
- [ ] Feed the standing-crop validation status into harvest readiness without changing the requirement for true target harvest events.
- [ ] Run the complete test suite and artifact assertions.

