# Chengdu Regional Yield Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate harvest timing from Chengdu regional evidence and rerun the six-season controller yield comparison with a traceable spring yield plausibility envelope.

**Architecture:** A source registry feeds a focused calibration module. The module derives thermal-time and partial-yield priors without changing target evidence status, while the existing cohort projector consumes the new maturity interval and writes separate regional-transfer artifacts.

**Tech Stack:** Python 3.12, pandas, NumPy, PyYAML, pytest, existing GreenLight cohort model.

---

### Task 1: Evidence registry and calibration API

**Files:**
- Create: `data/external/crops/chengdu_regional_tomato/source_registry.csv`
- Create: `configs/crops/chengdu_regional_yield_transfer.yml`
- Create: `tests/test_chengdu_regional_yield_transfer.py`
- Create: `experiments/crop/calibrate_chengdu_regional_yield_transfer.py`

- [ ] Write tests that reject target-eligible rows, derive deterministic positive thermal thresholds from complete baseline trajectories, and construct an ordered spring-only partial-yield envelope.
- [ ] Run `pytest tests/test_chengdu_regional_yield_transfer.py -q` and confirm failure because the module is absent.
- [ ] Implement registry loading, accumulated degree-day extraction, WUR time-warped cumulative fractions, prior serialization, and input validation.
- [ ] Rerun the focused test and confirm it passes.

### Task 2: Regional cohort projection

**Files:**
- Modify: `experiments/crop/project_controller_cohort_yield.py`
- Modify: `tests/test_controller_cohort_yield_projection.py`

- [ ] Add a failing test showing that an explicit maturity interval and transfer-status label are propagated into scenario and aggregate outputs.
- [ ] Run the focused test and confirm the new arguments are rejected.
- [ ] Add optional prior-path handling while preserving current defaults and mass-balance semantics.
- [ ] Rerun both crop projection test files.

### Task 3: Execute and audit

**Files:**
- Create generated priors under `data/processed/external/chengdu_regional_tomato/`.
- Create generated regional projection and audit artifacts in the guarded benchmark result directory.
- Update: `data/README.md`

- [ ] Generate regional priors from the frozen baseline trajectories and registered sources.
- [ ] Project all 78 controller episodes using 16 paired cohort draws per episode.
- [ ] Report first-harvest onset, spring envelope coverage, algorithm yield summaries, mass balance, and target-validation blockers.
- [ ] Run focused tests, then the complete suite, and independently audit output counts and numerical invariants.

