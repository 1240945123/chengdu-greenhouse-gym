# Chengdu Thermal Validation V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace invalid indoor-as-outdoor calibration trajectories with audited measured disturbances and establish continuous thermal-model validity gates before controller comparison.

**Architecture:** Strengthen the trajectory builder's data contract, generate a fingerprinted V2 dataset from the aligned database, then extend evaluation to true continuous state rollouts and baseline-relative physical gates. Upgrade the physics backend only if corrected-data evidence requires it.

**Tech Stack:** Python 3.12, pandas, NumPy, CasADi, pytest, YAML/JSON configuration.

---

### Task 1: Enforce measured-disturbance trajectory inputs

**Files:**
- Modify: `processing/chengdu_trajectory.py`
- Modify: `tests/data_pipeline.py`

- [ ] Add a failing test proving missing outdoor temperature/RH raises `ValueError` by default.
- [ ] Run `python -m pytest tests/data_pipeline.py -q` and verify the new test fails because indoor fallback is still active.
- [ ] Add explicit `require_measured_outdoor=True` behavior and preserve opt-in legacy fallback only for synthetic fixtures.
- [ ] Run the focused tests and verify they pass.

### Task 2: Build audited V2 trajectories

**Files:**
- Create: `processing/build_chengdu_trajectory_v2.py`
- Create: `tests/test_chengdu_trajectory_v2.py`
- Create: `data/processed/chengdu_agri/greenhouse_001/trajectories/v2/manifest.json`
- Create: `data/processed/chengdu_agri/greenhouse_001/trajectories/v2/train.csv`
- Create: `data/processed/chengdu_agri/greenhouse_001/trajectories/v2/val.csv`
- Create: `data/processed/chengdu_agri/greenhouse_001/trajectories/v2/test.csv`

- [ ] Add failing tests for exact one-hour adjacency, chronological splits, source hash, exclusion counts, and non-identical indoor/outdoor disturbances.
- [ ] Run the focused tests and verify failures identify missing V2 builder behavior.
- [ ] Implement the builder using `aligned/greenhouse_1h.csv`, retaining measured fields and control-quality columns.
- [ ] Run focused tests, generate artifacts, and verify manifest hashes against files.

### Task 3: Add continuous-rollout validation and baselines

**Files:**
- Modify: `experiments/reports/evaluate_chengdu_multistep.py`
- Create: `tests/test_chengdu_thermal_validation_v2.py`

- [ ] Add failing tests proving all model states propagate between steps and reporting includes MAE, RMSE, bias, extrema, finite-state and physical-envelope fields.
- [ ] Add failing tests for persistence and outdoor-following reference predictors.
- [ ] Run focused tests and verify expected failures.
- [ ] Implement continuous rollout and baseline-relative reporting without state clipping.
- [ ] Run focused tests and verify all pass.

### Task 4: Calibrate and select without leakage

**Files:**
- Create: `experiments/reports/calibrate_chengdu_thermal_v2.py`
- Create: `configs/models/chengdu_thermal_validation_v2.yml`
- Create: `tests/test_chengdu_thermal_calibration_v2.py`

- [ ] Add failing tests that training alone fits candidates, validation alone selects them, and test rows cannot affect selection.
- [ ] Implement constrained candidate evaluation with a score combining normalized 6/24/72-hour validation errors and a hard physical-envelope rejection.
- [ ] Run focused tests and then calibrate against V2 train/validation data.
- [ ] Evaluate the frozen selection once on V2 test data and save model/data fingerprints.

### Task 5: Decide and, only if required, implement ChengduPhysicsV2

**Files:**
- Conditionally create: `glassgym/models/ChengduPhysicsV2/ode.py`
- Conditionally create: `glassgym/models/ChengduPhysicsV2/utils.py`
- Conditionally create: `glassgym/models/ChengduPhysicsV2/__init__.py`
- Conditionally create: `tests/test_chengdu_physics_v2.py`

- [ ] Compare corrected-data legacy performance with validation gates and baselines.
- [ ] If legacy fails, first add failing energy-partition, positive-capacity, rain-safe exchange, and 72-hour stability tests.
- [ ] Implement the smallest versioned multi-node model that satisfies those tests and recalibrate only on train/validation.
- [ ] Keep `ChengduPhysics` unchanged and select V2 only through explicit configuration.

### Task 6: Run gates and document the decision

**Files:**
- Create: `docs/audits/stage_a_v2_package_g_thermal_validation.md`
- Modify if GO: `configs/envs/ChengduControllerBenchmark.yml`

- [ ] Run focused thermal/data tests.
- [ ] Run the complete pytest suite with workspace-local temporary storage.
- [ ] Run one-step, 6/12/24/72-hour validation and the rainy-day 24-hour controller smoke test.
- [ ] Record metrics, artifacts, fingerprints, limitations, and a GO/NO-GO decision. Switch the benchmark backend only on GO.
