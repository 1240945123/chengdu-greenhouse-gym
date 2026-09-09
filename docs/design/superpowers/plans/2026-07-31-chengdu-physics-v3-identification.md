# Chengdu Physics V3 Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify a roof/fan-separated Chengdu greenhouse model from measured training episodes and decide validation readiness without touching the final test split.

**Architecture:** Generate an audited V3 trajectory with physical device and sensor-quality fields, implement a versioned eight-input offline physics backend, then run bounded robust identification and regime-aware validation. Production controller interfaces remain unchanged until this stage passes.

**Tech Stack:** Python 3.12, pandas, NumPy, SciPy, CasADi, pytest, YAML/JSON.

---

### Task 1: Build the V3 trajectory and sensor-quality contract

**Files:**
- Create: `processing/build_chengdu_trajectory_v3.py`
- Create: `tests/test_chengdu_trajectory_v3.py`
- Create: `data/processed/chengdu_agri/greenhouse_001/trajectories/v3/manifest.json`

- [ ] Write a failing test requiring roof, fan, wet-pad, sensor-count, disagreement, and quality fields in every split.
- [ ] Run the focused test and verify it fails because the V3 builder does not exist.
- [ ] Implement chronological transition construction, 3 C disagreement exclusion for training eligibility, and split/source hashes.
- [ ] Run focused tests and generate the V3 artifact from `aligned/greenhouse_1h.csv`.

### Task 2: Implement the offline eight-input V3 backend

**Files:**
- Create: `glassgym/models/ChengduPhysicsV3/__init__.py`
- Create: `glassgym/models/ChengduPhysicsV3/ode.py`
- Create: `glassgym/models/ChengduPhysicsV3/utils.py`
- Create: `tests/test_chengdu_physics_v3.py`

- [ ] Write failing tests showing roof exchange responds to wind, fan exchange works with roof closed, infiltration remains when both are off, and pad parameters are fixed/non-identifiable.
- [ ] Run focused tests and verify missing-module failures.
- [ ] Implement V3 by preserving V2 thermal nodes and separating `uRoofVent`, `uFan`, and `uPad` fluxes.
- [ ] Run the V3 structural and 72-hour stability tests.

### Task 3: Add backend-aware V3 row conversion and continuous evaluation

**Files:**
- Modify: `experiments/reports/evaluate_chengdu_physics.py`
- Modify: `experiments/reports/evaluate_chengdu_multistep.py`
- Create: `tests/test_chengdu_physics_v3_evaluation.py`

- [ ] Write a failing test proving a V3 trajectory row becomes an eight-element control vector with independent roof, fan, and pad values.
- [ ] Run the test and verify the current six-input converter fails.
- [ ] Add an explicit backend control schema; do not infer dimensions from column presence.
- [ ] Run legacy and V3 evaluation tests to verify backward compatibility.

### Task 4: Implement bounded robust identification and sensitivity audit

**Files:**
- Create: `experiments/reports/identify_chengdu_physics_v3.py`
- Create: `configs/models/chengdu_physics_v3_identification.yml`
- Create: `tests/test_chengdu_physics_v3_identification.py`

- [ ] Write failing tests for positive bounds, robust-loss behavior, train-only fitting, validation-only selection, finite-difference rank reporting, and frozen pad parameters.
- [ ] Run focused tests and verify failures identify the missing identifier.
- [ ] Implement deterministic bounded multistart least squares with workspace-local artifacts and parameter fingerprints.
- [ ] Run identification on eligible V3 training transitions and select on V3 validation only.

### Task 5: Add regime and residual-autocorrelation diagnostics

**Files:**
- Create: `experiments/reports/analyze_chengdu_v3_residuals.py`
- Create: `tests/test_chengdu_v3_residuals.py`

- [ ] Write failing tests for lag-1/6/24 autocorrelation and day/night, roof-only, fan-only, both-on, and both-off groups.
- [ ] Implement diagnostics with explicit minimum-group-size reporting rather than silently dropping sparse regimes.
- [ ] Run focused tests and produce validation residual artifacts.

### Task 6: Run gates and document the decision

**Files:**
- Create: `docs/audits/stage_a_v2_package_h_v3_identification.md`

- [ ] Compare V3 with persistence, outdoor-following, corrected V1, and V2 on common validation starts.
- [ ] Run the rainy high-radiation diagnostic with the selected V3 parameters.
- [ ] Run focused tests and the complete pytest suite with workspace-local temporary storage.
- [ ] Record fingerprints, identifiability limits, metrics, residual diagnostics, and GO/NO-GO. Do not read the V3 test split unless all validation gates pass.
