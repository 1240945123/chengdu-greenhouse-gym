# Chengdu Temporal Residual Climate Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate leakage-safe small LSTM and TCN residual models on top of ChengduPhysicsV4.

**Architecture:** A focused temporal-residual module owns contiguous sequence construction, fit-only normalization, the two neural architectures and artifact loading. A training command selects architecture by median calibration performance across three seeds. A separate evaluator performs causal warm-up and continuous physical-state rollouts against existing V4, persistence and Ridge baselines.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, CasADi/CVODES, PyYAML, pytest.

---

### Task 1: Contiguous Temporal Dataset

**Files:**
- Create: `glassgym/models/residual/__init__.py`
- Create: `glassgym/models/residual/temporal.py`
- Create: `tests/test_chengdu_temporal_residual.py`

- [ ] Write failing tests requiring 24-row windows, residual targets and rejection of windows crossing hourly timestamp gaps.
- [ ] Run `python -m pytest tests/test_chengdu_temporal_residual.py -q` and confirm the module is missing.
- [ ] Implement `build_temporal_residual_sequences` using the existing leakage-safe feature builder.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Fit-Only Scaling And Architectures

**Files:**
- Modify: `glassgym/models/residual/temporal.py`
- Test: `tests/test_chengdu_temporal_residual.py`

- [ ] Write failing tests for scaler round trips, LSTM output shape, causal TCN output shape and deterministic parameter counts.
- [ ] Implement `ArrayStandardizer`, `LSTMResidualModel` and `TCNResidualModel` with two-output heads.
- [ ] Re-run focused tests and confirm all architecture tests pass.

### Task 3: Deterministic Training And Artifact

**Files:**
- Create: `experiments/predictors/train_chengdu_temporal_residual.py`
- Create: `configs/models/chengdu_temporal_residual.yml`
- Test: `tests/test_chengdu_temporal_residual.py`

- [ ] Write a failing synthetic test showing training reduces calibration residual error and architecture selection uses median seed score.
- [ ] Implement seeded training, gradient clipping, early stopping and JSON plus PyTorch artifact metadata.
- [ ] Run the synthetic training test and inspect the saved split and seed metadata.

### Task 4: Causal Multi-Step Evaluation

**Files:**
- Create: `experiments/reports/evaluate_chengdu_temporal_residual.py`
- Test: `tests/test_chengdu_temporal_residual.py`

- [ ] Write a failing test proving warm-up excludes the origin and future rows and that corrected predictions are written back to physical state.
- [ ] Implement context warm-up, continuous rollout and three-seed ensemble prediction.
- [ ] Implement common-origin comparisons with persistence, V4 and frozen Ridge.
- [ ] Re-run focused and existing Chengdu multi-step tests.

### Task 5: Train, Evaluate And Report

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual/`
- Create: `docs/audits/stage_a_v2_package_k_temporal_residual.md`

- [ ] Generate train one-step V4 predictions without changing the frozen V4 parameter artifact.
- [ ] Train LSTM and TCN for seeds 0, 1 and 2 using fit-only scalers and calibration early stopping.
- [ ] Select architecture by median calibration score and evaluate all seeds plus the ensemble on validation.
- [ ] Write the gate decision, limitations, hashes and full horizon metric table.

### Task 6: Verification

**Files:**
- Verify: temporal residual code, artifacts and unaffected controller configuration.

- [ ] Run focused temporal and Chengdu physics tests with workspace-local temporary directories.
- [ ] Run the complete pytest suite.
- [ ] Confirm no forbidden observed indoor feature enters the neural input and the production environment remains on its existing backend.

