# Chengdu V4 Grey-Box Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a leakage-safe periodic Ridge residual model on top of ChengduPhysicsV4.

**Architecture:** One module owns feature construction, Ridge fitting, correction and uncertainty calibration. A selection command creates V4 one-step train predictions, performs the blocked fit/calibration procedure and evaluates only finalists on validation with continuous state feedback. Existing legacy residual artifacts remain readable.

**Tech Stack:** Python 3.12, NumPy, pandas, CasADi/CVODES, PyYAML, pytest.

---

### Task 1: Leakage-Safe Features

**Files:**
- Modify: `experiments/reports/chengdu_residual_correction.py`
- Create: `tests/test_chengdu_greybox_residual.py`

- [ ] Write a failing test requiring timestamp periodic features, predicted/outdoor gaps and rejection of observed indoor feature names.

```python
features = build_residual_features(frame, FEATURE_SETS["periodic_interactions"])
assert "hour_sin" in features
with pytest.raises(ValueError, match="leakage"):
    validate_residual_features(["x_air_temperature"])
```

- [ ] Run `python -m pytest tests/test_chengdu_greybox_residual.py -q` and confirm the functions are absent.
- [ ] Implement `validate_residual_features()` and `build_residual_features()` with deterministic timestamp-derived columns.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Ridge Residual And Calibration

**Files:**
- Modify: `experiments/reports/chengdu_residual_correction.py`
- Test: `tests/test_chengdu_greybox_residual.py`

- [ ] Write failing tests showing Ridge learns residual rather than direct target, constant columns remain finite, and calibration quantiles are monotonic.

```python
model = fit_ridge_residual_corrector(frame, features, alpha=1.0)
corrected = apply_residual_corrector(frame, model, gain=1.0)
assert corrected["pred_air_temperature"].iloc[-1] > frame["pred_air_temperature"].iloc[-1]
assert model["uncertainty"]["air_temperature"]["q95"] >= model["uncertainty"]["air_temperature"]["q90"]
```

- [ ] Implement standardized closed-form Ridge with an unpenalized intercept, residual-target coefficients, gain and physical fallback semantics.
- [ ] Implement calibration quantiles from a separate frame and model serialization metadata.
- [ ] Run focused tests and legacy residual tests.

### Task 3: Stateful Hybrid Predictor

**Files:**
- Modify: `experiments/reports/evaluate_chengdu_multistep.py`
- Test: `tests/test_chengdu_greybox_residual.py`

- [ ] Write a failing two-step test proving the predictor never requests `x_` features and writes corrected temperature/RH back into physical state.
- [ ] Route schema `ridge_residual_v1` through the new feature builder while retaining the legacy coefficient path.
- [ ] Record fallback counts on the predictor and clip only to `[-10, 60] C` and `[0, 100] %RH`.
- [ ] Run focused and V1-V4 evaluator tests.

### Task 4: Blocked Selection

**Files:**
- Create: `experiments/reports/select_chengdu_v4_greybox_residual.py`
- Create: `configs/models/chengdu_v4_greybox_residual.yml`
- Create: `tests/test_chengdu_greybox_selection.py`

- [ ] Write failing tests for chronological 80/20 split, calibration-only base ranking, mandatory physical fallback, validation-only gain selection and absence of a test argument.
- [ ] Generate candidates from feature sets and alpha values, rank top two on calibration normalized MAE, then evaluate four gains plus fallback on validation.
- [ ] Save a model artifact with source/config/code hashes and explicit `fit`, `calibration`, `validation`, `unopened` roles.

```yaml
feature_sets: [base, periodic, periodic_interactions]
alphas: [0.1, 1.0, 10.0, 100.0]
gains: [0.25, 0.5, 0.75, 1.0]
fit_fraction: 0.8
base_top_k: 2
validation_horizons: [1, 6, 24, 72]
validation_stride: 6
```

### Task 5: Metrics And Audit

**Files:**
- Create: `experiments/reports/evaluate_chengdu_hybrid_uncertainty.py`
- Create: `docs/audits/stage_a_v2_package_j_v4_greybox_residual.md`
- Test: `tests/test_chengdu_greybox_residual.py`

- [ ] Write a failing test for empirical interval coverage and width at requested horizons.
- [ ] Implement fixed calibration-quantile interval metrics from saved rollout rows.
- [ ] Run standardized validation, corrected one-step residual diagnostics and rainy stress replay with physical fallback accounting.
- [ ] Compare V4 hybrid against persistence, V1, V2, V3 and uncorrected V4; record every gate and hash.

### Task 6: Verification

**Files:**
- Verify: all Package J code, configuration and artifacts.

- [ ] Run Package J focused tests with workspace-local `TEMP` and `TMP`.
- [ ] Run the complete pytest suite.
- [ ] Confirm the artifact test role is `unopened`, no residual feature is prohibited, and `configs/envs/ChengduControllerBenchmark.yml` remains six-input `ChengduPhysics`.

