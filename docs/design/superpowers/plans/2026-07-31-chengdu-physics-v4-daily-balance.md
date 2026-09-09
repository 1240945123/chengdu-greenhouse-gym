# ChengduPhysicsV4 Daily Balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate a minimal daily energy and moisture-balance extension without opening the Chengdu test split.

**Architecture:** ChengduPhysicsV4 preserves the V3 28-state, eight-control interface and adds five appended parameters. Shared evaluators gain an explicit V4 schema and deterministic sky-temperature forcing. A separate selection command ranks bounded candidates on train, chooses among finalists on validation, and emits a fully audited artifact.

**Tech Stack:** Python 3.12, CasADi/CVODES, NumPy, pandas, PyYAML, pytest.

---

### Task 1: Sky Forcing

**Files:**
- Modify: `experiments/reports/evaluate_chengdu_physics.py`
- Test: `tests/test_chengdu_physics_v4_evaluation.py`

- [ ] Write tests requiring a finite Brutsaert sky temperature below outdoor temperature and explicit V4 control schema support.

```python
def test_clear_sky_temperature_is_finite_and_below_air():
    value = estimate_clear_sky_temperature(20.0, 80.0)
    assert -40.0 < value < 20.0

def test_v4_uses_eight_control_columns():
    assert CONTROL_SCHEMAS["ChengduPhysicsV4"] == [
        "uBoil", "uCO2", "uThScr", "uRoofVent",
        "uLamp", "uBlScr", "uFan", "uPad",
    ]
```
- [ ] Run `python -m pytest tests/test_chengdu_physics_v4_evaluation.py -q` and confirm failure because V4 support is absent.
- [ ] Add `estimate_clear_sky_temperature()` and the explicit `ChengduPhysicsV4` eight-control schema; use the estimate only for V4 when measured sky temperature is absent.

```python
def estimate_clear_sky_temperature(t_air_c: float, rh_percent: float) -> float:
    t_k = np.clip(t_air_c + 273.15, 180.0, 330.0)
    vapor_hpa = np.clip(rh_percent, 0.0, 100.0) / 100.0 * saturation_pressure_pa(t_air_c) / 100.0
    emissivity = np.clip(1.24 * (vapor_hpa / t_k) ** (1.0 / 7.0), 0.2, 1.0)
    return float(np.clip(t_k * emissivity ** 0.25 - 273.15, -80.0, t_air_c))
```
- [ ] Re-run the focused test and confirm it passes.

### Task 2: V4 Physical Backend

**Files:**
- Create: `glassgym/models/ChengduPhysicsV4/__init__.py`
- Create: `glassgym/models/ChengduPhysicsV4/ode.py`
- Create: `glassgym/models/ChengduPhysicsV4/utils.py`
- Test: `tests/test_chengdu_physics_v4.py`

- [ ] Write tests for the eight-input contract, 225-parameter integration, V3-equivalent reference, and directional longwave, latent and moisture-buffer responses.

```python
def test_v4_reference_matches_v3_one_step(sample_row):
    p3 = reference_params(220)
    p4 = np.r_[p3, [0.0, 1.0, 0.0, 1.0, 0.0]]
    assert_predictions_close(run("ChengduPhysicsV3", p3), run("ChengduPhysicsV4", p4), atol=1e-5)
```
- [ ] Run the focused tests and confirm failure because the backend does not exist.
- [ ] Implement V4 by preserving V3 equations and adding cover longwave, floor capacity scaling, canopy latent cooling and conservative vapor-buffer exchange at indices 220-224.

```python
q_longwave = longwave_scale * 0.84 * sigma * area * (t_cover_k**4 - t_sky_k**4)
buffer_flux = moisture_rate * (vp_buffer - vp_air)
dxdt[15] += buffer_flux
dxdt[16] = -buffer_flux / moisture_capacity_ratio
```
- [ ] Re-run the focused tests and confirm they pass.

### Task 3: Backend-Aware Rollouts

**Files:**
- Modify: `experiments/reports/evaluate_chengdu_multistep.py`
- Modify: `experiments/reports/evaluate_chengdu_physics.py`
- Test: `tests/test_chengdu_physics_v4_evaluation.py`

- [ ] Add a failing test that builds a V4 predictor from a 225-value artifact and completes a two-step rollout.

```python
predictor = build_physics_predictor(np.ones(225), model_backend="ChengduPhysicsV4")
metrics, rows = evaluate_multistep_rollouts(frame, predictor, horizons=[1, 2], start_indices=[0])
assert metrics["all_predictions_finite"]
assert len(rows) == 2
```
- [ ] Extend parameter sizing and backend dispatch without changing V1-V3 behavior.
- [ ] Run V4 evaluation tests and all existing V1-V3 evaluation tests.

### Task 4: Leakage-Safe Candidate Selection

**Files:**
- Create: `experiments/reports/select_chengdu_physics_v4.py`
- Create: `configs/models/chengdu_physics_v4_selection.yml`
- Test: `tests/test_chengdu_physics_v4_selection.py`

- [ ] Write failing tests for deterministic training ranking, mandatory reference inclusion, validation-only final selection, physical-envelope rejection and absence of a test-data argument.

```python
def test_selection_always_retains_reference():
    finalists = rank_train_candidates(candidates, score_fn=lambda c: c["score"], top_k=2)
    assert any(item["candidate_kind"] == "v3_reference" for item in finalists)

def test_cli_has_no_test_csv_argument():
    assert "test_csv" not in inspect.signature(select_from_config).parameters
```
- [ ] Implement a bounded candidate generator, common-start multi-step scoring, train finalist ranking and validation selection.

```yaml
thermal_grid:
  longwave_scale: [0.0, 0.5, 1.0]
  floor_capacity_scale: [0.5, 1.0, 2.0]
  latent_heat_scale: [0.0, 0.5, 1.0]
moisture_grid:
  moisture_buffer_rate_s: [0.0, 0.0000462963, 0.0001388889]
  moisture_buffer_capacity_ratio: [1.0, 4.0, 8.0]
train_stride: 168
validation_stride: 6
top_k: 5
```
- [ ] Emit a 225-value artifact containing candidate audits, split hashes and implementation/config hashes.
- [ ] Run selection tests and confirm they pass.

### Task 5: Diagnostics And Decision

**Files:**
- Modify: `experiments/reports/analyze_chengdu_v3_residuals.py`
- Create: `docs/audits/stage_a_v2_package_i_v4_daily_balance.md`
- Test: `tests/test_chengdu_v3_residuals.py`

- [ ] Add a failing test that lets residual diagnostics label an arbitrary backend while retaining the existing output schema.

```python
diagnostics = residual_diagnostics(frame, minimum_group_size=2)
assert diagnostics["autocorrelation"]["temperature_error"]["lag_24"] is None
```
- [ ] Generalize the diagnostic CLI to accept V4 without changing the diagnostic calculations.
- [ ] Run train selection, standardized validation at common starts, residual diagnostics and the existing rainy stress replay.
- [ ] Compare V4 against persistence, V1, V2 and V3; record every GO/NO-GO gate and reproducibility hash.

### Task 6: Verification

**Files:**
- Verify: all Package I files and artifacts.

- [ ] Run focused Package I tests.
- [ ] Run `python -m pytest -q` with workspace-local `TEMP` and `TMP`.
- [ ] Verify artifact hashes, split roles and that the production six-input environment configuration is unchanged.
