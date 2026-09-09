# Vanthoor Crop Yield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the empirical Chengdu crop equations with the literature-backed Vanthoor/GreenLight tomato crop subsystem and report explicit harvested yield metrics.

**Architecture:** Keep the calibrated Chengdu climate derivatives for states 0-21 and reuse the existing GreenLight auxiliary flux model for crop states 22-27. Expose fruit harvest as an explicit diagnostic flow, integrate it per environment step, and report dry and fresh harvested yield without changing the climate-control reward.

**Tech Stack:** Python, CasADi, NumPy, pandas, pytest, YAML.

---

### Task 1: Reuse Vanthoor crop dynamics

**Files:**
- Create: `glassgym/models/GreenLight/crop.py`
- Modify: `glassgym/models/ChengduPhysics/ode.py`
- Test: `tests/test_vanthoor_crop_model.py`

- [ ] Write a failing test asserting that Chengdu states 22-27 have the same derivatives as the GreenLight crop subsystem for identical state, weather, controls, and parameters.
- [ ] Run `pytest tests/test_vanthoor_crop_model.py -v` and verify the empirical Chengdu equations fail the equality assertion.
- [ ] Add a focused crop-flux API backed by `GreenLight.aux_states.update` and use it in both the Chengdu crop derivatives and diagnostics.
- [ ] Run the focused test and confirm it passes with finite non-negative harvest flow.

### Task 2: Record harvested yield

**Files:**
- Modify: `glassgym/environments/greenlight_env.py`
- Modify: `experiments/controllers/tune_benchmark_classical.py`
- Modify: `experiments/controllers/train_benchmark_rl.py`
- Modify: `experiments/controllers/benchmark_metrics.py`
- Test: `tests/test_vanthoor_crop_model.py`
- Test: `tests/test_controller_benchmark_metrics.py`

- [ ] Write failing tests for per-step harvest diagnostics and episode-level dry/fresh yield conversion.
- [ ] Run the focused tests and verify the new fields are missing.
- [ ] Integrate harvest rate by the trapezoidal rule over each 900-second step and expose crop stocks and harvested dry matter through environment info.
- [ ] Add trajectory columns and metrics for harvested dry matter, fresh yield, harvest onset, peak daily yield, and final crop stocks.
- [ ] Run focused tests and confirm all yield fields are finite and correctly converted using `dmfm=0.0627`.

### Task 3: Add scientific provenance and regional validation metadata

**Files:**
- Create: `configs/crops/vanthoor_tomato.yml`
- Modify: `data/README.md`
- Test: `tests/test_vanthoor_crop_model.py`

- [ ] Write a failing test requiring model DOI, parameter source, dry-matter fraction, and Sichuan external reference range.
- [ ] Add machine-readable provenance that distinguishes literature parameterization from target-site calibration.
- [ ] Document that the Pengzhou range is an external plausibility check, not a calibration target.
- [ ] Run the focused tests.

### Task 4: Regression and experiment verification

**Files:**
- Modify only files required by failures discovered during verification.

- [ ] Run the complete test suite.
- [ ] Run a Chengdu smoke trajectory for baseline, PID, and MPC and verify finite crop/yield diagnostics.
- [ ] Generate benchmark metrics and inspect the yield values against the configured Sichuan reference range.
- [ ] Record limitations: no target-greenhouse harvest calibration, no cultivar-specific parameter fit, and no heating/CO2 control.
