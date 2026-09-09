# Controller Model V2 and Forecast Emulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate MPC from plant internals and provide a deterministic, role-safe Chengdu historical forecast emulator.

**Architecture:** A restricted immutable controller context feeds a frozen low-order climate transition model. A separate artifact builder fits persistence/climatology blends on 2023, selects on 2024, and exposes only issue-time forecasts during 2025 evaluation.

**Tech Stack:** Python 3.12, NumPy, pandas, PyYAML, pytest, existing GreenLight Gym interfaces.

---

### Task 1: Restricted MPC Context

**Files:**
- Modify: `glassgym/core/types.py`
- Modify: `experiments/controllers/tune_benchmark_classical.py`
- Test: `tests/test_controller_model_v2.py`

- [ ] Add a failing test constructing two plant environments with identical permitted indoor climate but poisoned latent states and parameters; assert the generated MPC contexts are equal and expose no `x`, `p`, `d`, or environment reference.
- [ ] Run `python -m pytest tests/test_controller_model_v2.py -q` and confirm the restricted-context import or assertions fail.
- [ ] Add frozen `ControllerStepContextV2` with `t`, `dt`, `indoor_temperature`, `indoor_relative_humidity`, `u`, `forecast`, `hour_of_day`, and `day_of_year`.
- [ ] Add `_mpc_step_context(env)` that derives only the permitted values and use it for MPC evaluation.
- [ ] Run the focused test and require a pass.

### Task 2: Independent Controller Model

**Files:**
- Create: `glassgym/components/controller_model.py`
- Modify: `glassgym/components/mpc.py`
- Modify: `configs/agents/mpc.yml`
- Test: `tests/test_controller_model_v2.py`

- [ ] Add failing tests for finite bounded one-step temperature/RH predictions, coefficient validation, no plant-model binding, and proposal invariance to poisoned plant internals.
- [ ] Run the tests and confirm failures caused by missing `ControllerModelV2`.
- [ ] Implement a versioned frozen coefficient schema and deterministic low-order temperature/RH transition using outside temperature/RH, radiation, wind, screens, ventilation, and lamps.
- [ ] Refactor `LightweightMPCController` to initialize this model from config and roll out two-state climate predictions from `ControllerStepContextV2`; remove `bind_env` and CasADi/true-parameter use.
- [ ] Run MPC, safety, and controller benchmark tests and require all passes.

### Task 3: Historical Forecast Emulator Artifact

**Files:**
- Extend: `glassgym/components/forecast.py`
- Create: `experiments/weather/fit_forecast_emulator.py`
- Create: `configs/forecasts/chengdu_historical_emulator_v2.yml`
- Test: `tests/test_forecast_emulator_v2.py`

- [ ] Add failing tests proving role validation accepts only 2023 fit and 2024 select sources, rejects 2025 fitting, produces deterministic checksums, and never receives post-issue rows.
- [ ] Add a failing synthetic test where the selected blend has weighted 2024 MAE no worse than pure persistence.
- [ ] Implement hourly/day-of-year climatology, persistence/climatology blend evaluation by lead, deterministic selection, and JSON artifact serialization with source hashes and metrics.
- [ ] Generate the frozen Chengdu artifact under `data/processed/chengdu_agri/greenhouse_001/forecasts/v2/` from existing 2023/2024 V2 weather files.
- [ ] Run focused tests and verify the artifact checksum and role manifest.

### Task 4: Runtime Provider and Benchmark Wiring

**Files:**
- Extend: `glassgym/components/forecast.py`
- Modify: `glassgym/environments/greenlight_env.py`
- Modify: `configs/envs/ChengduControllerBenchmark.yml`
- Modify: `experiments/controllers/benchmark_protocol.py`
- Test: `tests/test_forecast_emulator_v2.py`
- Test: `tests/test_controller_benchmark_integration.py`

- [ ] Add failing poisoned-future tests showing identical 2025 forecasts and MPC proposals for histories identical through issue time.
- [ ] Implement artifact loading and issue-time forecast generation with explicit model/version/error IDs.
- [ ] Configure the Chengdu benchmark to use the frozen emulator; retain persistence as an explicit diagnostic fallback only.
- [ ] Record provider identity and forecast mode in benchmark trajectories and summaries.
- [ ] Run focused integration tests.

### Task 5: Verification and Audit

**Files:**
- Create: `docs/audits/stage_a_v2_package_f_controller_model_forecast.md`

- [ ] Run the complete repository test suite with a workspace-local pytest temp directory.
- [ ] Run one short baseline/PID/MPC smoke comparison and verify finite reward, comfort, runtime, safety, forecast, and yield columns.
- [ ] Request an independent read-only adversarial review of latent-state access, data roles, future leakage, and metric propagation.
- [ ] Fix every P1/P2 finding with a failing regression test first.
- [ ] Write a GO/NO-GO audit that distinguishes the historical emulator from operational forecasts and lists remaining paper gates.

