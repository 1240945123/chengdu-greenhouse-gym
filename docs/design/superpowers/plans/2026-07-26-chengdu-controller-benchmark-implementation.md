# Chengdu Controller Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a reproducible Chengdu benchmark comparing rule-based, PID, MPC, PPO, and SAC under one climate-control protocol.

**Architecture:** A dedicated environment config fixes heating and CO2 at zero and exposes four controls. Shared benchmark modules own action adaptation, local RL training, evaluation, metrics, manifests, and reporting; thin CLI scripts orchestrate smoke and paper profiles without requiring W&B.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3, CasADi, NumPy, pandas, PyYAML, matplotlib, pytest.

---

## File Map

- Modify `glassgym/components/rewards.py`: add the Chengdu climate reward and registry entry.
- Create `configs/envs/ChengduControllerBenchmark.yml`: common four-action environment.
- Create `configs/benchmarks/chengdu_controllers.yml`: profiles, windows, seeds, metrics, and algorithm settings.
- Create `experiments/controllers/benchmark_protocol.py`: configuration, action adaptation, evaluation, and validation.
- Create `experiments/controllers/benchmark_metrics.py`: episode and aggregate statistics.
- Create `experiments/controllers/train_benchmark_rl.py`: local PPO/SAC train-save-load workflow.
- Create `experiments/controllers/tune_benchmark_classical.py`: validation-only PID/MPC selection.
- Create `experiments/controllers/run_chengdu_benchmark.py`: resumable orchestration and manifests.
- Create `experiments/controllers/report_chengdu_benchmark.py`: tables and plots.
- Create `tests/test_chengdu_climate_reward.py`: reward arithmetic tests.
- Create `tests/test_controller_benchmark_protocol.py`: action and split tests.
- Create `tests/test_controller_benchmark_metrics.py`: aggregation tests.
- Create `tests/test_controller_benchmark_integration.py`: five-controller smoke tests.
- Modify `data/README.md`: document benchmark inputs and result interpretation.

Git metadata is absent in this workspace. The commit steps below are recorded as
logical checkpoints; execute them only if `.git` becomes available, and never
initialize or rewrite repository history implicitly.

### Task 1: Chengdu Climate Reward

**Files:**
- Modify: `glassgym/components/rewards.py`
- Create: `tests/test_chengdu_climate_reward.py`

- [ ] **Step 1: Write failing reward tests**

Create tests that build a `RewardContext` with indoor observations ordered as
`[co2, temperature, rh]` and assert the exact decomposition:

```python
def test_chengdu_reward_is_zero_inside_comfort_with_idle_controls():
    reward, info = ChengduClimateReward(dt=900, p=np.zeros(208)).compute_reward(
        make_context(temp=24.0, rh=72.5, hour=12.0, u=np.zeros(6))
    )
    assert reward == pytest.approx(0.0)
    assert info["temperature_penalty"] == 0.0
    assert info["humidity_penalty"] == 0.0


def test_chengdu_reward_excludes_heating_and_co2_states():
    first = make_context(temp=30.0, rh=90.0, co2=300.0, u=np.zeros(6))
    second = make_context(temp=30.0, rh=90.0, co2=1600.0, u=np.array([1, 1, 0, 0, 0, 0]))
    reward_a, _ = ChengduClimateReward(dt=900, p=np.zeros(208)).compute_reward(first)
    reward_b, _ = ChengduClimateReward(dt=900, p=np.zeros(208)).compute_reward(second)
    assert reward_a == pytest.approx(reward_b)
```

- [ ] **Step 2: Verify the tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chengdu_climate_reward.py -q`

Expected: collection fails because `ChengduClimateReward` is not defined.

- [ ] **Step 3: Implement the reward**

Add a reward class with configurable day/night temperature bands, RH band, and
weights. Compute distance outside each band, divide temperature by `10` and RH by
`40`, then add lamp, allowed-action magnitude, and action-change terms. Return
`reward = -total_penalty` and these stable info keys:

```python
info = {
    "reward": reward,
    "temperature_penalty": temperature_penalty,
    "humidity_penalty": humidity_penalty,
    "lamp_penalty": lamp_penalty,
    "effort_penalty": effort_penalty,
    "action_change_penalty": action_change_penalty,
    "temperature_target": 0.5 * (temp_low + temp_high),
    "humidity_target": 0.5 * (rh_low + rh_high),
}
```

Register it as `REWARDS_MODULES["ChengduClimateReward"]`.

- [ ] **Step 4: Run focused and reward regression tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_chengdu_climate_reward.py tests/reward_test.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Record checkpoint**

When Git exists: `git commit -am "feat: add Chengdu climate benchmark reward"`.

### Task 2: Common Environment And Action Protocol

**Files:**
- Create: `configs/envs/ChengduControllerBenchmark.yml`
- Create: `configs/benchmarks/chengdu_controllers.yml`
- Create: `experiments/controllers/benchmark_protocol.py`
- Create: `tests/test_controller_benchmark_protocol.py`

- [ ] **Step 1: Write failing protocol tests**

Test that `full_control_target_to_action()` maps a six-control target to the four
incremental normalized actions and cannot expose heating or CO2:

```python
def test_target_adapter_enforces_disabled_controls():
    current = np.array([0, 0, 0.4, 0.3, 0.0, 0.2], dtype=np.float32)
    target = np.array([1, 1, 0.8, 0.0, 1.0, 0.2], dtype=np.float32)
    action = full_control_target_to_action(current, target, delta=0.1)
    assert action.tolist() == pytest.approx([1.0, -1.0, 1.0, 0.0])


def test_resolved_windows_are_chronological_and_fit_weather():
    protocol = load_benchmark_config("smoke")
    validate_windows(protocol, available_days=31)
    assert max(protocol.train_start_days) < protocol.validation_start_day
    assert protocol.validation_start_day < protocol.test_start_day
```

- [ ] **Step 2: Verify protocol tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_protocol.py -q`

Expected: import failure for `benchmark_protocol`.

- [ ] **Step 3: Add immutable benchmark configuration**

Set `controlled_inputs` to `[uThScr, uVent, uLamp, uBlScr]`, `season_length` to
`4`, normalized incremental actions to true, `reward_fn` to
`ChengduClimateReward`, training starts to `0..17`, validation start to `21`, and
test start to `26`. Configure smoke as one seed with `4096` PPO/SAC steps and paper
as five seeds with `2_000_000` steps per algorithm.

- [ ] **Step 4: Implement protocol helpers**

Define frozen config dataclasses, YAML loading, chronological validation, resolved
config serialization, six-to-four action adaptation, controller construction, and
one-episode evaluation. Each trajectory row must include six `u*` columns,
temperature, RH, reward components, wall time, algorithm, and seed. Assert
`uBoil == 0` and `uCO2 == 0` at every step.

- [ ] **Step 5: Run protocol and existing controller tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_protocol.py tests/pid_controller.py tests/mpc_controller.py -q`

Expected: all selected tests pass.

- [ ] **Step 6: Record checkpoint**

When Git exists: `git commit -am "feat: define common Chengdu controller protocol"`.

### Task 3: Metrics And Statistical Aggregation

**Files:**
- Create: `experiments/controllers/benchmark_metrics.py`
- Create: `tests/test_controller_benchmark_metrics.py`

- [ ] **Step 1: Write failing deterministic metric tests**

```python
def test_episode_metrics_integrate_fifteen_minute_steps():
    rows = synthetic_steps(rewards=[-1.0, -2.0], temp=[18.0, 36.0], rh=[70.0, 90.0])
    metrics = summarize_episode(rows, dt_seconds=900)
    assert metrics["cumulative_reward"] == pytest.approx(-3.0)
    assert metrics["temperature_violation_hours"] == pytest.approx(0.25)
    assert metrics["humidity_violation_hours"] == pytest.approx(0.25)


def test_aggregate_reports_seed_distribution():
    summary = aggregate_algorithms(pd.DataFrame({
        "algorithm": ["pid", "pid", "sac", "sac"],
        "cumulative_reward": [-4.0, -2.0, -3.0, -3.0],
    }), bootstrap_samples=1000, bootstrap_seed=7)
    assert summary.loc["pid", "reward_mean"] == pytest.approx(-3.0)
    assert summary.loc["sac", "reward_std"] == pytest.approx(0.0)
```

- [ ] **Step 2: Verify metric tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_metrics.py -q`

Expected: import failure for `benchmark_metrics`.

- [ ] **Step 3: Implement episode and aggregate metrics**

Use named columns, explicit `dt_seconds`, sample standard deviation, seeded
bootstrap resampling, and NaN rejection. Include all metrics from the design and
mark `paper_ready=False` unless all five algorithms and required seeds exist.

- [ ] **Step 4: Verify metric tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_metrics.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Record checkpoint**

When Git exists: `git commit -am "feat: add benchmark statistics"`.

### Task 4: Local PPO And SAC Training

**Files:**
- Create: `experiments/controllers/train_benchmark_rl.py`
- Create: `tests/test_controller_benchmark_integration.py`

- [ ] **Step 1: Write a failing minimal train-save-load test**

Parameterize the test over `ppo` and `sac`, train for the smallest valid algorithm
budget on a one-day test override, and assert these artifacts exist:

```python
assert (run_dir / "best_model.zip").exists()
assert (run_dir / "final_model.zip").exists()
assert (run_dir / "vecnormalize.pkl").exists()
assert (run_dir / "training_metadata.json").exists()
assert evaluate_saved_agent(run_dir, algorithm="ppo", seed=0)
```

- [ ] **Step 2: Verify the RL integration test fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k rl -q`

Expected: import failure for `train_benchmark_rl`.

- [ ] **Step 3: Implement local training**

Create flattened vector environments with `DummyVecEnv` for smoke and configurable
parallel environments for paper. Apply `VecNormalize(norm_obs=True,
norm_reward=True)` during training, share observation normalization with an
unnormalized-reward validation env, and use `EvalCallback` plus
`CheckpointCallback`. Disable W&B and write resolved hyperparameters, package
versions, seed, timestamps, validation scores, and completion status locally.

- [ ] **Step 4: Implement strict saved-agent evaluation**

Load the matching SB3 class and normalization statistics, set
`training=False/norm_reward=False`, use deterministic predictions, and reject
action-space or observation-space mismatch before evaluation.

- [ ] **Step 5: Run the minimal PPO/SAC cycle**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k rl -q`

Expected: PPO and SAC both train, save, reload, and finish evaluation.

- [ ] **Step 6: Record checkpoint**

When Git exists: `git commit -am "feat: add local PPO and SAC benchmark training"`.

### Task 5: Validation-Only PID And MPC Selection

**Files:**
- Create: `experiments/controllers/tune_benchmark_classical.py`
- Modify: `configs/benchmarks/chengdu_controllers.yml`
- Modify: `tests/test_controller_benchmark_integration.py`

- [ ] **Step 1: Write failing selection tests**

Provide two candidates with known validation scores and assert selection uses
reward first, then lower control variation as a deterministic tie-break. Assert
that trial rows contain only validation start day `21` and never test day `26`.

- [ ] **Step 2: Verify selection tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k classical -q`

Expected: missing tuner functions.

- [ ] **Step 3: Implement PID and MPC searches**

Use small explicit candidate grids from YAML. PID candidates vary ventilation
temperature/RH gains. MPC candidates vary horizons `2, 4, 8` and climate/action
weights while setting CO2, heat, and fruit weights to zero and
`include_conservative=false`. Save every trial and the selected YAML/JSON.

- [ ] **Step 4: Run selection and controller tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k classical tests/pid_controller.py tests/mpc_controller.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Record checkpoint**

When Git exists: `git commit -am "feat: tune PID and MPC on validation weather"`.

### Task 6: Resumable Five-Algorithm Orchestration

**Files:**
- Create: `experiments/controllers/run_chengdu_benchmark.py`
- Modify: `tests/test_controller_benchmark_integration.py`

- [ ] **Step 1: Write failing manifest tests**

Test transitions `pending -> running -> complete/failed`, profile hash checking,
resume behavior for complete runs, rerun behavior for failed runs, and refusal to
reuse an artifact whose resolved config hash differs.

- [ ] **Step 2: Verify manifest tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k manifest -q`

Expected: missing runner and manifest functions.

- [ ] **Step 3: Implement orchestration CLI**

Support:

```powershell
.venv\Scripts\python.exe -m experiments.controllers.run_chengdu_benchmark `
  --profile smoke --algorithms baseline pid mpc ppo sac --resume
```

Run classic tuning, RL training, fixed-window test evaluation, episode metrics,
and manifest updates. Write atomically via a temporary file in the result
directory and rename only after successful serialization.

- [ ] **Step 4: Verify manifest tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_integration.py -k manifest -q`

Expected: all selected tests pass.

- [ ] **Step 5: Record checkpoint**

When Git exists: `git commit -am "feat: orchestrate Chengdu controller benchmark"`.

### Task 7: Paper Tables And Figures

**Files:**
- Create: `experiments/controllers/report_chengdu_benchmark.py`
- Modify: `tests/test_controller_benchmark_metrics.py`
- Modify: `data/README.md`

- [ ] **Step 1: Write failing report tests**

Build a temporary complete result set and assert generation of
`comparison.csv`, `comparison.json`, `reward_comparison.png`,
`comfort_comparison.png`, `control_effort.png`, and
`test_trajectories.png`. Build an incomplete set and assert its JSON says
`paper_ready: false` with missing algorithms/seeds listed.

- [ ] **Step 2: Verify report tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_metrics.py -k report -q`

Expected: missing report module.

- [ ] **Step 3: Implement report generation**

Use deterministic plotting order `baseline, pid, mpc, ppo, sac`, include error
bars only where replication exists, label axes with physical units, and write the
single-test-weather limitation into the JSON metadata and README.

- [ ] **Step 4: Verify report tests pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_controller_benchmark_metrics.py -q`

Expected: all selected tests pass and images are nonempty.

- [ ] **Step 5: Record checkpoint**

When Git exists: `git commit -am "feat: report controller benchmark results"`.

### Task 8: End-To-End Verification And Runs

**Files:**
- Output: `results/chengdu_agri_greenhouse_001/controller_benchmark/smoke/`
- Output: `results/chengdu_agri_greenhouse_001/controller_benchmark/paper/`

- [ ] **Step 1: Run focused tests**

Run:

```powershell
.venv\Scripts\python.exe -m pytest `
  tests/test_chengdu_climate_reward.py `
  tests/test_controller_benchmark_protocol.py `
  tests/test_controller_benchmark_metrics.py `
  tests/test_controller_benchmark_integration.py `
  tests/pid_controller.py tests/mpc_controller.py tests/chengdu_physics_model.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run all five smoke experiments**

Run:

```powershell
$env:TEMP='E:\school\final paper\.tmp'
$env:TMP=$env:TEMP
$env:TORCH_HOME='E:\school\final paper\.cache\torch'
.venv\Scripts\python.exe -m experiments.controllers.run_chengdu_benchmark `
  --profile smoke --algorithms baseline pid mpc ppo sac --resume
```

Expected: manifest marks all five algorithms complete, all recorded heating and
CO2 controls are zero, and the report is labelled smoke rather than paper-ready.

- [ ] **Step 3: Inspect smoke outputs and numerical health**

Run a validation script that rejects non-finite states/rewards, incomplete
episodes, missing controls, nonzero disabled controls, or absent checkpoints.
Expected: zero validation errors.

- [ ] **Step 4: Launch the full paper profile**

Run:

```powershell
.venv\Scripts\python.exe -m experiments.controllers.run_chengdu_benchmark `
  --profile paper --algorithms baseline pid mpc ppo sac --resume
```

Expected: five complete PPO seeds, five complete SAC seeds, all classic-controller
test runs, and a comparison report with `paper_ready: true`. If execution is
interrupted externally, preserve the manifest and resume from the last complete
seed rather than reporting partial metrics as final.

- [ ] **Step 5: Run the broader regression suite**

Run: `.venv\Scripts\python.exe -m pytest tests -q`

Expected: benchmark tests pass. Report the six already documented legacy
`tests/env_test.py` interface failures separately if they remain unchanged.

- [ ] **Step 6: Publish the exact result status**

State which profile completed, training steps and seeds actually executed,
artifact paths, aggregate reward ranking, comfort/control trade-offs, and any
remaining failures. Never describe a smoke or interrupted paper run as a complete
paper result.
