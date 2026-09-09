# V5 Controller Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select pure PPO, pure SAC, and MPC candidates on validation weather and determine whether each legitimately outperforms frozen PID on untouched confirmation weather.

**Architecture:** Keep controller definitions, generic candidate ranking, and the experiment runner separate. Extend MPC with validated configuration parameters, add resumable RL checkpoint training to the existing RL utilities, and orchestrate selection/confirmation in one new script that writes machine-readable and Markdown outputs.

**Tech Stack:** Python 3.12, NumPy, pandas, CasADi, Gymnasium, Stable-Baselines3, pytest.

---

### Task 1: Configurable MPC

**Files:**
- Modify: `experiments/controllers/v5_smoke_controllers.py`
- Modify: `tests/test_v5_smoke_controllers.py`

- [ ] **Step 1: Write failing parameter validation and scoring tests**

Add tests asserting that custom actuator levels and four cost weights are retained,
generate the expected Cartesian candidate count, return finite actions, and reject
empty/out-of-range levels or negative weights.

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_v5_smoke_controllers.py -q`

Expected: failure because `V5HybridMPC` does not accept the new parameters.

- [ ] **Step 3: Implement the minimal configurable controller**

Give `V5HybridMPC.__init__` keyword parameters `levels`, `temperature_weight`,
`humidity_weight`, `effort_weight`, and `variation_weight`; validate all inputs and
replace the four hard-coded score coefficients with these values.

- [ ] **Step 4: Run the focused test**

Run: `pytest tests/test_v5_smoke_controllers.py -q`

Expected: all tests pass.

### Task 2: Validation ranking and promotion gates

**Files:**
- Create: `experiments/controllers/v5_controller_optimization.py`
- Create: `tests/test_v5_controller_optimization.py`

- [ ] **Step 1: Write failing ranking tests**

Test `aggregate_validation_candidates`, `select_best_candidate`, and
`assess_confirmation`. Verify mean aggregation across all three validation
scenarios, deterministic tie-breaking, rejection of incomplete/unstable candidates,
strict reward superiority to PID, climate tolerances, and safety tolerance.

- [ ] **Step 2: Run the new test and observe import failure**

Run: `pytest tests/test_v5_controller_optimization.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement pure table transformations**

Create constants for training, validation, confirmation, and excluded smoke-test
scenarios. Implement ranking using descending mean reward then ascending temperature
MAE, RH MAE, action variation, and inference time. Keep confirmation assessment in
a separate function so confirmation rows cannot enter selection.

- [ ] **Step 4: Run ranking tests**

Run: `pytest tests/test_v5_controller_optimization.py -q`

Expected: all tests pass.

### Task 3: Resumable RL checkpoints

**Files:**
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Modify: `tests/test_v5_controller_smoke.py`

- [ ] **Step 1: Write a failing small checkpoint test**

Train at cumulative checkpoints 32 and 64 for PPO and SAC. Assert each checkpoint
contains `model.zip`, `vecnormalize.pkl`, and metadata recording algorithm, seed,
cumulative steps, exact training scenarios, and schema version.

- [ ] **Step 2: Run the focused test and observe failure**

Run: `pytest tests/test_v5_controller_smoke.py -q`

Expected: failure because checkpoint training is not implemented.

- [ ] **Step 3: Implement checkpoint training**

Factor model construction into a helper and add `train_rl_checkpoints`. Train one
model incrementally with `reset_num_timesteps=False`, save normalizer and model after
each cumulative target, skip a checkpoint only when metadata exactly matches, and
return ordered metadata records.

- [ ] **Step 4: Run the RL utility tests**

Run: `pytest tests/test_v5_controller_smoke.py -q`

Expected: all tests pass for PPO and SAC.

### Task 4: Optimization runner

**Files:**
- Create: `experiments/controllers/run_v5_controller_optimization.py`
- Modify: `tests/test_v5_controller_optimization.py`

- [ ] **Step 1: Write failing protocol tests**

Assert the runner's manifest has disjoint training, validation, confirmation, and
historical-test roles; MPC candidate IDs are stable; and confirmation receives only
already selected candidate IDs.

- [ ] **Step 2: Run the protocol tests and observe failure**

Run: `pytest tests/test_v5_controller_optimization.py -q`

Expected: missing runner functions.

- [ ] **Step 3: Implement CLI orchestration**

Add CLI options for output directory, episode days, seeds, checkpoints, and resume.
Evaluate frozen PID on validation, search bounded MPC candidates, train/evaluate PPO
and SAC checkpoints, freeze one selection per algorithm, then evaluate only those
three selections and PID on confirmation. Write CSV trajectories/tables and JSON
manifests after every completed unit.

- [ ] **Step 4: Add report generation**

Generate `scientific_interpretation.md` with validation rankings, confirmation
metrics, per-gate pass/fail reasons, explicit data roles, and the statement that
one-day fruit-state change is not measured harvested yield.

- [ ] **Step 5: Run protocol tests**

Run: `pytest tests/test_v5_controller_optimization.py -q`

Expected: all tests pass.

### Task 5: Execute and verify the experiment

**Files:**
- Create outputs under: `results/chengdu_agri_greenhouse_001/controller_benchmark/v5_controller_optimization/`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_v5_smoke_controllers.py tests/test_v5_controller_smoke.py tests/test_v5_controller_optimization.py -q`

Expected: all pass.

- [ ] **Step 2: Run validation selection and confirmation**

Run: `python -m experiments.controllers.run_v5_controller_optimization --episode-days 1 --seeds 0 1 --checkpoints 8192 16384 32768 --resume`

Expected: complete candidate, selection, confirmation, and report artifacts without
numerical failures or residual fallback.

- [ ] **Step 3: Inspect output consistency**

Check that every selected ID exists in the validation table, no confirmation
scenario appears in validation rows, and every algorithm has exactly one
confirmation row.

- [ ] **Step 4: Run the full suite**

Run: `pytest -q`

Expected: the current suite plus new tests passes.

- [ ] **Step 5: Report honestly**

State which of PPO, SAC, and MPC passed every confirmation gate against PID. If an
algorithm fails, retain its results and recommend the separately labeled residual-RL
stage rather than changing PID or reselecting on confirmation.

### Task 6: Residual-PPO follow-up after pure PPO failure

**Files:**
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Create: `experiments/controllers/run_v5_residual_ppo_optimization.py`
- Modify: `tests/test_v5_controller_smoke.py`
- Modify: `tests/test_v5_controller_optimization.py`

- [ ] **Step 1: Write failing wrapper and metadata tests**

Assert a zero residual reproduces the frozen PID normalized action, residual actions
are clipped to the common action space, and checkpoint metadata records the residual
scale so incompatible artifacts cannot be reused.

- [ ] **Step 2: Implement PID-guided action wrapping**

Add a Gymnasium action wrapper and optional `residual_pid_scale` arguments to the
training/evaluation factories while preserving default pure-RL behavior.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest tests/test_v5_controller_smoke.py -q`

Expected: all tests pass for pure and residual paths.

- [ ] **Step 4: Train and select Residual-PPO**

Train seeds 0 and 1 at cumulative checkpoints 4,096, 8,192, and 16,384 using residual
scale 0.25. Select only on 2025 spring days 59, 74, and 89.

- [ ] **Step 5: Open a fresh unified confirmation**

Evaluate frozen PID, first-stage selected MPC and SAC, and selected Residual-PPO once
on 2025 spring day 119. Apply the same reward, climate, safety, and health gates.

- [ ] **Step 6: Preserve both conclusions**

Report that pure PPO failed first-stage confirmation and state the Residual-PPO
result separately. Do not replace the pure PPO row or claim that residual guidance
is the original PPO algorithm.
