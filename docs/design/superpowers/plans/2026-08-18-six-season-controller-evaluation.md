# Six-season Controller Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run frozen Baseline, PID, MPC, PPO, and SAC controllers over six complete 120-day Chengdu seasons and generate validated paper-ready comparisons.

**Architecture:** Keep the existing single-test benchmark intact, add backward-compatible scenario overrides to its evaluators, and place multi-season orchestration in a focused resumable runner. Every result is keyed by algorithm, seed, and season and is validated before aggregation.

**Tech Stack:** Python, pandas, NumPy, Gymnasium, Stable-Baselines3, pytest, GreenLight model.

---

### Task 1: Scenario override contract

**Files:**
- Modify: `tests/test_controller_benchmark.py`
- Modify: `experiments/controllers/tune_benchmark_classical.py`
- Modify: `experiments/controllers/train_benchmark_rl.py`

- [ ] Write tests that call the classical and saved-agent evaluators with explicit `growth_year` and `start_day` values and assert those values appear in every returned row.
- [ ] Run the focused tests and verify they fail because the overrides are not accepted or forwarded.
- [ ] Add optional keyword arguments and forward them to `build_environment` / `_make_eval_env`, retaining current defaults.
- [ ] Run the focused tests and verify they pass.

### Task 2: Multi-season protocol and artifact validation

**Files:**
- Create: `tests/test_six_season_controller_evaluation.py`
- Create: `experiments/controllers/evaluate_six_season_controllers.py`

- [ ] Write failing tests for the six canonical season definitions, collision-free unit keys and filenames, complete trajectory validation, and rejection of wrong-year or short trajectories.
- [ ] Run the test module and verify failures are caused by the missing runner API.
- [ ] Implement `EvaluationSeason`, canonical season expansion, unit identity helpers, and strict trajectory validation.
- [ ] Run the test module and verify these protocol tests pass.

### Task 3: Resume and aggregation

**Files:**
- Modify: `tests/test_six_season_controller_evaluation.py`
- Modify: `experiments/controllers/evaluate_six_season_controllers.py`

- [ ] Write failing tests showing that a valid complete unit is skipped, a stale unit is rerun, and records remain distinct by algorithm, seed, and season.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement atomic manifest transitions, metric upserts, aggregate-by-algorithm, and aggregate-by-season functions.
- [ ] Run the focused tests and verify they pass.

### Task 4: Controller execution CLI

**Files:**
- Modify: `tests/test_six_season_controller_evaluation.py`
- Modify: `experiments/controllers/evaluate_six_season_controllers.py`

- [ ] Write failing tests for job expansion, frozen artifact lookup, smoke mode, algorithm filters, and disabled-control checks.
- [ ] Run the focused tests and confirm the new behavior is absent.
- [ ] Implement CLI arguments `--profile`, `--algorithms`, `--seasons`, `--resume`, `--smoke-steps`, and `--max-workers`; load selected classical parameters and frozen RL artifacts from the paper benchmark directory.
- [ ] Run the focused tests and verify they pass.

### Task 5: Preflight and complete evaluation

**Files:**
- Generate: `results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d_guarded_v2/*`

- [ ] Run one short smoke unit for each algorithm and verify model loading, finite state, scenario identity, and zero heating/CO2.
- [ ] Run the complete six-season command with resume enabled and bounded parallelism.
- [ ] Validate all 78 expected episodes: 18 deterministic-controller episodes and 60 RL episodes, each with 11,520 steps.
- [ ] Generate algorithm and season comparison tables from only validated complete episodes.

### Task 6: Result audit

**Files:**
- Generate: `results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d/result_analysis.md`

- [ ] Check numerical stability, physical ranges, completion, controller fallback, safety intervention, comfort, runtime, and mass/yield metrics.
- [ ] Compare weather-season sensitivity and seed variability without test-set seed selection.
- [ ] State clearly which yield conclusions are simulator-relative and which require future Chengdu harvest validation.
- [ ] Run the focused tests and full repository test suite, recording exact outcomes.
