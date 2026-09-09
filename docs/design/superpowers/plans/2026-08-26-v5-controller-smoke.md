# V5 Controller Smoke Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run leakage-safe Baseline/PID/MPC/PPO/SAC smoke comparison in the V5 hybrid environment.

**Architecture:** A two-action controller module exposes a common target interface for deterministic policies. A separate benchmark runner trains compact SB3 agents on historical training weather, selects only on validation weather, evaluates once on test weather, and writes common metrics from executed controls.

**Tech Stack:** Python, NumPy, pandas, Gymnasium, Stable-Baselines3, pytest

---

### Task 1: Deterministic two-action controllers

**Files:**
- Create: `experiments/controllers/v5_smoke_controllers.py`
- Test: `tests/test_v5_smoke_controllers.py`

- [ ] Write failing tests for frozen baseline, PID high-temperature response, RH fan lockback, and MPC finite supported actions.
- [ ] Implement minimal stateless baseline, stateful PID, and short-horizon MPC.
- [ ] Run focused tests and verify they pass.

### Task 2: Common rollout and metrics

**Files:**
- Create: `experiments/controllers/run_v5_controller_smoke.py`
- Test: `tests/test_v5_controller_smoke.py`

- [ ] Write failing tests for executed-control rollout fields and promotion gates.
- [ ] Implement deterministic rollout, metrics, validation selection, and immutable test-role checks.
- [ ] Verify baseline/PID/MPC short rollouts complete.

### Task 3: PPO and SAC smoke training

**Files:**
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Modify: `tests/test_v5_controller_smoke.py`

- [ ] Write failing tests for compact PPO/SAC train-save-load-evaluate behavior using the V5 builder.
- [ ] Implement flattened vector environments, VecNormalize persistence, fixed seeds, and validation checkpoint selection.
- [ ] Run 2023-2024 training and 2025 spring validation.

### Task 4: Frozen test and report

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/controller_benchmark/v5_hybrid_smoke/*`

- [ ] Freeze promoted artifacts and hashes.
- [ ] Open the 2025 autumn test once for all five algorithms.
- [ ] Write per-step trajectories, summary tables, ranking, and limitations.

### Task 5: Verification

**Files:**
- Modify only files above if verification exposes a defect.

- [ ] Run focused controller, environment, safety, and RL integration tests.
- [ ] Run the complete pytest suite with zero failures.
