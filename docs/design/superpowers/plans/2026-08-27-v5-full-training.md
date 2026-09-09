# V5 Full Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver resumable, multi-seed V5 PPO/SAC/Residual-PPO training on four complete Chengdu weather seasons and frozen 120-day controller comparisons.

**Architecture:** First remove pandas from the V5 per-step residual hot path while preserving exact model behavior. Then add a focused full-training protocol module that owns temporal scenario roles, paper hyperparameters, checkpoint selection, resumable evaluation, and reporting while reusing the existing V5 environment and controller APIs.

**Tech Stack:** Python 3.12, NumPy, pandas, Gymnasium, Stable-Baselines3, PyTorch, pytest, YAML/JSON/CSV artifacts.

---

### Task 1: Accelerate V5 Residual Inference

**Files:**
- Modify: `experiments/controllers/v5_hybrid_environment.py`
- Modify: `tests/test_v5_hybrid_environment.py`

- [ ] Add a failing randomized equivalence test comparing the postprocessor result with `apply_ridge_residual_corrector` for the real selected model.
- [ ] Run `pytest tests/test_v5_hybrid_environment.py -q` and verify the missing fast inference API fails.
- [ ] Prevalidate model arrays in `V5ResidualClimatePostprocessor.__init__`, construct the 22 supported thermal-dynamics features as a NumPy vector, and preserve target gain, clipping, and fallback semantics.
- [ ] Run the focused tests and a 500-step throughput benchmark; require equivalent outputs and record before/after steps per second.

### Task 2: Define the Temporal Full-Training Protocol

**Files:**
- Create: `experiments/controllers/v5_full_training_protocol.py`
- Create: `configs/benchmarks/chengdu_v5_full_training.yml`
- Create: `tests/test_v5_full_training_protocol.py`

- [ ] Add failing tests for 14-day training-window expansion, non-overlapping 2025 validation windows, temporal isolation, immutable manifest fields, and checkpoint plateau decisions.
- [ ] Run `pytest tests/test_v5_full_training_protocol.py -q` and verify imports fail.
- [ ] Implement validated protocol dataclasses/loaders and explicit `train`, `validation`, `temporal_holdout`, and `descriptive` roles.
- [ ] Run protocol tests and serialize a resolved protocol preview.

### Task 3: Add Paper-Profile Resumable RL Training

**Files:**
- Modify: `experiments/controllers/run_v5_controller_smoke.py`
- Create: `experiments/controllers/run_v5_full_training.py`
- Modify: `tests/test_v5_controller_smoke.py`
- Create: `tests/test_v5_full_training.py`

- [ ] Add failing tests for paper PPO/SAC hyperparameters, multi-environment factories, checkpoint contract compatibility, Residual-PPO labelling, and completed-run reuse.
- [ ] Verify the new tests fail before implementation.
- [ ] Extend the reusable V5 trainer with explicit model profiles, deterministic CPU thread limits, atomic checkpoint artifacts, and cumulative resume metadata without changing smoke defaults.
- [ ] Implement the orchestrator for three algorithms, three seeds, and rollout-aligned 102.4k/307.2k/614.4k checkpoints with per-run logs and automatic continuation rules.
- [ ] Run focused tests and a short two-seed integration smoke run.

### Task 4: Select Checkpoints and Evaluate Full Seasons

**Files:**
- Create: `experiments/controllers/evaluate_v5_full_training.py`
- Create: `tests/test_v5_full_evaluation.py`

- [ ] Add failing tests for health gates, validation-only selection, deterministic-controller seed handling, resumable trajectories, role-aware aggregation, and disabled-control assertions.
- [ ] Verify tests fail before implementation.
- [ ] Evaluate validation windows, freeze selected checkpoints, then evaluate the 2025 autumn holdout and all six 120-day seasons for baseline/PID/MPC and selected RL agents.
- [ ] Persist trajectories atomically and aggregate reward, temperature/RH error, joint comfort, active safety, effort, action variation, crop carbon, and simulated production indicators.

### Task 5: Run, Verify, and Report

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/controller_benchmark/v5_full_training/`
- Create: `results/chengdu_agri_greenhouse_001/controller_benchmark/v5_full_training/result_analysis.md`

- [ ] Run preflight tests and throughput checks.
- [ ] Run the complete resumable training protocol; inspect every checkpoint metadata file and learning curve.
- [ ] Run frozen 120-day evaluations and regional simulated-yield projection.
- [ ] Run the full focused test suite, then repository tests.
- [ ] Write a result analysis separating statistical evidence, climate-control performance, simulated crop outcomes, runtime, limitations, and claims that remain prohibited.
