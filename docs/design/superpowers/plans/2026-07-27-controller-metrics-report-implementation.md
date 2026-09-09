# Controller Metrics Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conventional control-performance metrics and paper-ready comparison artifacts to the completed Chengdu benchmark.

**Architecture:** Extend episode summarization with error, comfort, actuator, and runtime metrics; make aggregation emit mean and standard deviation for all numeric metrics; then have the report layer select and format the paper-facing subset. Existing trajectories remain the source of truth.

**Tech Stack:** Python, pandas, NumPy, matplotlib, pytest

---

### Task 1: Episode Metrics

**Files:**
- Modify: `experiments/controllers/benchmark_metrics.py`
- Test: `tests/test_controller_benchmark_metrics.py`

- [ ] Add failing assertions for RMSE, separate comfort fractions, per-actuator means, and milliseconds per step.
- [ ] Run the focused test and confirm the new keys are absent.
- [ ] Implement the metrics from trajectory columns using finite numeric calculations.
- [ ] Run the focused metric tests and confirm they pass.

### Task 2: Algorithm Dispersion

**Files:**
- Modify: `experiments/controllers/benchmark_metrics.py`
- Test: `tests/test_controller_benchmark_metrics.py`

- [ ] Add a failing test requiring `<metric>_mean` and `<metric>_std` for every numeric episode metric.
- [ ] Run the test and confirm the standard-deviation column is absent.
- [ ] Add sample standard deviation for repeated runs and zero for one-run deterministic algorithms.
- [ ] Run the aggregation tests and confirm they pass.

### Task 3: Paper Report

**Files:**
- Modify: `experiments/controllers/report_chengdu_benchmark.py`
- Test: `tests/test_controller_benchmark_metrics.py`

- [ ] Add failing artifact checks for `paper_metrics.csv`, `paper_metrics.md`, `tracking_errors.png`, `comfort_rates.png`, and `runtime_comparison.png`.
- [ ] Run the report test and confirm the files are missing.
- [ ] Implement concise CSV/Markdown tables and dimensionally separated plots.
- [ ] Run the report tests and confirm they pass.

### Task 4: Regenerate And Verify

**Files:**
- Regenerate: `results/chengdu_agri_greenhouse_001/controller_benchmark/paper/*`

- [ ] Recompute `episode_metrics.csv` from all saved trajectories so newly added formulas are applied consistently.
- [ ] Generate the expanded report from the paper profile.
- [ ] Run the focused benchmark test suite.
- [ ] Inspect the resulting table and figures for finite values, complete algorithms, and readable labels.

