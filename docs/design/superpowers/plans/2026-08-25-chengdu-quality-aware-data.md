# Chengdu Quality-Aware Climate Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an audited V4 Chengdu trajectory with channel-aware sensor quality weights and heat-regime labels while preserving every valid observation.

**Architecture:** A focused sensor-quality module derives per-variable hourly metadata from existing channel and observed-fraction columns. A V4 trajectory builder merges current and target quality onto the established physical-control trajectory. A report command summarizes data support and stratifies frozen rollout errors without changing primary metrics.

**Tech Stack:** Python 3.12, pandas, NumPy, pytest.

---

### Task 1: Hourly Sensor Quality

**Files:**
- Create: `processing/chengdu_sensor_quality.py`
- Create: `tests/test_chengdu_sensor_quality.py`

- [ ] Write failing tests for effective coverage, single-sensor medium quality, agreeing dual-sensor high quality and disagreement low quality.
- [ ] Run the focused test and confirm the module is missing.
- [ ] Implement channel discovery and deterministic quality metadata.
- [ ] Re-run the focused tests.

### Task 2: V4 Transition Bundle

**Files:**
- Create: `processing/build_chengdu_trajectory_v4.py`
- Test: `tests/test_chengdu_sensor_quality.py`

- [ ] Write failing tests that preserve values and rows, attach current and target quality, retain extreme heat and compute the minimum transition weight.
- [ ] Implement the V4 builder on top of the existing valid-transition and physical-device contracts.
- [ ] Save chronological splits and a hashed manifest with quality counts.
- [ ] Re-run V2-V4 trajectory tests.

### Task 3: Stratified Quality Audit

**Files:**
- Create: `experiments/reports/audit_chengdu_data_quality_v4.py`
- Test: `tests/test_chengdu_sensor_quality.py`

- [ ] Write a failing test for MAE/RMSE/Bias grouped by quality tier and heat regime.
- [ ] Implement dataset and optional rollout summaries without replacing primary metrics.
- [ ] Generate the target-greenhouse audit artifacts.

### Task 4: Verification

**Files:**
- Create: `docs/audits/chengdu_data_quality_v4.md`

- [ ] Confirm V3 and V4 have identical valid row counts and canonical observations.
- [ ] Confirm all extreme rows remain present.
- [ ] Run focused tests and the complete repository suite.
- [ ] Record the quality distribution, hashes, limitations and recommended model-use policy.

