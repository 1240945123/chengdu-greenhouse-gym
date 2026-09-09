# Harvest Timing Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-season harvest timing metrics and acceptance gates.

**Architecture:** Compute timing metrics from already aligned daily batches in the common evaluation module, aggregate them season-wise, and enforce explicit thresholds in the Chengdu production acceptance function.

**Tech Stack:** Python, pandas, NumPy, pytest.

---

### Task 1: Single-season timing metrics

**Files:**
- Modify: `common/harvest_evaluation.py`
- Modify: `tests/test_harvest_evaluation.py`

- [x] Add failing analytically known Wasserstein and p50/p90 date tests.
- [x] Implement normalized cumulative-distribution distance and quantile-date lookup.
- [x] Verify zero predicted yield produces non-finite timing metrics.

### Task 2: Multi-season aggregation

**Files:**
- Modify: `common/harvest_evaluation.py`
- Modify: `tests/test_harvest_evaluation.py`

- [x] Add failing tests proving calendar gaps are excluded and mean/max summaries are season-local.
- [x] Aggregate mean and maximum timing metrics across seasons.
- [x] Run the complete harvest evaluation test module.

### Task 3: Acceptance and documentation

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`

- [x] Add failing acceptance tests for each timing gate.
- [x] Add default thresholds and finite-value gates.
- [x] Document the metric definitions and thresholds.

### Task 4: Verification

- [x] Run the complete test suite with temporary files on drive E.
- [x] Rebuild the current target readiness report.
- [x] Keep the full goal active until real complete target seasons are available.

No Git metadata is present in this workspace; tests and generated reports
replace commit checkpoints.
