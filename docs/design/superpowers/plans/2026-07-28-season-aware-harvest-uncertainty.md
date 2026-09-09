# Season-Aware Harvest Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce season-safe conditional uncertainty intervals for batch harvest and cumulative yield.

**Architecture:** A tested residual-block sampler enforces training-season boundaries. The bootstrap helper creates batch and per-season cumulative ensembles, while harvest evaluation scores both interval families without changing existing acceptance gates.

**Tech Stack:** Python, NumPy, pandas, pytest.

---

### Task 1: Season-aware residual sampler

**Files:**
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_harvest_calibration.py`

- [x] Add a failing test proving one block never mixes residuals from two training seasons.
- [x] Implement `_draw_season_residual_blocks` with within-season circular indexing.
- [x] Add a failing multiseason forecast test proving cumulative quantiles reset at each season.
- [x] Integrate the sampler into `residual_block_prediction_intervals` and run focused tests.

### Task 2: Batch intervals and evaluation

**Files:**
- Modify: `common/harvest_evaluation.py`
- Modify: `tests/test_harvest_evaluation.py`

- [x] Add failing tests for ordered batch quantiles and batch 80%/95% coverage, width, and Winkler metrics.
- [x] Add batch percentile generation and separate missing-day handling for batch versus cumulative intervals.
- [x] Generalize interval metric calculation while preserving existing cumulative metric names and values.
- [x] Run focused evaluation tests.

### Task 3: Pipeline metadata and audit

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`

- [x] Require the synthetic held-out pipeline to emit batch and cumulative interval metrics.
- [x] Record season-boundary and cumulative-reset policies in uncertainty metadata.
- [x] Document conditional scope and remaining excluded sources.
- [x] Run the full suite and an end-to-end synthetic artifact audit.

No Git metadata is present in this workspace; test and generated evidence replace commit checkpoints.
