# Harvest Season Completion Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require independent season-boundary and completion evidence before real harvest calibration or validation.

**Architecture:** A focused loader validates a one-row-per-season CSV and binds it to canonical harvest events. The production pipeline requires this manifest with event data, excludes ongoing seasons from fitting, and records completion audit metadata.

**Tech Stack:** Python, pandas, NumPy, pytest, CSV.

---

### Task 1: Season manifest and event binding

**Files:**
- Create: `processing/chengdu_harvest_seasons.py`
- Create: `tests/test_chengdu_harvest_seasons.py`
- Create: `data/raw/chengdu_agri/greenhouse_001/crop/harvest_seasons_template.csv`

- [x] Add failing tests for valid complete/ongoing rows and event binding.
- [x] Add failing tests for missing source, invalid bounds, identity mismatch, and out-of-bound events.
- [x] Implement manifest loading, binding, canonical columns, and audit summary.

### Task 2: Pipeline gate and readiness

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add a failing argument-contract test requiring a season manifest with harvest events.
- [x] Bind events, filter calibration to complete-season evidence, and report completion counts.
- [x] Ensure data sufficiency does not count ongoing seasons as usable.

### Task 3: Documentation and audit

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`

- [x] Document the season template and completion-source requirements.
- [x] Run complete tests and verify current readiness remains false with zero completed target seasons.

No Git metadata is present in this workspace; tests and generated evidence replace commit checkpoints.
