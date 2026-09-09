# Measured Harvest Trajectories Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export auditable physical-event and daily cumulative harvest trajectories.

**Architecture:** Add one pure processing module that validates canonical measured events and returns event, daily, and season-summary artifacts. Integrate it after event-to-season binding and before calibration.

**Tech Stack:** Python, pandas, NumPy, pytest, CSV, JSON.

---

### Task 1: Event and daily trajectories

**Files:**
- Create: `processing/chengdu_measured_harvest.py`
- Create: `tests/test_chengdu_measured_harvest.py`

- [x] Add failing multi-season tests with multiple same-day physical records.
- [x] Implement event sequence, daily aggregation, cumulative reset, and reconciliation.
- [x] Add failing invalid timestamp, mass, duplicate ID, and missing identity tests.

### Task 2: Season summary and semantics

**Files:**
- Modify: `processing/chengdu_measured_harvest.py`
- Modify: `tests/test_chengdu_measured_harvest.py`

- [x] Add failing tests for complete/ongoing summaries and missing source identifiers.
- [x] Implement serializable season summaries and explicit daily-batch semantics.

### Task 3: Production integration

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`
- Modify: `data/README.md`

- [x] Write both processed CSV artifacts when measured events exist.
- [x] Add the season summary and batch semantics to readiness JSON.
- [x] Document event versus daily model-batch interpretation.

### Task 4: Verification

- [x] Run focused and complete tests on drive E.
- [x] Rebuild the current no-event readiness report with empty measured summary.
- [x] Keep the full goal active until real target harvest seasons exist.

No Git metadata is present; test and artifact evidence replace commit
checkpoints.
