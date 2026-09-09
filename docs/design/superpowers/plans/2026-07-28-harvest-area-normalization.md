# Harvest Area Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize all target harvest batches to a fixed modeled production area while retaining actual-area diagnostics.

**Architecture:** The event loader receives a trusted greenhouse-code-to-area mapping, computes fixed-area model targets and local-area audit fields, and rejects inconsistent coverage. The pipeline derives the mapping from target-site configuration.

**Tech Stack:** Python, pandas, NumPy, pytest, YAML.

---

### Task 1: Loader semantics

**Files:**
- Modify: `processing/chengdu_harvest_events.py`
- Modify: `tests/test_chengdu_harvest_events.py`

- [x] Add a failing partial-area test proving model yield uses fixed area while local yield and coverage use actual area.
- [x] Add failing tests for missing normalization mapping and harvested area above production area.
- [x] Implement the trusted area mapping and output fields, then run focused tests.

### Task 2: Pipeline and documentation

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Pass the configured target greenhouse area to event loading and expose area semantics in readiness audit.
- [x] Document fixed-area versus local-area columns and prohibit sample extrapolation.
- [x] Run complete tests and a partial-area cumulative-yield audit.

No Git metadata is present in this workspace; tests and generated evidence replace commit checkpoints.
