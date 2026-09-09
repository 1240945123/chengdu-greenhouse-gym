# Harvest Acceptance Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make configured harvest acceptance thresholds authoritative end to end.

**Architecture:** Add a strict YAML-to-canonical threshold adapter, pass its output through seasonal evaluation, and recompute readiness gates from the recorded canonical thresholds.

**Tech Stack:** Python, YAML, NumPy, pytest.

---

### Task 1: Strict threshold adapter

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add failing tests for complete mapping, missing/unknown keys, and invalid values.
- [x] Implement directional key mapping and semantic validation.
- [x] Run focused adapter tests.

### Task 2: Experiment and readiness propagation

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add failing tests proving a custom threshold changes experiment acceptance.
- [x] Return canonical thresholds with experiment results and reuse them in readiness.
- [x] Pass the strict YAML mapping from `run_pipeline`.

### Task 3: Documentation and verification

**Files:**
- Modify: `data/README.md`

- [x] Document that YAML thresholds are authoritative and audited.
- [x] Run the complete test suite on drive E.
- [x] Rebuild current target readiness without real harvest events.

No Git metadata is present; tests and generated reports replace commit
checkpoints.
