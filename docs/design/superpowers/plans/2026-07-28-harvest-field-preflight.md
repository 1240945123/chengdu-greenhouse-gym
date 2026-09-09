# Harvest Field Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone field-workbook quality and measurement-readiness command.

**Architecture:** Compose existing canonical loaders, season binding, sufficiency assessment, and measured trajectory generation behind one focused CLI. Keep measurement coverage separate from crop-driver and model-validation readiness.

**Tech Stack:** Python, pandas, YAML, pytest, XLSX, JSON, CSV.

---

### Task 1: Preflight report core

**Files:**
- Create: `experiments/crop/validate_chengdu_harvest_workbook.py`
- Create: `tests/test_chengdu_harvest_preflight.py`

- [x] Add a failing two-complete-season XLSX test.
- [x] Implement configuration identity resolution, canonical loading, binding, trajectory generation, and coverage flags.
- [x] Write JSON/event/daily outputs and verify totals reconcile.

### Task 2: Ongoing seasons and traceability

**Files:**
- Modify: `tests/test_chengdu_harvest_preflight.py`
- Modify: `experiments/crop/validate_chengdu_harvest_workbook.py`

- [x] Add failing complete-plus-ongoing and missing-source-ID tests.
- [x] Add traceability warnings without promoting ongoing seasons.
- [x] Assert the report never claims crop-model or final target validation.

### Task 3: Documentation and verification

**Files:**
- Modify: `data/README.md`

- [x] Document the standalone command and report interpretation.
- [x] Run focused and complete tests on drive E.
- [x] Execute the command against a synthetic populated workbook artifact.

No Git metadata is present; tests and generated reports replace commit
checkpoints.
