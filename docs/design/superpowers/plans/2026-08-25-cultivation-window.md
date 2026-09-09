# Chengdu Cultivation Window Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a crop-valid Chengdu trajectory ending before July 12, 2026.

**Architecture:** Derive an immutable V5 bundle from V4, filter by target timestamp, chronologically re-split, and train new model artifacts under separate paths. Mark post-cultivation experiments invalid rather than deleting them.

**Tech Stack:** Python, pandas, pytest, existing V4 trajectory and grey-box evaluation pipeline.

---

### Task 1: V5 cultivation-window bundle

**Files:**
- Create: `processing/build_chengdu_trajectory_v5.py`
- Create: `tests/test_chengdu_trajectory_v5.py`

- [ ] Test target-time cutoff, value preservation, row accounting, and chronological splits.
- [ ] Implement V4 loading, strict cutoff filtering, split writing, hashing, and manifest generation.
- [ ] Generate the real V5 dataset.

### Task 2: Cultivation model configuration

**Files:**
- Create: `configs/models/chengdu_cultivation_greybox_residual.yml`
- Create: `results/chengdu_agri_greenhouse_001/cultivation_window_greybox/*`

- [ ] Point train and validation only to V5.
- [ ] Select the model without a test path.
- [ ] Freeze validation decision and evaluate V5 test afterward.

### Task 3: Invalidate post-cultivation conclusions

**Files:**
- Modify: `results/chengdu_agri_greenhouse_001/physics_v4_summer_online_adaptation/adaptation_decision.md`
- Create: `results/chengdu_agri_greenhouse_001/cultivation_window_greybox/cultivation_scope.md`

- [ ] Mark July 12 onward as empty-greenhouse data.
- [ ] Remove those metrics from crop-environment claims while retaining provenance.
- [ ] Report only V5 validation and test metrics.

### Task 4: Verification

**Files:**
- Verify all modified and generated files.

- [ ] Compare retained V4/V5 core values exactly.
- [ ] Run focused trajectory and model tests.
- [ ] Run the complete test suite and report exact results.
