# V5 Actuator Identifiability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a provenance-locked audit that determines which V5-observed actuators can safely enter controller optimization.

**Architecture:** A focused report module reads the V5 training split, computes empirical support, and probes the frozen ChengduPhysicsV4 plus Ridge residual predictor with local action perturbations. It writes tabular, JSON, and Markdown artifacts without modifying model or trajectory inputs.

**Tech Stack:** Python, pandas, NumPy, existing Chengdu physics/residual evaluator, pytest

---

### Task 1: Empirical actuator support

**Files:**
- Create: `experiments/reports/audit_chengdu_v5_actuators.py`
- Test: `tests/test_chengdu_v5_actuator_audit.py`

- [ ] Write a failing test with well-supported, sparse, and constant synthetic controls.
- [ ] Run `python -m pytest tests/test_chengdu_v5_actuator_audit.py -q` and verify the import fails.
- [ ] Implement deterministic support statistics and threshold classification.
- [ ] Re-run the focused test and verify it passes.

### Task 2: Cultivation boundary and counterfactual probe

**Files:**
- Modify: `experiments/reports/audit_chengdu_v5_actuators.py`
- Modify: `tests/test_chengdu_v5_actuator_audit.py`

- [ ] Add a failing test that rejects any target timestamp at or after the cultivation cutoff.
- [ ] Add a failing test using an injected predictor to verify low/high perturbations are isolated by actuator and row.
- [ ] Implement strict cutoff validation and local response summaries.
- [ ] Run the focused tests and verify they pass.

### Task 3: Reproducible artifacts and real audit

**Files:**
- Modify: `experiments/reports/audit_chengdu_v5_actuators.py`
- Create: `results/chengdu_agri_greenhouse_001/physics_v5_cultivation_greybox_residual/actuator_audit/*`

- [ ] Implement CLI arguments, hashes, CSV/JSON/Markdown writing, and recommended action-space output.
- [ ] Run the audit against V5 train and the frozen selected model.
- [ ] Inspect support counts and directional responses for scientific plausibility.

### Task 4: Verification

**Files:**
- Modify only if a test exposes a defect in files changed above.

- [ ] Run `python -m pytest tests/test_chengdu_v5_actuator_audit.py tests/test_chengdu_trajectory_v5.py tests/test_chengdu_greybox_residual.py -q`.
- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Record limitations and the promoted actuator set in the final report.
