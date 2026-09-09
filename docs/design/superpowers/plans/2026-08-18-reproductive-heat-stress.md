# Reproductive Heat-Stress Yield Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a v3 120-day yield projection that quantifies literature-constrained reproductive heat stress without target-yield fitting.

**Architecture:** A focused stress-response module calculates trailing-climate retention. The cohort simulator applies rejected allocation as an explicit mass-balance term, and the controller projection expands each preserved v2 parameter draw across three named heat-tolerance scenarios.

**Tech Stack:** Python, pandas, NumPy, SciPy, pytest, YAML/CSV/JSON artifacts

---

### Task 1: Implement the heat-retention response

**Files:**
- Create: `glassgym/models/harvest/reproductive_stress.py`
- Modify: `glassgym/models/harvest/__init__.py`
- Test: `tests/test_reproductive_heat_stress.py`

- [ ] Write failing tests for bounds, continuity, monotonicity, trailing-window behavior, and invalid parameters.
- [ ] Run the focused tests and confirm failure because the module is absent.
- [ ] Implement the immutable parameter object and vectorized trailing-temperature response.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Integrate stress into the cohort mass balance

**Files:**
- Modify: `experiments/crop/simulate_chengdu_greenlight_harvest_priors.py`
- Test: `tests/test_reproductive_heat_stress.py`

- [ ] Add failing tests proving no-stress equivalence and explicit rejected-allocation accounting.
- [ ] Apply retention before cohort allocation and add reproductive loss to driver and summary artifacts.
- [ ] Update the mass-balance equation and verify focused tests pass.

### Task 3: Add v3 controller projection and audit

**Files:**
- Create: `configs/crops/chengdu_reproductive_heat_stress.yml`
- Create: `experiments/crop/project_controller_stress_yield.py`
- Create: `experiments/crop/audit_controller_stress_yield.py`
- Test: `tests/test_controller_stress_yield_projection.py`

- [ ] Add failing tests for three scenario expansion, paired v2 parameters, and aggregate stress metrics.
- [ ] Implement projection and prefix-safe artifact writing.
- [ ] Implement completeness, mass-balance, timing, spring-envelope, and v2 comparison audits.
- [ ] Run focused tests and confirm they pass.

### Task 4: Register evidence and run the complete experiment

**Files:**
- Create: `data/external/crops/tomato_reproductive_stress/source_registry.csv`
- Create: `results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d_guarded_v2/regional_v3_*`

- [ ] Register source DOI, regime, transfer role, and target-ineligible status.
- [ ] Run all 78 episodes with 16 paired cohort draws and three stress scenarios.
- [ ] Verify expected rows, paired scenario coverage, and dry-matter balance.
- [ ] Generate v2/v3 comparison tables and interpret algorithm ranking changes.

### Task 5: Full verification

**Files:**
- Modify: `data/README.md`

- [ ] Document v3 inputs, outputs, and limitations.
- [ ] Run the full pytest suite.
- [ ] Run artifact-level consistency checks and record the final counts.

