# V5 Hybrid Greenhouse Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an optional V5 climate-corrected GreenLight environment for control preflight.

**Architecture:** GreenLightEnv gains a narrow optional state-postprocessing hook after ODE integration. A separate V5 component translates the eight-control environment state into frozen Ridge features, applies a timestep-scaled bounded correction, and a builder assembles the provenance-locked environment.

**Tech Stack:** Python, Gymnasium, CasADi, pandas, NumPy, pytest

---

### Task 1: State postprocessing contract

**Files:**
- Modify: `glassgym/environments/greenlight_env.py`
- Create: `glassgym/components/climate_postprocessor.py`
- Test: `tests/test_v5_hybrid_environment.py`

- [ ] Write a failing test showing that a postprocessor changes climate before observations and reward while the default path remains unchanged.
- [ ] Run the focused test and verify the missing constructor argument or hook failure.
- [ ] Implement the protocol, shape/finite validation, correction call, and transition metadata.
- [ ] Run the focused test and verify it passes.

### Task 2: Frozen V5 residual postprocessor

**Files:**
- Modify: `glassgym/components/climate_postprocessor.py`
- Modify: `tests/test_v5_hybrid_environment.py`

- [ ] Write failing tests for exact feature mapping, timestep gain scaling, and vapor-pressure consistency.
- [ ] Implement V5 row construction and Ridge application using the existing validated residual feature builder.
- [ ] Run focused tests and verify all pass.

### Task 3: Eight-control environment builder

**Files:**
- Create: `experiments/controllers/v5_hybrid_environment.py`
- Modify: `tests/test_v5_hybrid_environment.py`

- [ ] Write a failing integration test for 28 states, 8 controls, two-dimensional action space, 225 parameters, and finite reset/step outputs.
- [ ] Implement artifact validation, parameter loading, weather binding, and environment construction.
- [ ] Verify heating, CO2, lamp, and wet-pad pump remain zero during a step.

### Task 4: One-day policy preflight

**Files:**
- Create: `experiments/controllers/run_v5_hybrid_preflight.py`
- Test: `tests/test_v5_hybrid_preflight.py`

- [ ] Write a failing test for deterministic policy scenarios and required metric fields.
- [ ] Implement low, mid, and high roof/fan policy rollouts and provenance artifacts.
- [ ] Run the real one-day preflight and inspect physical/crop-state gates.

### Task 5: Verification

**Files:**
- Modify only files above if verification exposes a defect.

- [ ] Run all V5 hybrid, residual, safety, reward, and benchmark integration tests.
- [ ] Run the complete pytest suite and require zero failures.
- [ ] Record the promotion decision and remaining limitations.
