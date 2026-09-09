# Harvest Driver Ensemble Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate traceable GreenLight driver scenarios into held-out batch and cumulative tomato-harvest prediction intervals.

**Architecture:** Validate scenario ensembles against the point test driver, form a parameter-by-driver joint forecast ensemble, and retain season-aware residual blocks. Production input and uncertainty metadata remain explicit and optional.

**Tech Stack:** Python, pandas, NumPy, pytest, CSV, YAML/JSON audits.

---

### Task 1: Scenario validation

**Files:**
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_harvest_calibration.py`

- [x] Add failing tests for two valid aligned scenarios and for duplicate,
  missing, identity-altered, pick-altered, and unknown-source rows.
- [x] Implement a focused validator that returns canonical scenario rows and
  an audit containing IDs, counts, sources, and uniform sampling policy.
- [x] Run focused validation tests: 6 passed.

### Task 2: Joint driver/parameter/residual intervals

**Files:**
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_harvest_calibration.py`

- [x] Add a failing deterministic test where low/high driver growth produces a
  wider cumulative interval than a fixed driver.
- [x] Extend `joint_harvest_prediction_intervals` to build Cartesian
  driver-scenario and parameter-draw forecasts while preserving season resets.
- [x] Verify reproducibility, quantile order, and unchanged point-only output.

### Task 3: Production gate and artifacts

**Files:**
- Modify: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Modify: `tests/test_chengdu_harvest_pipeline.py`

- [x] Add `uncertainty_driver_scenarios` to formal calibration and an optional
  `--uncertainty-driver-scenarios` production CLI path.
- [x] Validate scenarios before bootstrap fitting, filter to the held-out test
  season, save the canonical CSV, and include the scenario audit in results.
- [x] Make uncertainty source accounting dynamic and require disjoint complete
  included/excluded source sets for target validation.

### Task 4: Configuration, docs, and verification

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`

- [x] Document scenario schema, source semantics, command usage, and why the
  current target remains fixed-driver conditional.
- [x] Run focused harvest uncertainty and production tests on drive E: 86 passed.
- [x] Run the full suite and record the exact count: 256 passed, 4 subtests passed.

No Git metadata is present; test output and generated audits replace commit
checkpoints.
