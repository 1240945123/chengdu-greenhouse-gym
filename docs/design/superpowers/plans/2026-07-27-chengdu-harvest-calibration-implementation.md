# Chengdu Harvest Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a target-greenhouse tomato harvest simulation pipeline that predicts harvest timing, batch fresh mass, cumulative yield, and uncertainty from Chengdu observations.

**Architecture:** Keep GreenLight/Vanthoor as the carbon-production model, standardize crop observations and true harvest events into separate canonical tables, and add a maturity-cohort observation layer that converts simulated fruit growth into harvest batches. Calibrate with time-blocked or season-held-out validation and label site transfer explicitly when only Xindu crop observations are available for a Pidu target.

**Tech Stack:** Python, NumPy, pandas, SciPy, PyYAML, pytest.

---

### Task 1: Canonical crop and harvest data

**Files:**
- Create: `processing/chengdu_crop_observations.py`
- Create: `data/schemas/crop_observation_schema.yml`
- Create: `data/schemas/harvest_event_schema.yml`
- Modify: `configs/datasets/chengdu_agri_greenhouse_001.yml`
- Test: `tests/test_chengdu_crop_observations.py`

- [ ] Write failing tests that extract `planting_info` and `plant_message_info`, convert plant-scale grams to area-scale kg/m2 using planting density, distinguish standing ripe-fruit observations from actual harvest events, and reject invalid dates or masses.
- [ ] Run `pytest tests/test_chengdu_crop_observations.py -v` and confirm failure because the extraction API is absent.
- [ ] Implement a streaming, table-whitelisted SQL extractor that does not expose unrelated or sensitive backup content.
- [ ] Add canonical schema/config metadata including greenhouse code, planting code, cultivar, observation scope, source site, and target-site eligibility.
- [ ] Run the focused tests and extract the available Chengdu observations into the processed crop directory.

### Task 2: Maturity-cohort harvest model

**Files:**
- Create: `glassgym/models/harvest/cohort_model.py`
- Create: `glassgym/models/harvest/__init__.py`
- Test: `tests/test_harvest_cohort_model.py`

- [ ] Write failing tests for mass conservation, temperature-driven maturity, explicit picking schedules, non-negative batch mass, and timestep invariance.
- [ ] Run the focused tests and confirm the model API is absent.
- [ ] Implement fruit dry-matter cohorts driven by GreenLight allocation, thermal age, local dry/fresh fraction, and configurable harvest weekdays/thresholds.
- [ ] Run the focused tests and verify cumulative harvested plus standing fruit equals cumulative allocated fruit within tolerance.

### Task 3: Calibration and validation metrics

**Files:**
- Create: `experiments/crop/calibrate_harvest_model.py`
- Create: `experiments/crop/__init__.py`
- Create: `common/harvest_evaluation.py`
- Test: `tests/test_harvest_calibration.py`

- [ ] Write failing tests for first-harvest-date error, batch/cumulative MAE and WMAPE, bias, R2, block-bootstrap prediction intervals, and train/validation/test separation.
- [ ] Implement bounded SciPy calibration with explicit parameter provenance and no use of test observations in fitting.
- [ ] Add uncertainty from parameter/bootstrap replicates and report 80% and 95% interval coverage.
- [ ] Run focused tests with a synthetic identifiable season and recover known parameters within tolerance.

### Task 4: End-to-end Chengdu experiment

**Files:**
- Create: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `configs/crops/vanthoor_tomato.yml`
- Modify: `data/README.md`
- Create: `experiments/crop/run_chengdu_harvest_pipeline.py`
- Test: `tests/test_chengdu_harvest_pipeline.py`

- [ ] Add a configuration that separates literature defaults, Xindu pre-calibration, and Pidu target calibration.
- [ ] Run the pipeline on existing Xindu observations and write canonical data, fit artifacts, predictions, metrics, and data-sufficiency status under `results/chengdu_agri_greenhouse_001/harvest_model/`.
- [ ] Verify the run refuses to label results target-validated when Pidu harvest events are absent.
- [ ] Run the relevant regression suite and document the exact target harvest fields required to unlock final validation.
