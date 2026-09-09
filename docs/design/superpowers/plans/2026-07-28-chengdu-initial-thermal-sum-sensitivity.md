# Chengdu Initial Thermal Sum Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a leakage-free phenology prior for unknown baseline canopy thermal sum and propagate it to the strict held-out standing-fruit result.

**Architecture:** A focused helper in `common/tomato_phenology.py` builds low, median, and high initial thermal-sum scenarios from baseline flower-truss replicates and an explicitly bounded early-temperature window. The existing target crop runner calibrates `rgFruit` only on 2026-04-27 for each scenario and evaluates 2026-05-19 afterward, writing separate diagnostic artifacts.

**Tech Stack:** Python, pandas, NumPy, pytest, CasADi/GreenLight crop replay.

---

### Task 1: Thermal-sum scenario helper

**Files:**
- Modify: `common/tomato_phenology.py`
- Modify: `tests/test_tomato_phenology.py`

- [x] Add a failing test that supplies three baseline flower-truss replicates and seven days of hourly temperature, then asserts ordered `low`, `median`, and `high` positive thermal sums and evidence metadata that excludes held-out mass.
- [x] Run `python -m pytest tests/test_tomato_phenology.py -q` and verify the missing helper causes failure.
- [x] Implement `derive_initial_thermal_sum_scenarios` using `thermal_sum = flower_trusses * T / max(0, -0.2903 + 0.1454*ln(T))`, temperature quantiles 0.10, 0.50, and 0.90, and strict finite/range validation.
- [x] Re-run the focused tests and verify they pass.

### Task 2: Scenario propagation

**Files:**
- Modify: `experiments/crop/run_chengdu_target_crop_validation.py`
- Modify: `tests/test_chengdu_target_crop_validation.py`

- [x] Add a failing unit test for a scenario-result summarizer that verifies calibration and held-out errors are kept separate and that envelope coverage uses only completed predictions.
- [x] Run the focused test and verify failure due to the missing summarizer.
- [x] Implement the summarizer and integrate three scenarios into `run_target_crop_validation`; copy the initial state per scenario, fit `rgFruit` on the calibration date only, and evaluate the validation date only after fitting.
- [x] Write `initial_thermal_sum_sensitivity.csv` and `initial_thermal_sum_sensitivity.json`; add the summary to `target_crop_validation_report.json` with `uncertainty_type=structural_sensitivity_not_predictive_interval` and `adoption_status=diagnostic_not_final_target_parameterization`.
- [x] Re-run target validation tests and verify they pass.

### Task 3: Formal artifacts and regression

**Files:**
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `data/README.md`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation/*`
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/readiness_report.json`

- [x] Run the formal target crop validation command and inspect scenario ordering, fit errors, holdout errors, and envelope coverage.
- [x] Update configuration and data documentation with measured values and the non-predictive interpretation.
- [x] Rebuild harvest readiness and verify `target_validated=false` and zero target harvest events.
- [x] Run `python -m pytest -q` and require zero failures.

The workspace copy has no `.git` directory, so commit steps are intentionally replaced by artifact and full-suite verification.
