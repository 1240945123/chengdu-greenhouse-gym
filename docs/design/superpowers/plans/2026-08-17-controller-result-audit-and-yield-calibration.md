# Controller Result Audit and Yield Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an evidence-labelled audit of the completed controller experiment and a reproducible WUR-informed calibration of the Chengdu harvest-process prior.

**Architecture:** A pure process-prior estimator reads immutable WUR climate, production, and quality files and emits a bootstrap calibration artifact. A separate audit module consumes existing benchmark and harvest artifacts, applies horizon- and evidence-aware gates, and writes machine-readable and paper-readable reports before the calibrated six-season simulator is rerun.

**Tech Stack:** Python 3.12, pandas, NumPy, PyYAML, pytest, JSON, Markdown.

---

### Task 1: WUR process-prior estimator

**Files:**
- Create: `tests/test_wur_process_priors.py`
- Create: `experiments/crop/estimate_wur_process_priors.py`

- [ ] Write a failing test with two miniature compartments containing climate,
  production, and quality rows. Assert Excel-date conversion, thermal-time
  integration, dry-matter normalization, compartment-block summaries, and
  deterministic bootstrap intervals.
- [ ] Run `python -m pytest tests/test_wur_process_priors.py -q` and verify the
  failure is caused by the missing estimator module.
- [ ] Implement validated CSV loading, event-level degree-day integration,
  compartment summaries, and deterministic compartment bootstrap.
- [ ] Run the focused test and require all assertions to pass.

### Task 2: Evidence-aware result audit

**Files:**
- Create: `tests/test_result_reasonableness_audit.py`
- Create: `experiments/reports/audit_controller_and_yield_results.py`

- [ ] Write failing tests that classify a 4-day single-scenario benchmark as
  climate-comparison complete but yield-non-evaluable, flag negative legacy net
  production wording, and preserve `target_validated=false` when target harvest
  events are missing.
- [ ] Run the focused test and verify it fails because the audit module is absent.
- [ ] Implement metric summaries, physical checks, scenario/horizon gates,
  evidence labels, severity-ranked findings, and JSON/Markdown writers.
- [ ] Run focused tests and require all assertions to pass.

### Task 3: Generate WUR calibration artifact

**Files:**
- Create: `data/processed/external/wur_agc2/wur_process_priors.json`

- [ ] Run `python -m experiments.crop.estimate_wur_process_priors --root data/external/crops/wur_agc2/extracted --observations data/processed/external/wur_agc2/external_harvest_observations.csv --output data/processed/external/wur_agc2/wur_process_priors.json --base-temperature-c 10 --bootstrap-samples 10000 --seed 20260817`.
- [ ] Verify at least five compartments and 50 maturity observations, finite
  intervals, dry-matter fractions in `(0, 1)`, and recorded DOI/provenance.

### Task 4: Register transfer-calibrated simulation bounds

**Files:**
- Modify: `configs/crops/chengdu_greenlight_harvest_prior_simulation.yml`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`
- Modify: `tests/test_chengdu_greenlight_harvest_prior_simulation.py`

- [ ] Add a failing config test requiring a `wur_process_prior` source and an
  explicit `external_transfer_calibrated_not_target_validated` status.
- [ ] Update maturity bounds from the process-prior 95% interval, retain the
  Chengdu Xindu dry-matter interval, and register the WUR picking cadence.
- [ ] Run the config and simulation tests and require them to pass.

### Task 5: Rerun six-season mechanistic prior

**Files:**
- Regenerate: `results/chengdu_agri_greenhouse_001/harvest_model/simulated_prior/greenlight_cohort/*`

- [ ] Run the existing GreenLight/cohort simulation command from its CLI help
  with the registered configuration and fixed seed.
- [ ] Verify six seasons, at least 72 paired scenarios, nonnegative harvest,
  finite states, and maximum mass-balance error no greater than `1e-9 kg m-2`.
- [ ] Regenerate the plausibility report and retain `target_validated=false`.

### Task 6: Produce final audit package

**Files:**
- Create: `results/chengdu_agri_greenhouse_001/result_audit/result_reasonableness.json`
- Create: `results/chengdu_agri_greenhouse_001/result_audit/result_reasonableness.md`
- Create: `results/chengdu_agri_greenhouse_001/result_audit/calibration_summary.csv`

- [ ] Run the audit against the completed paper benchmark, readiness report,
  target standing-crop report, WUR process prior, and regenerated six-season
  plausibility report.
- [ ] Run all focused crop/controller tests and then the full test suite.
- [ ] Report controller rankings separately from crop plausibility and list the
  exact target harvest evidence still required for a defensible Chengdu yield
  accuracy claim.

