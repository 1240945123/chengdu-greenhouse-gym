# External Harvest Prior and Chengdu Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add traceable public tomato harvest priors and clearly isolated Chengdu synthetic harvest scenarios without weakening target-validation gates.

**Architecture:** A source adapter downloads and audits the immutable WUR AGC2 archive, then a semantic normalizer emits external observations and priors only from explicit harvest fields. A separate Chengdu simulator consumes existing six-season weather/GreenLight drivers and writes synthetic, target-ineligible events with complete sampled-parameter provenance.

**Tech Stack:** Python, pandas, requests, py7zr, PyYAML, NumPy, JSON, pytest.

---

### Task 1: Reproducible WUR source acquisition

**Files:**
- Create: `processing/wur_agc2.py`
- Create: `tests/test_wur_agc2.py`
- Create: `configs/datasets/wur_agc2_tomato.yml`

- [x] Write tests for parsing 4TU metadata, rejecting a non-CC0 record, and
  rejecting archive size or MD5 mismatches.
- [x] Implement `fetch_article_metadata()`, `validate_article_metadata()`, and
  `download_verified_archive()` with article ID `12764777`, file ID `24757220`,
  expected MD5 `2a0c7f3332881caef54ca8f4dc60c9a3`, atomic writes, and SHA-256 audit.
- [x] Run `python -m pytest tests/test_wur_agc2.py -q` and require all tests to
  pass before downloading data.
- [x] Download to
  `data/external/crops/wur_agc2/raw/AutonomousGreenhouseChallenge_edition2.7z`
  and write `data/external/crops/wur_agc2/source_manifest.json`.

### Task 2: Archive inventory and semantic harvest normalization

**Files:**
- Modify: `processing/wur_agc2.py`
- Modify: `tests/test_wur_agc2.py`
- Create: `data/schemas/external_harvest_observation_schema.yml`

- [x] Add tests using a small in-memory archive fixture for path traversal,
  duplicate paths, encrypted files, unsupported file types, and inventory
  checksums.
- [x] Implement read-only extraction to a staging directory on E:, then create
  `archive_inventory.json` with path, bytes, extension, and SHA-256 per member.
- [x] Inventory CSV members and the PDF data dictionary before assigning
  harvest semantics.
- [x] After inspecting the real inventory, register an explicit WUR version-2
  mapping for batch date, compartment/team, fresh yield, cumulative flag, and
  cultivated-area basis. Unknown schema versions must fail.
- [x] Normalize to `external_harvest_observations.csv` with
  `evidence_class=external_observed`, original identifiers, source row, and no
  Pidu greenhouse identity.
- [x] Produce `normalization_audit.json` and preserve documented event-batch
  semantics without inventing cumulative source values.

### Task 3: External prior estimation

**Files:**
- Create: `experiments/crop/estimate_external_harvest_priors.py`
- Create: `tests/test_external_harvest_priors.py`

- [x] Write failing tests for compartment-block bootstrap, first-harvest timing,
  inter-pick intervals, seasonal cumulative yield, and fixed-seed determinism.
- [x] Estimate empirical distributions by compartment without pooling rows as
  independent observations.
- [x] Write `external_harvest_priors.json` containing quantiles, sample counts,
  source DOI, cultivar, site, greenhouse type, and transfer limitations.
- [x] Reject priors with fewer than three valid compartments or ambiguous mass
  semantics.

### Task 4: Evidence-class enforcement

**Files:**
- Modify: `processing/chengdu_harvest_events.py`
- Modify: `experiments/crop/calibrate_harvest_model.py`
- Modify: `tests/test_chengdu_harvest_events.py`
- Modify: `tests/test_harvest_calibration_pipeline.py`

- [x] Add tests proving `external_observed` and `simulated_prior` rows cannot
  become target eligible even if greenhouse identifiers are edited to match.
- [x] Require `evidence_class=target_observed` for target calibration rows while
  retaining backward compatibility only for audited target workbooks created by
  this repository.
- [x] Add rejected evidence-class counts to target harvest assessments.
- [x] Verify the current readiness report still has zero target events and
  `target_validated=false`.

### Task 5: Chengdu synthetic harvest ensemble

**Files:**
- Create: `experiments/crop/simulate_chengdu_harvest_priors.py`
- Create: `tests/test_chengdu_harvest_prior_simulation.py`
- Create: `configs/crops/chengdu_harvest_prior_simulation.yml`

- [x] Write tests for six weather-season groups, deterministic sampling,
  synthetic identities, target ineligibility, declared pick protocols, and
  dry-matter mass balance.
- [x] Read the six existing 2023-2025 Chengdu ERA5 seasons and accepted external
  prior intervals without copying WUR timestamps or yields.
- [x] Sample weather source, climate-model parameter scenario, initial thermal
  sum, maturity thermal time, dry-matter fraction, and picking schedule using a
  fixed seed and Latin hypercube draws.
- [x] Run GreenLight/harvest cohorts and write separate driver, event, seasonal
  summary, sampled-parameter, and audit artifacts below
  `results/chengdu_agri_greenhouse_001/harvest_model/simulated_prior/`.
- [x] Use identifiers prefixed `SIM_`, `target_eligible=false`, and
  `evidence_class=simulated_prior` on every synthetic event.

### Task 6: Plausibility report and documentation

**Files:**
- Create: `experiments/crop/report_harvest_prior_coverage.py`
- Create: `tests/test_harvest_prior_coverage.py`
- Modify: `data/README.md`
- Modify: `configs/crops/chengdu_tomato_harvest.yml`

- [x] Report first harvest date, inter-pick interval, batch yield, cumulative
  yield, peak harvest date, dry-matter mass balance, and 80/95% interval widths.
- [x] Compare simulations with WUR distributions and Pidu standing-crop states,
  labeling both comparisons as plausibility rather than target harvest accuracy.
- [x] Generate a coverage matrix with statuses `target_observed`,
  `external_prior`, `simulated`, and `missing_target_measurement`.
- [x] Document exact reproduction commands, DOI/API links, licenses, checksums,
  transfer limitations, and remaining field collection requirements.
- [x] Run focused tests, then the full suite (`303 passed, 4 subtests passed`). Confirm no
  synthetic artifact appears in the target raw harvest directory and readiness
  remains false.

Stage note: the fast weather-conditioned statistical ensemble is complete with
120 scenarios and 1,402 events. The separate higher-fidelity ensemble is also
complete with 18 ChengduPhysics/GreenLight trajectories, 72 paired cohort
scenarios, and 654 events. Both remain diagnostic simulated priors; neither is
target harvest calibration or validation evidence.

No Git metadata is present; test output and generated audits replace commit
checkpoints.
