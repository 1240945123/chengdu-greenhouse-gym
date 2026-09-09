# Three-Year Six-Season Weather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download auditable 2023-2025 ERA5 hourly weather, build six 120-day Chengdu greenhouse seasons, validate it against the 2026 observations, and integrate a leakage-free chronological benchmark split.

**Architecture:** Add one focused ERA5 ingestion module and one season/bias-processing module. Raw API responses and manifests remain separate from GreenLight-ready yearly and seasonal files. Benchmark configuration describes named train, validation, and test seasons and validates provenance before a run.

**Tech Stack:** Python, requests, pandas, NumPy, PyYAML, pytest, Open-Meteo Historical API with ERA5.

---

### Task 1: ERA5 API client and parser

**Files:**
- Create: `processing/chengdu_era5.py`
- Create: `tests/test_chengdu_era5.py`

- [ ] Write failing tests using an in-memory API payload with hourly `time`, `temperature_2m`, `relative_humidity_2m`, `shortwave_radiation`, `wind_speed_10m`, `precipitation`, and soil-temperature arrays. Assert strict equal lengths, unique continuous local timestamps, expected units, calendar-year row counts, and rejection of missing values.
- [ ] Run `python -m pytest tests/test_chengdu_era5.py -q` and confirm import/API failures.
- [ ] Implement `build_archive_request`, `parse_archive_payload`, `download_archive_year`, SHA-256 helpers, bounded retries, timeout handling, atomic writes, and a CLI accepting coordinates, years, raw root, timezone, and model.
- [ ] Run the targeted tests and confirm they pass.

### Task 2: GreenLight conversion, correction, and seasons

**Files:**
- Modify: `processing/chengdu_era5.py`
- Create: `tests/test_chengdu_era5_seasons.py`

- [ ] Write failing tests for calendar-aligned GreenLight conversion, deterministic overlap correction, night radiation remaining zero, bounded RH/non-negative wind, exact spring and autumn boundaries, and exactly 2,880 rows per season.
- [ ] Run the new test file and confirm failures.
- [ ] Implement `to_greenlight_weather`, `fit_local_bias_correction`, `apply_local_bias_correction`, `extract_season`, and `build_six_seasons`. Store correction confidence by month and avoid unsupported month-specific extrapolation.
- [ ] Run both ERA5 test files and confirm they pass.

### Task 3: Provenance and quality reporting

**Files:**
- Modify: `processing/chengdu_era5.py`
- Modify: `data/raw/chengdu_agri/greenhouse_001/metadata.yml`
- Modify: `data/README.md`

- [ ] Add tests that recompute raw/processed hashes, require API request metadata, verify all six season identities, and fail when a season crosses its configured boundary.
- [ ] Add a manifest containing source/model/API, coordinates, request URLs, retrieval UTC, units, hashes, season definitions, correction fit interval, and train/validation/test role.
- [ ] Document exact rebuild and validation commands and label ERA5 as reanalysis rather than observation.
- [ ] Run targeted tests.

### Task 4: Download and generate artifacts

**Files:**
- Create generated files under `data/external/weather/era5/chengdu_greenhouse_001/`
- Create generated files under `data/processed/chengdu_agri/greenhouse_001/weather_era5/`

- [ ] Resolve and record the best available base coordinate, with an explicit fallback label if approximate.
- [ ] Download 2023, 2024, and 2025 with `models=era5`, `timezone=Asia/Shanghai`, `wind_speed_unit=ms`, and the required hourly variables.
- [ ] Validate 8,760/8,784/8,760 hourly rows and raw hashes.
- [ ] Fit the local correction against the audited 2026 overlap, generate full-year GreenLight files and six season files, and write quality/provenance reports.
- [ ] Verify six files each contain 2,880 finite hourly rows and physically valid values.

### Task 5: Chronological benchmark integration

**Files:**
- Modify: `configs/benchmarks/chengdu_controllers.yml`
- Modify: `experiments/controllers/benchmark_protocol.py`
- Modify: `experiments/controllers/train_benchmark_rl.py`
- Modify: `experiments/controllers/run_chengdu_benchmark.py`
- Modify: `tests/test_controller_benchmark_protocol.py`
- Modify: `tests/test_controller_benchmark_integration.py`

- [ ] Write failing tests for four 2023-2024 training seasons, spring-2025 validation, autumn-2025 testing, year/start pairing, and rejection of a synthetic or validation/test source in training augmentation.
- [ ] Extend the protocol from independently sampled years/start days to explicit weather scenarios so invalid cross-products cannot occur.
- [ ] Keep fixed validation and test samplers and include all ERA5 manifests/files in the experiment fingerprint.
- [ ] Run controller protocol/integration tests.

### Task 6: End-to-end verification

**Files:**
- Generated outputs: `results/chengdu_agri_greenhouse_001/controller_benchmark/smoke/`

- [ ] Run `python -m pytest tests --ignore=tests/env_test.py -q`.
- [ ] Archive the current smoke directory without deleting it.
- [ ] Run `python -m experiments.controllers.run_chengdu_benchmark --profile smoke --no-resume`.
- [ ] Verify all five algorithms complete against autumn 2025, heating and CO2 remain zero, manifests match current hashes, and comparison metrics are finite.
- [ ] Report that smoke establishes pipeline validity only; paper-scale multi-seed training remains separate.

## Plan Review

The plan covers every source, season, correction, provenance, split, and verification requirement in the approved design. File responsibilities and function names are consistent, and no unspecified implementation placeholders remain. The repository has no Git metadata, so commit steps are intentionally omitted.
