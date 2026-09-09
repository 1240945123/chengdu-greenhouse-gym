# Three-Month Weather And Training Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the full Chengdu weather file and add reproducible block-bootstrap weather augmentation for controller training.

**Architecture:** Keep conversion of audited real weather in `chengdu_weather.py`, isolate stochastic generation and provenance in `chengdu_weather_augmentation.py`, and let benchmark configuration distinguish the real evaluation year from training-only synthetic years. Existing environment loading remains unchanged because every generated scenario uses the GreenLight CSV schema.

**Tech Stack:** Python, pandas, NumPy, PyYAML, pytest, Stable-Baselines3

---

### Task 1: Full Real Weather Conversion

**Files:**
- Modify: `processing/chengdu_weather.py`
- Test: `tests/data_pipeline.py`

- [ ] Add a failing test that passes both indoor and outdoor columns and requires outdoor temperature/RH, 400 ppm CO2, hourly continuity, and no missing output values.
- [ ] Run `python -m pytest tests/data_pipeline.py -q` and confirm the new outdoor-mapping assertion fails.
- [ ] Add `convert_aligned_weather_dataframe(df)` with explicit outdoor mappings and physical validation.
- [ ] Run the focused test and confirm it passes.

### Task 2: Moving-Block Weather Generator

**Files:**
- Create: `processing/chengdu_weather_augmentation.py`
- Create: `tests/test_chengdu_weather_augmentation.py`

- [ ] Add failing tests for deterministic output, different-seed diversity, exact 110-day length, hourly time, bounds, and source-block provenance.
- [ ] Run the new test file and confirm import failure.
- [ ] Implement daily-block extraction, seeded 3-day block sampling, bounded correlated perturbations, output writing, and manifest hashing.
- [ ] Run the new tests and confirm they pass.

### Task 3: Benchmark Training Weather Separation

**Files:**
- Modify: `configs/benchmarks/chengdu_controllers.yml`
- Modify: `experiments/controllers/benchmark_protocol.py`
- Modify: `experiments/controllers/train_benchmark_rl.py`
- Test: `tests/test_controller_benchmark_protocol.py`

- [ ] Add failing tests requiring 110 real days, non-overlapping train/validation/test windows, and separate training weather years.
- [ ] Run the protocol tests and confirm the new configuration fields are absent.
- [ ] Add `training_growth_years` to `BenchmarkConfig` and use it only in `_make_training_env`; retain `growth_year=2026` for validation and test.
- [ ] Run protocol and RL smoke tests and confirm they pass.

### Task 4: Generate Artifacts

**Files:**
- Regenerate: `data/processed/chengdu_agri/greenhouse_001/weather/Chengdu/2026.csv`
- Generate: synthetic numeric-year CSV files and `synthetic_weather_manifest.json`
- Modify: `data/README.md`

- [ ] Convert all 2,650 aligned hourly rows into the real weather file.
- [ ] Generate ten 110-day synthetic training years from source days 0 through 69 using seeds 0 through 9.
- [ ] Record file hashes and generation settings in the manifest.
- [ ] Update data documentation with real/synthetic separation and commands.

### Task 5: Verification

**Files:**
- Verify existing tests and smoke training output.

- [ ] Validate all real and synthetic files for row counts, hourly monotonic time, finite values, and physical bounds.
- [ ] Run PPO and SAC smoke training against the augmented training-year list.
- [ ] Run the benchmark-relevant test suite excluding the known legacy `tests/env_test.py` contract tests.
- [ ] Report that existing paper checkpoints are stale after the weather/config fingerprint changes and must be retrained before publication.
