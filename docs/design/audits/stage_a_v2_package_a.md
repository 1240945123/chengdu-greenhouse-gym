# Stage A V2 Package A Go/No-Go Report

> Superseded for current status by
> `docs/audits/stage_a_v2_package_ab_foundations.md`. This file preserves the
> earlier Package A checkpoint and its original evidence.

Date: 2026-07-30

## Decision

**GO for the reviewed weather-substrate portion of Package A.** The exact weather grid,
schema-aware radiation resampling, cache identity, cache isolation, cross-year
loading, and versioned weather scenario manifest pass their declared tests and
the unchanged repository regression suite.

This decision does not authorize paper-profile controller runs. Solver-failure
semantics, benchmark migration, safety interfaces, forecast isolation, and the
180-day controller protocol remain gated by later work packages.

## Implemented

- `load_weather_data_v2` creates an exact half-open solver grid with
  `ceil((season_days + forecast_days) * 86400 / timestep_seconds)` rows.
- Interval-mean radiation is resampled by conservative integration. Temperature,
  RH, wind, and sky temperature use state interpolation with physical bounds.
- Cross-year and leap-year sources are loaded without replacing the requested
  solver timestep with source cadence. Incomplete starts, interior gaps, and
  incomplete calendar years required for cross-year loading fail explicitly.
- The original `load_weather_data` name retains the previous interpolation
  behavior; `load_weather_data_legacy` is an alias. V2 is opt-in and therefore
  cannot silently reinterpret historical configurations.
- `WeatherScenarioV2` records dimensions, timing, loader identity, all required
  source paths/checksums, expected row count, and a stable SHA-256 fingerprint.
- `WeatherRepository` caches by the V2 fingerprint and returns independent
  arrays so one environment cannot mutate another environment's weather.
- `WeatherRepository.from_config` defaults old configurations to legacy and
  requires `schema_version: v2` for migration. V2 results must match the
  manifest shape and contain only finite values.
- Negative radiation sensor noise is zeroed before conservative integration,
  and soil-temperature phase resets on each actual calendar year.

## TDD Evidence

Initial red tests reproduced the defects:

- 180.5-day cross-year hourly source at 15-minute solver cadence returned 17,332
  rows instead of 17,328.
- One day returned 100 rows instead of 96.
- Different season lengths reused one cache entry.
- Mutating one returned array changed the cached array.
- Editing a source CSV did not invalidate the cache.
- `WeatherRepository.describe()` did not exist.
- Review-driven red tests also reproduced rounded-endpoint checksum omission,
  same-size/same-timestamp source replacement, partial-year compression,
  negative-radiation energy bias, unvalidated loader output, and leap-year soil
  phase shift.

Final focused result: `32 passed` in `tests/test_weather_substrate_v2.py`.

Final repository verification:

```text
python -m compileall -q glassgym tests
python -m pytest -q
335 passed, 4 subtests passed in 84.39s
```

## Provenance

- Pre-change manifest: `outputs/audit/stage_a_v2/source_manifest_pre_package_a.json`
- Pre-change manifest SHA-256:
  `3FB61FB19F69738B8159B6BF5987E63F83C53E8E95304DC4A066C01D9BD5D7E2`
- Final reviewed manifest:
  `outputs/audit/stage_a_v2/source_manifest_post_package_a_reviewed.json`
- Post-change manifest SHA-256:
  `9D7BE5B4B01DEF620B3375A3AA7CEDDF5AF0F8F6B309A80FDD5AFE60AF731DC7`
- Changed source/test files before this report: 5; removed files: 0.
- Runtime: Python 3.12.3 in the repository-local E: `.venv`.

## Independent Review

A separate read-only code review identified two high-, five medium-, and one
low-severity issue. The implementation now addresses all concrete weather-core
findings: rounded target edges determine source-year checksums; legacy naming is
preserved; radiation is sanitized before integration; checksums are recomputed
from content; V2 shape/finite contracts are enforced; source coverage is strict;
and leap-year soil phase is calendar-aware. The review correctly noted that
`DisturbanceSchemaV2` and precipitation/action semantics are still outside this
partial GO decision.

A narrower read-only re-review confirmed those corrections and found two more
medium issues: arbitrary positive timesteps could break legacy day/night
smoothing, and partial-day requests labelled partial radiation as full-day DLI.
V2 now has a timestep-independent day/night derivation and computes DLI from
complete source calendar days. Both were reproduced by red tests before repair;
the final full-suite result above was recorded after these changes.

## Remaining Gates

- Add V2 solver-failure termination and bounded, non-exploitable failure return.
- Add `DisturbanceSchemaV2`, including precipitation semantics and conservative
  accumulated-rain handling; action hold/transition semantics belong to the
  action/safety package.
- Migrate the benchmark protocol from hard-coded 120-day roles to versioned
  season manifests without reinterpreting legacy outputs.
- Introduce the common safety projection and forecast provider; remove future-
  weather and latent-plant access from controller evaluation.
- Validate real Chengdu climate coverage before any paper-profile ranking.
