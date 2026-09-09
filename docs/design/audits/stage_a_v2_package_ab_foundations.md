# Stage A V2 Package A/B Foundations Go/No-Go Report

Date: 2026-07-30

## Decision

**GO for the reviewed weather substrate, disturbance schema, and numerical-
failure foundation.** This decision covers exact V2 weather grids,
`DisturbanceSchemaV2`, conservative precipitation handling, and structured
solver-failure termination. It does not authorize paper-profile controller
ranking because the common safety layer, forecast isolation, season-manifest
migration, and real Chengdu climate gate remain incomplete.

## Added Contracts

- `DisturbanceSchemaV2` has canonical immutable 10- and 11-column forms. The
  compatibility form preserves the existing GreenLight inputs. Column 10 in the
  extended form is precipitation rate in `mm/h`.
- The source column `precipitation accumulation` is interpreted as millimetres
  accumulated over each source interval. V2 converts it to an interval rate and
  conservatively allocates it to solver intervals, preserving total millimetres
  without interpolation into adjacent dry intervals.
- Eleven-column loading fails when precipitation is missing, non-finite, or
  negative. Cross-year source files must use one common regular cadence.
- Numerical or integration failure returns `truncated=true`,
  `terminated=false`, an in-space failure observation, and structured failure
  metadata. Normal reward computation is skipped and the last valid model state
  is restored.
- The failure reward is strictly below the worst valid remaining undiscounted
  return under frozen finite per-step bounds. Reward outputs are required to be
  finite and inside those bounds; configurations that cannot produce a finite,
  representably lower failure return are rejected.
- Exception handling is limited to the integrator call. Crop-model and other
  programming errors outside that boundary propagate instead of being
  mislabeled as numerical failures.

## TDD And Review Evidence

Initial tests reproduced missing precipitation output, missing source-policy
errors, mutable/custom schema construction, cross-year cadence corruption,
normal reward execution after failure, swallowed programming errors, non-finite
solver states, reward-bound exploitability, floating-point overflow/equality,
and out-of-space failure observations.

Final focused result:

```text
tests/test_weather_substrate_v2.py
tests/test_solver_failure_semantics_v2.py
51 passed in 6.96s
```

Final repository verification:

```text
.venv/Scripts/python.exe -m pytest -q
354 passed, 4 subtests passed in 90.42s
```

Reviewed source manifest:
`outputs/audit/stage_a_v2/source_manifest_post_package_ab_reviewed.json`
(`SHA-256 7006288595F243084D477181DBFFD7F35D9BB7945AA785816BE34918DD673BA9`).

An independent read-only review found three P1 and three P2 issues. All six
were reproduced with targeted tests and fixed. The re-review reported no
remaining concrete findings in scope.

## Data Migration Status

The current Chengdu `weather_era5/full_years` CSV files remain in the 10-column
source profile and do not yet contain `precipitation accumulation`. Existing
benchmarks therefore remain explicitly compatible and are not silently treated
as rain-aware. Before enabling `nd: 11`, the ERA5 conversion pipeline must emit
per-source-interval precipitation accumulation and migrate the environment to
the V2 repository schema.

## Residual Risks

- A programming defect occurring inside the integrator callable may be exposed
  by that backend as `RuntimeError`, which is indistinguishable from a numerical
  integration failure at this API boundary.
- Failure sentinels are implemented and tested for `gymnasium.spaces.Box`, the
  space type used by current observation modules. Other space types require an
  explicit sentinel policy before use.
- Action hold/transition semantics, rain/wind roof restrictions, wet-pad
  interlocks, fallback policy, and vectorized propagation remain part of the
  broader Package B safety work.

## Remaining Gates

- Emit and audit rain-aware Chengdu full-year weather files before selecting the
  11-column schema.
- Add `ActionSchemaV2` and the common safety projection below every controller.
- Add issue-time forecast providers and leakage tests for MPC and learned agents.
- Migrate benchmark roles to immutable complete-season manifests.
- Pass the real Chengdu climate coverage and calibration gate before any paper-
  profile ranking.
