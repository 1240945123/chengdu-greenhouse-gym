# Stage A V2 Package B Safety Go/No-Go Report

Date: 2026-07-30

## Decision

**GO for rain-aware weather ingestion and the reviewed primary-controller
safety layer.** The Chengdu benchmark now uses 11-dimensional V2 weather,
exposes current precipitation to every controller, and applies one common
execution projection below Baseline, PID, MPC, PPO, and SAC.

This is not a GO for paper-profile controller ranking. The wind threshold is a
versioned engineering default rather than a target-equipment certification;
forecast isolation, complete-season protocol migration, and the real Chengdu
climate gate remain incomplete.

## Weather Migration

- Local archived ERA5 JSON is the source; no replacement weather was downloaded.
- Open-Meteo ERA5 `precipitation` is verified as hourly interval accumulation in
  `mm` and is written as `precipitation accumulation` in each GreenLight source
  CSV. The V2 loader converts it conservatively to `mm/h`.
- Annual accumulated precipitation is 915.3 mm for 2023, 1127.0 mm for 2024,
  and 1363.6 mm for 2025.
- The benchmark config now explicitly selects `schema_version: v2`,
  `load_weather_data_v2`, and `nd: 11`; old 10-dimensional configurations remain
  legacy and are not reinterpreted.
- `WeatherObservationsV2` exposes radiation, outdoor temperature, RH, CO2, wind,
  and precipitation rate to all benchmark controllers.

Artifact SHA-256 values:

- 2023 CSV: `F52C3236E49D669922E64D26086E00507A0E1CF176B360D70DC656E2B2123F24`
- 2024 CSV: `7D5128A570530661D2F7D6CC43A147AC5A5B65A743434E2BB66C7D187CDA4F3B`
- 2025 CSV: `AC3D38ADAFD1057ADCF0450E7B71FCFAD3D664EB44ABD7659539BF2134C33DEC`
- ERA5 manifest: `5C9BF91CD19A280B698547DD6AA2FBCDAAD77B2773168EA1A0B64707A52238D1`

## Safety Contract

- `ActionSchemaV2` defines canonical immutable six-control primary and reserved
  eight-control wet-pad forms. Heating and CO2 are disabled in the primary
  profile.
- Invalid or non-finite proposals fall back to the previous executed action and
  record the invalid indices. Logged proposals remain finite so recovered
  episodes can still be summarized.
- Equipment bounds, rain and wind roof closure, heating/CO2 disablement, lamp
  schedule, normal slew limits, emergency vent slew, optional high-RH wet-pad
  lockout, and fan-pump interlock are deterministic and tested.
- Each transition records proposed and executed controls, intervention reasons,
  fallback use, fallback duration, and the pre-reset indoor climate.
- Episode metrics include intervention fraction/count, rain and wind closure
  steps, fallback steps/duration, and proposed-to-executed L1 projection.

The primary `ChengduPhysics` backend no longer couples `uVent` to an uncalibrated
18 kW wet-pad term. `ChengduPhysicsLegacy` preserves that behavior only for
explicit historical reproduction. The eight-control wet-pad interface remains
disabled in the primary benchmark until its cooling, humidification, and energy
dynamics are calibrated.

## Verification

Focused Package B regression:

```text
88 passed in 44.44s
```

Final repository regression after review fixes:

```text
.venv/Scripts/python.exe -m pytest -q
363 passed, 4 subtests passed in 112.11s
```

`compileall` passed for `glassgym`, `processing`, `experiments`, and `tests`.
ERA5 provenance validation passed against the updated manifest.

Reviewed source manifest:
`outputs/audit/stage_a_v2/source_manifest_post_package_b_reviewed.json`
(`SHA-256 5ADF8275E763607BDC15CFCE372B941510EC39696FAE33A5827172AB259821D2`).

An independent read-only review found two P1 and two P2 issues: non-finite
fallback logging, incomplete reserved wet-pad wiring, terminal RL climate
logging after vectorized auto-reset, and inaccurate intervention reasons. All
four were reproduced and repaired. Re-review reported no remaining concrete
findings and independently ran 30 relevant tests successfully.

## Remaining Gates

- Replace the 10 m/s wind closure default with limits certified for the target
  roof-window hardware and operating procedure.
- Add timeout handling and rule-baseline fallback around controller inference,
  including vectorized propagation tests.
- Implement an issue-time forecast provider and remove future-truth and latent-
  plant leakage from MPC and learned-controller evaluation.
- Migrate the 120-day benchmark roles to immutable complete-season manifests.
- Calibrate wet-pad physics before enabling `uPadFan` or `uPadPump`.
- Pass the real Chengdu climate coverage/calibration gate before controller
  rankings are presented as paper results.
