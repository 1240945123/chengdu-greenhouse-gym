# Stage A V2 Package J: V4 Grey-Box Residual

## Decision

- **GO:** leakage-safe periodic Ridge residual architecture.
- **GO:** blocked train fit/calibration and validation-only candidate selection.
- **GO:** validation gate authorizing one frozen held-out test evaluation.
- **GO:** reporting the hybrid as a research comparison model.
- **NO-GO:** replacing the production six-input controller environment.
- **NO-GO:** changing the frozen model after observing test results.

## Leakage Audit

The legacy linear residual corrector includes `x_air_temperature` and `x_relative_humidity` in its default feature list. The stateful multi-step evaluator supplies future trajectory rows, so those fields are future measured indoor states. Package J therefore does not reuse that feature contract.

The new allowlist contains only V4-generated temperature/RH predictions, outdoor temperature/RH, radiation, wind, roof/fan/screen controls and timestamp-derived periodic terms. Prefixes `x_`, `next_x_`, `true_` and `corrected_` are rejected. Predicted-minus-outdoor gaps and radiation/hour interactions use only model outputs and exogenous inputs.

## Fit And Selection Protocol

The 1,787-row chronological train split is divided into 1,429 fit rows and 358 calibration rows. Fit estimates feature normalization and Ridge coefficients; calibration ranks base models and estimates absolute-error quantiles. Twelve base combinations cover three feature sets and alpha values 0.1, 1, 10 and 100. The two best calibration models proceed to validation with gains 0.25, 0.5, 0.75 and 1.0, plus a mandatory gain-zero physical fallback.

The selected model is:

| Property | Value |
|---|---:|
| Feature set | periodic interactions |
| Ridge alpha | 10.0 |
| Hourly feedback gain | 1.0 |
| Temperature q90 radius | 2.239 C |
| RH q90 radius | 8.841 %RH |

The residual is trained on hourly transitions. At solver timestep `dt`, feedback is scaled by `min(1, dt/3600)`. The 900-second controller stress replay therefore uses gain scale 0.25; hourly validation and test use scale 1.0.

## Validation Gate

All values use 52 common validation starts at six-hour stride.

| Model | 24 h T / RH MAE | 72 h T / RH MAE | 72 h mean MAE |
|---|---:|---:|---:|
| Persistence | 1.84 C / 7.20 %RH | 1.87 C / 6.62 %RH | 4.24 |
| Corrected V1 | 1.62 C / 5.42 %RH | 1.74 C / 5.85 %RH | 3.79 |
| V4 physical | 2.22 C / 7.89 %RH | 2.49 C / 9.08 %RH | 5.79 |
| V4 grey-box | **1.20 C / 4.96 %RH** | **1.39 C / 5.91 %RH** | **3.65** |

Additional hybrid validation MAE is 0.888 C / 3.646 %RH at one hour and 1.178 C / 4.248 %RH at six hours.

Validation lag-24 residual autocorrelation falls from V4's 0.600/0.577 to 0.249 for temperature and 0.282 for RH. Q90 coverage is 0.846/0.865 at 24 hours and 0.827/0.808 at 72 hours. Physical envelope and rainy stress gates pass. The frozen validation gate records all eleven criteria as true and authorizes opening the test split.

## Rainy Stress Replay

Using the same 2025 day 226 weather, 900-second solver grid and recorded baseline controls as Packages H and I:

- maximum temperature = 38.346 C;
- minimum temperature = 21.215 C;
- RH range = 52.122-100.000%;
- residual fallback count = 0;
- maximum temperature at or below 40 C = pass.

## Frozen Held-Out Test

The selected-model SHA-256 was frozen before test evaluation. No model, feature, gain or threshold was changed afterward. The primary test retains all 384 transitions and uses 52 requested starts; timestamp gaps reduce completed rollout rows to 3,103.

| Model | 24 h T / RH MAE | 72 h T / RH MAE | 72 h mean MAE |
|---|---:|---:|---:|
| Persistence | 2.86 C / **6.98 %RH** | 4.60 C / 9.14 %RH | 6.87 |
| V4 physical | 3.35 C / 10.17 %RH | 3.39 C / 9.70 %RH | 6.55 |
| V4 grey-box | **2.63 C** / 7.17 %RH | **2.30 C / 6.60 %RH** | **4.45** |

The hybrid substantially improves V4 and wins both 72-hour targets. At 24 hours it improves temperature and total mean MAE, but RH is 0.19 %RH worse than persistence. Test q90 coverage is 0.667/0.689 at 24 hours and 0.743/0.714 at 72 hours, below the 0.75 requirement. Test lag-24 autocorrelation is 0.732 for temperature and 0.312 for RH. The full primary test therefore fails the production gate.

## Test Data Quality Sensitivity

The held-out period runs from 4 July to 20 July 2026. Of 384 rows, 206 use a single indoor temperature sensor; 63 rows exceed 40 C and the maximum reported temperature is 59.574 C. These are the previously documented e3038-only high observations, but they were not removed from the primary test.

A post-hoc sensitivity requiring at least two indoor temperature sensors at both forecast origin and target is labeled non-primary. It leaves only 26 rows at 24 hours and 18 at 72 hours. Hybrid MAE becomes 1.48 C / 6.53 %RH at 24 hours and 1.12 C / 4.90 %RH at 72 hours. This supports a sensor-quality effect but is too small and post-hoc to replace the primary result.

## Reproducibility

- Residual correction SHA-256: `8a56302ab6d88901d66511ef23e7ba9ec2319775b62236fb36b7a1284ba8ad07`.
- Stateful evaluator SHA-256: `9ce32c2374fefde3436b783ce9b66fde42a5c190a6aa263748e2a4ea15ac9d11`.
- Selector SHA-256: `916bc384eeba3e75dbc27b744016a26beec31ccc2f7cb5052f8a566485e540a2`.
- Uncertainty evaluator SHA-256: `e788da91e6281be9b915c787d8e9d91c04797ca49a2f60630a8acde754c07f29`.
- Config SHA-256: `b835500a6dcefa49655fdead5e9bd41e9dc9392011b2e4e08a23d92d22d3df6f`.
- Frozen selected model SHA-256: `e0029f90d65ebf4baaf4e9a8bdb9d8ecd81d8ec149a1685aff1efc75d4f75d10`.
- Validation gate SHA-256: `8fc921c6e093614ec93e0154f45b8baeee0b8fe065422bb6564227799b5d9471`.
- Held-out test decision SHA-256: `58c5a25e3b8685556004567006dbbf4371f049bd38afe31672b8885fed0edf33`.

## Verification

- Package J and affected Chengdu physics tests: `29 passed` in 10.51 s.
- Full repository suite: `454 passed`, `4 subtests passed` in 97.92 s.
- Full-suite warnings: 152 third-party deprecation/future warnings; no test failures.
- The full suite used `py7zr==1.1.3` from workspace-local `.deps`; package cache and temporary files remained under the repository.
- The selected feature allowlist contains no `x_`, `next_x_`, `true_` or `corrected_` inputs.
- Production environment remains `nu: 6` with backend `ChengduPhysics`.
- This workspace is not a Git working tree, so no branch, commit or pull-request operation applies.

## Next Evidence

Do not tune Package J against the opened test. The next defensible step is a new, independently collected Chengdu season with two shielded indoor temperature/RH sensors, continuous radiation metadata and explicit sensor-maintenance records. Package J may be reported as the strongest current hybrid research model, but the production controller benchmark remains on the unchanged six-input environment until independent-season replication.
