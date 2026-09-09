# Stage A V2 Package G: Chengdu Thermal Validation

## Decision

- **GO:** measured-disturbance trajectory V2 and provenance.
- **GO:** continuous 1/6/12/24/72-hour validation workflow.
- **GO:** `ChengduPhysicsV2` as an experimental, versioned structural candidate.
- **NO-GO:** using either Chengdu physics backend for final PID/MPC/PPO/SAC ranking.
- **NO-GO:** opening the V2 test split. Validation criteria have not been met.

## Root Cause Corrected

The previous trajectory artifacts silently copied indoor temperature and relative humidity into the outdoor disturbance columns. They contained 743 transitions from 2026-04-05 through 2026-05-05, and calibration used only the first 240 training rows with a one-step objective. This made envelope and ventilation terms weakly identifiable and concealed long-horizon energy accumulation.

`processing.chengdu_trajectory.build_trajectory_dataframe` now requires measured outdoor temperature and humidity by default. Legacy fallback is possible only through an explicit argument for old synthetic fixtures.

## V2 Data Artifact

Source: `data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv`

- Source rows: 2,650 hourly rows.
- Source interval: 2026-04-01 through 2026-07-20.
- Candidate transitions: 2,649.
- Excluded for missing required measurement: 96.
- Valid transitions: 2,553.
- Chronological rows: train 1,787; validation 382; test 384.
- Source SHA-256: `54100bdbb6aa8e480ef0cf7d242badc950efa0a63b348f77da7b1e8846d6eb2d`.
- Indoor-as-outdoor fallback: prohibited.

The artifact manifest and split hashes are in `data/processed/chengdu_agri/greenhouse_001/trajectories/v2/manifest.json`.

## Evaluation Repair

The old multistep evaluator propagated only temperature and relative humidity and reinitialized the other 26 states every hour. The new evaluator propagates all 28 model states, stops at timestamp gaps, and reports MAE, RMSE, bias, observed/predicted extrema, finite-state status, and physical-envelope status. Persistence and outdoor-following references use identical rollout starts.

Calibration uses training data for candidate fitting and validation data for final selection. The test split is not accepted by the selection API and remains unopened.

## Corrected Validation Results

The table reports MAE on 52 common validation rollout starts with six-hour stride.

| Model | 1 h T / RH | 24 h T / RH | 72 h T / RH | 72 h mean |
|---|---:|---:|---:|---:|
| Persistence | 0.99 C / 4.23 %RH | 1.84 C / 7.20 %RH | 1.87 C / 6.62 %RH | 4.24 |
| Outdoor following | 1.79 C / 6.93 %RH | 1.90 C / 6.38 %RH | 1.98 C / 6.84 %RH | 4.41 |
| ChengduPhysics, corrected calibration | 1.58 C / 6.17 %RH | 1.62 C / 5.42 %RH | 1.74 C / 5.85 %RH | 3.79 |
| ChengduPhysicsV2, corrected calibration | 1.56 C / 6.23 %RH | 2.19 C / 7.97 %RH | 2.37 C / 8.77 %RH | 5.57 |

The V1 corrected calibration is more accurate than the references on 24/72-hour mean MAE, but predicts a validation maximum of 38.97 C versus 35.05 C observed. Its one-day rainy high-radiation controller diagnostic reaches 48.81 C and yields temperature MAE 10.49 C. It therefore fails extrapolation validity despite acceptable in-domain averages.

V2 partitions solar energy among air, canopy, and floor nodes, adds cover/floor heat capacities, retains wind-dependent infiltration when the roof is closed, and removes a duplicate dimensionally inconsistent roof dehumidification term. On the same rainy diagnostic it reduces maximum temperature to 39.44 C and temperature MAE to 8.19 C, but its measured validation accuracy is worse and RH MAE remains excessive. Structural plausibility improved; empirical adequacy did not yet pass.

## Reproducibility

- V2 ODE SHA-256: `7197bb1480e6aa2b3814d345db9879c7736848d8c56cb5e6a719fbb382d35855`.
- V2 integrator SHA-256: `1efdab74c17b29fff776c43563c64762c533d8f7bee8419ce93b906e53680723`.
- Calibration config SHA-256: `54415a301de2ceca90a18d46231f1d05cdd8b47a28e3bdf6cfe2c8d90c4bc402`.
- Selected V2 parameter artifact SHA-256: `15cacdbe9fa0fc1f8e87180642e6fca7748c9dcb02995942e9b54d010a78bac5`.
- Selected V2 multipliers: p208=8.0, p209=0.6, p210=2.5, p211=8.0, p212=0.6, p213=1.0, p214=0.6, p215=3.0.
- Focused verification: 29 passed.
- Full regression: 404 passed and 4 subtests passed in 83.92 s.

## Required Next Gate

Do not switch `ChengduControllerBenchmark.yml` yet. The next stage must identify V2 heat- and moisture-transfer parameters with constrained optimization on contiguous measured episodes, add fan/wet-pad and side-opening controls as distinct exchange paths, and compare residual autocorrelation by day/night and actuator regime. The validation target is to beat persistence at 24 and 72 hours for both temperature and RH while keeping the rainy diagnostic below 40 C without clipping. Only then may the frozen test split be opened and controller experiments rerun.
