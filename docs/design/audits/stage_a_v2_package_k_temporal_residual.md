# Stage A V2 Package K: Temporal Residual Climate Models

## Decision

- **GO:** retain the LSTM and TCN implementations as research comparison models.
- **GO:** report all three seeds and architecture ensembles on common validation origins.
- **NO-GO:** replace the frozen V4 plus Ridge hybrid.
- **NO-GO:** retrain PPO/SAC against the temporal residual environment.
- **NO-GO:** select `lstm_seed_1` after seeing validation merely because it is the best individual run.

## Protocol

The 1,787-row chronological train split was divided into 1,429 fit rows and 358 calibration rows. A 24-hour lookback produced 940 contiguous fit sequences and 335 contiguous calibration sequences. Input and residual-target scalers were estimated on fit only. LSTM and TCN each used fixed seeds 0, 1 and 2, Adam, gradient clipping and calibration early stopping.

Architecture selection used median normalized calibration MAE across all three seeds. TCN was selected with median score 0.4600 versus 0.4703 for LSTM. The test suffix was not used because it had already been opened in Package J.

Every token contains only V4 predictions, outdoor weather, controls and timestamp-derived terms. Observed or target indoor prefixes are rejected. Validation warm-up contains only rows before each origin; forecast residuals are fed back into the V4 temperature and vapor-pressure states.

## Calibration

| Architecture | Seed | Parameters | Best epoch | T residual MAE | RH residual MAE | Score |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 0 | 2,210 | 51 | 1.005 C | 4.062 %RH | 0.4544 |
| LSTM | 1 | 2,210 | 50 | 0.974 C | 4.536 %RH | 0.4703 |
| LSTM | 2 | 2,210 | 26 | 1.095 C | 4.052 %RH | 0.4763 |
| TCN | 0 | 1,874 | 64 | 0.948 C | 3.665 %RH | 0.4203 |
| TCN | 1 | 1,874 | 31 | 1.013 C | 4.134 %RH | 0.4600 |
| TCN | 2 | 1,874 | 9 | 1.037 C | 4.318 %RH | 0.4752 |

## Common-Origin Validation

All models use the same 48 origins. Every origin completes all requested horizons.

| Model | 1 h T / RH MAE | 6 h T / RH MAE | 24 h T / RH MAE | 72 h T / RH MAE |
|---|---:|---:|---:|---:|
| Persistence | 1.14 / 4.00 | 4.45 / 16.87 | 1.87 / 6.89 | 1.90 / 7.36 |
| V4 physical | 1.61 / 5.28 | 2.19 / 7.62 | 2.24 / 8.06 | 2.53 / 9.35 |
| V4 + Ridge | 0.93 / 3.46 | 1.35 / 4.83 | **1.25 / 5.09** | **1.51 / 6.45** |
| LSTM ensemble | 0.82 / 3.29 | 1.33 / 4.58 | 1.38 / **4.96** | 1.74 / **6.15** |
| TCN ensemble | **0.72 / 3.11** | **1.31 / 4.73** | 1.39 / 5.25 | 1.82 / 6.45 |

Units are C for temperature and percentage points for RH. All temporal predictions are finite and remain inside the physical envelope.

The selected TCN ensemble improves the one-hour MAE from Ridge's 0.93 C / 3.46 %RH to 0.72 C / 3.11 %RH. It does not preserve the improvement in long rollouts: at 24 hours it reaches 1.39 C / 5.25 %RH, and at 72 hours 1.82 C / 6.45 %RH. The strict four-part long-horizon gate therefore fails.

## Stability And Interpretation

Calibration ranking does not transfer cleanly to validation: TCN is selected on calibration, while LSTM performs better at long validation horizons. LSTM seed-1 reaches 1.22 C / 4.68 %RH at 24 hours and 1.55 C / 5.75 %RH at 72 hours, but LSTM seed-2 degrades to 1.63 C / 6.11 %RH and 2.04 C / 7.46 %RH. Reporting only seed-1 would hide material initialization sensitivity.

The likely causes are limited seasonal coverage, only 940 fit sequences after leakage-safe blocking, one-step teacher-state training followed by rolled-state inference, and changing sensor quality through the 2026 record. Increasing network depth would raise variance without adding independent information.

## Recommended Direction

Keep V4 plus Ridge as the current primary climate model. Use the temporal models only as ablations demonstrating short-horizon residual structure. The next evidence should be an independently collected crop season with two shielded indoor temperature/RH sensors, sensor-maintenance flags, reliable radiation measurements, and explicit irrigation or substrate-water observations. Once that exists, repeat blocked season-level validation and test a shrinkage-gated Ridge plus LSTM correction rather than replacing the physical model.

Do not restart controller training until the climate model passes both 24- and 72-hour temperature/RH gates on an independent season. This avoids optimizing PPO/SAC against neural rollout artifacts that are not yet stable.

## Artifacts

- Configuration: `configs/models/chengdu_temporal_residual.yml`
- Training metadata: `results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual/training_summary.json`
- Validation metrics: `results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual/validation_metrics.json`
- Common-origin rollouts: `results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual/validation_rollouts.csv`
- Six model checkpoints: `results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual/*_seed_*.pt`

## Verification

- Focused temporal and grey-box tests: 22 passed.
- Full repository suite: 520 passed, 4 subtests passed in 145.79 seconds.
- Python compilation check passed for the new model, training and evaluation modules.
- Forbidden observed/target feature count in the saved artifact: zero.
- Ridge and TCN each completed all 48 common 72-hour validation origins.
- The production controller configuration remains `nu: 6` with `model_backend: ChengduPhysics`.
- The workspace is not a Git working tree, so branch, commit and pull-request operations do not apply.
