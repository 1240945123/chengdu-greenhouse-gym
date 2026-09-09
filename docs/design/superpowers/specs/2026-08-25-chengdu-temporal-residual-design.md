# Chengdu Temporal Residual Climate Model Design

## Objective

Evaluate whether a small causal LSTM or TCN can improve the leakage-safe ChengduPhysicsV4 plus Ridge residual benchmark without using future indoor observations. The model predicts the one-hour temperature and relative-humidity residual left by the physical model and is evaluated in continuous 1, 6, 24 and 72 hour rollouts.

## Evidence Constraint

The available target trajectory contains 2,553 hourly transitions from 1 April to 20 July 2026. This is less than one independent crop season and the already-opened test suffix contains substantial single-sensor temperature exposure and values near 59.5 C. A neural residual result is therefore a candidate comparison, not evidence of independent-season deployment readiness.

## Data Roles

- Fit: the first 80% of the existing chronological train split.
- Calibration: the final 20% of the existing chronological train split; used for early stopping and architecture selection.
- Validation: the existing chronological validation split; used once for model reporting and the environment-model gate.
- Test: not used by this stage because it has already been opened and inspected.

Input and target scalers are fitted on the fit block only. Sequence windows may not cross timestamp gaps or role boundaries.

## Inputs And Targets

Each hourly token uses only values available during forecasting:

- V4-predicted indoor temperature and RH;
- outdoor temperature, RH, radiation and wind;
- roof vent, fan, blackout screen and thermal screen commands;
- sine/cosine hour terms;
- predicted indoor-minus-outdoor gaps;
- radiation-by-hour interactions.

Observed indoor columns (`x_`, `next_x_`, `true_`, `corrected_`) are forbidden. The target is `observed next state - V4 one-step prediction` for temperature and RH. Lookback is 24 hours.

## Models

Two deliberately small models are compared:

- LSTM: one recurrent layer, 16 hidden units, final-token two-output head.
- TCN: two causal residual convolution blocks, 16 channels, final-token two-output head.

Both predict standardized residuals and use MSE training, Adam, gradient clipping, deterministic seeds `[0, 1, 2]`, and calibration early stopping. Architecture selection uses the median calibration score across all seeds, where score is the mean of temperature MAE divided by 2 C and RH MAE divided by 10 percentage points. No best-seed-only reporting is allowed.

## Stateful Rollout

At each forecast origin, the temporal context is built only from rows before the origin. Historical tokens use one-step V4 predictions initialized from then-observed states, which are legitimate past information. Forecast tokens use the continuously rolled V4 state and known weather/control rows. The predicted residual is fed back into the physical temperature and vapor-pressure states. Context warm-up may not consume rows at or after the forecast origin.

## Comparisons And Metrics

Use identical validation origins and report pure V4, persistence, frozen Ridge, LSTM seeds, TCN seeds and architecture ensembles. At 1, 6, 24 and 72 hours report temperature/RH MAE, RMSE and bias, physical-envelope pass, and requested/completed rollout counts. Also report parameter count, training rows, calibration score and runtime metadata.

## Gate

The temporal candidate passes only if its three-seed ensemble:

- is finite and remains inside `[-10, 60] C` and `[0, 100] %RH`;
- beats frozen Ridge on both temperature and RH MAE at 24 and 72 hours;
- does not select an architecture from a single lucky seed;
- records every split, seed, scaler and source hash.

Failure keeps frozen Ridge as the strongest current hybrid and directs work toward better sensors and an independent season rather than additional network tuning.

