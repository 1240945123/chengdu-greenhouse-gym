# Chengdu V4 Grey-Box Residual Design

## Objective

Package J tests whether a leakage-safe, low-capacity residual model can remove the daily error left by ChengduPhysicsV4. The hybrid must retain continuous physical-state rollouts, expose uncertainty, preserve a zero-correction fallback and keep the held-out test split unopened.

## Rejected Alternatives

The legacy linear corrector is not accepted because its default features include future-row indoor temperature and RH. During stateful multi-step evaluation those values are measured future states and therefore leak the target trajectory into the forecast.

A recurrent neural residual model is also rejected for this stage. The measured trajectory has only 2,553 valid hourly transitions, so an LSTM or Transformer residual would add capacity and tuning variance without independent seasons for reliable selection.

The selected approach is a standardized Ridge model of the one-step V4 residual. It is deterministic, auditable and constrained to exogenous or model-generated features.

## Data Roles

The chronological V3 trajectory boundaries remain unchanged. The train split is divided chronologically into an 80% fit block and a 20% calibration block. The fit block estimates feature normalization and Ridge coefficients. The calibration block ranks base models and estimates absolute-error quantiles. Validation selects the final feature set, regularization and feedback gain. Test is not accepted by the training command and remains unopened.

## Allowed Features

No feature may begin with `x_`, `next_x_`, `true_` or `corrected_`. Allowed raw features are:

- V4 one-step predictions for air temperature and RH;
- outdoor temperature, RH, radiation and wind;
- roof, fan, blackout-screen and thermal-screen commands;
- timestamp-derived sine and cosine of hour of day.

Derived features may include predicted indoor-minus-outdoor gaps and radiation-by-hour interactions. All derived values must be computable from the current predicted state, known disturbance forecast, known control sequence and timestamp.

Three nested feature sets are evaluated: `base`, `periodic` and `periodic_interactions`. Ridge alpha values are `[0.1, 1, 10, 100]`. Base models are ranked by normalized calibration MAE; the top two plus the physical fallback proceed to validation. Feedback gains `[0.25, 0.5, 0.75, 1.0]` scale the predicted residual before it is written back into the V4 temperature and vapor-pressure state.

## Model Form

For each target, the residual is

`r = y_next - y_v4`.

Features are standardized using fit-block means and nonzero standard deviations. Ridge coefficients solve

`beta = argmin ||X beta - r||^2 + alpha ||beta||^2`,

with the intercept excluded from regularization. The hybrid prediction is

`y_hybrid = y_v4 + gain * r_hat`.

The fitted residual has a one-hour time base. For a solver step `dt`, the applied feedback gain is multiplied by `min(1, dt / 3600)`. Thus the 900-second controller environment applies one quarter of the hourly correction per step, while the hourly validation protocol applies the selected gain unchanged.

Temperature and RH are clipped only to the same broad physical envelope used by evaluation, not to comfort bands. A nonfinite feature or prediction causes the residual path to fall back to the uncorrected V4 output and increments a fallback counter.

## Uncertainty

For every candidate, the calibration block computes target-specific absolute-error quantiles at 80%, 90% and 95%. Validation reports empirical coverage and mean interval width at 1, 6, 24 and 72 hours using fixed symmetric intervals around the hybrid prediction. These are split-conformal-style marginal diagnostics, not claims of conditional coverage or future-season validity.

## Evaluation And Gates

All models use the same 52 validation starts at six-hour stride and continuous 72-hour physical-state feedback. Package J is GO for opening the sealed test only if:

- predictions are finite and within the physical envelope;
- both temperature and RH MAE beat persistence at 24 and 72 hours;
- both lag-24 one-step residual autocorrelations are below 0.5;
- 90% interval coverage for both targets at 24 and 72 hours is at least 0.75;
- the selected artifact records fit, calibration, validation and unopened-test roles plus source hashes.

Failure leaves the production six-input environment unchanged.

## Artifacts

Package J writes the selected model, candidate audit, validation rollouts, uncertainty metrics, corrected one-step residual diagnostics and a stage audit under `results/chengdu_agri_greenhouse_001/physics_v4_greybox_residual/` and `docs/audits/`.
