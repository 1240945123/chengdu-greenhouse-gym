# Stage A V2 Package B: Controller Execution and Forecast Isolation

Date: 2026-07-30

## Decision

**GO for bounded controller-evaluation smoke tests and leak-free persistence-
forecast diagnostics.** This is not a GO for paper-profile ranking. The current
provider is a persistence baseline, not a calibrated operational forecast; MPC
still uses the plant integrator and true model parameters, which remain gated by
the later `ControllerModelV2` isolation package.

## Controller Execution Contract

- Classical and saved PPO/SAC evaluation proposals pass through the same
  `ControllerExecutorV2` wall-clock boundary.
- Exceptions, invalid shapes, and non-finite proposals invoke the frozen rule
  baseline. If that fallback also fails, the current/no-change finite action is
  used before environment safety projection.
- A timed-out primary controller is disabled for the rest of the episode. This
  prevents repeated calls and accumulating non-cancellable worker threads.
- Proposal and fallback conversion, shape validation, and finite-value checks
  execute inside the same timeout boundary as controller inference.
- PPO/SAC validation used to select `best_model.zip` uses a protected SB3
  callback and records validation controller calls, fallbacks, and timeouts.
  Timeout latches reset for each validation event so an early checkpoint cannot
  force later checkpoints to be ranked as the rule baseline.
- Trajectories record inference time, failure kind, fallback use, fallback
  failure kind, and consecutive fallback duration.
- Episode summaries report mean inference time, controller fallback fraction,
  timeout fraction, fallback steps, and maximum fallback duration separately
  from environment safety interventions.

## Forecast Isolation Contract

- `PersistenceForecastProviderV2.issue` accepts exactly `history[:issue+1]` and
  rejects any history containing post-issue rows.
- Each immutable issued forecast records issue timestep, valid timesteps, lead
  steps, source/model version, and optional error-model ID.
- Benchmark `StepContext` objects and environment observations receive forecasts
  from the configured provider.
- `LightweightMPCController` reads issued forecast values. Missing forecasts use
  current-weather persistence for compatibility, never future realized values.
- Benchmark controller contexts contain an immutable history ending at issue
  time, and MPC retains only the transition function rather than the environment
  or its complete weather matrix.
- Observation and reward contexts use the same causal history. The DLI field is
  recomputed as radiation accumulated through issue time rather than exposing
  the complete calendar-day integral.
- `WeatherForecastObservationsV2` requires a matching issue-time forecast.
  Direct realized-future slicing remains available only under the explicit
  `WeatherForecastObservationsLegacy` name and is absent from the benchmark.
- Poisoned-future tests prove forecasts, V2 observations, and MPC rollout
  disturbances are unchanged when post-issue realized weather is replaced.

## Verification

- Focused controller/safety regression: `76 passed, 6 subtests passed`.
- Full repository regression: `376 passed, 4 subtests passed` in 108.86 seconds.
- Static scan found no unlabelled `WeatherForecastObservations` registration and
  no direct `ctx.d[ctx.t+k]` use outside the explicit legacy observer.

## Remaining Gates

- Fit and freeze a realistic common forecast/error model using permitted
  historical-fit and select roles; keep perfect forecast as a labelled upper
  bound only.
- Separate `ControllerModelV2` from the plant model so MPC cannot use the plant
  transition function, true latent parameters, or hidden plant state.
- Add multi-environment paper-profile propagation tests and reject any paper run
  with nonzero solver failure, controller timeout, or invalid fallback rates.
