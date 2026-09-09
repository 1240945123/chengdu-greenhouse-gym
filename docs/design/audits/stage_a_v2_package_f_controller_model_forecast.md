# Stage A V2 Package F: Controller Model and Forecast Emulator

Date: 2026-07-31

## Decision

**GO for interface isolation and forecast-emulator diagnostics. NO-GO for
controller ranking or paper-profile training.** MPC is now isolated from the
plant transition and hidden state. The historical forecast emulator is frozen
and role-safe. However, a one-day 2025 benchmark exposes an implausible long-
horizon thermal response in `ChengduPhysics`, so reward and comfort differences
cannot yet be interpreted as controller performance.

## ControllerModelV2

- `ControllerStepContextV2` exposes only current indoor temperature/RH, current
  executed controls, time, and an issue-time forecast.
- MPC has no `bind_env`, CasADi plant transition, full state, true parameter
  vector, weather repository, crop mass, or posterior identity.
- The frozen two-state controller surrogate uses versioned temperature and
  humidity coefficients. Poisoning hidden plant states and true parameters does
  not change MPC proposals.
- Heating and CO2 remain disabled by the common benchmark action/safety layer.

## Historical Forecast Emulator

- Label: `historical-forecast-emulator-v2`; explicitly not operational NWP.
- Fit role/year: 2023. Select role/year: 2024. Retrospective year: 2025.
- Candidate persistence/climatology weights: 0, 0.25, 0.5, 0.75, 1.
- Selected weight: 0.75.
- 2024 normalized 48-step weighted MAE: 0.6189950661 versus persistence
  0.9361561474 after explicitly removing February 29 from selection.
- Artifact fingerprint:
  `b1e101f9763e7b13ef6434c4066c05f2bb0ed9a0e83821c1d4bf46fb9e7a32a8`.
- Payload SHA-256:
  `d02249094a632b58bd021c6c12713cd2063d794614880199c52121e696b7de85`.
- Runtime providers receive only issue-time causal history; poisoned future
  values do not change forecasts. Trajectories record model version and artifact
  identity.
- Runtime loading recomputes the artifact fingerprint and verifies the payload,
  roles, and 2023/2024 source hashes. Metric variable indices/scales and the
  leap-day policy are stored in the manifest.
- Trusted benchmark configuration pins the artifact fingerprint and both source
  hashes. Trajectories and RL metadata carry the model label, metric definition,
  source hashes, roles, leap policy, and controller-model fingerprint.

## Verification

- Focused controller, forecast, safety, and benchmark suite:
  `90 passed, 6 subtests passed`.
- Full repository suite: `392 passed, 4 subtests passed` in 87.19 seconds.
- Static scan found no `ctx.x`, `ctx.p`, `env.F`, `bind_env`, or `weather_data`
  access in MPC or `ControllerModelV2`.

## One-Day Diagnostic

All methods completed 96 steps with zero controller fallback and zero timeout.

| Algorithm | Reward | Temp MAE (C) | RH MAE (%pt) | Joint comfort | Safety intervention | Inference (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | -152.158 | 17.732 | 16.795 | 0.010 | 0.948 | 0.544 |
| PID | -150.226 | 17.684 | 17.009 | 0.000 | 0.854 | 0.542 |
| MPC | -149.955 | 17.707 | 16.894 | 0.000 | 0.583 | 2.432 |

Fresh yield is zero for all methods because a one-day episode is too short to
represent a harvest season. These numbers are pipeline diagnostics, not efficacy
results.

The reviewed MPC controller-model fingerprint is
`8cc29cd311263e7cc73df6875727021e34c24ff8b6b3e16e4436c83e69ce5b4f`.

## Thermal Root Cause and Gate

The test-day indoor temperature reaches about 95.2 C for every controller. In
`ChengduPhysics`, `0.62 * area * radiation` is injected directly into the air
node while the dominant heat capacity is scaled air volume. During precipitation
the safety projection closes the only represented ventilation actuator. At
roughly 1,166 W/m2 this creates a high closed-roof equilibrium that is not a
credible greenhouse climate. The previous one-step calibration does not validate
this long-horizon energy balance.

Before ranking controllers:

1. Split rain-sensitive roof ventilation from rain-safe leakage/side ventilation.
2. Add canopy, floor, structure, and cover thermal storage or calibrate an
   equivalent multi-node energy balance.
3. Validate 24-hour and multi-day extrema, energy conservation, and rain-event
   responses against Chengdu measurements.
4. Re-run the paired one-day diagnostic, then 180-day smoke/development gates.
