# Controller Model V2 and Forecast Emulator Design

## Scope

This package removes plant-model and latent-state access from benchmark MPC and
adds a reproducible issue-time weather forecast emulator for Chengdu. It does
not claim access to archived operational forecasts and does not authorize the
paper profile.

## Controller Boundary

`ControllerStepContextV2` contains only timestep, control interval, current
indoor air temperature and relative humidity, current executed controls,
issue-time forecast, hour, and day. It contains no full plant state, plant
parameters, weather repository, transition function, harvest state, or posterior
draw identity.

`ControllerModelV2` is a frozen low-order temperature and humidity transition
model. Its coefficients are supplied by a versioned configuration and are never
read from the evaluation environment. It predicts the next indoor climate from
the restricted state, candidate controls, and one issued weather row. Heating
and CO2 remain disabled. MPC scores only the frozen comfort/resource/action
objective and applies the same action slew limits used by the benchmark.

Poison tests change all hidden plant states and true plant parameters while
holding permitted observations fixed. MPC proposals must remain identical.

## Forecast Emulator

Because archived Chengdu operational forecasts are unavailable, the main
diagnostic provider is explicitly named `historical-forecast-emulator-v2`. It
combines issue-time persistence with a day-of-year/hour climatology built only
from permitted years. Candidate blend weights are fitted on 2023 and selected
by multi-lead temperature, humidity, radiation, and wind MAE on 2024. The frozen
artifact stores source checksums, roles, variables, lead horizon, blend weight,
metrics, and schema version.

At runtime the provider receives only history through the issue timestep and the
frozen artifact. It cannot inspect 2025 values after issue time. Precipitation is
forecast conservatively by persistence at zero unless rain is already observed;
the provider is not used to make equipment-certification claims. A separately
labelled perfect-forecast provider may be added later as an upper bound.

## Gates

- Restricted MPC contexts contain no forbidden fields.
- Poisoned hidden state, parameters, and future realized weather do not change
  MPC proposals.
- Forecast fitting rejects 2025 sources and role violations.
- Forecast artifacts are deterministic and checksum-addressed.
- 2024 selection metrics must beat or equal pure persistence on the declared
  weighted MAE; otherwise persistence remains selected.
- All repository tests pass, followed by an independent adversarial review.

## Limitations

The controller model is an engineering surrogate until calibrated against
independent actuator-response experiments. The forecast emulator estimates
forecast uncertainty from historical weather structure, not archived numerical
weather prediction errors. Both labels must appear in reports.

