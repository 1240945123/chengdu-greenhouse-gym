# Stage A V2 Package H: V3 Device-Separated Identification

## Decision

- **GO:** V3 physical-device trajectory and sensor-quality provenance.
- **GO:** eight-input offline roof/fan-separated model interface.
- **GO:** bounded robust identification, physical-reference comparison, and sensitivity audit tooling.
- **NO-GO:** accepting the fitted parameter candidates.
- **NO-GO:** moving V3 into the production PID/MPC/PPO/SAC benchmark.
- **NO-GO:** opening or reporting model metrics on the held-out test split.

## Data And Control Identifiability

The V3 trajectory retains 2,553 valid hourly transitions with the same chronological boundaries as V2: train 1,787, validation 382, and sealed test 384. Sensor/control quality gates leave 1,742 training and 359 validation transitions eligible for identification.

Physical command support in the 2,650-row source is:

| Device | Active rows | Active fraction | Treatment |
|---|---:|---:|---|
| Roof window | 1,605 | 60.57% | Separate free exchange path |
| Fan | 1,167 | 44.04% | Separate free exchange path |
| Wet-pad pump | 666 | 25.13% | Preserved as provenance |
| Wet-pad film | 328 | 12.38% | Preserved as provenance |
| Strict fan+pump+film pad state | 10 | 0.38% | Efficiency fixed at zero; not identifiable |
| Roof thermal screen | 264 | 9.96% | Tied to side screen |
| Side thermal screen | 265 | 10.00% | Tied to roof screen because commands are collinear |

There is no measured side-opening actuator. Background infiltration is retained as a physical term and is not labeled a side-window control.

## Sensor Audit

Only configured indoor equipment e3038 and e3044 contributes to the indoor consistency fields; outdoor equipment e3036 is explicitly excluded. Channel e3038 and e3044 overlap for 956 hourly samples with about 0.99 C mean absolute disagreement and 9.91 C maximum disagreement. Identification eligibility requires disagreement at or below 3 C and high roof/fan command-state confidence. Single-sensor rows remain eligible but are flagged.

The sustained 53-60 C e3038 observations after e3044 coverage ends remain in source provenance. They are not silently deleted or used to tune validation thresholds.

## V3 Model Contract

The offline V3 input order is:

`[uBoil, uCO2, uThScr, uRoofVent, uLamp, uBlScr, uFan, uPad]`

V3 preserves V2 air/canopy/floor/cover thermal nodes. Roof exchange responds to roof command and wind; fan exchange is a separate forced-air term; wind-dependent infiltration remains with both actuators off. Pad sensible and latent parameters are fixed at zero in this stage because observed strict pad operation is insufficient.

The production environment remains six-input and unchanged.

## Identification Method

Eight free parameters were bounded: air capacity, solar gain, envelope conductance, roof exchange, transpiration, vapor exchange, background infiltration, and fan exchange. CO2 terms and two wet-pad terms were fixed. Fitting used deterministic train-only subsamples, normalized temperature/RH residuals, bounded `soft_l1` least squares, two starts, and a finite-difference step of `1e-4` that is resolvable above CVODES tolerances.

An initial implementation used SciPy's default difference step near `1e-8`; CVODES returned numerically indistinguishable trajectories and the optimizer falsely terminated without moving. A regression test now protects the corrected step size.

Every selection pool includes an unfitted physical-reference candidate. The fitted candidates lowered robust one-step training cost but greatly worsened continuous validation, reaching roughly 25-28 %RH MAE at 24-72 hours. Validation therefore selected the unfitted physical reference. No fitted coefficient set is accepted as an identified model.

At the selected reference, the sensitivity matrix has rank 8, nonzero condition number 272.36, and no exactly zero numerical column. Fan exchange has the lowest sensitivity norm, 0.124, and lacks a both-actuators-off validation regime, so practical identifiability remains limited despite full numerical rank.

## Common-Start Validation

MAE values use 52 common validation starts at six-hour stride.

| Model | 24 h T / RH | 72 h T / RH | 72 h mean |
|---|---:|---:|---:|
| Persistence | 1.84 C / 7.20 %RH | 1.87 C / 6.62 %RH | 4.24 |
| Corrected ChengduPhysics V1 | 1.62 C / 5.42 %RH | 1.74 C / 5.85 %RH | 3.79 |
| ChengduPhysicsV2 | 2.19 C / 7.97 %RH | 2.37 C / 8.77 %RH | 5.57 |
| ChengduPhysicsV3 reference | 2.51 C / 9.21 %RH | 2.75 C / 10.06 %RH | 6.40 |

V3 fails the requirement to beat persistence for both targets at 24 and 72 hours.

## Residual Diagnostics

- Temperature autocorrelation: lag 1 = 0.812, lag 6 = -0.077, lag 24 = 0.566.
- RH autocorrelation: lag 1 = 0.803, lag 6 = -0.013, lag 24 = 0.556.
- Day temperature bias: -1.18 C.
- Night temperature bias: +0.69 C.
- Day RH bias: +2.22 %RH.
- Night RH bias: -4.99 %RH.
- Validation control regimes: roof only 43 rows, fan only 107, both on 232, both off 0.

Both lag-24 values exceed the 0.5 gate. Daily heat storage/radiation and moisture-buffering dynamics remain systematically unresolved.

## Stress Gate

Replaying the reviewed rainy high-radiation weather and executed baseline controls, with V2 `uVent` mapped to V3 roof opening and fan/pad held off, gives:

- Maximum temperature: 39.44 C.
- Minimum temperature: 21.34 C.
- RH range: 50.13-100.00%.
- Finite physical envelope: pass.
- Maximum temperature at or below 40 C: pass.

This is a stress diagnostic, not a validation-accuracy result.

## Reproducibility

- V3 trajectory manifest SHA-256: `672ffc7990efce129bedbf3c6553eb0aa8cc906f80bc89019f71bd149652aae1`.
- V3 ODE SHA-256: `fba6bfce1176309c2503c88910466d6c418445f7c0b8723280e314936ddf5511`.
- V3 integrator SHA-256: `e9417a04aa72bb2bae67fa54dd223d8af569d89dc95d88007c1913e11b7c7149`.
- Identification config SHA-256: `21bdc061a57de88e21f879fb31a6c58c7a7fc14f5d034bb85b413c824b1ad9a1`.
- Canonical selected-parameter artifact SHA-256: `3be85abb2bee8829c78057f633935df1e8dfdd1f924ea36fdafd24712ace76c4`.
- Focused verification: 28 passed.
- Full regression: 423 passed and 4 subtests passed in 126.66 seconds.

`selected_params_reduced.json` and `selected_params_final.json` are retained as trace artifacts; the canonical path used by scripts is `selected_params.json`, which contains the validation-selected physical reference.

## Next Evidence Required

The next model stage should not expand the optimizer. It must address the observed daily structure: measured/estimated sky longwave forcing, sensor radiation-exposure quality, soil/floor heat storage, canopy latent heat, and greenhouse moisture buffering. A dedicated experiment is required to identify pad efficiency and to create roof/fan-off periods. Until those data or dynamics are available, the controller benchmark remains blocked from final paper ranking.
