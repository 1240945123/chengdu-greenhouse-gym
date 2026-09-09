# Stage A V2 Package I: V4 Daily Energy And Moisture Balance

## Decision

- **GO:** preserving the V3 observations without an unsupported radiation correction.
- **GO:** V4 daily energy and lumped moisture-buffer structure.
- **GO:** leakage-safe staged train ranking and validation-only final selection.
- **GO:** rainy high-radiation physical stress gate.
- **NO-GO:** replacing the production controller-benchmark environment.
- **NO-GO:** opening the held-out test split.

## Training-Period Sensor Check

The proposed solar-exposure correction was tested before changing the target. On the training interval, indoor channels e3038/s1545 and e3044/s1561 have 397 simultaneous radiation-qualified observations. Their difference has 0.865 C MAE, +0.289 C mean bias and -0.162 correlation with global radiation. Mean e3038-minus-e3044 bias is +0.418 C below 5 W m-2 and only +0.068 C above 300 W m-2. This evidence does not support daytime solar overheating of the retained target, so no observation was altered.

## Physical Basis And Contract

V4 retains 28 states and the eight-input offline order:

`[uBoil, uCO2, uThScr, uRoofVent, uLamp, uBlScr, uFan, uPad]`

The extension follows the energy and vapor balances used by process-based greenhouse models: cover-to-sky longwave radiation, floor thermal storage, crop latent cooling and bidirectional moisture storage. Clear-sky temperature is estimated from outdoor temperature and vapor pressure using the Brutsaert emissivity expression when measured sky temperature is absent ([Brutsaert, 1975](https://doi.org/10.1029/WR011i005p00742)). The greenhouse balance structure is consistent with the design model described by [Vanthoor et al., 2011](https://doi.org/10.1016/j.biosystemseng.2011.02.006).

The five appended parameters at indices 220-224 are longwave scale, floor-capacity scale, moisture-buffer rate, moisture-buffer capacity ratio and latent-heat scale. `[0, 1, 0, 1, 0]` disables the extension and reproduces V3 temperature and vapor predictions to numerical tolerance.

State x[16] is a lumped canopy/substrate moisture potential. It is explicitly a storage proxy because irrigation, substrate water content and canopy transpiration were not measured.

## Selection Protocol

The selector has no test-data argument. It evaluated 27 thermal candidates on chronological train rollouts, combined the three train finalists with 27 moisture candidates, retained five train-ranked finalists including the V3 reference, and evaluated those candidates once on validation. Common validation starts use a six-hour stride.

The selected extension is:

| Parameter | Value |
|---|---:|
| Longwave scale | 1.0 |
| Floor capacity scale | 2.0 |
| Moisture-buffer rate | 0.0001388889 s-1 |
| Moisture-buffer capacity ratio | 4.0 |
| Latent-heat scale | 0.5 |

The moisture exchange time constant is approximately two hours. Selection means this candidate is best among the bounded table on the current validation interval; it does not establish that the proxy parameters are uniquely identifiable physical measurements.

## Common-Start Validation

| Model | 24 h T / RH MAE | 72 h T / RH MAE | 72 h mean MAE |
|---|---:|---:|---:|
| Persistence | 1.84 C / 7.20 %RH | 1.87 C / 6.62 %RH | 4.24 |
| Corrected ChengduPhysics V1 | 1.62 C / 5.42 %RH | 1.74 C / 5.85 %RH | 3.79 |
| ChengduPhysicsV2 | 2.19 C / 7.97 %RH | 2.37 C / 8.77 %RH | 5.57 |
| ChengduPhysicsV3 | 2.51 C / 9.21 %RH | 2.75 C / 10.06 %RH | 6.40 |
| ChengduPhysicsV4 | 2.22 C / 7.89 %RH | 2.49 C / 9.08 %RH | 5.79 |

V4 improves over V3 at both horizons but does not beat persistence for either target. It therefore fails the predeclared accuracy gate.

Additional V4 values are:

- 1 h: 1.502 C and 5.791 %RH MAE.
- 6 h: 2.095 C and 7.223 %RH MAE.
- finite predictions and physical validation envelope: pass.

## Residual Gate

- Temperature autocorrelation: lag 1 = 0.822, lag 6 = -0.073, lag 24 = 0.600.
- RH autocorrelation: lag 1 = 0.811, lag 6 = -0.011, lag 24 = 0.577.
- Day temperature bias = -1.589 C; night temperature bias = +0.445 C.
- Day RH bias = +3.893 %RH; night RH bias = -3.698 %RH.

Both lag-24 values remain above 0.5 and are slightly worse than V3. The added states improve mean rollout error but do not explain the repeating daily residual structure.

## Rainy Stress Replay

The Package H replay was independently reconstructed from the v2 weather loader, 2025 day 226, 900-second integration and the recorded baseline controls. It reproduces the V3 maximum exactly at 39.442655 C. With the selected V4 parameters:

- maximum temperature = 36.758 C;
- minimum temperature = 21.358 C;
- RH range = 56.461-100.000%;
- physical envelope = pass;
- maximum temperature at or below 40 C = pass.

This is a physical stress diagnostic, not a measured-accuracy result.

## Reproducibility

- V4 ODE SHA-256: `d6e2f9eb586b89471d9287bc419a2cecc90c41eca419e53903827edcbab52e3b`.
- Selector SHA-256: `a8b1287a7a732ecc65d99af0d94f13647a95e372c0a3a872b8f20ecf66da853d`.
- Selection config SHA-256: `c61a62cd2301fc4243698bb0151f52672cb6c375194890ee21d8cdc20bc3fa05`.
- Selected artifact SHA-256: `58c68a9c40c576c3f883e70ff7f6b690c4c79445cf462b09b2e8b3cba97d7598`.
- Validation metrics SHA-256: `74ef2e6425c3df48410cbf674b3bac378c683e061b4bda6395b087c8b0e5b71f`.
- Residual diagnostics SHA-256: `fbc87373cecfb28c79a15630bfe4ee8abbcdf31b4951fa1f2adea3db84b42458`.
- Stress metrics SHA-256: `13bb24eb2f7c83e6cab612280134bc1789dc27bda472a5fc0206ae6f465e4b5d`.
- Package I focused verification: 31 passed in 8.95 seconds.
- Full regression verification: 437 passed and 4 subtests passed in 483.71 seconds.

## Next Stage

Further unmeasured physical states should not be added to the optimizer. Package J should test a leakage-safe hybrid residual model on top of V4, with hour-of-day, radiation, outdoor climate and actuator regime as exogenous features. It must use blocked train fitting, validation-only model selection, continuous multi-step feedback and uncertainty reporting. The physical V4 forecast remains available as a fallback, and the test split remains sealed until the hybrid beats persistence at 24 and 72 hours for both targets.
