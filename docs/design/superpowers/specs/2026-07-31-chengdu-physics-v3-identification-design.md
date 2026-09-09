# Chengdu Physics V3 Identification Design

## Goal

Build an offline, auditable parameter-identification stage that separates measured roof-window and fan effects before changing the production controller environment. The stage must identify only parameters supported by the available 110-day dataset and must leave the final test split sealed until validation gates pass.

## Evidence And Identifiability

The aligned hourly source contains independent physical device columns that were collapsed in the current trajectory contract. Roof-window commands are active for 60.6 percent of hours and fan commands for 44.0 percent; their correlation is -0.129, so their effects can be investigated separately. The strict fan, pump, and wet-pad-film intersection `uPad` is active for only about 0.38 percent of hours, so wet-pad efficiency is not identifiable from this dataset. Roof and side thermal-screen commands are effectively perfectly correlated and cannot be estimated separately. There is no measured side-opening actuator; leakage is therefore a physical background term, not a fabricated side-window command.

Indoor channel e3038 and e3044 overlap for 956 hourly samples with about 0.99 C mean absolute disagreement and up to 9.91 C disagreement. After e3044 ends, e3038 reports sustained 53-60 C periods during high radiation. Those observations remain in the source audit, but single-sensor and disagreement flags must accompany the target so sensor exposure effects are not silently treated as greenhouse air truth.

## Architecture

1. Create a V3 trajectory artifact from the aligned hourly source. Preserve roof window, fan, wet-pad pump, wet-pad film, strict pad activation, physical screen states, per-device known/uncertain fractions, sensor count, and sensor disagreement. Keep chronological train/validation/test boundaries and source fingerprints.
2. Create `ChengduPhysicsV3` without modifying V1 or V2. Its offline control vector is `[uBoil, uCO2, uThScr, uRoofVent, uLamp, uBlScr, uFan, uPad]`. Roof ventilation follows wind/buoyancy exchange; fan exchange is a separate forced-air term; background infiltration is independent. Pad cooling/moisture parameters are present but fixed and marked non-identifiable in this stage.
3. Fit a compact parameter set with bounded robust least squares on training episodes only. Candidate parameterizations are shortlisted using training loss, then selected by continuous 6/24/72-hour validation metrics. Test data is inaccessible to the selection API.
4. Compute a finite-difference sensitivity matrix and report singular values, condition number, rank, and per-parameter sensitivity. Weak parameters are frozen rather than reported as estimates.
5. Report residual bias, MAE/RMSE, and lag-1/6/24 autocorrelation by day/night and by roof/fan regime. Compare against persistence, outdoor-following, corrected V1, and V2 on common validation starts.

## Sensor Target Policy

The canonical indoor target remains the median of available indoor sensors. V3 adds provenance rather than rewriting raw observations: number of available indoor channels, maximum pairwise disagreement, single-sensor flag, and a high-disagreement flag at 3 C. Training uses robust loss and excludes transitions with disagreement above 3 C. Single-sensor rows remain eligible but are reported separately. No test-period sensor values are used to tune the threshold or parameters.

## Parameter Scope

Freely identifiable candidates are effective air capacity, solar transmission, envelope conductance, roof-window exchange, fan exchange, background infiltration, transpiration source, and vapor exchange. Positive bounds are defined in configuration. Thermal-screen roof/side coefficients remain tied. Wet-pad sensible and latent efficiency remain fixed at zero for fitting because the observed active support is insufficient; later activation requires manufacturer specifications or a dedicated excitation experiment.

## Validation Gates

- Data: physical controls preserved, exact one-hour adjacency, no indoor-as-outdoor substitution, and complete fingerprints.
- Identification: finite solution, no bound/NaN failure, reported sensitivity rank, and no unsupported parameter labeled identified.
- Accuracy: outperform persistence for both temperature and RH MAE at 24 and 72 hours on validation.
- Residuals: absolute 24-hour residual autocorrelation below 0.5 for both targets; otherwise systematic daily dynamics remain unresolved.
- Physics: all states finite, temperature within -10 to 60 C, RH within 0 to 100 percent without post-step clipping.
- Stress: rainy high-radiation diagnostic maximum at or below 40 C.

Failure of any gate keeps the production benchmark on NO-GO and leaves the test split unopened.

## Deferred Production Change

This stage does not change `ChengduControllerBenchmark.yml`, RL observation/action dimensions, PID/MPC interfaces, or published benchmark outputs. If V3 passes validation, a separate versioned stage will add fan and pad actions to the environment and update every controller under one common execution contract.

