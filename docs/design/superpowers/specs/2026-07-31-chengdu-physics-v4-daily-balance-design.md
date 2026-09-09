# ChengduPhysicsV4 Daily Balance Design

## Objective

Package I tests whether the unresolved 24-hour temperature and humidity structure can be reduced by a minimal physical extension of ChengduPhysicsV3. The production six-input environment remains unchanged, and the held-out test split remains unopened.

## Evidence And Choice

Three approaches were considered:

1. A learned residual corrector would likely improve validation metrics but would weaken physical interpretation and extrapolation to new Chengdu seasons.
2. A large greenhouse reconstruction with new canopy, substrate, cloud and irrigation states would be closer to a full process model but is not identifiable from the current measurements.
3. The selected hybrid approach preserves the V3 state and control contract, adds only daily energy and water-buffer processes, and retains a V3-equivalent zero-extension candidate.

Training-period overlap between indoor channels e3038/s1545 and e3044/s1561 does not support a solar-heating correction to the observed temperature. Their high-radiation bias is about +0.07 C, night bias about +0.42 C, and difference-versus-radiation correlation -0.16. Observations therefore remain unchanged.

The physical additions follow established balance-model structure: greenhouse models include sky thermal radiation, soil heat storage, crop latent heat and vapor balances; Brutsaert's clear-sky emissivity provides a physically grounded estimate when measured longwave radiation is absent (Brutsaert, 1975, doi:10.1029/WR011i005p00742; Vanthoor et al., 2011, doi:10.1016/j.biosystemseng.2011.02.006).

## Model Contract

The V4 state dimension remains 28 and the offline input order remains:

`[uBoil, uCO2, uThScr, uRoofVent, uLamp, uBlScr, uFan, uPad]`

State `x[16]`, unused by V3 predictions except as a passive vapor follower, becomes a lumped canopy/substrate moisture-buffer vapor potential in V4. It exchanges vapor bidirectionally with greenhouse air. This is a storage proxy, not a claim that irrigation or crop transpiration was measured.

Five extension parameters are appended:

| Index | Parameter | Role |
|---:|---|---|
| 220 | `longwave_scale` | Cover-to-sky Stefan-Boltzmann exchange multiplier |
| 221 | `floor_capacity_scale` | Floor thermal storage multiplier |
| 222 | `moisture_buffer_rate_s` | Air-buffer vapor exchange rate |
| 223 | `moisture_buffer_capacity_ratio` | Buffer capacity relative to greenhouse air |
| 224 | `latent_heat_scale` | Fraction of transpiration vapor source coupled as canopy latent cooling |

The V3-compatible reference is `[0, 1, 0, 1, 0]`. With these values, V4 must reproduce V3 air temperature and humidity to numerical tolerance.

## Forcing And Balances

When measured sky temperature is absent, evaluation estimates clear-sky emissivity from outdoor temperature and vapor pressure using the Brutsaert form and converts downwelling longwave flux to an effective sky temperature. The value is bounded to a physically finite range. This forcing is deterministic and uses no future targets.

Cover energy receives net longwave loss

`Q_lw = longwave_scale * emissivity * sigma * area * (T_cover_K^4 - T_sky_K^4)`.

The existing transpiration vapor-pressure source is converted to water mass flow through the ideal-gas relation and to latent heat through water's vaporization enthalpy. The scaled heat is removed from the canopy energy balance.

Moisture-buffer exchange is

`J_buffer = k_buffer * (vp_buffer - vp_air)`

with equal and opposite storage change divided by the capacity ratio. Zero exchange exactly disables the mechanism.

## Candidate Selection

No broad nonlinear optimizer is used. A deterministic, physically bounded candidate table is evaluated on chronological training rollouts. The top training candidates plus the V3-compatible reference are evaluated once on validation. The test split is not accepted as a function argument by the selection command.

The candidate table contains ablations for thermal-only, moisture-only and combined extensions. Selection score is the normalized mean of 24-hour and 72-hour temperature and RH MAE. Failed integrations and nonphysical predictions are rejected.

## Acceptance Gates

V4 is accepted for controller benchmarking only if all conditions hold on common validation starts:

- finite predictions and physical temperature/RH envelope;
- lower temperature and RH MAE than persistence at both 24 and 72 hours;
- temperature and RH lag-24 one-step residual autocorrelation below 0.5;
- no hidden test-set access;
- the selected artifact records data, config and implementation hashes.

Failing any gate leaves the production model unchanged and records Package I as NO-GO.

## Tests And Artifacts

Tests cover Brutsaert forcing bounds, V3-equivalent zero extension, direction of longwave/latent/buffer fluxes, explicit V4 backend selection, train/validation role separation, deterministic candidate selection and artifact schema. Outputs are written under `results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/` with a Package I audit in `docs/audits/`.

