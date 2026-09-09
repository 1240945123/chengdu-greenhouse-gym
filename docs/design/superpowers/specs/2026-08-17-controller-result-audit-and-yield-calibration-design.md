# Controller Result Audit and Yield Calibration Design

## Goal

Determine which completed controller results are scientifically interpretable,
replace ambiguous short-window yield claims with explicit biomass accounting,
and calibrate the identifiable tomato harvest-process priors before rerunning
the six Chengdu 120-day seasons.

## Evidence Contract

The workflow keeps three evidence classes separate:

- `target_observed`: measurements from Pidu greenhouse 63. Only these may
  establish target-site harvest accuracy.
- `external_observed`: the WUR AGC2 six-compartment tomato dataset. These data
  may calibrate transferable process priors and validate parsers, but may not
  set `target_validated=true`.
- `simulated_prior`: Chengdu weather-driven GreenLight and cohort outputs.
  These support controller comparison, sensitivity analysis, and experiment
  preflight only.

All reports must retain the current production gate: zero dated target harvest
events means the model is not target-harvest calibrated or validated.

## Result Audit

The completed 4-day controller benchmark is audited as a short-horizon climate
control experiment. Its valid outcomes are reward, temperature and humidity
tracking, comfort fractions, actuator effort, safety intervention, fallback,
and runtime. It is not a complete-season yield experiment because it contains
one fixed 2025 autumn weather window and no meaningful harvest horizon.

The audit distinguishes:

- harvested fresh yield, which must be non-negative;
- gross allocated fruit dry matter, which must be non-negative;
- fruit inventory change, which may be negative over a short interval;
- accounted net fruit biomass change, defined as harvest plus final standing
  fruit minus initial standing fruit, which may also be negative.

The existing `simulated_fresh_fruit_production_kg_m2` name is treated as an
ambiguous legacy metric when its value is negative. Reports must not describe
that value as physical negative production.

## Calibration Scope

Only parameters identifiable from available observations are calibrated:

1. fruit development duration and maturity thermal time, using WUR reported
   truss development time combined with compartment air temperature;
2. fruit dry-matter fraction, using observed fruit-quality dry matter values
   as an external prior and the existing Chengdu Xindu fresh/dry pairs as the
   transfer prior for Pidu;
3. picking interval and observed batch-mass distribution, using WUR production
   events;
4. standing-fruit allocation multiplier, retaining the existing one-date Pidu
   fit and one-date holdout as preliminary evidence only.

Photosynthesis, maintenance respiration, organ allocation, maturity, and
initial state are not jointly fitted to the three Pidu crop dates. That fit is
structurally underdetermined.

## Calibration Method

For each WUR compartment, Excel serial timestamps are converted with the
documented 1899-12-30 origin. For every harvest event with a valid truss
development duration, maturity thermal time is the integral of
`max(Tair - 10 degC, 0)` over the preceding reported development interval.
Estimates are aggregated by compartment, and uncertainty uses a deterministic
compartment-block bootstrap so dense 5-minute climate rows are not treated as
independent biological replicates.

The process-prior artifact records estimates, 80% and 95% intervals,
compartment counts, event counts, missingness, source DOI, cultivar, facility,
base temperature, and transfer limitations. Chengdu simulation configuration
uses the calibrated WUR maturity interval, the Chengdu source-site dry-matter
interval, and WUR picking cadence. This is transfer calibration, not Pidu
harvest calibration.

## Evaluation Protocol

The six Chengdu seasons remain the complete-season evaluation substrate:
2023-2025 spring and autumn, 120 days each. The rerun produces at least 12
paired parameter scenarios per season and reports:

- cumulative and batch fresh yield;
- first-harvest crop day and inter-pick interval;
- harvest-event count and peak daily harvest;
- standing and harvested fruit dry matter;
- dry-matter mass-balance error;
- season, parameter, and weather uncertainty intervals;
- external WUR plausibility distances;
- Pidu standing-crop WMAPE, bias, and evidence sufficiency.

The controller benchmark and crop-yield evaluation are reported in separate
tables. A controller may rank well on 4-day climate reward without a validated
yield advantage.

## Acceptance Gates

- Every completed controller run and expected model artifact is present.
- No harvested or allocated mass is negative and no state is non-finite.
- Short-window yield claims are labelled non-evaluable.
- WUR process-prior estimation covers at least five compartments and 50 valid
  development observations.
- Six Chengdu seasons are present, all mass-balance errors are at most
  `1e-9 kg m-2`, and every synthetic row is target-ineligible.
- External plausibility is reported separately from target accuracy.
- `target_validated` remains false until dated harvest masses from at least two
  complete Pidu seasons support a chronological held-out season.

## Authoritative Basis

- Vanthoor et al. (2011), tomato yield model description and validation,
  DOI `10.1016/j.biosystemseng.2011.08.005`.
- Katzin et al. (2020), GreenLight process-based greenhouse model,
  DOI `10.1016/j.biosystemseng.2020.03.010`.
- Hemming et al. (2020), WUR Autonomous Greenhouse Challenge tomato dataset,
  DOI `10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7`.
- Hemming et al. (2020), observed autonomous tomato production study,
  DOI `10.3390/s20226430`.

