# Reproductive Heat-Stress Yield Design

## Objective

Add a literature-constrained reproductive heat-stress layer to the existing
120-day Chengdu cohort projection. The layer explains potential fruit dry matter
that does not become set fruit under chronic warm conditions without fitting a
post-hoc multiplier to the Pengzhou yield target.

## Evidence

- Sato, Peet, and Thomas (2000), DOI
  `10.1046/j.1365-3040.2000.00589.x`, compared optimal 28/22 C and chronic
  32/26 C regimes and attributed reduced tomato fruit set primarily to pollen
  release, viability, and anther development.
- Peet, Willits, and Gardner (1997) reported reproductive decline as daily mean
  temperature rose through approximately 25-29 C and identified the pre-anthesis
  period as temperature-sensitive.
- Pham et al. (2020), DOI `10.1016/j.envexpbot.2020.104150`, found that prolonged
  35 C stress damaged about half of pollen viability and demonstrated strong
  genotype dependence.
- Boote et al. (2012), DOI `10.21273/hortsci.47.8.1038`, showed that revised
  temperature cardinal parameters improved CROPGRO-Tomato predictions of fruit
  number and fruit dry matter across independent experiments.

These sources constrain model shape and sensitivity scenarios. They are not
target-greenhouse calibration observations and remain target-ineligible.

## Model

For each hourly fruit-allocation increment, calculate the trailing 10-day mean
air temperature. Reproductive retention is 1.0 at or below 25 C, decreases with
a smoothstep response between 25 and 29 C, and reaches a scenario-specific floor
at or above 29 C. Three explicit sensitivity scenarios use floor retentions of
0.25, 0.50, and 0.75 and are labelled heat-sensitive, central, and heat-tolerant.
These are sensitivity levels, not confidence intervals.

The retained fraction enters the existing age-structured harvest cohort. The
rejected fraction is recorded as reproductive heat-stress dry-matter loss. The
balance becomes:

`initial + gross allocation - source loss - reproductive stress loss - standing - harvested = 0`.

Relative humidity and VPD exposure are reported diagnostically. They do not alter
yield in v3 because the available regional evidence does not provide a defensible
quantitative target response for this greenhouse and cultivar.

## Outputs

- A source registry for reproductive-stress evidence.
- A versioned v3 stress configuration.
- `regional_v3_` scenario, episode, season, and algorithm projection artifacts.
- Stress exposure and loss summaries by algorithm and season.
- A v3 audit comparing timing, yield, spring plausibility, and mass balance with
  v2 while retaining all v1/v2 files.

## Acceptance

- Retention is finite, bounded in [floor, 1], monotonic with temperature, and
  continuous at 25 and 29 C.
- No-stress mode reproduces the v2 cohort result.
- All three stress scenarios reuse the same v2 maturity and harvest parameters.
- Every canonical controller episode produces all required scenarios.
- Maximum dry-matter balance error remains below `1e-9 kg/m2`.
- Reports separate potential yield, stress-adjusted yield, and reproductive loss.
- No artifact is labelled target-validated.

