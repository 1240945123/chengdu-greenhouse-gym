# Plan: Stage A authoritative greenhouse tomato simulation and controller benchmark
_Locked via grill by Codex + user_

## Goal

Upgrade the repository from a six-season, 120-day diagnostic prior into a
paper-grade Stage A study of Chengdu greenhouse tomato production. Stage A will
use authoritative public observations, ERA5-Land weather, mechanistic
greenhouse and crop models, and clearly labelled simulations to pretrain and
stress-test the system before sufficient Pidu harvest observations exist. It
will model six approximately 180-day spring/autumn crop cycles from 2023-2025,
create at least 60 validated weather variants, estimate transferable crop and
harvest parameter distributions with a literature-standard Bayesian workflow,
and compare Baseline, PID, MPC, PPO, and SAC under one frozen protocol. Stage A
must not claim accuracy for real Pidu harvests; that claim remains gated on at
least two complete target seasons, one of which is genuinely unseen and acquired
after the validation protocol is frozen. The two 2025 seasons have already been
inspected in earlier repository experiments, so they are locked retrospective
paired case studies, not untouched tests or independent replicates supporting
population-level controller superiority claims.

## Approach

1. **Freeze provenance and experiment identity.**
   - Initialize repository version control or an equivalent immutable source
     manifest because the current folder has no Git metadata.
   - Preserve all existing 120-day outputs as a legacy diagnostic experiment;
     never overwrite them with the 180-day study.
   - Assign every data and experiment artifact a version, checksum, source,
     license, configuration snapshot, dependency manifest, seed, and run ID.
   - Write and checksum a machine-readable `ProtocolFreezeV2` before any locked
     2025 evaluation. It fixes data roles, estimands, metrics, acceptance gates,
     reward weights, exclusions, multiplicity families, controller settings,
     seeds, and the permitted report claims. It also records an exact dependency
     lockfile or container digest, Python/CasADi/native solver versions, OS,
     CPU/GPU, thread counts, numerical tolerances, and deterministic settings.
     Any amendment creates a new protocol version and invalidates confirmatory
     language for affected runs.
   - Store all environments, caches, temporary outputs, and large artifacts on
     the E: workspace drive.

2. **Build an authoritative evidence registry.**
   - Search peer-reviewed papers, institutional repositories, and official
     agricultural guidance for Sichuan protected-tomato crop calendars,
     greenhouse tomato growth/harvest observations, GreenLight parameters,
     and cultivar/management priors.
   - Include WUR Autonomous Greenhouse Challenge data, additional public
     greenhouse tomato datasets, Chinese greenhouse tomato studies, and
     Sichuan or Southwest China cultivation guidance where licensing and field
     semantics are auditable.
   - Keep source observations separate by dataset, greenhouse, cultivar,
     structure, area basis, and season. Never concatenate incompatible rows as
     exchangeable observations.
   - Apply the evidence hierarchy: Pidu target observations; comparable
     Sichuan/Southwest observations; other authoritative observations;
     peer-reviewed parameter priors; mechanistic simulations; expert
     assumptions. Lower levels cannot overwrite higher-level evidence.

3. **Resolve six complete Chengdu crop calendars.**
   - Derive spring and autumn planting/clearing ranges from authoritative local
     guidance rather than cutting each calendar year in half for convenience.
   - Create 2023-2025 spring/autumn cycles of approximately 180 days. Permit
     autumn/winter cycles to cross calendar years and fetch the required next
     year's weather.
   - Represent early, typical, and late planting as sensitivity scenarios when
     guidance provides ranges rather than one date.
   - Validate greenhouse occupancy intervals including clearing and sanitation
     buffers. If authoritative spring and autumn cycles overlap, run them as
     explicitly independent counterfactual simulations rather than claiming
     that one physical greenhouse produced both crops sequentially.
   - Record planting, flowering/fruit-set assumptions, first harvest, peak
     harvest, final harvest, and clearing dates. A fixed 120-day window remains
     only an early-yield diagnostic.

4. **Repair and version the simulation substrate before extending seasons.**
   - In `glassgym/environments/utils.py`, keep source cadence and requested solver
     timestep as separate variables; resample onto an exact half-open target
     grid with `ceil((season + forecast horizon) * 86400 / requested_dt)` rows.
     Resampling is schema-specific: conservatively allocate accumulated rainfall
     and energy, preserve integrals for interval-mean radiation/flux variables,
     use bounded interpolation for state variables such as temperature and RH,
     and use declared hold/transition semantics for actions. Test energy/water
     integral conservation, bounds, event peaks, and no rain smearing across dry
     intervals.
     Add tests for hourly sources at 15-minute and hourly solver steps, leap
     years, cross-year seasons, forecast padding, missing next-year files, and
     exact endpoint alignment.
   - In `glassgym/components/weather.py`, replace the current
     `(location, year, start_day)` cache key with an immutable key containing
     location, year, start, season length, forecast horizon, timestep,
     disturbance count, loader/schema version, and source-file checksums. Return
     read-only arrays or copies so one environment cannot mutate another.
   - Introduce versioned `WeatherScenarioV2`, `DisturbanceSchemaV2`, and
     `ControllerBenchmarkV2` manifests. Keep the present 120-day profile as a
     read-only legacy schema and add explicit migration readers; never reinterpret
     old artifacts under V2 semantics.
   - Remove the hard-coded 120-day assertion from
     `experiments/controllers/benchmark_protocol.py` in V2 only. Migrate and
     test `configs/benchmarks/chengdu_controllers.yml`, protocol loading,
     fingerprints, completeness checks, `run_chengdu_benchmark.py`, report
     labels, and output namespaces for explicit season manifests and roles.
   - In `glassgym/environments/greenlight_env.py`, replace the bare exception
     handler with explicit integration/numerical exceptions. On failure, do not
     compute a normal reward or observation from stale state: emit structured
     metadata, set `truncated=true`, and terminate that rollout. Derive the
     failure continuation penalty from frozen finite per-step reward bounds so
     it is lower by `epsilon` than the worst valid discounted continuation from
     the same prefix for every remaining horizon. Exclude failed evaluation
     rollouts from normal ranking, report them as failures, and reject any paper-
     profile policy with a nonzero failure rate. Add adversarial policies that
     try to gain return by numerical failure, plus finite-state, solver,
     timeout/fallback, and vectorized propagation tests.
   - Gate: no 180-day data generation or controller work begins until exact
     weather length/cache isolation tests and solver-failure integration tests
     pass alongside the unchanged legacy 120-day regression suite.

5. **Acquire and validate historical weather.**
   - Use hourly ERA5-Land as the primary historical source, retaining UTC and
     converting a separate timestamp to Asia/Shanghai.
   - Use on-site weather, authoritative Chinese station data, or NASA POWER as
     comparison/bias-audit sources when available; do not silently substitute
     them for ERA5-Land.
   - Enforce continuous time grids, unit/range checks, cross-year continuity,
     no duplicate timestamps, and complete source manifests.
   - Assign roles independently within each pipeline rather than treating years
     as universal roles. Weather/controller uses pre-2023
     `historical_fit_reference`, 2023 `fit`, 2024 `select`, and 2025
     `retrospective`. Local climate uses contiguous `climate_fit`,
     `climate_select`, and `climate_holdout` blocks inside sufficiently covered
     on-site records. External crop data uses source-level `crop_prior_fit` and
     rotated `crop_transfer_holdout` folds. Future Pidu harvest data uses
     `target_update` and a later, genuinely unseen `independent_test` season.
     No artifact descendant may move to a more permissive role.
   - Create a machine-readable lineage matrix with one row per artifact,
     pipeline namespace, role, parent checksums, and permitted operations.
     Weather/forecast distributions may use `historical_fit_reference` plus
     weather `fit` and are checked/tuned only on weather `select`; controller
     training uses controller `fit`; model and hyperparameter choice uses the
     corresponding `select`; each crop transfer fold excludes its held-out
     source from prior estimation. Every 2025 access is logged and
     non-decisional after `ProtocolFreezeV2`; prior access is audited.

6. **Generate at least 60 weather variants without leakage.**
   - Generate at least ten temporally coherent variants per base season while
     preserving diurnal cycles, autocorrelation, cross-variable dependence,
     seasonal trends, and physically valid temperature, humidity, radiation,
     wind, and rainfall combinations.
   - Add explicit heatwave, consecutive-cloudy/high-humidity, wind, and heavy
     rain stress scenarios.
   - Validate monthly/seasonal moments and quantiles, autocorrelation,
     cross-correlation, extremes, event duration, and physical constraints
     against held-out historical weather using methods justified by primary
     literature.
   - Train controllers only on 2023-origin variants. Use 2024-origin variants
     for model selection and 2025-origin variants only for post-freeze,
     descriptive robustness analysis. The Stage A main retrospective benchmark
     uses the unperturbed 2025 ERA5-Land seasons.

7. **Calibrate and gate the greenhouse climate model.**
   - Calibrate the Chengdu physical climate backend using real indoor/outdoor
     observations, preserving a chronological held-out interval.
   - Before calibration, require aligned indoor/outdoor/action observations
     spanning at least one complete spring and one complete autumn/winter crop
     period, at least 90% of expected timestamps in every split, at least 30
     independent hourly windows in each predeclared hot, cold, high-humidity,
     high-radiation, and high-wind regime, and at least 100 separated transitions
     with nontrivial range for each calibrated actuator. Record sensor accuracy,
     maintenance, outages, and action saturation. If a condition is unmet,
     narrow the validated domain to supported regimes or defer calibration; the
     current approximately 2,650-hour 2026 record alone cannot establish annual
     or autumn/winter validity.
   - Compare ChengduPhysics with PINN and Transformer predictors using rolling
     temporal validation at one-step, 24-hour, and 72-hour horizons.
   - For the mechanistic model, infer a declared sensitive parameter subset and
     state uncertainty with a Bayesian state-space model: bounded physical
     priors, sensor-accuracy observation terms, multivariate Student-t transition
     residuals, elapsed-time discrepancy correlation, and explicit initial-state
     uncertainty. Fit only on `climate_fit`, choose structure on
     `climate_select`, and reserve `climate_holdout` for frozen checks. Use the
     MCMC diagnostics and synthetic-recovery requirements in Step 10 and persist
     parameter, state, and discrepancy draws as the climate posterior. If this
     model or its diagnostics fail, do not relabel a bootstrap ensemble as a
     posterior and block the paper profile.
   - Report MAE, RMSE, bias, interval coverage, extreme-value error, comfort
     classification, and long-rollout drift.
   - Climate model gates are: one-step temperature/RH MAE <= 1.5 C/5 %RH;
     24-hour <= 2.0 C/7 %RH; 72-hour <= 2.5 C/10 %RH. Cite or justify every
     configured threshold. Failure of coverage, excitation, or accuracy gates
     blocks paper-profile controller comparisons, rankings, and publication
     claims; only labelled diagnostic plumbing runs may proceed.

8. **Perform multi-weather global sensitivity analysis.**
   - Apply variance-based global sensitivity analysis across multiple years,
     planting dates, and weather conditions before crop parameter calibration,
     using Sobol only when the declared latent inputs are independent, following
     greenhouse tomato uncertainty literature such as
     https://doi.org/10.1016/j.compag.2024.109324.
   - Include GreenLight/Vanthoor growth, phenology, allocation, temperature
     response, dry-matter, maturity, and harvest parameters in the candidate
     set. Freeze insensitive parameters at documented values and calibrate only
     identifiable sensitive parameters.
   - Pre-register a dimensionless output vector covering first-harvest timing,
     weekly and final standing-fruit/organ biomass, weekly and cumulative
     harvest, and climate-stress integrals, with each output scaled by its
     declared observational or physiological range and seasons weighted equally.
     Use latent-space Saltelli/Sobol screening only when the transform is
     component-wise, so each latent factor maps to exactly one physical
     parameter. If the valid joint physiological prior requires a transform that
     mixes parameters or otherwise imposes dependence, compute dependence-aware
     effects such as physical-parameter Shapley effects and freeze parameters
     only from those parameter-level indices; never attribute mixed latent
     indices to individual physical parameters. Record the joint prior,
     transform, and attribution rule. For the Saltelli path, double the base
     sample size from 512 until two successive doublings give
     maximum absolute total-effect change < 0.02 and top-effect rank correlation
     > 0.95. Bootstrap sensitivity uncertainty by base sample.
   - Freeze a parameter only when the upper 95% bootstrap bound of its total-
     effect index is < 0.01 for every predeclared output and season; otherwise
     retain, pool, or investigate it. Report first-order/total-order indices,
     interactions, convergence, and sensitivity stability across seasons.

9. **Build and independently validate a Bayesian emulator.**
   - Use space-filling Latin hypercube simulator runs over sensitive parameter
     priors and weather conditions to train a Gaussian-process or comparably
     interpretable emulator.
   - Select the emulator by predeclared, held-out parameter-weather tests; do
     not choose it on posterior results. Require held-out R2 >= 0.98 and
     normalized RMSE <= 5% for each calibrated output, plus interval calibration
     and acceptable worst-case residuals.
   - Estimate output- and region-specific emulator error covariance solely from
     independent held-out simulator runs, freeze or tightly inform it before
     fitting observations, and propagate it through the Bayesian likelihood.
     Estimate source-transfer discrepancy separately only after this freeze.
     Verify posterior regions with fresh full GreenLight runs. Fall back to
     direct or sequential simulation if the emulator gate fails.

10. **Estimate transferable parameters with hierarchical Bayesian MCMC.**
   - Use physiologically bounded priors from GreenLight and peer-reviewed tomato
     studies, with dataset-, greenhouse-, cultivar-, and structure-level random
     effects. Reserve a future Pidu target level for Stage B updating.
   - Pre-register the executable observation model before fitting. Positive
     organ dry masses and standing fruit mass use source-specific log-scale
     Student-t longitudinal likelihoods around emulator/full-model predictions.
     Primitive harvest increments use a hurdle/censoring model: structural zeros
     before harvest readiness or at a declared no-pick event, observed zero or
     below-detection batches as left-censored measurements with source-specific
     detection limits, and positive batches as log-scale Student-t observations.
     Missing records remain distinct from every zero class. Phenology/timing uses
     interval-censored day likelihoods; indoor temperature and vapor pressure
     use a bivariate residual model. Irregular longitudinal residuals use an
     elapsed-time continuous AR(1)/Ornstein-Uhlenbeck correlation
     `corr(i,j) = exp(-abs(t[i]-t[j]) / tau)`; discrete AR(1) is allowed only on
     verified regular grids with explicit gap segmentation. Source-reported
     measurement error is fixed where documented;
     unknown observation scales receive weakly informative half-normal priors.
     Missing observations contribute no likelihood and are never zero-filled.
     Cumulative harvest, as a deterministic sum of batch increments, is used
     only for posterior predictive checks unless a source supplies an
     independently measured cumulative quantity with declared joint covariance.
   - Parameter hierarchy is defined on transformed bounded scales:
     global tomato effects plus dataset, cultivar, greenhouse-structure, and
     season deviations with regularizing non-centered random effects. Source
     influence follows declared measurement/model-discrepancy variances, not an
     arbitrary row weight. Emulator discrepancy and source-transfer discrepancy
     are separate variance terms so one cannot hide the other.
   - Run prior predictive checks before fitting and weak/informative-prior
     sensitivity analyses after fitting.
   - Use at least four chains and rank-normalized split R-hat, bulk/tail ESS,
     MCSE, rank/trace plots, energy diagnostics, and posterior predictive checks
     following Vehtari et al. and Stan/PyMC official guidance. Formal outputs
     require R-hat < 1.01, bulk and tail ESS >= 400, zero divergences, and no
     maximum-tree-depth failures; otherwise reparameterize rather than merely
     extending chains.
   - Retain bootstrap only as a robustness comparison. Persist posterior draws,
     diagnostics, predictive draws, prior sensitivity, and full-simulator
     verification results.
   - Before real fitting, require parameter-recovery tests on synthetic data,
     simulation-based calibration rank diagnostics, prior predictive physical
     bounds, and identifiability checks. A prior-dominated or non-identifiable
     parameter is fixed, pooled more strongly, or reported unresolved rather
     than promoted to a calibrated result.

11. **Validate crop and harvest transfer without circular evidence.**
    - Gate crop growth before harvest fitting: at least three observed growth
      dates, standing-fruit WMAPE <= 20%, phenology WMAPE <= 10%, and organ
      biomass NRMSE <= 25%. A harvest parameter may not compensate for a failed
      crop-growth trajectory.
    - Define one authoritative fruit ledger. GreenLight supplies gross fruit
      allocation and source non-harvest loss fractions; the cohort ledger alone
      owns standing mature fruit and scheduled harvest removal. GreenLight's
      native harvest flux is diagnostic and must not also reduce cohort mass or
      contribute to reward. Prove per-step and whole-season conservation:
      initial fruit + allocation - non-harvest loss = standing fruit + cohort
      harvest, with a configured numerical tolerance and failure-on-breach.
    - Perform complete-season temporal validation inside each external dataset
      and leave-one-dataset-out transfer validation. When WUR is held out, omit
      it from prior estimation for that fold.
    - Report first/peak/final harvest dates, batch and cumulative WMAPE/MAE/RMSE,
      final bias, R2, timing Wasserstein distance, interval coverage/width,
      calibration scores, and failure cases. Simulated Chengdu yields are
      mechanism checks, not observed accuracy evidence.

12. **Extend the environment through one common safety layer.**
    - Primary action space remains roof vent, shading screen, thermal screens,
      and supplemental lighting with heating and CO2 permanently disabled.
      First remove the current uncalibrated 18 kW fan-pad cooling term from
      `uVent` in `glassgym/models/ChengduPhysics/ode.py`; `uVent` represents roof
      ventilation only in V2. Preserve the old coupled behavior only in the
      explicitly named legacy backend for reproduction.
    - Add a secondary wet-pad experiment only after calibrating fan/pump power,
      cooling, and humidification. Extend DisturbanceSchemaV2 with precipitation
      and ActionSchemaV2 with independent `uPadFan` and `uPadPump`; define units,
      defaults, bounds, missing-data policy, observation exposure, energy terms,
      state equations, serialization, and backward-compatible readers. Enforce
      fan-pump interlocking. Irrigation is not a control action until root-zone
      water and water-stress models exist.
    - Apply universal bounds, slew limits, rain/wind roof restrictions,
      high-humidity wet-pad restrictions, lighting schedules, timeout handling,
      finite-value checks, and rule-baseline fallback below every controller.
      Freeze one deterministic lexicographic projection: invalid/late proposals
      invoke fallback; emergency weather constraints define the feasible set;
      equipment bounds and fan-pump interlocks are enforced; schedules are
      applied; then normal slew limits and minimum-distance tie-breaking are
      applied. Emergency closure may use a separately specified emergency slew
      rate. Define conflict rules and require deterministic, idempotent, bounded,
      interlock-preserving property tests for every controller adapter.
    - Log proposed actions, executed actions, safety interventions, solver
      failures, and fallback duration.

13. **Freeze the reward, forecast, and controller protocol.**
    - Define the exact per-step reward as time-scaled comfort, resource,
      action-change, and constraint terms plus `w_h * (H[t+1] - H[t])`, where
      `H` is cohort-ledger cumulative harvest, and optional potential-based
      shaping `gamma * P(s[t+1]) - P(s[t])`. Set terminal potential to zero and
      include no separate terminal cumulative-harvest reward. Save every
      component and prove by tests that shaping does not alter the frozen base
      objective or controller ordering under equivalent timestep, horizon, and
      discount conventions.
    - Freeze Baseline rules before retrospective evaluation. Tune PID and MPC
      only on `fit` and `select` seasons, preserving default and tuned results
      plus all search trials.
    - Give every controller the same observations, action/safety interface,
      weather, initial crop state, horizon, and disabled-control constraints.
      Version `InitialCropStateV2` with transplant age, thermal time, organ dry
      masses, leaf area, standing-fruit/cohort inventory, cultivar, measurement
      source or prior-draw ID, units, and covariance. Derive season-specific
      states from measurements where available or frozen biological priors;
      sample paired state draws once per scenario and reuse them for every
      controller. Reject mass-inconsistent or phenologically impossible states.
    - Separate hidden `PlantModelV2` dynamics and posterior-draw parameters from
      a frozen `ControllerModelV2`. MPC receives only permitted observations,
      estimated state, the issue-time forecast, and controller-model parameters
      frozen from fit/select data; it cannot access `env.F`, true `ctx.p`, hidden
      plant state, or retrospective draw IDs. Poison latent plant parameters and
      assert unchanged MPC proposals when permitted observations are held fixed.
    - MPC prediction transitions execute the identical versioned safety
      projection, actuator dynamics, delays, and fallback semantics used by the
      plant interface, using only information available at forecast issue time.
      Test predicted versus executed action identity and constraint handling for
      every safety rule.
    - Evaluate MPC under a realistic common weather forecast in the main benchmark
      and a separately labelled perfect-forecast upper bound. Report horizon,
      replanning interval, solve time, timeout rate, and infeasibility.
    - Replace direct access to `ctx.d[t+k]` and `WeatherForecastObservations`
      realized-future slices with a `ForecastProviderV2` interface. Each forecast
      carries issue time, valid time, lead, source/model version, and ensemble or
      error-model ID. The main provider is fitted on
      `historical_fit_reference` plus 2023 `fit` forecast errors, selected and
      frozen on 2024 `select`, and cannot read 2025 realized values after issue
      time. The perfect-forecast provider is a separate upper-bound experiment.
      Add sentinel tests that poison future ERA5 values and prove retrospective
      MPC/agents cannot observe them, plus forecast-horizon exhaustion and
      missing-issue fallbacks.

14. **Train PPO and SAC reproducibly.**
    - Use smoke, development, and paper profiles. Only frozen paper-profile
      runs enter final tables.
    - Train at least five seeds per algorithm with identical scenario and step
      budgets. Select hyperparameters and a labelled seed using `select` data
      only; never inspect retrospective outcomes for selection.
    - Main tables report all five seeds, mean, standard deviation, minimum, and
      maximum. A predeclared Student-t 95% interval around the seed mean may be
      shown only as a seed-variability summary conditional on each retrospective
      season, not as a population or biological confidence interval. A secondary
      table reports the `validation-selected seed`, without calling it a
      performance ceiling or using it for primary conclusions.
    - Save checkpoints, normalization statistics, training curves, evaluation
      trajectories, failures, and configuration fingerprints.

15. **Run a frozen five-controller benchmark.**
    - Compare Baseline, PID, MPC, PPO, and SAC on identical original 2025 spring
      and autumn locked retrospective seasons, then on post-freeze 2025-origin
      weather variants and stress scenarios.
    - Report total fresh harvest, estimated marketable yield only as a labelled
      sensitivity quantity, batch uniformity, first/peak/final harvest timing,
      climate violations, actuator/resource use, action variation, switches,
      safety interventions, reward components, runtime, and robustness.
    - Form paired records by weather, initial state, and evaluation seed, but do
      not treat seeds or perturbations descended from one base season as
      independent biological/weather replicates. For the two locked 2025
      seasons, report complete paired raw differences and within-case seed
      variability as descriptive case studies only. Do not report population-
      level p-values, Holm-adjusted superiority claims, or base-season bootstrap
      confidence intervals until additional independent held-out years exist.
      Separate spring and autumn conclusions.
    - Propagate epistemic uncertainty in stratified batches of 30 draws from the
      frozen crop/climate posterior plus explicit model-discrepancy scenarios,
      reusing each draw across all controllers. Freeze a discrepancy-innovation
      path and seed for every paired scenario and replay the identical process-
      noise realization for every controller; report parameter uncertainty and
      process/discrepancy uncertainty as separate crossed factors. For each
      predeclared pairwise
      metric, robust direction requires at least 0.95 of paired draws to have the
      same sign, binomial Monte Carlo standard error <= 0.02, and probability
      change < 0.02 across two successive batches. Increase to at most 300 draws;
      otherwise report the comparison as precision-limited and non-robust.
      Report posterior-median-parameter results separately from paired draw
      distributions, and never pool parameter draws with weather cases or RL
      seeds as independent replicates.

16. **Produce publication and audit artifacts.**
    - Generate immutable raw/processed manifests, evidence and data-quality
      matrices, weather validation, Sobol indices, emulator diagnostics, MCMC
      diagnostics, posterior checks, external transfer validation, controller
      trajectories, paired metrics, uncertainty tables, ablations, figures, and
      an experiment-flow browser report.
    - Run ablations for external observations only, simulations only, external
      plus simulation, hierarchical versus pooled calibration, physical versus
      PINN/Transformer climate prediction, controller safety, and reward terms.
    - State prominently that Stage A is migration pretraining and simulation.
      Keep `target_validated=false` until Stage B has at least two complete Pidu
      harvest seasons and passes the locked independent-test gates.

17. **Execute through file-level, resumable work packages.**
    - Package A, weather substrate: modify `glassgym/environments/utils.py`,
      `glassgym/components/weather.py`, and versioned schemas/config loaders; add
      unit tests for grids/cache/checksums and integration tests for 180-day
      cross-year episodes. Rollback is selection of the unchanged legacy schema.
    - Package B, protocol and safety: modify benchmark config/protocol/runner,
      `greenlight_env.py`, reward failure semantics, action/disturbance schemas,
      and controller adapters; add leakage sentinels, failure propagation,
      fallback, and legacy trajectory regression tests.
    - Package C, evidence/weather data: add source adapters, immutable manifests,
      calendar registry, lineage matrix, weather generator, validation report,
      and fit/selection/retrospective role enforcement. No downstream package
      consumes an artifact whose audit gate is false.
    - Package D, crop inference: add versioned observation schemas, Sobol runner,
      emulator dataset/trainer/validator, hierarchical model specification,
      climate and crop state-space/hierarchical inference, synthetic recovery/SBC,
      MCMC diagnostics, posterior verification, and leave-one-source-out
      evaluation.
    - Package E, fruit ledger: implement and test the single cohort-owned harvest
      ledger, reward integration, conservation assertions, event schemas, and
      180-day crop/harvest reports.
    - Package F, control: implement ForecastProviderV2, PlantModelV2 versus
      ControllerModelV2 isolation, safety-aware frozen Baseline/PID/MPC,
      PPO/SAC smoke/development/paper profiles, checkpoints, latent-parameter
      leakage tests, and paired descriptive/epistemic evaluation.
    - Package G, publication: generate tables, figures, ablations, limitations,
      browser flow, reproduction script, and final evidence audit.
    - Every package follows failing-test-first implementation, writes to a new
      schema/run namespace, supports restart from atomic shards, and has a
      go/no-go report. Estimate CPU/GPU hours and disk before its paper run;
      exceedance requires a versioned budget revision, not silent sample cuts.

## Key decisions & tradeoffs

- **Stage A first:** useful simulation and transfer work proceeds now, while
  real Pidu harvest validation is deferred rather than fabricated.
- **External and simulated data may train the model:** evidence classes remain
  separate, and only target observations can support the later Pidu accuracy
  claim.
- **Complete seasons replace 120-day main windows:** approximately 180-day
  locally sourced calendars improve yield interpretation but require cross-year
  weather and materially more compute.
- **Authoritative Bayesian workflow:** multi-weather Sobol screening, a
  independently validated emulator, hierarchical Bayesian MCMC, and posterior
  predictive checks replace point calibration as the main method. This is more
  expensive and complex but supports transfer and uncertainty claims.
- **Mechanistic main model, learned comparisons:** ChengduPhysics and GreenLight
  remain the main interpretable chain; PINN and Transformer are forecast
  comparisons, not substitutes for harvest evidence.
- **Paired controller protocol:** all controllers share one environment and
  safety interface. A validation-selected RL seed may be shown only as a
  secondary labelled result; primary conclusions use all seeds.
- **Four-action primary study, wet-pad secondary study:** this preserves current
  comparability while allowing a calibrated realistic cooling extension.
- **Total harvest is primary:** marketable yield remains an explicitly modelled
  sensitivity estimate until fruit grading observations exist.
- **Chronological and source-level isolation:** no random row splits, no
  2025-derived weather in fitting or selection, and no source can serve
  simultaneously as prior evidence and an independent held-out fold.

## Risks / open questions

- Authoritative Sichuan crop-calendar sources and additional reusable harvest
  datasets may be sparse, paywalled, differently licensed, or semantically
  incompatible. The evidence registry must report gaps instead of silently
  harmonizing them.
- Six recent base seasons alone do not characterize long-term weather. The
  weather generator may need a longer pre-2023 ERA5-Land reference period while
  preserving the frozen 2023-fit/2024-select/2025-retrospective roles.
- The current GreenLight crop model may not represent low-technology Pidu
  greenhouse physiology, water stress, wet-pad behavior, or cultivar effects.
  Model discrepancy must remain explicit in the posterior and conclusions.
- A Gaussian-process emulator may not scale or pass the worst-case gate. The
  fallback is a smaller sensitive parameter set, local emulators, active
  learning, or direct sequential calibration, not relaxed validation.
- Bayesian parameters may remain non-identifiable with available external data.
  Strong posterior correlations or prior-dominated posteriors must be reported;
  they cannot be hidden by a point estimate.
- Sixty 180-day weather scenarios, Sobol runs, MCMC verification, and ten RL
  training groups are computationally expensive. Runs need resumable shards,
  checkpoints, bounded worker counts, and early smoke/development gates.
- Controller rankings may be sensitive to model error and reward design. Stress
  tests, reward-component audits, model ensembles, and Stage B target data are
  required before operational claims.
- Exact citations and thresholds for weather-generation validation, emulator
  calibration, and crop/harvest acceptance must be finalized during the source
  audit and versioned with the corresponding configuration.

## Out of scope

- A claim that Stage A accurately predicts real Pidu harvests.
- Online deployment to greenhouse equipment or autonomous field trials.
- Irrigation optimization, root-zone water control, or water-stress policy
  learning.
- Heating and CO2 control.
- Unqualified marketable-yield claims without grading, fruit size, defect, or
  quality measurements.
- Tuning any model, controller, reward, source weight, or seed on test-season
  outcomes.
- Replacing missing target harvest dates, batch weights, or areas with invented
  values.
- Stage B target updating, independent Pidu harvest validation, and subsequent
  safety-controlled field deployment.
