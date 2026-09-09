# Chengdu Greenhouse Controller Benchmark Design

## Objective

Build a reproducible, paper-ready benchmark for five greenhouse controllers:
rule-based baseline, PID, MPC, PPO, and SAC. All controllers must run closed-loop
against the same calibrated Chengdu greenhouse model and be evaluated with the
same weather windows, action constraints, reward, seeds, and metrics.

PINN and Transformer remain dynamics-prediction experiments. Their prediction
errors are reported separately and are not included as controller reward rows.

## Scientific Scope

The benchmark evaluates simulated control performance, not real-world causal
performance. Historical Chengdu measurements are used to construct weather,
calibrate the physics model, and define chronological train/validation/test
windows. Since historical logs contain only the actions that were actually
issued, they cannot directly estimate outcomes for alternative controller
actions.

Heating and CO2 injection are excluded, following the project scope. The full
GreenLight input vector remains six-dimensional for model compatibility, but
`uBoil` and `uCO2` are always zero. Controllers act on:

- `uThScr`: combined roof and side thermal-screen command
- `uVent`: roof-window, fan, and wet-pad ventilation proxy
- `uLamp`: supplemental lighting
- `uBlScr`: external-shade proxy

Every controller passes through the same action adapter, clipping, and slew-rate
rules. The output logs retain all six model controls so fixed and controlled
dimensions are auditable.

## Experiment Windows

The available weather file contains 31 days from 2026-04-05 through 2026-05-05.
Use four-day episodes so chronological separation is possible:

- Training: start days 0 through 17, sampled only from the training interval
- Validation: fixed start day 21
- Test: fixed start day 26

The final comparison uses the fixed test window and seeds `0, 1, 2, 3, 4`.
Validation data selects PID/MPC parameters and PPO/SAC checkpoints. Test reward
must not influence tuning or checkpoint selection.

## Environment And Reward

Create a dedicated benchmark environment configuration derived from
`ChengduSingleGreenhouseEnv.yml` with:

- calibrated `ChengduPhysics` backend
- 15-minute model step
- four-day season length
- four controlled inputs
- fixed zero heating and CO2 controls
- chronological training and evaluation weather scenarios
- common observations for all learning controllers

The existing economic `GreenhouseReward` is not the primary benchmark reward
because it assumes crop revenue, heating, and CO2 costs that are not supported by
the Chengdu dataset. Add a `ChengduClimateReward` with explicit, logged terms:

- temperature comfort error and out-of-bound violation
- relative-humidity comfort error and out-of-bound violation
- lighting electricity proxy
- action magnitude proxy
- action-change penalty

The reward is the negative weighted sum of these terms, so higher is better and
zero is the theoretical optimum. Temperature and humidity targets may follow a
day/night schedule, but they are identical for every controller. CO2 is logged as
a state diagnostic and contributes neither reward nor violation penalty.

## Controller Integration

### Rule-Based Baseline

Reuse the existing rule controller through a benchmark adapter. Force heating
and CO2 to zero before action submission. Existing thermal-screen, ventilation,
lamp, and blackout-screen logic remains the baseline policy.

### PID

Reuse the existing PID loops for ventilation and humidity control. The benchmark
adapter disables heating and CO2 outputs. PID gains are selected on validation
only from a small deterministic search space. The selected parameters and all
trial scores are saved.

### MPC

Use the existing lightweight receding-horizon controller with candidate actions
restricted to the four allowed inputs. Its objective must use the same climate
targets and cost priorities as the benchmark reward. Horizon and objective
weights are selected on validation only. Solver failure produces a logged,
deterministic fallback action rather than terminating the run.

### PPO And SAC

Train both Stable-Baselines3 agents with identical environment observations,
action space, training weather sampler, validation scenario, seed set, and total
environment-step budget. Use their algorithm-appropriate hyperparameters, save
periodic validation results, the best checkpoint, the final checkpoint, and
normalization statistics. Final evaluation is deterministic and uses the best
validation checkpoint.

For feasibility, provide a smoke profile and a paper profile. Smoke runs verify
the complete pipeline with a small step budget. Paper runs use the configured full
budget and five independent training seeds. A result is labelled paper-ready only
when every required seed and artifact is present.

## Benchmark Runner And Outputs

Add one orchestration entry point that can run `smoke`, `paper`, or selected
algorithms. It must support resuming completed RL runs and must not silently mix
profiles or stale checkpoints.

Store outputs under:

`results/chengdu_agri_greenhouse_001/controller_benchmark/<profile>/`

Required artifacts are:

- immutable resolved experiment configuration
- training metadata and learning curves for PPO/SAC
- selected PID/MPC parameters and validation trials
- per-step test trajectories, including states, controls, reward, and components
- per-episode metrics for every algorithm and seed
- aggregate comparison CSV and JSON
- reward, comfort, control-effort, and trajectory comparison plots
- machine-readable run manifest with completion and failure status

## Metrics

The primary metric is undiscounted cumulative test reward. Report mean, standard
deviation, median, and 95% bootstrap confidence interval across seeds.

Secondary metrics are:

- air-temperature MAE to target and violation hours
- relative-humidity MAE to target and violation hours
- joint comfort fraction
- mean and total actuator effort
- total action variation and switching count
- lamp-use proxy
- controller wall-clock time per environment step
- episode completion and numerical-failure counts

The report must state that seed repetitions over one test weather window quantify
algorithmic/model-initialization variability, not weather generalization.

## Reliability And Validation

Unit tests cover reward arithmetic, fixed-control enforcement, controller action
mapping, chronological scenario validation, metric aggregation, and manifest
completeness. Integration tests run short episodes for all five controllers and a
minimal PPO/SAC train-save-load-evaluate cycle.

Before a paper run, execute the focused benchmark tests and existing relevant
controller/environment tests. Existing unrelated legacy `tests/env_test.py`
interface failures remain documented and are not treated as benchmark failures.

The runner fails clearly for missing weather, invalid windows, non-finite states
or rewards, absent checkpoints, incompatible action dimensions, and incomplete
seed sets. Aggregation refuses to label incomplete results as paper-ready.

## Acceptance Criteria

The work is complete when:

1. All five algorithms finish the smoke profile end to end.
2. Heating and CO2 remain exactly zero in every recorded trajectory.
3. Every algorithm uses the same test scenario, reward, action limits, and metrics.
4. PPO and SAC can train, save, reload, and evaluate deterministically.
5. The benchmark produces auditable per-step data, aggregate tables, and plots.
6. A paper profile can be launched or resumed with five seeds per learning method.
7. The final report distinguishes completed smoke evidence from completed full
   training evidence and never presents partial runs as final results.
