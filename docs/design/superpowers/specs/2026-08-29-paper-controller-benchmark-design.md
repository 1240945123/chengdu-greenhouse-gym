# Chengdu Greenhouse Paper Controller Benchmark Design

## Objective

Build a complete, reproducible, and fair controller-comparison experiment for a
graduation thesis, with a paper-ready subset when the evidence is strong enough.
The main contribution is the comparison protocol on the Chengdu greenhouse
digital twin, not a predetermined ranking.

The study tests the hypothesis that a well-tuned reinforcement-learning
controller can outperform classical PID and MPC control. Classical controllers
must not be weakened, unfavorable seeds must not be hidden, and final claims
must follow the frozen results.

## Scope and Claims

The main comparison includes a fixed baseline, PID, MPC, PPO, SAC, and
PID-guided Residual-PPO. TD3, DDPG, A2C, and RecurrentPPO enter a low-budget
screening stage. Promoted methods receive the same full-training and evaluation
protocol as the existing RL methods.

Heating and CO2 remain disabled. All compared controllers own only roof
ventilation and independent fan control. Target-site harvest calibration is out
of scope because dated Pidu harvest events are unavailable. Crop carbon, fruit
state, and regional transferred yield are exploratory simulated indicators and
must not be described as measured Chengdu harvest accuracy.

## Evidence Roles and Time Scales

There is no universal greenhouse-control industry standard for episode length.
The public GreenLight-Gym task uses a 60-day production-control horizon, a
peer-reviewed greenhouse RL-MPC study uses 40-day cycles, and the WUR Autonomous
Greenhouse Challenge evaluates a complete six-month tomato crop. This study
adopts the horizon that matches its primary climate-control objective.

- Training episodes: 14 days, sampled from 2023-2024 spring and autumn weather.
- Model-selection episodes: the fixed non-overlapping 14-day windows in 2025
  spring.
- Primary temporal holdout: two consecutive, non-overlapping 60-day windows in
  2025 autumn, starting on day 226 and day 286, opened only after all
  selections are frozen.
- Descriptive robustness: one fixed 60-day evaluation at the start of each of
  the six configured 2023-2025 spring and autumn weather seasons.
- Complete cultivation-cycle evaluation: deferred until a target crop and
  harvest model is accepted.

Training-weather robustness rows must be labeled descriptive and must not be
presented as independent generalization evidence. Overlapping training windows
increase weather-state coverage but are not independent replicates.

Reference sources:

- GreenLight-Gym public task: https://github.com/BartvLaatum/GreenLight-Gym2
- RL-MPC 40-day evaluation: https://doi.org/10.1016/j.atech.2024.100751
- WUR six-month tomato challenge dataset:
  https://research.wur.nl/en/datasets/autonomous-greenhouse-challenge-second-edition-2019/

## Experimental Pipeline

### 1. Foundation repair

Fix rollout failure semantics before any new long training. Every attempted
step, including a failed ODE integration step, must produce an auditable failure
record. A rollout has exactly one terminal status: `complete`,
`numerical_failure`, `timeout`, or `invalid_action`.

Baseline and PID must complete every planned 60-day preflight scenario with
finite states before algorithm screening starts. Existing incomplete 120-day
baseline and PID trajectories are invalid and are excluded from thesis tables.

### 2. Classical reference

Keep the baseline fixed. Tune PID and MPC only with 2023-2024 training weather
and 2025 spring validation windows. Evaluate PID on the exact RL validation
windows to create the frozen relative safety reference.

### 3. RL screening

Screen TD3, DDPG, A2C, and RecurrentPPO with the same state, action, reward,
weather roles, low training budget, and multiple screening seeds. Preserve every
screening result, including non-promoted algorithms, for the thesis appendix.

Candidates first pass hard health gates: complete episodes, finite values, no
numerical failures, no critical safety errors, valid crop-carbon tolerance, and
no residual fallback. Safety non-inferiority requires both (a) mean active
material intervention no more than 0.05 absolute fraction above PID and (b) no
validation window more than 0.10 absolute fraction above its paired PID window.

Climate non-inferiority requires mean temperature-band MAE no more than 0.5 C
above PID, mean relative-humidity-band MAE no more than 3 percentage points
above PID, and mean joint comfort no more than 0.05 absolute fraction below
PID. These margins are frozen before the new screening trajectories are run.

Among candidates passing health, safety, and climate gates, maximize mean
validation reward. Candidates within 1% of the highest reward are ordered by
lower safety intervention, lower total action variation, and then lower
inference time. Algorithm-level screening ranks the mean across screening seeds,
not the best seed. Promote the top two new algorithms. Promote an additional
method only when its mean reward exceeds the better existing PPO/SAC reference
by at least 2%, the paired reward difference is positive in at least 75% of
seed-window pairs, and all non-inferiority gates pass.

### 4. Full training and selection

Train PPO, SAC, Residual-PPO, and promoted algorithms with seeds 0, 1, and 2.
The minimum full budget is 307,200 environment steps. Extend an algorithm to
614,400 steps only when its frozen validation learning curve has not plateaued.
Checkpoint and seed selection uses 2025 spring validation data only.

Freeze the algorithm registry, hyperparameters, checkpoints, selection rules,
PID safety reference, metric definitions, and protocol hashes before opening
the 2025 autumn holdout.

### 5. Final evaluation

Evaluate every frozen controller on both 2025 autumn 60-day holdout windows.
After the primary results are immutable, run the six-season 60-day descriptive
robustness evaluation. A method with any incomplete primary holdout rollout is
excluded from the formal main ranking and reported as a reliability failure.

## Metrics and Statistical Analysis

### Primary performance

- Mean daily reward and 60-day cumulative reward.
- Paired reward difference and percentage change relative to PID and MPC.
- An RL method is described as comprehensively superior only when reward
  improves without a preregistered material climate or safety degradation.

### Climate and control quality

- Temperature and relative-humidity MAE and RMSE.
- Joint comfort fraction, exceedance duration, cumulative exceedance magnitude,
  and severe-exceedance fraction.
- Separate day and night climate metrics.
- Mean actuator effort, total variation, action-rate distribution, and
  high-frequency switching count.
- Active material safety intervention fraction, projection magnitude, and
  reason counts.
- Controller inference time, MPC solve time, and real-time feasibility.

### Reliability and crop diagnostics

- Completion, early termination, ODE failure, invalid action, non-finite state,
  and fallback counts.
- Crop-carbon tolerance and simulated fruit-state change.
- Regional transferred yield, if retained, is labeled simulated and is neither
  a promotion gate nor evidence of target harvest accuracy.

### Statistical reporting

Use all three RL seeds in the main tables. Report mean, standard deviation,
median, interquartile range, and 95% confidence intervals. Compute paired
differences on identical weather windows and standardized effect sizes. Use a
hierarchical bootstrap over weather windows and RL seeds, and Holm correction
for multiple algorithm comparisons.

Because the primary holdout contains only two independent weather windows,
confidence and significance claims must remain conservative. Best-seed
trajectories are illustrative figures only and never replace all-seed tables.

## Code Boundaries

Add focused modules rather than expanding the current monolithic scripts:

- `paper_benchmark_protocol.py`: immutable evidence roles, windows, budgets,
  metrics, and thresholds.
- `paper_algorithm_registry.py`: uniform classical and RL algorithm adapters.
- `paper_screening.py`: low-budget experiments and promotion decisions.
- `paper_training.py`: full training, checkpointing, pause, and resume.
- `paper_evaluation.py`: validation, holdout, and robustness rollouts.
- `paper_statistics.py`: paired comparisons, confidence intervals, effect
  sizes, and multiplicity correction.
- `paper_report.py`: paper tables, figures, CSV, JSON, and Markdown reports.

## Artifacts and Provenance

Write new results under
`results/chengdu_agri_greenhouse_001/controller_benchmark/paper_benchmark_v1/`.
Do not overwrite V5 artifacts. The run root contains the resolved protocol,
source/data/model SHA256 hashes, classical tuning results, all screening rows,
checkpoints, frozen selections, trajectories, metrics, statistical outputs,
figures, timing data, and a reasonableness report.

Use temporary files followed by atomic replacement. Reuse an artifact only when
its row count, terminal status, protocol hash, and model identity all match.
Parallel task failures must preserve completed task outputs but prevent the
stage from being marked complete.

The workspace is initialized as a new local Git repository. The design document
is the first tracked artifact. Source tracking and ignore rules for datasets,
models, caches, and generated results are established before implementation
changes are committed.

## Resource Policy

Run at most three worker processes. Use below-normal process priority and a
bounded CPU affinity so the workstation remains responsive. Every long-running
unit supports checkpoints, pause, resume, and validated artifact reuse.

## Verification Gates

Long training cannot start until all of the following pass:

1. Failed-step recording and terminal-status unit tests.
2. Sixty-day trajectory completeness and solver-stability tests.
3. PID-relative safety-gate tests.
4. Multi-metric screening, promotion, and frozen-selection tests.
5. Short train/save/resume/evaluate tests for TD3, DDPG, A2C, and RecurrentPPO.
6. Checkpoint-resume equivalence tests.
7. Statistical aggregation and paper-table tests.
8. Baseline and PID 60-day preflights on every planned weather scenario.

## Completion Criteria

The benchmark is complete when every required algorithm has a frozen selection,
all primary holdout rollouts are either complete or explicitly classified as
failures, all metrics and statistical outputs are reproducible from immutable
trajectories, and the report distinguishes primary holdout evidence,
training-weather robustness, and simulated crop diagnostics.
