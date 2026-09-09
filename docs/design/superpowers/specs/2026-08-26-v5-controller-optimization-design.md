# V5 Controller Optimization Design

## Objective

Optimize pure PPO, pure SAC, and MPC against the frozen PID controller in the V5
Chengdu hybrid greenhouse environment. An algorithm passes only when its reward is
strictly greater than PID on an untouched confirmation scenario while climate,
safety, numerical-stability, and crop-state gates also pass.

The experiment must not weaken PID, alter rewards by algorithm, select from the
confirmation result, or reuse the previously opened autumn smoke-test scenario for
tuning.

## Experimental roles

- Training: 2023-2024 Chengdu spring and autumn scenarios.
- Validation: 2025 spring start days 59, 74, and 89. These scenarios may be used
  repeatedly for checkpoint and hyperparameter selection.
- Confirmation: 2025 spring start day 104. This scenario is evaluated once after
  each algorithm's candidate is frozen.
- Historical smoke test: 2025 autumn start day 226 remains descriptive evidence
  only and is excluded from optimization.

All roles use the same V5 residual model, physics model, weather interface, safety
projector, action limits, reward implementation, time step, and episode duration.
PID is instantiated with its current frozen defaults.

## Candidate optimization

### MPC

Expose prediction horizon, discrete actuator levels, and four nonnegative cost
weights: temperature violation, humidity violation, actuator effort, and actuator
variation. Search a bounded candidate grid on all validation scenarios and select
the feasible candidate with the largest mean validation reward. Ties are resolved
by lower climate-band error, then lower actuator variation, then lower inference
time.

### PPO and SAC

Train each pure algorithm from multiple fixed seeds on the training scenario
sampler. Save model and observation-normalization state at cumulative checkpoints.
Evaluate every checkpoint on all validation scenarios and select one checkpoint per
algorithm by the same deterministic ranking rule. Confirmation results must not
change the selected seed or checkpoint.

The first optimization budget uses seeds 0 and 1 with checkpoints at 8,192, 16,384,
and 32,768 environment steps. Existing 8,192-step seed-0 smoke artifacts may be
reused only when their training metadata exactly matches the frozen protocol.

## Promotion gates

Selection feasibility requires completed episodes, finite values, zero numerical
failures, zero residual fallbacks, and nonnegative crop carbon within the existing
`1e-6 mg/m2` tolerance. Final promotion additionally requires:

- confirmation cumulative reward greater than frozen PID;
- temperature-band MAE no more than PID plus 0.25 C;
- RH-band MAE no more than PID plus 1 percentage point;
- safety-intervention fraction no more than PID plus 0.02;
- no crop-state or episode-completion regression.

Failure is retained and reported. It is not repaired by changing the baseline or
opening more confirmation scenarios. If pure PPO or SAC fails after this budget,
the next separately named experiment may evaluate Residual-PPO or Residual-SAC with
PID as the teacher/controller prior.

## Residual-PPO follow-up

The completed first-stage confirmation is immutable: MPC and SAC passed, while pure
PPO failed reward and temperature gates. The follow-up therefore keeps the pure PPO
failure in the report and trains a separately named `Residual-PPO` policy. At each
step, frozen PID supplies the base normalized incremental action and PPO supplies a
bounded residual; the executed proposal is `clip(pid_action + scale * residual,
-1, 1)`. The common environment reward and safety projection remain unchanged.

Residual-PPO checkpoint selection reuses only validation start days 59, 74, and 89.
Because start day 104 has already been opened, it cannot be used again as an
untouched claim set. A second, unified confirmation evaluates frozen PID, the
already selected MPC and SAC candidates, and selected Residual-PPO exactly once on
2025 spring start day 119. The result is reported as Residual-PPO, not pure PPO.

## Outputs

Store immutable candidate metadata, per-scenario trajectories, validation tables,
the selected configuration for each algorithm, confirmation tables, and a Markdown
reasonableness report under a new `v5_controller_optimization` result directory.
Report reward, temperature/RH band MAE, joint comfort fraction, safety intervention,
actuator effort and variation, inference time, episode completion, residual
fallbacks, carbon-state validity, and fruit-state change. One-day optimization
episodes do not support a yield claim.

## Testing and failure handling

Unit tests cover configurable MPC parameters, deterministic candidate ranking,
strict separation of validation and confirmation roles, promotion gates, checkpoint
metadata, and artifact compatibility. Small end-to-end tests train and reload both
RL algorithms. Long runs write each checkpoint atomically enough to resume from the
last complete candidate after interruption. The full repository suite runs before
results are accepted.
