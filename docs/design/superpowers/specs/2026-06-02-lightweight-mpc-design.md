# Lightweight MPC Controller Design

**Goal:** Add a lightweight model predictive controller as a deterministic, explainable comparison method alongside baseline, PID, PPO, and SAC.

**Scope:** This design implements candidate-action rolling-horizon MPC. It does not implement a full continuous nonlinear MPC optimization problem.

## Context

The project already supports deterministic controllers through a simple interface:

```python
controller.predict(ctx: StepContext) -> np.ndarray
```

The returned array is a raw six-element control vector:

```text
[uBoil, uCO2, uThScr, uVent, uLamp, uBlScr]
```

The existing baseline and PID evaluators run controllers with `normalize_actions=False`, step the `GreenLightEnv`, and save comparable per-timestep CSV files. MPC should follow the same interface and evaluation shape.

The environment exposes a CasADi GreenLight integrator through `env.F`, and each step uses:

```python
p_dyn = ca.vertcat(ca.DM(env.weather_data[env.timestep]), env.p)
res = env.F(x0=ca.DM(env.x), u=ca.DM(env.u), p=p_dyn)
```

The lightweight MPC will reuse this model for short lookahead prediction.

## Recommended Approach

Implement MPC as a standalone controller and evaluation entrypoint:

- `glassgym/components/mpc.py`
- `configs/agents/mpc.yml`
- `experiments/evaluate_mpc.py`
- `tests/mpc_controller.py`

This mirrors PID and avoids changing existing baseline/PPO/SAC/PID paths.

## Controller Design

### CandidateActionGenerator

`CandidateActionGenerator` produces a small, deterministic set of candidate raw controls in `[0, 1]`.

Inputs:

- current full control vector `ctx.u`
- step size values from config
- action bounds `[0, 1]`

Default candidates should include:

- hold current control
- heating low/medium/high
- ventilation low/medium/high
- CO2 low/medium/high
- lamp on/off
- a conservative combined climate action

The generator must keep the number of candidates small enough for a full 60-day episode to run in reasonable time.

### LightweightMPCController

`LightweightMPCController.predict(ctx)` does:

1. Generate candidate controls.
2. For each candidate, simulate a fixed control sequence over a short horizon.
3. Score the predicted trajectory using a lightweight stage objective.
4. Return the first candidate with the highest total score.

The first version should use constant control over the horizon rather than optimizing a sequence. This makes the controller faster and easier to test.

### Prediction Horizon

Default:

- `horizon_steps: 4`
- environment step `dt = 900 s`
- total lookahead = 1 hour

This is short enough to be feasible but long enough to capture near-term effects of heating, ventilation, CO2, and lighting.

### Stage Objective

The internal MPC score is a proxy objective, not the exact environment reward. The environment reward is still used for final evaluation.

Stage score:

```text
score =
  - temperature_error_weight * abs(t_air - temp_setpoint)
  - co2_error_weight * abs(co2_ppm - co2_setpoint) / 1000
  - rh_violation_weight * max(0, rh - rh_max) / 100
  - heat_cost_weight * uBoil
  - co2_cost_weight * uCO2
  - lamp_cost_weight * uLamp
  - action_change_weight * mean(abs(u - previous_u))
  + fruit_growth_weight * max(0, predicted_fruit_dw_growth)
```

The objective should be configurable in `configs/agents/mpc.yml`.

The stage objective intentionally stays simple and robust. It avoids directly invoking `GreenhouseReward` because the reward depends on observation modules and previous/current state bookkeeping that is awkward inside many hypothetical rollouts.

### Measurements

Use the same state conversions as the PID and rule-based controllers:

- Indoor air temperature: `x[2]`
- Indoor CO2 ppm: `co2dens2ppm(x[2], 1e-6 * x[0])`
- Relative humidity: `100 * x[15] / satVp(x[2])`
- Fruit dry weight proxy: `x[25]`
- Outdoor radiation: `d[0]`
- Day/night indicator: `d[8]`

### Setpoints

Default setpoints:

- daytime temperature: `19.5`
- nighttime temperature: `16.5`
- daytime CO2: `800 ppm`
- RH maximum: `85%`

Day/night uses weather disturbance `isDay` plus optional lamp time rule if needed.

## Evaluation Design

Add `experiments/evaluate_mpc.py` with the same CLI shape as `evaluate_pid.py`:

```bash
python -m experiments.evaluate_mpc \
  --env_id GreenLightEnv \
  --env_config glassgym/configs/envs/ \
  --mpc_config configs/agents/ \
  --n_sims 1 \
  --save_dir results/mpc/
```

The script should save:

```text
results/mpc/mpc_GreenLightEnv.csv
```

The output should include the same reward/info columns used by the baseline and PID scripts, plus scenario metadata:

- `sim`
- `location`
- `growth_year`
- `start_day`

## Testing Design

Add focused tests that do not require a full 60-day run:

1. Candidate generator includes the current control vector.
2. Candidate generator clips every candidate to `[0, 1]`.
3. `LightweightMPCController.predict(ctx)` returns shape `(6,)`.
4. MPC controls are all within `[0, 1]`.
5. A short environment rollout with MPC runs for several steps without crashing.
6. A tiny horizon/candidate config can evaluate quickly in unit tests.

The full episode is a verification run, not a unit test.

## Expected Result

MPC should generate a complete default-scenario CSV. It does not need to beat PPO or SAC. For论文对比, success means:

- deterministic and explainable control
- full-episode completion
- comparable CSV output
- reward summary can be reported alongside baseline, PID, PPO, and SAC

## Performance Constraints

Full-season evaluation has 5760 environment steps. If the default config uses 12 candidates and 4 horizon steps, the controller may perform about 276,480 model integrations. That is feasible but heavier than PID/baseline.

The config should therefore support smaller values for debugging:

- `horizon_steps: 1`
- fewer candidates

## Non-Goals

- No full nonlinear CasADi NLP solver in this task.
- No optimization over arbitrary continuous control sequences.
- No changes to PPO/SAC training code.
- No replacement of baseline or PID.

## Open Decisions Resolved

- MPC is built as a lightweight candidate-action rolling-horizon controller.
- The first version uses constant candidate actions over the horizon.
- The internal objective is a configurable proxy; final comparison uses true environment reward.
- MPC gets its own evaluation script and result directory.
