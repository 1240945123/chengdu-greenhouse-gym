# PID Controller Design

**Goal:** Add a PID-based greenhouse controller as a non-learning comparison method for the existing baseline, PPO, and SAC experiments.

**Scope:** This design covers PID only. MPC will be designed separately after PID can run a full episode and produce comparable reward CSV output.

## Context

The project already has a rule-based controller in `glassgym/components/rule_based.py`. It exposes a simple controller interface:

```python
controller.predict(ctx: StepContext) -> np.ndarray
```

The returned array is a raw six-element control vector:

```text
[uBoil, uCO2, uThScr, uVent, uLamp, uBlScr]
```

`experiments/evaluate_baseline.py` evaluates this controller by setting `normalize_actions=False`, resetting the environment for each weather scenario, calling `predict(ctx)` at each step, stepping the environment, and saving a per-timestep CSV.

PID should follow the same shape so it can be compared directly with:

- `results/baseline/rb_baseline_GreenLightEnv.csv`
- `results/rl/rl_ppo_GreenLightEnv.csv`
- `results/rl/rl_sac_GreenLightEnv.csv`

## Recommended Approach

Implement PID as a standalone controller and evaluation entrypoint:

- `glassgym/components/pid.py`
- `configs/agents/pid.yml`
- `experiments/evaluate_pid.py`
- `tests/pid_controller.py`

This keeps PID separate from the existing rule-based baseline and avoids changing the already working baseline/PPO/SAC reproduction path.

## Controller Design

### PIDLoop

`PIDLoop` is a small reusable class that computes a bounded PID output from:

- setpoint
- measured value
- timestep length `dt`

It stores:

- proportional gain `kp`
- integral gain `ki`
- derivative gain `kd`
- output bounds
- integral state
- previous error
- optional integral bounds for anti-windup

The sign convention should be explicit:

```text
error = setpoint - measurement
output = kp * error + ki * integral + kd * derivative
```

For variables where high measurement should increase the output, such as ventilation for high humidity, the controller can pass reversed error inputs or use a negative gain. The implementation should prefer positive gains and explicit error direction for readability.

### PIDController

`PIDController.predict(ctx)` returns a raw control vector in `[0, 1]` with length 6.

The first version should prioritize stable full-episode execution and interpretable comparison, not optimal greenhouse tuning.

Control mapping:

| Output | Strategy |
|---|---|
| `uBoil` | Temperature PID. Increase heating when indoor air temperature is below day/night setpoint. |
| `uCO2` | CO2 PID. Increase dosing when indoor CO2 ppm is below daytime setpoint. Keep near zero at night. |
| `uVent` | Combine high-temperature PID and high-RH PID by taking the maximum. |
| `uLamp` | Simple deterministic rule using time of day, outdoor radiation, and daily radiation sum. |
| `uThScr` | Simple deterministic rule based on outdoor temperature and day/night. |
| `uBlScr` | Simple deterministic rule: close during night lighting when configured. |

The controller should clip all outputs to `[0, 1]`.

### Measurements

Use the same state conversions already used by `RuleBasedController`:

- Indoor air temperature: `ctx.x[2]`
- Indoor CO2 ppm: `co2dens2ppm(ctx.x[2], 1e-6 * ctx.x[0])`
- Relative humidity: `100 * ctx.x[15] / satVp(ctx.x[2])`
- Outdoor radiation: `ctx.d[ctx.t][0]`
- Outdoor temperature: `ctx.d[ctx.t][1]`
- Daily light integral: `ctx.d[ctx.t][7]`
- Day/night indicator: `ctx.d[ctx.t][8]`
- Smooth day/night indicator: `ctx.d[ctx.t][9]`

### Setpoints

Default PID configuration should mirror the rule-based baseline where useful:

- daytime temperature setpoint: `19.5`
- nighttime temperature setpoint: `16.5`
- CO2 daytime setpoint: `800`
- RH maximum setpoint: `85`
- lamp on hour: `0`
- lamp off hour: `18`
- lamp outdoor radiation cutoff: `400`
- lamp DLI cutoff: `10`

## Evaluation Design

Add `experiments/evaluate_pid.py` with the same CLI shape as `evaluate_baseline.py`:

```bash
python -m experiments.evaluate_pid \
  --env_id GreenLightEnv \
  --env_config glassgym/configs/envs/ \
  --pid_config configs/agents/ \
  --n_sims 1 \
  --save_dir results/pid/
```

The script should save:

```text
results/pid/pid_GreenLightEnv.csv
```

The CSV should include the same per-step reward/info columns used by the baseline evaluator, plus scenario metadata:

- `sim`
- `location`
- `growth_year`
- `start_day`

This allows direct reward aggregation:

```python
df["reward"].sum()
df["reward"].mean()
```

## Testing Design

Add focused tests that do not require a full 60-day run:

1. `PIDLoop` returns bounded outputs.
2. `PIDLoop.reset()` clears integral and derivative state.
3. `PIDController.predict(ctx)` returns shape `(6,)`.
4. PID controls are all within `[0, 1]`.
5. Low indoor temperature increases `uBoil` relative to a warmer state.
6. Low indoor CO2 during daytime increases `uCO2`.
7. High RH increases `uVent`.
8. A short environment rollout with PID runs for several steps without crashing.

The full experiment run is a verification step, not a unit test.

## Expected Result

PID should produce a complete CSV on the default Amsterdam 2010 day-59 scenario. It does not need to beat PPO or SAC. For论文对比, success means:

- the controller is deterministic and explainable
- it completes the full episode
- it writes a comparable CSV
- its reward can be reported alongside baseline, PPO, and SAC

## Non-Goals

- No MPC implementation in this PID task.
- No automatic PID tuning in the first version.
- No changes to PPO/SAC training code.
- No replacement of the existing rule-based baseline.

## Open Decisions Resolved

- PID is being built as a论文对比算法, not as a production-quality greenhouse controller.
- The first version uses PID for temperature, CO2, and humidity, with simple deterministic rules for lamp and screens.
- PID gets its own evaluation script to minimize risk to existing results.
