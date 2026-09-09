# PID Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic PID controller that can run the GreenLight environment and produce a reward CSV comparable with baseline, PPO, and SAC.

**Architecture:** Add a standalone PID controller module with the same `predict(ctx: StepContext) -> np.ndarray` interface as the existing rule-based controller. Add a PID YAML config and a dedicated evaluation script that mirrors `experiments/evaluate_baseline.py` while writing `results/pid/pid_GreenLightEnv.csv`.

**Tech Stack:** Python 3.12, NumPy, Gymnasium environment already in `glassgym`, YAML config loading through `RL.utils`, unittest for focused tests.

---

## Files

- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\components\pid.py`
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\pid.yml`
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_pid.py`
- Create: `E:\school\final paper\GreenLight-Gym2\tests\pid_controller.py`
- No changes to PPO/SAC training code.
- No changes to existing baseline evaluation code.

The workspace is not a git repository because it was downloaded as a zip snapshot, so commit steps are replaced with file/status verification commands.

---

### Task 1: PID Unit Tests

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\tests\pid_controller.py`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\pid_controller.py`

- [ ] **Step 1: Write failing tests for PIDLoop and PIDController**

Create `tests/pid_controller.py` with:

```python
import unittest

import numpy as np

from glassgym.components.pid import PIDController, PIDLoop
from glassgym.components.weather import WeatherRepository
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import load_weather_data
from glassgym.core.types import StepContext
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


class TestPIDLoop(unittest.TestCase):
    def test_output_is_clipped_to_bounds(self):
        loop = PIDLoop(kp=10.0, ki=0.0, kd=0.0, output_min=0.0, output_max=1.0)
        self.assertEqual(loop.compute(setpoint=10.0, measurement=0.0, dt=1.0), 1.0)

    def test_reset_clears_state(self):
        loop = PIDLoop(kp=1.0, ki=1.0, kd=1.0, output_min=0.0, output_max=10.0)
        loop.compute(setpoint=2.0, measurement=1.0, dt=1.0)
        loop.reset()
        self.assertEqual(loop.integral, 0.0)
        self.assertIsNone(loop.prev_error)


class TestPIDController(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params("GreenLightEnv", "glassgym/configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env_kwargs["normalize_actions"] = False
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)
        params = load_model_hyperparams("pid", "GreenLightEnv")
        self.controller = PIDController(**params)

    def _ctx(self):
        return StepContext(
            t=self.env.timestep,
            dt=self.env.dt,
            Np=self.env.Np,
            x_prev=self.env.x_prev,
            x=self.env.x,
            u=self.env.u,
            p=self.env.p,
            d=self.env.weather_data,
            hour_of_day=self.env.hour_of_day,
            day_of_year=self.env.day_of_year,
        )

    def test_predict_returns_six_controls(self):
        u = self.controller.predict(self._ctx())
        self.assertEqual(u.shape, (6,))

    def test_predict_returns_controls_in_unit_interval(self):
        u = self.controller.predict(self._ctx())
        self.assertTrue(np.all(u >= 0.0), u)
        self.assertTrue(np.all(u <= 1.0), u)

    def test_low_temperature_increases_heating(self):
        cold_ctx = self._ctx()
        warm_ctx = self._ctx()
        cold_ctx.x = cold_ctx.x.copy()
        warm_ctx.x = warm_ctx.x.copy()
        cold_ctx.x[2] = 14.0
        warm_ctx.x[2] = 22.0

        self.controller.reset()
        cold_u = self.controller.predict(cold_ctx)
        self.controller.reset()
        warm_u = self.controller.predict(warm_ctx)

        self.assertGreater(cold_u[0], warm_u[0])

    def test_low_co2_increases_dosing_during_day(self):
        low_ctx = self._ctx()
        high_ctx = self._ctx()
        low_ctx.x = low_ctx.x.copy()
        high_ctx.x = high_ctx.x.copy()
        low_ctx.hour_of_day = 12.0
        high_ctx.hour_of_day = 12.0
        low_ctx.x[0] = 0.15
        high_ctx.x[0] = 1.2

        self.controller.reset()
        low_u = self.controller.predict(low_ctx)
        self.controller.reset()
        high_u = self.controller.predict(high_ctx)

        self.assertGreaterEqual(low_u[1], high_u[1])

    def test_high_rh_increases_ventilation(self):
        humid_ctx = self._ctx()
        dry_ctx = self._ctx()
        humid_ctx.x = humid_ctx.x.copy()
        dry_ctx.x = dry_ctx.x.copy()
        humid_ctx.x[15] = 2500.0
        dry_ctx.x[15] = 500.0

        self.controller.reset()
        humid_u = self.controller.predict(humid_ctx)
        self.controller.reset()
        dry_u = self.controller.predict(dry_ctx)

        self.assertGreater(humid_u[3], dry_u[3])

    def test_short_rollout_does_not_crash(self):
        for _ in range(10):
            ctx = self._ctx()
            u = self.controller.predict(ctx)
            _obs, _reward, terminated, truncated, _info = self.env.step(u.astype(np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail because PID does not exist**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pid_controller
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'glassgym.components.pid'`.

- [ ] **Step 3: Verify no existing files were modified**

Run:

```powershell
Get-ChildItem -LiteralPath tests\pid_controller.py | Select-Object FullName,Length
```

Expected: the new test file exists.

---

### Task 2: PID Controller Implementation

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\components\pid.py`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\pid_controller.py`

- [ ] **Step 1: Implement PIDLoop and PIDController**

Create `glassgym/components/pid.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glassgym.core.types import StepContext
from glassgym.environments.utils import co2dens2ppm, satVp


@dataclass
class PIDLoop:
    kp: float
    ki: float
    kd: float
    output_min: float = 0.0
    output_max: float = 1.0
    integral_min: float = -10.0
    integral_max: float = 10.0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        dt = max(float(dt), 1e-9)
        error = float(setpoint) - float(measurement)
        self.integral = float(np.clip(
            self.integral + error * dt,
            self.integral_min,
            self.integral_max,
        ))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))


class PIDController:
    def __init__(
        self,
        temp_day_setpoint: float,
        temp_night_setpoint: float,
        temp_deadzone: float,
        co2_day_setpoint: float,
        rh_max: float,
        heat_pid: dict,
        vent_temp_pid: dict,
        co2_pid: dict,
        vent_rh_pid: dict,
        lamps_on: float,
        lamps_off: float,
        lamps_off_sun: float,
        lamp_rad_sum_limit: float,
        thscr_temp_threshold_day: float,
        thscr_temp_threshold_night: float,
        thscr_value: float,
        use_bl_scr: float,
    ):
        self.temp_day_setpoint = temp_day_setpoint
        self.temp_night_setpoint = temp_night_setpoint
        self.temp_deadzone = temp_deadzone
        self.co2_day_setpoint = co2_day_setpoint
        self.rh_max = rh_max
        self.lamps_on = lamps_on
        self.lamps_off = lamps_off
        self.lamps_off_sun = lamps_off_sun
        self.lamp_rad_sum_limit = lamp_rad_sum_limit
        self.thscr_temp_threshold_day = thscr_temp_threshold_day
        self.thscr_temp_threshold_night = thscr_temp_threshold_night
        self.thscr_value = thscr_value
        self.use_bl_scr = use_bl_scr

        self.heat_loop = PIDLoop(**heat_pid)
        self.vent_temp_loop = PIDLoop(**vent_temp_pid)
        self.co2_loop = PIDLoop(**co2_pid)
        self.vent_rh_loop = PIDLoop(**vent_rh_pid)

    def reset(self):
        self.heat_loop.reset()
        self.vent_temp_loop.reset()
        self.co2_loop.reset()
        self.vent_rh_loop.reset()

    def predict(self, ctx: StepContext):
        d = ctx.d[ctx.t]
        t_air = float(ctx.x[2])
        co2_ppm = float(co2dens2ppm(ctx.x[2], 1e-6 * ctx.x[0]))
        rh = float(100.0 * ctx.x[15] / satVp(ctx.x[2]))
        dt_hours = float(ctx.dt) / 3600.0

        is_day_inside = max(float(d[8]), self._lamp_time_active(ctx.hour_of_day))
        temp_setpoint = (
            is_day_inside * self.temp_day_setpoint
            + (1.0 - is_day_inside) * self.temp_night_setpoint
        )

        u = np.zeros(6, dtype=np.float32)
        u[0] = self.heat_loop.compute(temp_setpoint, t_air, dt_hours)
        u[1] = self._co2_control(co2_ppm, is_day_inside, dt_hours)

        vent_temp = self.vent_temp_loop.compute(
            measurement=temp_setpoint + self.temp_deadzone,
            setpoint=t_air,
            dt=dt_hours,
        )
        vent_rh = self.vent_rh_loop.compute(
            measurement=self.rh_max,
            setpoint=rh,
            dt=dt_hours,
        )
        u[3] = max(vent_temp, vent_rh)

        u[4] = self._lamp_control(ctx, d, t_air, rh)
        u[2] = self._thermal_screen_control(d)
        u[5] = self.use_bl_scr * (1.0 - float(d[9])) * u[4]
        return np.clip(u, 0.0, 1.0).astype(np.float32)

    def _co2_control(self, co2_ppm: float, is_day_inside: float, dt_hours: float) -> float:
        if is_day_inside <= 0.0:
            return 0.0
        return self.co2_loop.compute(self.co2_day_setpoint, co2_ppm, dt_hours)

    def _lamp_time_active(self, hour_of_day: float) -> float:
        if self.lamps_on == self.lamps_off:
            return 0.0
        if self.lamps_on < self.lamps_off:
            return float(self.lamps_on < hour_of_day < self.lamps_off)
        return float(hour_of_day > self.lamps_on or hour_of_day < self.lamps_off)

    def _lamp_control(self, ctx: StepContext, d: np.ndarray, t_air: float, rh: float) -> float:
        time_ok = self._lamp_time_active(ctx.hour_of_day)
        radiation_ok = float(d[0] < self.lamps_off_sun)
        dli_ok = float(d[7] < self.lamp_rad_sum_limit)
        temp_ok = float(t_air < self.temp_day_setpoint + self.temp_deadzone)
        rh_ok = float(rh < self.rh_max + 10.0)
        return float(time_ok * radiation_ok * dli_ok * temp_ok * rh_ok)

    def _thermal_screen_control(self, d: np.ndarray) -> float:
        is_day = float(d[8])
        threshold = (
            is_day * self.thscr_temp_threshold_day
            + (1.0 - is_day) * self.thscr_temp_threshold_night
        )
        return float(self.thscr_value if d[1] < threshold else 0.0)
```

- [ ] **Step 2: Run PID tests and capture the next failure**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pid_controller
```

Expected: ERROR because `configs/agents/pid.yml` does not exist yet.

---

### Task 3: PID Configuration

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\pid.yml`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\pid_controller.py`

- [ ] **Step 1: Add PID defaults**

Create `configs/agents/pid.yml` with:

```yaml
GreenLightEnv:
  temp_day_setpoint: 19.5
  temp_night_setpoint: 16.5
  temp_deadzone: 5.0
  co2_day_setpoint: 800.0
  rh_max: 85.0

  heat_pid:
    kp: 0.18
    ki: 0.015
    kd: 0.02
    output_min: 0.0
    output_max: 1.0
    integral_min: -20.0
    integral_max: 20.0

  vent_temp_pid:
    kp: 0.12
    ki: 0.005
    kd: 0.01
    output_min: 0.0
    output_max: 1.0
    integral_min: -20.0
    integral_max: 20.0

  co2_pid:
    kp: 0.002
    ki: 0.0001
    kd: 0.0
    output_min: 0.0
    output_max: 1.0
    integral_min: -2000.0
    integral_max: 2000.0

  vent_rh_pid:
    kp: 0.08
    ki: 0.002
    kd: 0.0
    output_min: 0.0
    output_max: 1.0
    integral_min: -50.0
    integral_max: 50.0

  lamps_on: 0.0
  lamps_off: 18.0
  lamps_off_sun: 400.0
  lamp_rad_sum_limit: 10.0
  thscr_temp_threshold_day: 5.0
  thscr_temp_threshold_night: 10.0
  thscr_value: 0.8
  use_bl_scr: 1.0
```

- [ ] **Step 2: Run targeted PID tests**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.pid_controller
```

Expected: PASS for all PID tests. If a numeric assertion fails, inspect the measured controls and adjust only the relevant PID default in `configs/agents/pid.yml`.

---

### Task 4: PID Evaluation Script

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_pid.py`
- Test: manual one-scenario smoke run

- [ ] **Step 1: Write the evaluation script**

Create `experiments/evaluate_pid.py` with:

```python
"""
Evaluate the PID controller on GreenLightEnv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from glassgym.components.pid import PIDController
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


def build_step_context(env: GreenLightEnv) -> StepContext:
    return StepContext(
        t=env.timestep,
        dt=env.dt,
        Np=env.Np,
        x_prev=env.x_prev,
        x=env.x,
        u=env.u,
        p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day,
        day_of_year=env.day_of_year,
    )


def evaluate_episode(env: GreenLightEnv, controller: PIDController) -> list[dict]:
    steps: list[dict] = []
    controller.reset()
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        _obs, reward, terminated, truncated, info = env.step(u.astype(np.float32))
        steps.append({
            "timestep": env.timestep,
            "reward": float(reward),
            **{k: v for k, v in info.items() if k != "controls"},
        })
        if terminated or truncated:
            break
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the PID controller on the GreenLight environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="glassgym/configs/envs/")
    parser.add_argument("--pid_config", default="configs/agents/")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parameter_provider", type=str, default=None)
    parser.add_argument("--parameter_provider_kwargs", type=str, default=None)
    parser.add_argument("--reward_kwargs", type=str, default=None)
    parser.add_argument("--save_dir", default="results/pid/")
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, default_scenarios = build_env_kwargs(env_kwargs)
    env_kwargs["normalize_actions"] = False

    scenarios = default_scenarios
    if args.scenarios is not None:
        scenarios = json.loads(args.scenarios)

    if args.parameter_provider is not None:
        env_kwargs["parameter_provider"] = args.parameter_provider
    if args.parameter_provider_kwargs is not None:
        env_kwargs["parameter_provider_kwargs"] = json.loads(args.parameter_provider_kwargs)
    if args.reward_kwargs is not None:
        overrides = json.loads(args.reward_kwargs)
        env_kwargs.setdefault("reward_kwargs", {}).update(overrides)

    pid_params = load_model_hyperparams("pid", args.env_id)
    env = GreenLightEnv(**env_kwargs)
    controller = PIDController(**pid_params)

    os.makedirs(args.save_dir, exist_ok=True)
    all_results: list[dict] = []

    for scenario in scenarios:
        desc = f"{scenario['location']}/{scenario['growth_year']}d{scenario['start_day']}"
        for sim in tqdm(range(args.n_sims), desc=desc):
            env.reset(seed=args.seed + sim, options={"scenario": scenario})
            episode_steps = evaluate_episode(env, controller)
            for step in episode_steps:
                step.update(sim=sim, **scenario)
            all_results.extend(episode_steps)

    df = pd.DataFrame(all_results)
    save_path = os.path.join(args.save_dir, f"pid_{args.env_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a short import smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_pid --help
```

Expected: CLI help output includes `Evaluate the PID controller`.

---

### Task 5: Full PID Reproduction Run

**Files:**
- Output: `E:\school\final paper\GreenLight-Gym2\results\pid\pid_GreenLightEnv.csv`

- [ ] **Step 1: Run PID evaluation on default scenario**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m experiments.evaluate_pid `
  --env_id GreenLightEnv `
  --env_config glassgym/configs/envs/ `
  --pid_config configs/agents/ `
  --n_sims 1 `
  --save_dir results/pid/
```

Expected: exit code 0 and output similar to:

```text
Saved 5761 rows to results/pid/pid_GreenLightEnv.csv
```

- [ ] **Step 2: Summarize PID reward**

Run:

```powershell
@'
import pandas as pd
p='results/pid/pid_GreenLightEnv.csv'
df=pd.read_csv(p)
print('rows', len(df))
print('cols', len(df.columns))
print('reward_sum', round(df['reward'].sum(), 6))
print('reward_mean', round(df['reward'].mean(), 6))
print('scenario', df[['location','growth_year','start_day']].drop_duplicates().to_dict('records'))
'@ | .\.venv\Scripts\python.exe -
```

Expected:

- rows is `5761` if the environment records the final terminal step like baseline.
- scenario is Amsterdam 2010 start day 59.
- reward values are numeric and finite.

---

### Task 6: Compare PID With Existing Results

**Files:**
- Read: `E:\school\final paper\GreenLight-Gym2\results\baseline\rb_baseline_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\rl\rl_ppo_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\rl\rl_sac_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\pid\pid_GreenLightEnv.csv`

- [ ] **Step 1: Run comparison summary**

Run:

```powershell
@'
import pandas as pd

paths = {
    'baseline': 'results/baseline/rb_baseline_GreenLightEnv.csv',
    'pid': 'results/pid/pid_GreenLightEnv.csv',
    'ppo': 'results/rl/rl_ppo_GreenLightEnv.csv',
    'sac': 'results/rl/rl_sac_GreenLightEnv.csv',
}

for name, path in paths.items():
    df = pd.read_csv(path)
    print(
        name,
        'rows', len(df),
        'reward_sum', round(df['reward'].sum(), 6),
        'reward_mean', round(df['reward'].mean(), 6),
    )
'@ | .\.venv\Scripts\python.exe -
```

Expected: all four rows print successfully. PID does not have to beat baseline, PPO, or SAC for this task to count as implemented. The success criterion is a complete, deterministic, comparable PID result.

- [ ] **Step 2: Record final file list**

Run:

```powershell
Get-ChildItem -File -LiteralPath 'glassgym\components\pid.py','configs\agents\pid.yml','experiments\evaluate_pid.py','tests\pid_controller.py','results\pid\pid_GreenLightEnv.csv' |
  Select-Object FullName,Length,LastWriteTime
```

Expected: all five files exist.

---

## Completion Criteria

PID implementation is complete when:

- `python -m unittest tests.pid_controller` passes.
- `python -m experiments.evaluate_pid --n_sims 1 --save_dir results/pid/` exits with code 0.
- `results/pid/pid_GreenLightEnv.csv` exists.
- PID reward summary can be printed alongside baseline, PPO, and SAC.
- Any warnings or limitations are reported in the final response.
