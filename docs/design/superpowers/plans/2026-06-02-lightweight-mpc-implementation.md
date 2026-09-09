# Lightweight MPC Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight candidate-action MPC controller that can run the GreenLight environment and produce a reward CSV comparable with baseline, PID, PPO, and SAC.

**Architecture:** Add a standalone MPC controller module that uses the same `predict(ctx: StepContext) -> np.ndarray` interface as existing deterministic controllers. The controller receives the environment integrator through `bind_env(env)`, enumerates a small candidate action set, predicts each candidate over a short horizon using `env.F`, scores the trajectory with a configurable proxy objective, and returns the best first action.

**Tech Stack:** Python 3.12, NumPy, CasADi integrator already exposed by `GreenLightEnv`, YAML config loading through `RL.utils`, unittest.

---

## Files

- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\components\mpc.py`
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\mpc.yml`
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_mpc.py`
- Create: `E:\school\final paper\GreenLight-Gym2\tests\mpc_controller.py`
- No changes to PPO/SAC/PID/baseline training or evaluation code.

The workspace is not a git repository because it was downloaded as a zip snapshot, so commit steps are replaced with file/status verification commands.

---

### Task 1: MPC Unit Tests

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\tests\mpc_controller.py`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\mpc_controller.py`

- [ ] **Step 1: Write failing tests for candidate generation and MPC prediction**

Create `tests/mpc_controller.py` with:

```python
import unittest

import numpy as np

from glassgym.components.mpc import CandidateActionGenerator, LightweightMPCController
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


class TestCandidateActionGenerator(unittest.TestCase):
    def test_includes_current_control(self):
        generator = CandidateActionGenerator(
            levels={
                "uBoil": [0.0, 0.5, 1.0],
                "uCO2": [0.0, 0.5],
                "uVent": [0.0, 0.5],
                "uLamp": [0.0, 1.0],
            },
            include_current=True,
            include_conservative=True,
        )
        current = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        candidates = generator.generate(current)
        self.assertTrue(any(np.allclose(c, current) for c in candidates))

    def test_candidates_are_clipped_to_unit_interval(self):
        generator = CandidateActionGenerator(
            levels={"uBoil": [-1.0, 2.0], "uCO2": [2.0], "uVent": [-1.0], "uLamp": [2.0]},
            include_current=True,
            include_conservative=True,
        )
        current = np.array([-5, 2, 0, 3, -1, 8], dtype=np.float32)
        candidates = generator.generate(current)
        self.assertTrue(np.all(candidates >= 0.0), candidates)
        self.assertTrue(np.all(candidates <= 1.0), candidates)


class TestLightweightMPCController(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params("GreenLightEnv", "glassgym/configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env_kwargs["normalize_actions"] = False
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        params["horizon_steps"] = 1
        params["candidate_levels"] = {
            "uBoil": [0.0, 0.5],
            "uCO2": [0.0, 0.5],
            "uVent": [0.0, 0.5],
            "uLamp": [0.0],
        }
        self.controller = LightweightMPCController(**params)
        self.controller.bind_env(self.env)

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

    def test_predict_requires_bound_environment(self):
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        controller = LightweightMPCController(**params)
        with self.assertRaises(RuntimeError):
            controller.predict(self._ctx())

    def test_short_rollout_does_not_crash(self):
        for _ in range(3):
            ctx = self._ctx()
            u = self.controller.predict(ctx)
            _obs, _reward, terminated, truncated, _info = self.env.step(u.astype(np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail because MPC does not exist**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.mpc_controller
```

Expected: FAIL or ERROR with `ModuleNotFoundError: No module named 'glassgym.components.mpc'`.

---

### Task 2: MPC Controller Implementation

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\glassgym\components\mpc.py`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\mpc_controller.py`

- [ ] **Step 1: Implement candidate generation and lightweight MPC**

Create `glassgym/components/mpc.py` with:

```python
from __future__ import annotations

import itertools
from typing import Any

import casadi as ca
import numpy as np

from glassgym.core.types import StepContext
from glassgym.environments.utils import co2dens2ppm, satVp


CONTROL_INDEX = {
    "uBoil": 0,
    "uCO2": 1,
    "uThScr": 2,
    "uVent": 3,
    "uLamp": 4,
    "uBlScr": 5,
}


class CandidateActionGenerator:
    def __init__(
        self,
        levels: dict[str, list[float]],
        include_current: bool = True,
        include_conservative: bool = True,
    ):
        self.levels = levels
        self.include_current = include_current
        self.include_conservative = include_conservative

    def generate(self, current_u: np.ndarray) -> np.ndarray:
        current = np.clip(np.asarray(current_u, dtype=np.float32), 0.0, 1.0)
        candidates: list[np.ndarray] = []

        if self.include_current:
            candidates.append(current.copy())

        base = current.copy()
        for name, values in self.levels.items():
            idx = CONTROL_INDEX[name]
            for value in values:
                candidate = base.copy()
                candidate[idx] = float(value)
                candidates.append(candidate)

        if self.include_conservative:
            conservative = np.zeros(6, dtype=np.float32)
            conservative[0] = 0.4
            conservative[2] = 0.5
            conservative[3] = 0.1
            candidates.append(conservative)

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for candidate in candidates:
            clipped = np.clip(candidate, 0.0, 1.0).astype(np.float32)
            key = tuple(np.round(clipped, 4))
            if key not in seen:
                seen.add(key)
                unique.append(clipped)

        return np.asarray(unique, dtype=np.float32)


class LightweightMPCController:
    def __init__(
        self,
        horizon_steps: int,
        candidate_levels: dict[str, list[float]],
        objective_weights: dict[str, float],
        temp_day_setpoint: float,
        temp_night_setpoint: float,
        co2_day_setpoint: float,
        rh_max: float,
        include_current: bool = True,
        include_conservative: bool = True,
        fallback_control: list[float] | None = None,
    ):
        self.horizon_steps = int(horizon_steps)
        self.objective_weights = objective_weights
        self.temp_day_setpoint = temp_day_setpoint
        self.temp_night_setpoint = temp_night_setpoint
        self.co2_day_setpoint = co2_day_setpoint
        self.rh_max = rh_max
        self.generator = CandidateActionGenerator(
            levels=candidate_levels,
            include_current=include_current,
            include_conservative=include_conservative,
        )
        self.fallback_control = np.asarray(
            fallback_control if fallback_control is not None else [0.5, 0.0, 0.5, 0.1, 0.0, 0.0],
            dtype=np.float32,
        )
        self.env = None

    def bind_env(self, env: Any):
        self.env = env

    def reset(self):
        pass

    def predict(self, ctx: StepContext) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("LightweightMPCController must be bound to an environment with bind_env(env).")

        candidates = self.generator.generate(ctx.u)
        best_score = -np.inf
        best_u = candidates[0] if len(candidates) else self.fallback_control

        for candidate in candidates:
            score = self._rollout_score(ctx, candidate)
            if score > best_score:
                best_score = score
                best_u = candidate

        return np.clip(best_u, 0.0, 1.0).astype(np.float32)

    def _rollout_score(self, ctx: StepContext, candidate_u: np.ndarray) -> float:
        x = np.asarray(ctx.x, dtype=np.float64).copy()
        prev_x = np.asarray(ctx.x_prev, dtype=np.float64).copy()
        total = 0.0

        for k in range(self.horizon_steps):
            t = min(ctx.t + k, len(ctx.d) - 1)
            d = np.asarray(ctx.d[t], dtype=np.float64)
            total += self._stage_score(
                x=x,
                prev_x=prev_x,
                u=candidate_u,
                prev_u=ctx.u,
                d=d,
            )
            try:
                prev_x = x.copy()
                p_dyn = ca.vertcat(ca.DM(d), ca.DM(ctx.p))
                res = self.env.F(x0=ca.DM(x), u=ca.DM(candidate_u), p=p_dyn)
                x = res["xf"].full().flatten()
                if not np.all(np.isfinite(x)):
                    return -np.inf
            except Exception:
                return -np.inf

        return float(total)

    def _stage_score(
        self,
        x: np.ndarray,
        prev_x: np.ndarray,
        u: np.ndarray,
        prev_u: np.ndarray,
        d: np.ndarray,
    ) -> float:
        weights = self.objective_weights
        t_air = float(x[2])
        co2_ppm = float(co2dens2ppm(x[2], 1e-6 * x[0]))
        rh = float(100.0 * x[15] / satVp(x[2]))
        is_day = float(d[8])
        temp_setpoint = is_day * self.temp_day_setpoint + (1.0 - is_day) * self.temp_night_setpoint
        co2_setpoint = is_day * self.co2_day_setpoint

        fruit_growth = max(0.0, float(x[25] - prev_x[25]))
        score = 0.0
        score -= weights["temperature_error"] * abs(t_air - temp_setpoint)
        score -= weights["co2_error"] * abs(co2_ppm - co2_setpoint) / 1000.0
        score -= weights["rh_violation"] * max(0.0, rh - self.rh_max) / 100.0
        score -= weights["heat_cost"] * float(u[0])
        score -= weights["co2_cost"] * float(u[1])
        score -= weights["lamp_cost"] * float(u[4])
        score -= weights["action_change"] * float(np.mean(np.abs(u - prev_u)))
        score += weights["fruit_growth"] * fruit_growth
        return float(score)
```

- [ ] **Step 2: Run tests and capture the next failure**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.mpc_controller
```

Expected: ERROR because `configs/agents/mpc.yml` does not exist yet.

---

### Task 3: MPC Configuration

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\configs\agents\mpc.yml`
- Test: `E:\school\final paper\GreenLight-Gym2\tests\mpc_controller.py`

- [ ] **Step 1: Add default MPC parameters**

Create `configs/agents/mpc.yml` with:

```yaml
GreenLightEnv:
  horizon_steps: 4

  candidate_levels:
    uBoil: [0.0, 0.25, 0.5, 0.75, 1.0]
    uCO2: [0.0, 0.25, 0.5, 0.75, 1.0]
    uVent: [0.0, 0.25, 0.5, 0.75, 1.0]
    uLamp: [0.0, 1.0]

  include_current: true
  include_conservative: true
  fallback_control: [0.5, 0.0, 0.5, 0.1, 0.0, 0.0]

  temp_day_setpoint: 19.5
  temp_night_setpoint: 16.5
  co2_day_setpoint: 800.0
  rh_max: 85.0

  objective_weights:
    temperature_error: 1.0
    co2_error: 0.25
    rh_violation: 4.0
    heat_cost: 0.25
    co2_cost: 0.15
    lamp_cost: 0.3
    action_change: 0.05
    fruit_growth: 0.000001
```

- [ ] **Step 2: Run targeted MPC tests**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m unittest tests.mpc_controller
```

Expected: PASS for all MPC tests.

---

### Task 4: MPC Evaluation Script

**Files:**
- Create: `E:\school\final paper\GreenLight-Gym2\experiments\evaluate_mpc.py`
- Test: CLI help and one default evaluation run

- [ ] **Step 1: Write the evaluation script**

Create `experiments/evaluate_mpc.py` with:

```python
"""
Evaluate the lightweight MPC controller on GreenLightEnv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from glassgym.components.mpc import LightweightMPCController
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


def evaluate_episode(env: GreenLightEnv, controller: LightweightMPCController) -> list[dict]:
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
        description="Evaluate the lightweight MPC controller on the GreenLight environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="glassgym/configs/envs/")
    parser.add_argument("--mpc_config", default="configs/agents/")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon_steps", type=int, default=None)
    parser.add_argument("--parameter_provider", type=str, default=None)
    parser.add_argument("--parameter_provider_kwargs", type=str, default=None)
    parser.add_argument("--reward_kwargs", type=str, default=None)
    parser.add_argument("--save_dir", default="results/mpc/")
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

    mpc_params = load_model_hyperparams("mpc", args.env_id)
    if args.horizon_steps is not None:
        mpc_params["horizon_steps"] = args.horizon_steps

    env = GreenLightEnv(**env_kwargs)
    controller = LightweightMPCController(**mpc_params)
    controller.bind_env(env)

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
    save_path = os.path.join(args.save_dir, f"mpc_{args.env_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run CLI help smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_mpc --help
```

Expected: CLI help output includes `Evaluate the lightweight MPC controller`.

---

### Task 5: Full MPC Reproduction Run

**Files:**
- Output: `E:\school\final paper\GreenLight-Gym2\results\mpc\mpc_GreenLightEnv.csv`

- [ ] **Step 1: Run MPC evaluation on default scenario**

Run:

```powershell
$env:PIP_CACHE_DIR='E:\school\final paper\GreenLight-Gym2\.pip-cache'
$env:TEMP='E:\school\final paper\GreenLight-Gym2\.tmp'
$env:TMP=$env:TEMP
.\.venv\Scripts\python.exe -m experiments.evaluate_mpc `
  --env_id GreenLightEnv `
  --env_config glassgym/configs/envs/ `
  --mpc_config configs/agents/ `
  --n_sims 1 `
  --save_dir results/mpc/
```

Expected: exit code 0 and output similar to:

```text
Saved 5761 rows to results/mpc/mpc_GreenLightEnv.csv
```

If the default horizon is too slow, rerun with:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluate_mpc `
  --env_id GreenLightEnv `
  --env_config glassgym/configs/envs/ `
  --mpc_config configs/agents/ `
  --horizon_steps 2 `
  --n_sims 1 `
  --save_dir results/mpc/
```

Report clearly if the fallback horizon was used.

- [ ] **Step 2: Summarize MPC reward**

Run:

```powershell
@'
import pandas as pd
p='results/mpc/mpc_GreenLightEnv.csv'
df=pd.read_csv(p)
print('rows', len(df))
print('cols', len(df.columns))
print('reward_sum', round(df['reward'].sum(), 6))
print('reward_mean', round(df['reward'].mean(), 6))
print('scenario', df[['location','growth_year','start_day']].drop_duplicates().to_dict('records'))
'@ | .\.venv\Scripts\python.exe -
```

Expected:

- rows is `5761` if the environment records the final terminal step like baseline/PID.
- scenario is Amsterdam 2010 start day 59.
- reward values are numeric and finite.

---

### Task 6: Compare MPC With Existing Results

**Files:**
- Read: `E:\school\final paper\GreenLight-Gym2\results\baseline\rb_baseline_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\pid\pid_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\rl\rl_ppo_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\rl\rl_sac_GreenLightEnv.csv`
- Read: `E:\school\final paper\GreenLight-Gym2\results\mpc\mpc_GreenLightEnv.csv`

- [ ] **Step 1: Run comparison summary**

Run:

```powershell
@'
import pandas as pd

paths = {
    'baseline': 'results/baseline/rb_baseline_GreenLightEnv.csv',
    'pid': 'results/pid/pid_GreenLightEnv.csv',
    'mpc': 'results/mpc/mpc_GreenLightEnv.csv',
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

Expected: all five rows print successfully. MPC does not have to beat PPO or SAC for this task to count as implemented.

- [ ] **Step 2: Record final file list**

Run:

```powershell
Get-ChildItem -File -LiteralPath 'glassgym\components\mpc.py','configs\agents\mpc.yml','experiments\evaluate_mpc.py','tests\mpc_controller.py','results\mpc\mpc_GreenLightEnv.csv' |
  Select-Object FullName,Length,LastWriteTime
```

Expected: all five files exist.

---

## Completion Criteria

MPC implementation is complete when:

- `python -m unittest tests.mpc_controller` passes.
- `python -m experiments.evaluate_mpc --n_sims 1 --save_dir results/mpc/` exits with code 0.
- `results/mpc/mpc_GreenLightEnv.csv` exists.
- MPC reward summary can be printed alongside baseline, PID, PPO, and SAC.
- Any runtime or horizon fallback is reported in the final response.
