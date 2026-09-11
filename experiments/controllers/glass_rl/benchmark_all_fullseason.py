"""玻璃温室全算法统一评估（完整生长季 + 优化 reward）。

口径（102 天完整生长季）：
- day0(4/1 定植苗) 起 102 天到 day102(7/12 清棚)
- seedling 起点、禁补光/CO2、yield_weight=1.0、cooling_weight=0.5
- 所有策略在同一 MultiDiscrete 环境逐时评估

策略列表：
- human（真实精确档位）
- 经典控制器：baseline / rule / pid / mpc
- RL 算法：ppo / a2c / dqn / sac / ddpg / td3 / trpo / recurrent_ppo
- multiagent（独立 IPPO）

用法：
  python -m experiments.controllers.glass_rl.benchmark_all_fullseason \
      [--strategies human baseline rule pid mpc ppo ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from experiments.controllers.glass_rl.full_season_benchmark_v3 import (
    load_human_actions, ACT_COLS)

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
ALGO_MODEL = OUT_ROOT / "algorithms_fullseason"

START_DAY = 0
DAYS = 102
MONTH_BINS = [-1, 30, 61, 91, 103]
MONTH_LABELS = ["4月", "5月", "6月", "7月"]


def build_env():
    from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv
    return GlassGreenhouseEnv(
        episode_days=DAYS, start_day_index=START_DAY,
        yield_weight=1.0, temperature_weight=1.0, humidity_weight=1.0,
        effort_weight=0.2, cooling_weight=0.5,
        crop_start="seedling", disable_supplements=True,
    )


def load_rl_model(name: str):
    import stable_baselines3 as sb3
    import sb3_contrib
    path = ALGO_MODEL / name / "model.zip"
    if name == "ppo":
        return sb3.PPO.load(str(path)), "multi"
    if name == "a2c":
        return sb3.A2C.load(str(path)), "multi"
    if name == "dqn":
        return sb3.DQN.load(str(path)), "flat"
    if name == "sac":
        return sb3.SAC.load(str(path)), "cont"
    if name == "ddpg":
        return sb3.DDPG.load(str(path)), "cont"
    if name == "td3":
        return sb3.TD3.load(str(path)), "cont"
    if name == "trpo":
        return sb3_contrib.TRPO.load(str(path)), "multi"
    if name == "recurrent_ppo":
        return sb3_contrib.RecurrentPPO.load(str(path)), "multi"
    raise ValueError(name)


def rl_to_levels(action, kind: str) -> np.ndarray:
    from experiments.controllers.glass_rl.env_variants import (
        continuous_to_levels, decode_levels)
    if kind == "multi":
        return np.asarray(action, dtype=int).reshape(-1)
    if kind == "flat":
        return decode_levels(int(np.asarray(action).reshape(-1)[0]))
    if kind == "cont":
        return continuous_to_levels(np.asarray(action, dtype=float).reshape(-1))
    raise ValueError(kind)


def summarize(name: str, temps, rhs, rewards, efforts, fruits, fails: int = 0) -> dict:
    temps = np.asarray(temps); rhs = np.asarray(rhs)
    df = pd.DataFrame({"hour": np.tile(np.arange(24), DAYS)[: len(temps)],
                       "temp": temps, "rh": rhs,
                       "reward": rewards, "effort": efforts})
    day = (df["hour"] >= 6) & (df["hour"] < 20)
    t_ok = np.where(day, df["temp"].between(20, 28), df["temp"].between(16, 24))
    rh_ok = df["rh"].between(60, 85)
    comfort = (t_ok & rh_ok).mean() * 100
    fg = fruits[-1] - fruits[0]
    monthly = []
    df["day"] = np.arange(len(df)) // 24
    df["month"] = pd.cut(df["day"], bins=MONTH_BINS, labels=MONTH_LABELS)
    for m, g in df.groupby("month", observed=True):
        d = (g["hour"] >= 6) & (g["hour"] < 20)
        monthly.append({
            "month": m, "mean_temp": round(float(g["temp"].mean()), 2),
            "max_temp": round(float(g["temp"].max()), 2),
            "comfort": round(float((np.where(d, g["temp"].between(20, 28), g["temp"].between(16, 24)) & g["rh"].between(60, 85)).mean()) * 100, 1),
            "total_reward": round(float(g["reward"].sum()), 1),
        })
    return {
        "name": name,
        "mean_temp": round(float(temps.mean()), 2),
        "max_temp": round(float(temps.max()), 2),
        "comfort": round(float(comfort), 1),
        "total_reward": round(float(np.sum(rewards)), 1),
        "mean_effort": round(float(np.mean(efforts)), 4),
        "fruit_growth_kg_m2": round(float(fg * 1e-6 / 0.081), 3),
        "integration_failures": fails,
        "monthly": monthly,
    }


def run_strategy(name: str, act_fn) -> dict:
    env = build_env()
    env.reset(seed=0)
    env._prev_fruit = float(env.x[25])
    temps, rhs, rewards, efforts, fruits, actions = [], [], [], [], [], []
    fails = 0
    for i in range(DAYS * 24):
        a = act_fn(env)
        actions.append(np.asarray(a, dtype=int).reshape(-1).tolist())
        obs, r, term, tr, info = env.step(a)
        if info.get("integration_failed"):
            fails += 1
        if not np.isfinite(env.x).all():
            print(f"[{name}] NaN at {i}, abort"); break
        temps.append(info["temperature"]); rhs.append(info["rh"])
        rewards.append(r); efforts.append(info["effort"])
        fruits.append(info["fruit_mg"])
    s = summarize(name, temps, rhs, rewards, efforts, fruits, fails)
    print(f"[{name:14s}] 均温{s['mean_temp']:5.1f} max{s['max_temp']:5.1f} "
          f"舒适{s['comfort']:5.1f}% reward{s['total_reward']:8.1f} "
          f"能耗{s['mean_effort']:.4f} 果实{s['fruit_growth_kg_m2']:+.3f} "
          f"积分失败{fails}")
    return s, temps, rhs, actions


def _next(arr, state):
    i = state[0]
    state[0] += 1
    return arr[i]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+",
                        default=["human", "baseline", "rule", "pid", "mpc",
                                 "ppo", "a2c", "dqn", "sac", "ddpg", "td3",
                                 "trpo", "recurrent_ppo", "multiagent"])
    args = parser.parse_args()

    results = {}
    trajectories = {}

    def _record(name: str, act_fn) -> None:
        s, temps, rhs, actions = run_strategy(name, act_fn)
        results[name] = s
        trajectories[name] = (temps, rhs, actions)

    if "human" in args.strategies:
        ha = load_human_actions().iloc[: DAYS * 24].reset_index(drop=True)
        human_acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int)
                      for i in range(len(ha))]
        state = [0]

        def human_act(env):
            i = state[0]
            state[0] += 1
            return human_acts[i]

        _record("human", human_act)

    from experiments.controllers.glass_rl.classical_controllers import (
        GlassBaseline, GlassRuleBased, GlassPID, GlassMPC)
    classics = {"baseline": GlassBaseline, "rule": GlassRuleBased,
                "pid": GlassPID, "mpc": GlassMPC}
    for name in args.strategies:
        if name in classics:
            _record(name, classics[name]().predict_action)

    rl_names = ["ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo"]
    for name in args.strategies:
        if name not in rl_names:
            continue
        try:
            model, kind = load_rl_model(name)
        except Exception as e:
            print(f"[{name}] 模型加载失败，跳过: {e}")
            continue

        def act_fn(env, model=model, kind=kind):
            a, _ = model.predict(env._get_obs(), deterministic=True)
            return rl_to_levels(a, kind)

        _record(name, act_fn)

    if "multiagent" in args.strategies:
        from experiments.controllers.glass_rl.benchmark_all import load_multiagent
        try:
            joint_predict = load_multiagent()
        except Exception as e:
            print(f"[multiagent] 模型加载失败，跳过: {e}")
        else:
            _record("multiagent", lambda env: joint_predict(env._get_obs()))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / f"benchmark_fullseason_d0_{DAYS}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    traj_out = OUT_ROOT / f"benchmark_fullseason_d0_{DAYS}_trajectories.json"
    traj_save = {}
    for k, (temps, rhs, actions) in trajectories.items():
        traj_save[k] = {
            "temp": [round(float(t), 2) for t in temps],
            "rh": [round(float(r), 1) for r in rhs],
            "action": actions,
        }
    with open(traj_out, "w", encoding="utf-8") as f:
        json.dump(traj_save, f, ensure_ascii=False)
    print(f"\n已保存 {out} + {traj_out}")


if __name__ == "__main__":
    main()
