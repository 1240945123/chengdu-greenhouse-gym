"""PPO 最终模型完整生长季评估（标定后环境，102 天确定性 rollout）。

口径与之前 benchmark 一致：
- seedling 起点（4/1 定植苗）、禁补光/CO2、标定参数、cooling_bonus
- 从 day0(4/1) 起连续 102 天（4/1-7/12）
- 确定性策略（deterministic=True）

输出：均温/最高温/舒适率(温湿联合+分月)/累计reward/果实鲜重/能耗。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

MODEL_PATH = "results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final/model"


def load_human_actions():
    """人工档位日志（与 benchmark_all_fullseason 一致）。"""
    import pandas as pd
    from experiments.controllers.glass_rl.benchmark_all_fullseason import ACT_COLS, load_human_actions as _load
    return _load()


def comfort(temp: float, rh: float, hour: float) -> bool:
    day = 6.0 <= hour < 20.0
    t_lo, t_hi = (20.0, 28.0) if day else (16.0, 24.0)
    return (t_lo <= temp <= t_hi) and (60.0 <= rh <= 85.0)


def evaluate(act_fn, label: str, days: int = 102) -> dict:
    from glass_env import GlassGreenhouseEnv

    env = GlassGreenhouseEnv(
        episode_days=days, start_day_index=0,
        crop_start="seedling", disable_supplements=True, cooling_weight=0.5,
    )
    obs, _ = env.reset(seed=0)
    fruit0 = float(env.x[25])

    temps, rhs, hours = [], [], []
    total_reward = 0.0
    total_effort = 0.0
    month_comfort = {m: [] for m in range(4, 8)}
    month_bounds = {4: 0, 5: 30, 6: 61, 7: 91}

    for i in range(days * 24):
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        total_effort += info.get("effort", 0.0)
        temps.append(info["temperature"])
        rhs.append(info["rh"])
        hours.append(info["hour"])
        day = i // 24
        for m in range(4, 8):
            lo = month_bounds[m]
            hi = month_bounds.get(m + 1, 103)
            if lo <= day < hi:
                month_comfort[m].append(comfort(info["temperature"], info["rh"], info["hour"]))

    fruit_final = float(env.x[25])
    fruit_kg = (fruit_final - fruit0) * 1e-6 / 0.081  # 干重 mg/m² → 鲜重 kg/m²

    temps = np.array(temps)
    rhs = np.array(rhs)
    all_comfort = np.mean([comfort(t, rh, h) for t, rh, h in zip(temps, rhs, hours)]) * 100

    return {
        "label": label,
        "mean_temp": float(temps.mean()),
        "max_temp": float(temps.max()),
        "min_temp": float(temps.min()),
        "comfort_pct": float(all_comfort),
        "month_comfort": {m: float(np.mean(v) * 100) for m, v in month_comfort.items()},
        "reward": float(total_reward),
        "fruit_kg": float(fruit_kg),
        "effort": float(total_effort),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--with-human", action="store_true", help="同时评估人工对照组")
    args = parser.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.model)

    def rl_act(obs):
        return model.predict(obs, deterministic=True)[0]

    rl = evaluate(rl_act, "PPO(标定后)")

    print("=" * 70)
    print("PPO 最终模型完整生长季评估（102 天，标定后环境）")
    print("=" * 70)
    print(f"  均温        : {rl['mean_temp']:.2f} °C")
    print(f"  最高温      : {rl['max_temp']:.2f} °C")
    print(f"  最低温      : {rl['min_temp']:.2f} °C")
    print(f"  舒适率      : {rl['comfort_pct']:.1f} % (温湿联合)")
    print(f"  累计 reward : {rl['reward']:.1f}")
    print(f"  果实鲜重    : {rl['fruit_kg']:.3f} kg/m²")
    print(f"  累计能耗    : {rl['effort']:.3f}")
    print("  分月舒适率  :", "  ".join(f"{m}月{rl['month_comfort'][m]:.1f}%" for m in range(4, 8)))

    if args.with_human:
        from experiments.controllers.glass_rl.full_season_benchmark_v3 import load_human_actions, ACT_COLS
        days = 102
        # load_human_actions 已做 FAN_MAP(0/1/2/4→0-3) 与 roof_vent 四舍五入映射
        ha = load_human_actions().iloc[: days * 24].reset_index(drop=True)
        human_acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
        state = {"i": 0}

        def human_act(obs):
            a = human_acts[state["i"]]
            state["i"] += 1
            return a

        hu = evaluate(human_act, "人工(标定后)")
        print("-" * 70)
        print(f"  人工对照组  : 均温 {hu['mean_temp']:.2f}°C  最高 {hu['max_temp']:.2f}°C  "
              f"舒适 {hu['comfort_pct']:.1f}%  reward {hu['reward']:.1f}  果实 {hu['fruit_kg']:.3f} kg/m²")


if __name__ == "__main__":
    main()
