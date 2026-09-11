"""最终三方对比评估：PPO v6 vs SAC v3 vs 人工（修复后环境，102天）。

统一口径：overheat + humidity_weight=2.0 + obs 8维 + 分温度 screen_pen。
PPO/人工用 MultiDiscrete 环境；SAC 用 Continuous 环境。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL_DIR = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT_JSON = RL_DIR / "final_comparison_ppo_sac_human.json"

ENV_KW = dict(episode_days=102, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5)


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def summarize(df, fruit):
    df = df.copy()
    df["is_day"] = (df.hour >= 6) & (df.hour < 20)
    df["month"] = pd.cut(df.day, bins=[-1, 30, 61, 91, 102], labels=["4月", "5月", "6月", "7月"])
    day, night = df[df.is_day], df[~df.is_day]
    comfort_pct = np.mean([comfort(r.t, r.rh, r.hour) for r in df.itertuples()]) * 100
    mc = {m: np.mean([comfort(r.t, r.rh, r.hour) for r in g.itertuples()]) * 100
          for m, g in df.groupby("month", observed=True)}
    return {
        "comfort_pct": float(comfort_pct),
        "mean_temp": float(df.t.mean()),
        "max_temp": float(df.t.max()),
        "rh_over85_day": float((day.rh > 85).mean() * 100),
        "rh_over85_night": float((night.rh > 85).mean() * 100),
        "fruit_kg": float(fruit),
        "month_comfort": {k: float(v) for k, v in mc.items()},
    }


def run_multi(model, env_kind="multi"):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    recs = []
    for i in range(102 * 24):
        a = model.predict(env._get_obs(), deterministic=True)[0]
        obs, r, term, tr, info = env.step(a)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    return summarize(pd.DataFrame(recs), fruit)


def run_sac(model):
    from experiments.controllers.glass_rl.env_variants import GlassGreenhouseEnvContinuous
    env = GlassGreenhouseEnvContinuous(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    recs = []
    for i in range(102 * 24):
        a = model.predict(env._get_obs(), deterministic=True)[0]
        obs, r, term, tr, info = env.step(a)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    return summarize(pd.DataFrame(recs), fruit)


def run_human():
    from glass_env import GlassGreenhouseEnv
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import load_human_actions, ACT_COLS
    ha = load_human_actions().iloc[: 102 * 24].reset_index(drop=True)
    acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    recs = []
    for i in range(102 * 24):
        obs, r, term, tr, info = env.step(acts[i])
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    return summarize(pd.DataFrame(recs), fruit)


def main():
    from stable_baselines3 import PPO, SAC
    print("评估 PPO v6...", flush=True)
    ppo = run_multi(PPO.load(str(RL_DIR / "ppo_final_v6/model")))
    print("评估 SAC v3...", flush=True)
    sac = run_sac(SAC.load(str(RL_DIR / "sac_final_v3/model")))
    print("评估 人工...", flush=True)
    human = run_human()

    rows = [("PPO v6", ppo), ("SAC v3", sac), ("人工", human)]
    print("\n" + "=" * 78)
    print("最终三方对比（102 天，修复后环境）")
    print("=" * 78)
    print(f"{'指标':<18} | {'PPO v6':>10} | {'SAC v3':>10} | {'人工':>10}")
    for name, key in [("舒适率 %", "comfort_pct"), ("均温 °C", "mean_temp"),
                      ("最高温 °C", "max_temp"), ("果实 kg/m²", "fruit_kg"),
                      ("白天湿度>85%", "rh_over85_day"), ("夜间湿度>85%", "rh_over85_night")]:
        print(f"{name:<18} | {ppo[key]:>10.1f} | {sac[key]:>10.1f} | {human[key]:>10.1f}")
    print("\n分月舒适率:")
    for m in ["4月", "5月", "6月", "7月"]:
        print(f"  {m}: PPO {ppo['month_comfort'][m]:.1f}% | SAC {sac['month_comfort'][m]:.1f}% | 人工 {human['month_comfort'][m]:.1f}%")

    OUT_JSON.write_text(json.dumps(
        {"ppo_v6": ppo, "sac_v3": sac, "human": human}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
