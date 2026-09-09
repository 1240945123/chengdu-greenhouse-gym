"""玻璃温室干净评估（CO2 修复后，生长季起点，禁补光/CO2）。

人工（真实精确档位动作） vs RL（glass_ppo_clean）。
口径：day30(5/1) 起 72 天到 day102(7/12 清棚)，early_fruiting 坐果初株，禁补光/CO2。
用法：python .tmp/full_season_benchmark_v3.py --model results/.../glass_ppo_clean/model.zip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, ".tmp")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
DATA = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
ACTUATOR_STATES = "results/chengdu_agri_greenhouse_001/real_greenhouse/aligned_actuator_states.csv"

ACT_COLS = ["shade_cloth", "top_insulation", "side_insulation", "roof_vent",
            "grow_lamp", "co2_gen", "pad_fan", "pad_pump", "pad_curtain"]
FAN_MAP = {0: 0, 1: 1, 2: 2, 4: 3}


def load_human_actions() -> pd.DataFrame:
    a = pd.read_csv(ACTUATOR_STATES)
    a["timestamp"] = pd.to_datetime(a["timestamp"])
    out = {"timestamp": a["timestamp"].to_numpy()}
    for col in ACT_COLS:
        vals = a[col].to_numpy(dtype=float)
        if col == "pad_fan":
            out[col] = np.array([FAN_MAP.get(int(v), 0) if not pd.isna(v) else 0 for v in vals])
        elif col == "roof_vent":
            out[col] = np.array([int(round(v)) if not pd.isna(v) else 0 for v in vals])
        else:
            out[col] = np.where(pd.isna(vals), 0, np.where(vals > 0, 1, 0)).astype(int)
    return pd.DataFrame(out).set_index("timestamp")


def run_strategy(name, actions, env_kwargs) -> dict:
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**env_kwargs)
    env.reset(seed=0)
    env._prev_fruit = float(env.x[25])
    temps, rhs, rewards, efforts, fruits, hours = [], [], [], [], [], []
    for i in range(len(actions)):
        obs, r, term, tr, info = env.step(actions[i])
        if not np.isfinite(env.x).all():
            print(f"[{name}] NaN at {i}, abort"); break
        temps.append(info["temperature"]); rhs.append(info["rh"])
        rewards.append(r); efforts.append(info["effort"])
        fruits.append(info["fruit_mg"]); hours.append(float(info["hour"]))
    df = pd.DataFrame({"hour": np.tile(np.arange(24), env_kwargs["episode_days"])[: len(temps)],
                       "temp": temps, "rh": rhs, "reward": rewards, "effort": efforts})
    day = (df["hour"] >= 6) & (df["hour"] < 20)
    comfort = (np.where(day, df["temp"].between(20, 28), df["temp"].between(16, 24)) & df["rh"].between(60, 85)).mean() * 100
    fg = fruits[-1] - fruits[0]
    monthly = []
    df["day"] = np.arange(len(df)) // 24
    df["month"] = pd.cut(df["day"], bins=[-1, 30, 60, 200], labels=["5月", "6月", "7月"])
    for m, g in df.groupby("month", observed=True):
        day = (g["hour"] >= 6) & (g["hour"] < 20)
        monthly.append({"month": m, "mean_temp": round(float(g["temp"].mean()), 2),
                        "max_temp": round(float(g["temp"].max()), 2),
                        "comfort": round(float((np.where(day, g["temp"].between(20, 28), g["temp"].between(16, 24)) & g["rh"].between(60, 85)).mean()) * 100, 1),
                        "total_reward": round(float(g["reward"].sum()), 1)})
    return {"name": name,
            "season_mean_temp": round(float(np.mean(temps)), 2),
            "season_max_temp": round(float(np.max(temps)), 2),
            "season_comfort": round(float(comfort), 1),
            "season_total_reward": round(float(np.sum(rewards)), 1),
            "season_mean_effort": round(float(np.mean(efforts)), 4),
            "fruit_start_mg": round(float(fruits[0]), 0), "fruit_end_mg": round(float(fruits[-1]), 0),
            "fruit_growth_kg_m2": round(float(fg * 1e-6 / 0.081), 3),
            "monthly": monthly}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="RL model zip 路径")
    parser.add_argument("--start-day", type=int, default=30)
    parser.add_argument("--days", type=int, default=72)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    data = pd.read_csv(DATA)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    ha = load_human_actions()

    env_kwargs = dict(episode_days=args.days, start_day_index=args.start_day,
                      yield_weight=1.0, crop_start="early_fruiting", disable_supplements=True)
    model = PPO.load(args.model)

    # human 动作
    ha2 = ha.iloc[args.start_day * 24: (args.start_day + args.days) * 24].reset_index(drop=True)
    human_acts = [np.array([ha2.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha2))]

    out_name = f"benchmark_v3_human_vs_rl_d{args.start_day}_{args.days}"
    results = {}
    results["human"] = run_strategy("human", human_acts, env_kwargs)
    print(f"[human] 均温{results['human']['season_mean_temp']}°C 舒适{results['human']['season_comfort']}% "
          f"reward{results['human']['season_total_reward']} 果实{results['human']['fruit_growth_kg_m2']} kg/m2")

    rl_acts = []
    env_tmp = __import__("glass_env").GlassGreenhouseEnv(episode_days=args.days, start_day_index=args.start_day,
                                                          yield_weight=1.0, crop_start="early_fruiting", disable_supplements=True)
    env_tmp.reset(seed=0)
    for i in range(args.days * 24):
        rl_acts.append(model.predict(env_tmp._get_obs(), deterministic=True)[0])
        env_tmp.step(rl_acts[-1])
    results["rl"] = run_strategy("rl", rl_acts, env_kwargs)
    print(f"[rl] 均温{results['rl']['season_mean_temp']}°C 舒适{results['rl']['season_comfort']}% "
          f"reward{results['rl']['season_total_reward']} 果实{results['rl']['fruit_growth_kg_m2']} kg/m2")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / f"{out_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"已保存 {out_name}.json")


if __name__ == "__main__":
    main()
