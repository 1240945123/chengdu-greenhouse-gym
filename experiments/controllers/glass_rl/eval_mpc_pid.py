"""经典控制器全季评估（PID / MPC / 规则 / 固定基准），统一 ENV_KW 口径。

输出：final_comparison_classical.json（指标 + 逐时轨迹）
MPC 为逐时候选搜索，较慢，建议后台运行。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT_JSON = RL / "final_comparison_classical.json"
DAYS = 102

ENV_KW = dict(episode_days=DAYS, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5)


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def run(ctrl, name):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    recs = []
    t0 = time.time()
    for i in range(DAYS * 24):
        a = ctrl.predict_action(env)
        obs, r, term, tr, info = env.step(a)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"],
                     "rh": info["rh"], "co2": float(env.x[0])})
        if (i + 1) % (24 * 10) == 0:
            print(f"  [{name}] day {(i+1)//24}/{DAYS} ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    df["is_day"] = (df.hour >= 6) & (df.hour < 20)
    df["month"] = pd.cut(df.day, bins=[-1, 30, 61, 91, 102], labels=["4月", "5月", "6月", "7月"])
    day, night = df[df.is_day], df[~df.is_day]
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    mc = {m: float(np.mean([comfort(r.t, r.rh, r.hour) for r in g.itertuples()]) * 100)
          for m, g in df.groupby("month", observed=True)}
    return {
        "comfort_pct": float(np.mean([comfort(r.t, r.rh, r.hour) for r in df.itertuples()]) * 100),
        "mean_temp": float(df.t.mean()), "max_temp": float(df.t.max()),
        "rh_over85_day": float((day.rh > 85).mean() * 100),
        "rh_over85_night": float((night.rh > 85).mean() * 100),
        "fruit_kg": float(fruit), "elapsed_s": round(time.time() - t0, 1),
        "month_comfort": mc,
        "traj_temp": [round(float(v), 2) for v in df.t.values],
        "traj_rh": [round(float(v), 1) for v in df.rh.values],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controllers", nargs="+", default=["baseline", "rule", "pid", "mpc"])
    args = p.parse_args()

    from experiments.controllers.glass_rl.classical_controllers import (
        GlassBaseline, GlassRuleBased, GlassPID, GlassMPC)
    reg = {"baseline": GlassBaseline, "rule": GlassRuleBased, "pid": GlassPID, "mpc": GlassMPC}

    out = {}
    if OUT_JSON.exists():
        out = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    for name in args.controllers:
        print(f"评估 {name} ...", flush=True)
        out[name] = run(reg[name](), name)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        s = out[name]
        print(f"  {name}: 舒适率 {s['comfort_pct']:.1f}% 均温 {s['mean_temp']:.1f} "
              f"最高 {s['max_temp']:.1f} 果实 {s['fruit_kg']:.2f} ({s['elapsed_s']}s)", flush=True)
    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
