"""玻璃温室消融与敏感性分析（B4）。

Part 1 控制动作消融：以规则控制器为基策略，逐个禁用执行器，测性能降幅
        → 量化每个执行器对温控/产量的贡献。
Part 2 关键参数敏感性（OAT）：对 GlassGreenhouse 关键物理参数 ±20% 扰动，
        以真实室内温度为参照，测白天/夜间 RMSE 变化。

输出：ablation_*.json + 控制台表
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT_JSON = RL / "ablation_glass.json"
DAYS = 102

ENV_KW = dict(episode_days=DAYS, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5)

ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]
# 可消融的执行器（排除补光/CO2——训练期本就禁用）
ABLATE = [0, 1, 2, 3, 6, 7, 8]


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def rollout_rule(disable_idx: int | None = None, param_scale: dict | None = None):
    from glass_env import GlassGreenhouseEnv
    from experiments.controllers.glass_rl.classical_controllers import GlassRuleBased
    env = GlassGreenhouseEnv(**ENV_KW)
    if param_scale:
        for k, v in param_scale.items():
            env.params[int(k)] *= float(v)
    env.reset(seed=0)
    ctrl = GlassRuleBased()
    fruit0 = float(env.x[25])
    temps, rhs = [], []
    for i in range(DAYS * 24):
        a = ctrl.predict_action(env)
        if disable_idx is not None:
            a = np.asarray(a).copy()
            a[disable_idx] = 0
        obs, r, term, tr, info = env.step(a)
        temps.append(info["temperature"])
        rhs.append(info["rh"])
    temps = np.array(temps); rhs = np.array(rhs)
    hours = np.tile(np.arange(24), DAYS)[:len(temps)]
    d = (hours >= 6) & (hours < 20)
    return {
        "comfort_pct": float(np.mean([comfort(t, r, h) for t, r, h in zip(temps, rhs, hours)]) * 100),
        "fruit_kg": float((float(env.x[25]) - fruit0) * 1e-6 / 0.081),
        "max_temp": float(temps.max()),
        "mean_temp": float(temps.mean()),
        "day_mean_t": float(temps[d].mean()),
        "day_dry_pct": float((rhs[d] < 60).mean() * 100),
        "temps": temps, "rhs": rhs,
    }


def main():
    out = {}

    # ---------- Part 1: 控制动作消融 ----------
    print("=== Part 1 控制动作消融（基策略=规则）===", flush=True)
    base = rollout_rule()
    print(f"基准(全开): 舒适率 {base['comfort_pct']:.1f}% 果实 {base['fruit_kg']:.2f} 最高温 {base['max_temp']:.1f}", flush=True)
    ab = {}
    for idx in ABLATE:
        r = rollout_rule(disable_idx=idx)
        ab[ACT_NAMES[idx]] = {
            "comfort_pct": r["comfort_pct"], "fruit_kg": r["fruit_kg"], "max_temp": r["max_temp"],
            "d_comfort": r["comfort_pct"] - base["comfort_pct"],
            "d_fruit": r["fruit_kg"] - base["fruit_kg"],
            "d_max_temp": r["max_temp"] - base["max_temp"],
        }
        print(f"  禁用{ACT_NAMES[idx]:<6}: 舒适率 {r['comfort_pct']:5.1f}% (Δ{r['comfort_pct']-base['comfort_pct']:+6.1f}) "
              f"果实 {r['fruit_kg']:.2f} (Δ{r['fruit_kg']-base['fruit_kg']:+.2f}) 最高温 {r['max_temp']:.1f} (Δ{r['max_temp']-base['max_temp']:+.1f})",
              flush=True)
    out["action_ablation"] = {"baseline": {k: base[k] for k in ("comfort_pct", "fruit_kg", "max_temp")},
                              "disabled": ab}

    # ---------- Part 2: 关键参数敏感性（OAT ±20%）----------
    print("\n=== Part 2 关键参数敏感性（OAT，±20%）===", flush=True)
    params = [
        (209, "solar_gain_scale 辐射增益"),
        (210, "ua_scale 围护传热"),
        (211, "vent_scale 顶窗通风"),
        (220, "longwave_scale 长波辐射"),
        (208, "heat_capacity_scale 热容量"),
    ]
    sens = {}
    for idx, name in params:
        row = {}
        for tag, sc in [("-20%", 0.8), ("+20%", 1.2)]:
            r = rollout_rule(param_scale={idx: sc})
            row[tag] = {"comfort_pct": r["comfort_pct"], "fruit_kg": r["fruit_kg"],
                        "max_temp": r["max_temp"], "day_mean_t": r["day_mean_t"],
                        "day_dry_pct": r["day_dry_pct"]}
        span_t = abs(row["+20%"]["day_mean_t"] - row["-20%"]["day_mean_t"])
        span_c = abs(row["+20%"]["comfort_pct"] - row["-20%"]["comfort_pct"])
        row["span_day_temp"] = span_t
        row["span_comfort"] = span_c
        sens[name] = row
        print(f"  {name:<26}: 白天均温 span {span_t:.2f}°C  舒适率 span {span_c:.1f}pp  "
              f"(±20%: {row['-20%']['day_mean_t']:.1f}→{row['+20%']['day_mean_t']:.1f}°C)", flush=True)
    out["param_sensitivity"] = sens

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
