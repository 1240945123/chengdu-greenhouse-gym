"""PPO 三版优化对比图（v6 基线 / v7 comfort0.3+smooth0.1 / v8 comfort0.6+smooth0）。

输出：ppo_variants_metrics.png（关键指标 + 失配分解）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "figures_optimization"
OUT.mkdir(parents=True, exist_ok=True)

ENV_KW = dict(episode_days=102, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5)
VARIANTS = [("PPO v6", "ppo_final_v6/model", "#79addc"),
            ("PPO v7", "ppo_final_v7/model", "#f0b27a"),
            ("PPO v8", "ppo_final_v8/model", "#2c6fbb")]


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def ev(path):
    from stable_baselines3 import PPO
    from glass_env import GlassGreenhouseEnv
    m = PPO.load(str(path))
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    f0 = float(env.x[25])
    rec = []
    for i in range(102 * 24):
        a = m.predict(env._get_obs(), deterministic=True)[0]
        o, r, te, tr, info = env.step(a)
        rec.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    df = pd.DataFrame(rec)
    df["is_day"] = (df.hour >= 6) & (df.hour < 20)
    d, n = df[df.is_day], df[~df.is_day]
    return {
        "comfort": float(np.mean([comfort(x.t, x.rh, x.hour) for x in df.itertuples()]) * 100),
        "fruit": float((float(env.x[25]) - f0) * 1e-6 / 0.081),
        "max_t": float(df.t.max()), "mean_t": float(df.t.mean()),
        "冷<20": float((d.t < 20).mean() * 100), "热>28": float((d.t > 28).mean() * 100),
        "干<60": float((d.rh < 60).mean() * 100), "湿>85": float((d.rh > 85).mean() * 100),
        "夜冷<16": float((n.t < 16).mean() * 100), "夜热>24": float((n.t > 24).mean() * 100),
        "夜湿>85": float((n.rh > 85).mean() * 100),
    }


res = {}
for label, rel, c in VARIANTS:
    print(f"eval {label} ...", flush=True)
    res[label] = ev(RL / rel)
(RL / "ppo_variants.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

labels = [v[0] for v in VARIANTS]
cols = [v[2] for v in VARIANTS]

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
panels = [("comfort", "舒适率 %"), ("fruit", "果实 kg/m²"), ("max_t", "最高温 °C"), ("mean_t", "均温 °C"),
          ("冷<20", "白天冷<20 %"), ("热>28", "白天热>28 %"), ("干<60", "白天干<60 %"), ("夜湿>85", "夜间湿>85 %")]
for ax, (k, nm) in zip(axes.ravel(), panels):
    vals = [res[l][k] for l in labels]
    bars = ax.bar(labels, vals, color=cols)
    ax.set_title(nm); ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("PPO 三版优化对比（v6 基线 / v7 comfort0.3+smooth0.1 / v8 comfort0.6+smooth0）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "ppo_variants_metrics.png", dpi=130)
plt.close(fig)
print("已生成:", OUT / "ppo_variants_metrics.png")
