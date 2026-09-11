"""四者对比评估：PPO v6 vs SAC v3 vs 规则(rule) vs 固定基准(baseline)。

在修复后环境 + 统一配置下评估 rule/baseline，复用已计算的 PPO/SAC 结果，
输出四者对比 JSON + 对比图（关键指标 + 分月舒适率）。
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
OUT_JSON = RL_DIR / "final_comparison_four_way.json"
FIG_DIR = RL_DIR / "figures_four_way"

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


def run_controller(ctrl):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    recs = []
    for i in range(102 * 24):
        a = ctrl.predict_action(env)
        obs, r, term, tr, info = env.step(a)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    return summarize(pd.DataFrame(recs), fruit)


def main():
    from experiments.controllers.glass_rl.classical_controllers import GlassRuleBased, GlassBaseline

    prev = json.loads((RL_DIR / "final_comparison_ppo_sac_human.json").read_text(encoding="utf-8"))
    ppo, sac = prev["ppo_v6"], prev["sac_v3"]

    print("评估 规则控制(rule)...", flush=True)
    rule = run_controller(GlassRuleBased())
    print("评估 固定基准(baseline)...", flush=True)
    baseline = run_controller(GlassBaseline())

    data = {"ppo_v6": ppo, "sac_v3": sac, "rule": rule, "baseline": baseline}
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台表 ----
    order = [("PPO v6", "ppo_v6"), ("SAC v3", "sac_v3"), ("规则控制", "rule"), ("固定基准", "baseline")]
    print("\n" + "=" * 82)
    print("四者对比（102 天，修复后环境）")
    print("=" * 82)
    header = f"{'指标':<16} | " + " | ".join(f"{n:>9}" for n, _ in order)
    print(header)
    for name, key in [("舒适率 %", "comfort_pct"), ("均温 °C", "mean_temp"),
                      ("最高温 °C", "max_temp"), ("果实 kg/m²", "fruit_kg"),
                      ("白天湿度>85%", "rh_over85_day"), ("夜间湿度>85%", "rh_over85_night")]:
        vals = " | ".join(f"{data[k][key]:>9.1f}" for _, k in order)
        print(f"{name:<16} | {vals}")
    print("\n分月舒适率:")
    for m in ["4月", "5月", "6月", "7月"]:
        vals = " | ".join(f"{data[k]['month_comfort'][m]:>7.1f}%" for _, k in order)
        print(f"  {m}: {vals}")
    print(f"\n已保存: {OUT_JSON}")

    # ---- 对比图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labels = [n for n, _ in order]
    colors = ["#2c6fbb", "#e67e22", "#27ae60", "#95a5a6"]

    # 图1：关键指标（6 子图）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("comfort_pct", "舒适率 %", "越高越好"),
        ("fruit_kg", "果实 kg/m²", "越高越好"),
        ("max_temp", "最高温 °C", "越低越好"),
        ("mean_temp", "均温 °C", "接近舒适区"),
        ("rh_over85_day", "白天湿度>85% %", "越低越好"),
        ("rh_over85_night", "夜间湿度>85% %", "越低越好"),
    ]
    for ax, (k, name, note) in zip(axes.ravel(), metrics):
        vals = [data[kk][k] for _, kk in order]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(f"{name}（{note}）")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("四者对比：PPO v6 vs SAC v3 vs 规则控制 vs 固定基准（102天，修复后环境）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "four_way_metrics.png", dpi=130)
    plt.close(fig)

    # 图2：分月舒适率
    fig, ax = plt.subplots(figsize=(10, 5))
    months = ["4月", "5月", "6月", "7月"]
    x = np.arange(len(months))
    w = 0.2
    for i, ((name, kk), c) in enumerate(zip(order, colors)):
        vals = [data[kk]["month_comfort"][m] for m in months]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylabel("舒适率 %")
    ax.set_title("分月舒适率对比：PPO v6 vs SAC v3 vs 规则控制 vs 固定基准")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "four_way_month_comfort.png", dpi=130)
    plt.close(fig)
    print(f"已生成对比图: {FIG_DIR}")


if __name__ == "__main__":
    main()
