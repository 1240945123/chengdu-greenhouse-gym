"""优化前后对比 + 核心四强对比图集。

Part 1 优化效果：PPO v6→v7、SAC v3→v4（reward 塑形优化）
Part 2 核心四强：PPO v7 / SAC v4 / MPC / PID —— 指标 + 状态轨迹 + 控制

统一评估口径（ENV_KW，comfort/smooth 权重置 0 以保证 reward 可比、物理指标不受影响）。
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
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
              comfort_weight=0.0, smooth_weight=0.0)

COLORS = {"PPO v6": "#79addc", "PPO v7": "#2c6fbb", "SAC v3": "#f0b27a", "SAC v4": "#e67e22",
          "MPC": "#c0392b", "PID": "#8e44ad", "规则控制": "#27ae60", "固定基准": "#95a5a6"}
ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]
KEY_ACT = [("a0", "外遮阳"), ("a3", "顶窗"), ("a6", "风机"), ("a7", "水泵")]


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def make_act(path, kind):
    if kind == "ppo":
        from stable_baselines3 import PPO
        m = PPO.load(str(path))
        return lambda e: m.predict(e._get_obs(), deterministic=True)[0]
    if kind == "sac":
        from stable_baselines3 import SAC
        from experiments.controllers.glass_rl.env_variants import continuous_to_levels
        m = SAC.load(str(path))
        return lambda e: continuous_to_levels(m.predict(e._get_obs(), deterministic=True)[0])
    if kind == "mpc":
        from experiments.controllers.glass_rl.classical_controllers import GlassMPC
        c = GlassMPC()
        return lambda e: c.predict_action(e)
    if kind == "pid":
        from experiments.controllers.glass_rl.classical_controllers import GlassPID
        c = GlassPID()
        return lambda e: c.predict_action(e)
    from experiments.controllers.glass_rl.classical_controllers import GlassRuleBased
    c = GlassRuleBased()
    return lambda e: c.predict_action(e)


def rollout(path, kind, label):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    act = make_act(path, kind)
    recs = []
    for i in range(102 * 24):
        w = env.weather.iloc[min(env._w_idx, len(env.weather) - 1)]
        lv = np.asarray(act(env)).reshape(-1)
        obs, r, term, tr, info = env.step(lv)
        row = {"day": i // 24, "hour": i % 24, "t": info["temperature"],
               "rh": info["rh"], "out_t": float(w["temp"]), "out_rh": float(w["rh"])}
        for j in range(9):
            row[f"a{j}"] = int(lv[j])
        recs.append(row)
    df = pd.DataFrame(recs)
    df["is_day"] = (df.hour >= 6) & (df.hour < 20)
    df["month"] = pd.cut(df.day, bins=[-1, 30, 61, 91, 102], labels=["4月", "5月", "6月", "7月"])
    day, night = df[df.is_day], df[~df.is_day]
    mc = {m: float(np.mean([comfort(r.t, r.rh, r.hour) for r in g.itertuples()]) * 100)
          for m, g in df.groupby("month", observed=True)}
    s = {
        "comfort_pct": float(np.mean([comfort(r.t, r.rh, r.hour) for r in df.itertuples()]) * 100),
        "mean_temp": float(df.t.mean()), "max_temp": float(df.t.max()),
        "rh_over85_day": float((day.rh > 85).mean() * 100),
        "rh_over85_night": float((night.rh > 85).mean() * 100),
        "fruit_kg": float((float(env.x[25]) - fruit0) * 1e-6 / 0.081),
        "month_comfort": mc,
    }
    return s, df


def main():
    specs = [
        ("PPO v6", RL / "ppo_final_v6/model", "ppo"),
        ("PPO v7", RL / "ppo_final_v7/model", "ppo"),
        ("SAC v3", RL / "sac_final_v3/model", "sac"),
        ("SAC v4", RL / "sac_final_v4/model", "sac"),
        ("MPC", None, "mpc"),
        ("PID", None, "pid"),
        ("规则控制", None, "rule"),
    ]
    res, dfs = {}, {}
    for label, path, kind in specs:
        print(f"rollout {label} ...", flush=True)
        s, df = rollout(path, kind, label)
        res[label] = s
        dfs[label] = df
        print(f"  {label}: 舒适率 {s['comfort_pct']:.1f}% 果实 {s['fruit_kg']:.2f} "
              f"最高温 {s['max_temp']:.1f}", flush=True)
    (RL / "final_comparison_optimization.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    days = np.arange(102)
    METRICS = [("comfort_pct", "舒适率 %"), ("fruit_kg", "果实 kg/m²"),
               ("max_temp", "最高温 °C"), ("mean_temp", "均温 °C"),
               ("rh_over85_day", "白天湿度>85% %"), ("rh_over85_night", "夜间湿度>85% %")]

    # ===== Part 1：优化前后 =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, nm) in zip(axes.ravel(), METRICS):
        groups = [("PPO", ["PPO v6", "PPO v7"]), ("SAC", ["SAC v3", "SAC v4"])]
        xs, labels, vals, cols = [], [], [], []
        pos = 0
        for gname, members in groups:
            for m in members:
                xs.append(pos); labels.append(m.replace(" ", "\n")); vals.append(res[m][k])
                cols.append(COLORS[m]); pos += 1
            pos += 0.6
        bars = ax.bar(xs, vals, color=cols)
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(nm); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("优化前后对比：PPO v6→v7、SAC v3→v4（comfort_bonus + smooth_pen）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "opt_metrics.png", dpi=130); plt.close(fig)

    # ===== Part 2：核心四强 =====
    core = ["PPO v7", "SAC v4", "MPC", "PID"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (k, nm) in zip(axes.ravel(), METRICS):
        vals = [res[m][k] for m in core]
        bars = ax.bar(core, vals, color=[COLORS[m] for m in core])
        ax.set_title(nm); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("核心四强对比：PPO v7 vs SAC v4 vs MPC vs PID（102天，修复后环境）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "core_metrics.png", dpi=130); plt.close(fig)

    # 状态：温度
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(20, 28, color="#e8f5e9", alpha=0.6, label="白天舒适区 20-28°C")
    ax.plot(days, dfs[core[0]].groupby("day")["out_t"].mean().values,
            color="#888780", ls="--", lw=1.5, label="室外气温")
    for m in core:
        ax.plot(days, dfs[m].groupby("day")["t"].mean().values, color=COLORS[m], lw=2, label=m)
    ax.set_xlabel("定植后天数（4/1 起）"); ax.set_ylabel("逐日均温 °C")
    ax.set_title("温室温度轨迹：核心四强对比"); ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "core_state_temperature.png", dpi=130); plt.close(fig)

    # 状态：湿度
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(60, 85, color="#e8f5e9", alpha=0.6, label="舒适湿度 60-85%")
    ax.plot(days, dfs[core[0]].groupby("day")["out_rh"].mean().values,
            color="#888780", ls="--", lw=1.5, label="室外湿度")
    for m in core:
        ax.plot(days, dfs[m].groupby("day")["rh"].mean().values, color=COLORS[m], lw=2, label=m)
    ax.set_xlabel("定植后天数（4/1 起）"); ax.set_ylabel("逐日均湿 %")
    ax.set_title("温室湿度轨迹：核心四强对比"); ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "core_state_humidity.png", dpi=130); plt.close(fig)

    # 控制
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    for ax, (col, nm) in zip(axes.ravel(), KEY_ACT):
        for m in core:
            ax.plot(days, dfs[m].groupby("day")[col].mean().values, color=COLORS[m], lw=1.8, label=m)
        ax.set_title(nm); ax.grid(alpha=0.3); ax.set_xlabel("天数")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("控制动作逐日档位：核心四强对比", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "core_control_lines.png", dpi=130); plt.close(fig)

    # 分月舒适率
    fig, ax = plt.subplots(figsize=(10, 5))
    months = ["4月", "5月", "6月", "7月"]; x = np.arange(4); w = 0.2
    for i, m in enumerate(core):
        vals = [res[m]["month_comfort"][mm] for mm in months]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(months); ax.set_ylabel("舒适率 %")
    ax.set_title("分月舒适率：核心四强对比"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "core_month_comfort.png", dpi=130); plt.close(fig)

    print(f"\n完成，图集: {OUT}")


if __name__ == "__main__":
    main()
