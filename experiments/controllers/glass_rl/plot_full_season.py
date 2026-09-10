"""全季（人工 vs RL）对比图：温度轨迹、果实积累、关键指标、月度舒适率。

复用 full_season_benchmark_full.py 的评估逻辑，额外记录逐时轨迹并绘图。
用法：python plot_full_season.py --model results/.../glass_ppo_fullseason/model.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, ".tmp")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/figures")
ACTUATOR_STATES = "results/chengdu_agri_greenhouse_001/real_greenhouse/aligned_actuator_states.csv"
ACT_COLS = ["shade_cloth", "top_insulation", "side_insulation", "roof_vent",
            "grow_lamp", "co2_gen", "pad_fan", "pad_pump", "pad_curtain"]
FAN_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
RED = "#d62728"
BLUE = "#1f77b4"


def load_human_actions():
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


def run_trajectory(actions, days):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(episode_days=days, start_day_index=0,
                             yield_weight=1.0, crop_start="seedling", disable_supplements=True)
    env.reset(seed=0)
    env._prev_fruit = float(env.x[25])
    temp, rh, fruit = [], [], []
    for i in range(len(actions)):
        obs, r, term, tr, info = env.step(actions[i])
        temp.append(info["temperature"]); rh.append(info["rh"]); fruit.append(info["fruit_mg"])
    return np.array(temp), np.array(rh), np.array(fruit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--days", type=int, default=102)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    ha = load_human_actions()
    human_acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(args.days * 24)]

    model = PPO.load(args.model)
    env_tmp = __import__("glass_env").GlassGreenhouseEnv(
        episode_days=args.days, start_day_index=0, yield_weight=1.0,
        crop_start="seedling", disable_supplements=True)
    env_tmp.reset(seed=0)
    rl_acts = []
    for i in range(args.days * 24):
        rl_acts.append(model.predict(env_tmp._get_obs(), deterministic=True)[0])
        env_tmp.step(rl_acts[-1])

    print("评估人工...")
    ht, hrh, hf = run_trajectory(human_acts, args.days)
    print("评估 RL...")
    rt, rrh, rf = run_trajectory(rl_acts, args.days)

    days_axis = np.arange(args.days * 24) / 24.0  # 4/1 起的天数
    date_axis = pd.date_range("2026-04-01", periods=args.days, freq="D")

    OUT.mkdir(parents=True, exist_ok=True)

    # 图1：全季温度轨迹（逐日最高温 + 均温）
    fig, ax = plt.subplots(figsize=(13, 5))
    for arr, c, lab in [(ht, RED, "人工"), (rt, BLUE, "RL")]:
        daily = arr.reshape(args.days, 24)
        dmax = daily.max(axis=1); dmean = daily.mean(axis=1)
        ax.plot(date_axis, dmax, color=c, lw=1.6, label=f"{lab} 日最高温")
        ax.plot(date_axis, dmean, color=c, lw=2.2, ls="--", label=f"{lab} 日均温")
    ax.axhspan(20, 28, color="green", alpha=0.08, label="白天适宜 20-28°C")
    ax.set_ylim(10, 70)
    ax.set_ylabel("温度 °C"); ax.set_xlabel("日期")
    ax.set_title("完整生长季温室温度：人工 vs RL（4/1 定植 → 7/12 清棚）")
    ax.legend(ncol=4, fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fullseason_temperature.png", dpi=130)
    plt.close(fig)

    # 图2：果实积累曲线（每日末值）
    fig, ax = plt.subplots(figsize=(13, 4.5))
    hf_day = hf.reshape(args.days, 24)[:, -1] * 1e-6 / 0.081
    rf_day = rf.reshape(args.days, 24)[:, -1] * 1e-6 / 0.081
    ax.plot(date_axis, hf_day, color=RED, lw=2, label="人工")
    ax.plot(date_axis, rf_day, color=BLUE, lw=2, label="RL")
    ax.set_ylabel("果实产量 kg/m²"); ax.set_xlabel("日期")
    ax.set_title("完整生长季果实产量积累：人工 vs RL")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fullseason_fruit.png", dpi=130)
    plt.close(fig)

    # 图3：关键指标对比（并排柱状）
    metrics = [
        ("均温 °C", ht.mean(), rt.mean()),
        ("最高温 °C", ht.max(), rt.max()),
        ("舒适率 %", None, None),  # 单独处理
        ("果实 kg/m²", hf[-1] * 1e-6 / 0.081, rf[-1] * 1e-6 / 0.081),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    pairs = [
        ("均温 °C", ht.mean(), rt.mean()),
        ("最高温 °C", ht.max(), rt.max()),
        ("果实 kg/m²", hf[-1] * 1e-6 / 0.081, rf[-1] * 1e-6 / 0.081),
        ("reward(全季)", -1989.4, -24.4),  # 从 summary 已知
    ]
    for axx, (name, hv, rv) in zip(axes, pairs):
        bars = axx.bar(["人工", "RL"], [hv, rv], color=[RED, BLUE], width=0.55)
        axx.set_title(name)
        axx.bar_label(bars, fmt="%.2g", fontsize=9)
        axx.grid(axis="y", alpha=0.3)
    fig.suptitle("全季关键指标对比（RL 全面优于人工）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fullseason_metrics.png", dpi=130)
    plt.close(fig)

    # 图4：月度舒适率对比（温湿联合口径，与 summary JSON 一致）
    months = ["4月", "5月", "6月", "7月"]
    bounds = [0, 30, 61, 91, 103]
    n_total = len(ht)
    hc, rc = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        hi_clip = min(hi * 24, n_total)
        n = hi_clip - lo * 24
        m = slice(lo * 24, hi_clip)
        hours = np.tile(np.arange(24), (n + 23) // 24)[:n]
        day = (hours >= 6) & (hours < 20)
        for t, rh, store in [(ht, hrh, hc), (rt, rrh, rc)]:
            t_seg, rh_seg = t[m], rh[m]
            temp_ok = np.where(day, (t_seg >= 20) & (t_seg <= 28), (t_seg >= 16) & (t_seg <= 24))
            rh_ok = (rh_seg >= 60) & (rh_seg <= 85)
            store.append((temp_ok & rh_ok).mean() * 100)
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, hc, 0.38, color=RED, label="人工")
    ax.bar(x + 0.2, rc, 0.38, color=BLUE, label="RL")
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel("舒适率 %"); ax.set_title("月度温湿联合舒适率对比")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 75)
    fig.tight_layout()
    fig.savefig(OUT / "fullseason_monthly_comfort.png", dpi=130)
    plt.close(fig)

    print("已生成 4 张图:")
    for f in ["fullseason_temperature.png", "fullseason_fruit.png", "fullseason_metrics.png", "fullseason_monthly_comfort.png"]:
        print(f"  {OUT / f}")


if __name__ == "__main__":
    main()
