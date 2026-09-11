"""PPO 优化前后 A/B 对比图（基线 ppo_final_v2 vs 优化 ppo_final_v3）。

生成：
1. 训练 reward 收敛曲线对比（滑动平均）
2. 关键指标分组柱状图（舒适率/湿度超标/过冷/果实）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = BASE / "ppo_final_v3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

RED = "#c0392b"
BLUE = "#2c6fbb"


def smooth(x, w=11):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def main():
    a = pd.read_csv(BASE / "ppo_final_v2" / "training_curve.csv")
    b = pd.read_csv(BASE / "ppo_final_v3" / "training_curve.csv")
    ta = a["timestep"].values / 1000
    tb = b["timestep"].values / 1000

    # ---------- 图1：reward 收敛对比 ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    ra, rb = a["rollout/ep_rew_mean"].values, b["rollout/ep_rew_mean"].values
    ax.plot(ta, smooth(ra), color=BLUE, lw=2.5, label="基线A（7维obs + 室外温差降温奖励）")
    ax.plot(tb, smooth(rb), color=RED, lw=2.5, label="优化B（8维obs + 过热缓解 + 湿度×2）")
    ax.set_xlabel("训练步数 (×1000)")
    ax.set_ylabel("平均 episode reward（滑动平均）")
    ax.set_title("PPO 训练收敛对比：基线 vs 优化")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ab_reward_curve.png", dpi=130)
    plt.close(fig)

    # ---------- 图2：关键指标对比 ----------
    metrics = [
        ("舒适率 %", 23.0, 49.3),
        ("白天湿度>85% %", 48.5, 22.0),
        ("夜间湿度>85% %", 75.3, 37.8),
        ("白天<20°C %", 43.1, 23.0),
        ("果实 kg/m²", 7.4, 8.2),
    ]
    names = [m[0] for m in metrics]
    a_vals = [m[1] for m in metrics]
    b_vals = [m[2] for m in metrics]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.2, a_vals, 0.38, color=BLUE, label="基线A")
    ax.bar(x + 0.2, b_vals, 0.38, color=RED, label="优化B")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("数值")
    ax.set_title("PPO 优化前后关键指标对比（越低越好，除舒适率/果实）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (va, vb) in enumerate(zip(a_vals, b_vals)):
        ax.text(i - 0.2, va + 1, f"{va:.0f}", ha="center", fontsize=9)
        ax.text(i + 0.2, vb + 1, f"{vb:.0f}", ha="center", fontsize=9, color=RED)
    fig.tight_layout()
    fig.savefig(OUT / "ab_metrics.png", dpi=130)
    plt.close(fig)

    print("已生成:", OUT / "ab_reward_curve.png")
    print("已生成:", OUT / "ab_metrics.png")


if __name__ == "__main__":
    main()
