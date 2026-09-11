"""PPO 训练过程全参数曲线绘制。

读取 train_ppo_final.py 产出的 training_curve.csv，绘制全部关键指标：
- rollout/ep_rew_mean（平均 episode reward，主收敛指标）
- train/loss、value_loss、policy_gradient_loss（三类损失）
- train/entropy_loss（探索熵）、approx_kl（KL）、clip_fraction（裁剪比例）
- train/learning_rate、explained_variance

输出一张 3×3 大图 + 一张 reward 大图。
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

CSV = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v2/training_curve.csv")
OUT = CSV.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def smooth(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def main():
    df = pd.read_csv(CSV)
    ts = df["timestep"].values / 1000  # 千步

    # ---------- 图1：reward 收敛（主图） ----------
    fig, ax = plt.subplots(figsize=(12, 5))
    if df["rollout/ep_rew_mean"].notna().any():
        ax.plot(ts, df["rollout/ep_rew_mean"], color="#2c6fbb", alpha=0.35, lw=1, label="原始")
        ax.plot(ts, smooth(df["rollout/ep_rew_mean"], 11), color="#2c6fbb", lw=2.5,
                label="滑动平均(11 rollout)")
    ax.set_xlabel("训练步数 (×1000)")
    ax.set_ylabel("平均 episode reward")
    ax.set_title("PPO 训练收敛曲线（修复后环境）")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "training_reward.png", dpi=130)
    plt.close(fig)

    # ---------- 图2：3×3 全参数 ----------
    panels = [
        ("rollout/ep_rew_mean", "平均 reward", "tab:blue", None),
        ("train/loss", "总损失 train/loss", "tab:red", None),
        ("train/value_loss", "价值损失 value_loss", "tab:orange", None),
        ("train/policy_gradient_loss", "策略梯度损失", "tab:green", None),
        ("train/entropy_loss", "熵损失（探索）", "tab:purple", None),
        ("train/approx_kl", "近似 KL 散度", "tab:brown", None),
        ("train/clip_fraction", "裁剪比例", "tab:pink", None),
        ("train/learning_rate", "学习率", "tab:cyan", None),
        ("train/explained_variance", "解释方差", "tab:gray", None),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (col, title, color, ylim) in zip(axes.flat, panels):
        if col in df.columns and df[col].notna().any():
            ax.plot(ts, df[col], color=color, lw=1.5)
            if col == "rollout/ep_rew_mean":
                ax.plot(ts, smooth(df[col], 11), color=color, lw=2.5, alpha=0.7)
        else:
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("步数(×1000)")
        ax.grid(alpha=0.3)
    fig.suptitle("PPO 训练过程全参数曲线（60 万步，修复后环境）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "training_all_params.png", dpi=130)
    plt.close(fig)

    print("已生成：", OUT / "training_reward.png")
    print("已生成：", OUT / "training_all_params.png")
    print("共", len(df), "个 rollout 记录点")


if __name__ == "__main__":
    main()
