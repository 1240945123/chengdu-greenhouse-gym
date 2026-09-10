"""60 天窗口（5/1-6/30）多算法对比完整图集。

覆盖所有曲线维度：
1. 关键指标汇总（6 子图）
2. 日平均温度轨迹（11 策略）+ 日最高温
3. 日平均湿度轨迹（11 策略）
4. 控制曲线（关键执行器，经典控制 + RL 分组）
5. 分月舒适率 + 分月最高温

注：本结果为「冠层过热 + 遮阳透光 bug 修复前」的 60 天窗口；温湿度控制指标可靠，
果实受遮阳 bug 影响被系统性低估（见 04 号文档）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")
from plotting import COLORS, LABELS, ORDER, _label, _ordered_keys

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = "results/chengdu_agri_greenhouse_001/real_greenhouse/rl"
FIG = Path(RL) / "figures_60day"
FIG.mkdir(parents=True, exist_ok=True)

ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO₂", "风机", "水泵", "卷膜"]
KEY_ACTS = [0, 3, 6, 7, 8]  # 遮阳/顶窗/风机/水泵/卷膜


def save(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {p}")
    return p


def main():
    bm = json.load(open(f"{RL}/benchmark_all_d30_60.json", encoding="utf-8"))
    tr = json.load(open(f"{RL}/benchmark_all_d30_60_trajectories.json", encoding="utf-8"))
    keys = _ordered_keys(bm)
    n = len(next(iter(tr.values()))["temp"])
    days = n / 24.0
    hours = np.arange(n) / 24.0

    # 图1：关键指标汇总（6 子图）
    metrics = [
        ("mean_temp", "平均温度", "°C"),
        ("max_temp", "最高温度", "°C"),
        ("comfort", "舒适率", "%"),
        ("total_reward", "累计奖励", ""),
        ("mean_effort", "平均能耗", ""),
        ("fruit_growth_kg_m2", "果实增长", "kg/m²"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title, unit) in zip(axes.ravel(), metrics):
        vals = [bm[k][key] for k in keys]
        colors = [COLORS[k] for k in keys]
        bars = ax.bar([_label(k) for k in keys], vals, color=colors, edgecolor="white")
        ax.set_title(f"{title}（{unit}）" if unit else title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for b, v in zip(bars, vals):
            fmt = f"{v:.4f}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 100 else f"{v:.0f}")
            ax.text(b.get_x() + b.get_width() / 2, v, fmt,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.suptitle("60 天窗口（5/1-6/30）各控制策略关键指标对比", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "01_controller_metrics")

    # 图2：温度轨迹（日平均 + 日最高温）
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    ax1, ax2 = axes
    for k in keys:
        t = np.asarray(tr[k]["temp"])
        d = t.reshape(-1, 24)
        ax1.plot(np.arange(d.shape[0]), d.mean(axis=1), color=COLORS[k], lw=1.4, label=_label(k))
        ax2.plot(np.arange(d.shape[0]), d.max(axis=1), color=COLORS[k], lw=1.4, alpha=0.9)
    for ax in axes:
        ax.axhspan(20, 28, color="green", alpha=0.06)
        ax.grid(alpha=0.3, linestyle="--")
    ax1.axhline(28, color="red", ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(28, color="red", ls="--", lw=0.8, alpha=0.5)
    ax1.set_ylabel("日均温 °C"); ax2.set_ylabel("日最高温 °C")
    ax1.set_title("日平均温度轨迹（11 策略）", fontsize=13)
    ax2.set_title("日最高温度轨迹（11 策略）", fontsize=13)
    ax1.set_ylim(10, 42); ax2.set_ylim(10, 52)
    ax1.legend(ncol=6, fontsize=8, loc="upper left")
    ax2.set_xlabel("天数（自 5/1 起）")
    fig.suptitle("60 天窗口温度轨迹对比", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "02_temperature_trajectory")

    # 图3：湿度轨迹（日平均）
    fig, ax = plt.subplots(figsize=(15, 5.5))
    for k in keys:
        r = np.asarray(tr[k]["rh"])
        d = r.reshape(-1, 24).mean(axis=1)
        ax.plot(np.arange(d.shape[0]), d, color=COLORS[k], lw=1.4, label=_label(k))
    ax.axhspan(60, 85, color="blue", alpha=0.06)
    ax.axhline(60, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(85, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("天数（自 5/1 起）"); ax.set_ylabel("日平均相对湿度 %")
    ax.set_title("60 天窗口湿度轨迹对比（11 策略，适宜 60-85%）", fontsize=14, fontweight="bold")
    ax.legend(ncol=6, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    save(fig, "03_humidity_trajectory")

    # 图4：控制曲线（关键执行器，分经典/RL 两组）
    classic = [k for k in keys if k in ("human", "baseline", "rule", "pid", "mpc")]
    rl_keys = [k for k in keys if k not in classic]
    for grp_name, grp in [("经典控制", classic), ("RL", rl_keys)]:
        fig, axes = plt.subplots(len(KEY_ACTS), 1, figsize=(15, 2.4 * len(KEY_ACTS)), sharex=True)
        for ax, ai in zip(axes, KEY_ACTS):
            for k in grp:
                a = np.asarray(tr[k]["action"])
                if a.ndim == 1:
                    a = a.reshape(-1, 9)
                daily = a[:, ai].reshape(-1, 24).mean(axis=1)
                ax.plot(np.arange(daily.shape[0]), daily, color=COLORS[k], lw=1.2,
                        label=_label(k), drawstyle="steps-post")
            ax.set_ylabel(f"{ACT_NAMES[ai]}档位", fontsize=10)
            ax.set_ylim(-0.1, 3.4)
            ax.legend(ncol=len(grp), fontsize=8, loc="upper right")
            ax.grid(alpha=0.3, linestyle="--")
        axes[-1].set_xlabel("天数（自 5/1 起）")
        fig.suptitle(f"关键执行器控制曲线（{grp_name}，日平均档位）", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        save(fig, f"04_control_{'classic' if grp_name == '经典控制' else 'rl'}")

    # 图5：分月舒适率 + 分月最高温
    months = ["5月", "6月"]
    x = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, metric, title, ylim in [(axes[0], "comfort", "分月舒适率 %", 80),
                                    (axes[1], "max_temp", "分月最高温 °C", 48)]:
        for i, m in enumerate(months):
            vals = []
            for k in keys:
                mm = {mmx["month"]: mmx[metric] for mmx in bm[k].get("monthly", [])}
                vals.append(mm.get(m, np.nan))
            ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, label=m, edgecolor="white")
        if metric == "max_temp":
            ax.axhline(28, color="red", ls="--", lw=1, label="舒适上限 28°C")
        ax.set_xticks(x); ax.set_xticklabels([_label(k) for k in keys], fontsize=8, rotation=45)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, ylim)
    fig.suptitle("60 天窗口分月对比（5 月 / 6 月）", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "05_monthly_comparison")

    print("\n全部图已生成完毕")


if __name__ == "__main__":
    main()
