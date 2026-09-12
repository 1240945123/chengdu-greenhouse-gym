"""五维系统评估可视化：性能 / 成功率 / 稳定性 / 样本效率 / 计算开销。

输入：eval_systematic.json
输出：systematic_dims.png（6 子图）+ systematic_radar.png（雷达图）
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "figures_systematic"
OUT.mkdir(parents=True, exist_ok=True)

d = json.loads((RL / "eval_systematic.json").read_text(encoding="utf-8"))
order = [k for k in ["PPO v6", "PPO v8", "SAC v3", "SAC v4", "残差PPO", "MPC", "规则", "人工"] if k in d]
COLORS = {"PPO v6": "#79addc", "PPO v8": "#2c6fbb", "SAC v3": "#f0b27a", "SAC v4": "#e67e22",
          "残差PPO": "#8e44ad", "MPC": "#c0392b", "规则": "#27ae60", "人工": "#95a5a6"}
KEY_TINT = ["#378ADD", "#EF9F27"]  # 双指标配色（第一/第二）


def grouped(ax, keys, title, logy=False):
    n = len(keys)
    x = np.arange(len(order))
    w = 0.36 if n == 2 else 0.6
    for i, (lab, k) in enumerate(keys):
        vals = [float(d[s].get(k) or 0) for s in order]
        pos = x + (i - (n - 1) / 2) * (w + 0.04)
        bars = ax.bar(pos, vals, w, label=lab,
                      color=[KEY_TINT[i]] * len(order) if n > 1 else [COLORS[s] for s in order])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=9)
    ax.set_title(title); ax.grid(axis="y", alpha=0.3)
    if logy:
        ax.set_yscale("log")
    if n > 1:
        ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 3, figsize=(17, 9))

grouped(axes[0, 0], [("舒适率 %", "comfort_pct"), ("果实 kg/m²", "fruit_kg")], "① 性能")
grouped(axes[0, 1], [("无过热日 %", "success_no_overheat_day"), ("舒适达标日 %", "success_comfort_day")], "② 成功率")
grouped(axes[0, 2], [("日均温标准差", "daily_t_std"), ("日内温度标准差", "intraday_t_std")], "③ 稳定性（越低越好）")

ax = axes[1, 0]
vals = [d[s]["decision_ms_mean"] for s in order]
bars = ax.bar(order, vals, color=[COLORS[s] for s in order])
ax.set_yscale("log"); ax.set_title("④ 计算开销：每步决策延迟（ms，对数轴）"); ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, (f"{v:.3f}" if v < 1 else f"{v:.1f}"), ha="center", va="bottom", fontsize=8)

ax = axes[1, 1]
vals = [float(d[s].get("train_minutes") or 0) for s in order]
bars = ax.bar(order, vals, color=[COLORS[s] for s in order])
ax.set_title("⑤ 样本效率：训练耗时（min；非学习类无训练）"); ax.grid(axis="y", alpha=0.3)
for b, v in zip(bars, vals):
    if v > 0:
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

ax = axes[1, 2]
learn = [(s, d[s]["sample_eff"].get("steps_to_80pct")) for s in order]
learn = [(s, v) for s, v in learn if v]
bars = ax.bar([s for s, _ in learn], [v / 1000 for _, v in learn], color=[COLORS[s] for s, _ in learn])
ax.set_title("⑥ 样本效率：达最终收益 80% 所需步数（×1000）"); ax.set_ylabel("步数 (×1000)")
ax.grid(axis="y", alpha=0.3)
for b, (s, v) in zip(bars, learn):
    lab = f"{v/1000:.0f}k" + ("*" if d[s]["sample_eff"].get("curve_start", 0) > 10000 else "")
    ax.text(b.get_x() + b.get_width() / 2, b.get_height(), lab, ha="center", va="bottom", fontsize=9)
ax.text(0.02, 0.96, "*SAC v4 为续训曲线（仅 300k–600k 段），步数为段内计数",
        transform=ax.transAxes, fontsize=8, va="top", color="#a32d2d")

fig.suptitle("五维系统评估：PPO / SAC / MPC / 规则 / 人工（102 天，统一环境与任务）", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "systematic_dims.png", dpi=130)
plt.close(fig)

# ===== 雷达图 =====
dims = ["性能\n(舒适率)", "成功率\n(舒适达标日)", "稳定性\n(1/日均温std)", "运行效率\n(1/决策延迟)", "训练效率\n(1/训练耗时)"]


def norm_hi(vals):
    v = np.asarray(vals, dtype=float)
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn) if mx > mn else np.ones_like(v)


M = np.vstack([
    norm_hi([d[s]["comfort_pct"] for s in order]),
    norm_hi([d[s]["success_comfort_day"] for s in order]),
    norm_hi([1.0 / d[s]["daily_t_std"] for s in order]),
    norm_hi([1.0 / (d[s]["decision_ms_mean"] + 1e-4) for s in order]),
    norm_hi([1.0 / (float(d[s].get("train_minutes") or 0) + 1.0) for s in order]),
]).T

ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
ang += ang[:1]
fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True))
for i, s in enumerate(order):
    v = M[i].tolist() + [M[i][0]]
    ax.plot(ang, v, color=COLORS[s], lw=2, label=s)
    ax.fill(ang, v, color=COLORS[s], alpha=0.07)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(dims, fontsize=10)
ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=8)
ax.set_title("五维能力雷达图（各维归一化，越外越好）", fontsize=13, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "systematic_radar.png", dpi=130)
plt.close(fig)
print(f"已生成: {OUT}")
