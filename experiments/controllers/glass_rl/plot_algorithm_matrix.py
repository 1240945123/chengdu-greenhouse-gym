"""全算法矩阵出图（读 algorithm_matrix.json）。

四张图：
  ① 舒适率排行（按算法族配色，标注模型预测类 / RL 各家族）
  ② 舒适率 × 果实 —— 二维性能面（标出 Pareto 前沿）
  ③ 舒适率 × 单次决策延迟（对数横轴）—— 成本-性能前沿（核心论点可视化）
  ④ 关键策略分月舒适率热力图 —— 暴露「7 月高温」为共同短板

用法：
  python -m experiments.controllers.glass_rl.plot_algorithm_matrix
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "algorithm_matrix"

# 算法族 → (显示名, 颜色)
FAMILY = {
    "human": ("人工", "#7f7f7f"),
    "baseline": ("经典", "#1f77b4"), "rule": ("经典", "#1f77b4"),
    "pid": ("经典", "#1f77b4"), "mpc": ("模型预测", "#d62728"),
    "lookahead_h1": ("模型预测", "#d62728"),
    "trpo": ("on-policy 离散", "#2ca02c"), "ppo_v9": ("on-policy 离散", "#2ca02c"),
    "ppo_v6": ("on-policy 离散", "#2ca02c"), "a2c": ("on-policy 离散", "#2ca02c"),
    "ppo": ("on-policy 离散", "#2ca02c"),
    "recurrent_ppo": ("on-policy 离散", "#2ca02c"),
    "sac": ("off-policy 连续", "#9467bd"), "dqn": ("off-policy 离散", "#8c564b"),
    "ddpg": ("off-policy 连续", "#9467bd"), "td3": ("off-policy 连续", "#9467bd"),
    "tqc": ("off-policy 连续", "#9467bd"), "crossq": ("off-policy 连续", "#9467bd"),
    "masked_ppo": ("P0/其他", "#e377c2"), "multiagent": ("P0/其他", "#e377c2"),
    "dagger_v1": ("P0/其他", "#e377c2"),
}
LABEL = {
    "human": "人工", "baseline": "Baseline", "rule": "规则", "pid": "PID",
    "mpc": "MPC(原)", "lookahead_h1": "前瞻H=1", "trpo": "TRPO", "ppo_v9": "PPO v9(obs16)",
    "ppo_v6": "PPO v6(obs8)", "ppo": "PPO(600k同协议)", "a2c": "A2C", "recurrent_ppo": "RecurrentPPO",
    "sac": "SAC", "dqn": "DQN", "ddpg": "DDPG", "td3": "TD3", "tqc": "TQC",
    "crossq": "CrossQ", "masked_ppo": "MaskablePPO", "multiagent": "多智能体IPPO",
    "dagger_v1": "DAgger v1",
}


def pareto(points):
    """返回 (舒适率↑, 果实↑) 二维上的非支配点集合。"""
    pts = np.asarray(points, dtype=float)
    keep = []
    for i in range(len(pts)):
        dominated = any(
            (pts[j, 0] >= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
             (pts[j] > pts[i]).any()) for j in range(len(pts)) if j != i)
        if not dominated:
            keep.append(i)
    return keep


def load_latency() -> dict:
    """空闲状态实测延迟（median ms）；缺失则回退到评估时的 decision_ms_mean。"""
    p = OUT / "latency_idle.json"
    if not p.exists():
        return {}
    return {k: v["median_ms"] for k, v in
            json.loads(p.read_text(encoding="utf-8")).items()}


def main() -> None:
    d = json.loads((OUT / "algorithm_matrix.json").read_text(encoding="utf-8"))
    lat = load_latency()
    keys = [k for k in d if k in FAMILY]

    def ms_of(k):
        return lat.get(k, d[k]["decision_ms_mean"])

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # ---------- ① 舒适率排行 ----------
    ax = axes[0, 0]
    order = sorted(keys, key=lambda k: d[k]["comfort_pct"])
    vals = [d[k]["comfort_pct"] for k in order]
    cols = [FAMILY[k][1] for k in order]
    bars = ax.barh([LABEL.get(k, k) for k in order], vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", fontsize=8.5)
    ax.axvline(d["human"]["comfort_pct"], color="#7f7f7f", ls="--", lw=1,
               label=f"人工 {d['human']['comfort_pct']:.1f}%")
    ax.set_xlabel("舒适率 %（102 天）")
    ax.set_title(f"① 舒适率排行（同协议 {len(keys)} 策略）", fontweight="bold")
    ax.set_xlim(0, 80)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # ---------- ② 舒适 × 果实 ----------
    ax = axes[0, 1]
    for k in keys:
        ax.scatter(d[k]["comfort_pct"], d[k]["fruit_kg"],
                   color=FAMILY[k][1], s=70, zorder=3)
        ax.annotate(LABEL.get(k, k), (d[k]["comfort_pct"], d[k]["fruit_kg"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    pidx = pareto([(d[k]["comfort_pct"], d[k]["fruit_kg"]) for k in keys])
    pf = sorted([(d[keys[i]]["comfort_pct"], d[keys[i]]["fruit_kg"]) for i in pidx])
    ax.plot([p[0] for p in pf], [p[1] for p in pf], "k--", lw=1.2, alpha=0.6,
            label="Pareto 前沿")
    ax.axhline(d["human"]["fruit_kg"], color="#7f7f7f", ls=":", lw=1)
    ax.set_xlabel("舒适率 %"); ax.set_ylabel("果实 kg/m²")
    ax.set_title("② 舒适 × 产量（二维性能面）", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ---------- ③ 舒适 × 决策延迟 ----------
    ax = axes[1, 0]
    for k in keys:
        ms = max(ms_of(k), 1e-3)
        ax.scatter(ms, d[k]["comfort_pct"], color=FAMILY[k][1], s=70, zorder=3)
        ax.annotate(LABEL.get(k, k), (ms, d[k]["comfort_pct"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("单次决策耗时 ms（对数轴；空闲实测中位数）")
    ax.set_ylabel("舒适率 %")
    ax.set_title("③ 成本-性能前沿（PPO ≈ MPC 的 1/88）", fontweight="bold")
    ax.grid(alpha=0.3, which="both")

    # ---------- ④ 分月舒适率热力图 ----------
    ax = axes[1, 1]
    show = ["human", "rule", "mpc", "lookahead_h1", "trpo", "ppo", "ppo_v9", "crossq",
            "multiagent", "tqc", "sac", "dqn"]
    show = [k for k in show if k in d]
    M = np.array([[d[k]["month_comfort"][m] for m in ("4", "5", "6", "7")]
                  for k in show], dtype=float)
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=85)
    ax.set_xticks(range(4)); ax.set_xticklabels(["4月", "5月", "6月", "7月"])
    ax.set_yticks(range(len(show))); ax.set_yticklabels([LABEL.get(k, k) for k in show])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("④ 分月舒适率 %（7 月高温为共同短板）", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("郫都玻璃温室 · 全算法统一矩阵（102 天完整生长季 · 同协议）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = OUT / "algorithm_matrix.png"
    fig.savefig(p, dpi=150)
    print(f"已写入 {p}")


if __name__ == "__main__":
    main()
