"""玻璃温室实验统一绘图模块（标准化图表风格）。

统一规范：
- 字体：Microsoft YaHei（中文）
- 配色：固定策略→颜色映射（colorblind-safe 色系）
- 单位：温度 °C、湿度 %、reward（无量纲）、果实 kg/m²、能耗（无量纲）
- 尺度：同一类图固定坐标范围，便于横向比较
- 输出：dpi=300 PNG，保存到 results/.../real_greenhouse/figures/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse")
FIG_DIR = OUT_ROOT / "figures"

# 固定策略配色（colorblind-safe）
COLORS = {
    "human": "#7f7f7f", "baseline": "#1f77b4", "rule": "#2ca02c", "pid": "#ff7f0e",
    "mpc": "#d62728", "ppo": "#9467bd", "a2c": "#8c564b", "dqn": "#17becf",
    "sac": "#e377c2", "ddpg": "#bcbd22", "td3": "#7f7f7f", "trpo": "#2e8b57",
    "recurrent_ppo": "#8b0000", "multiagent": "#4a4a4a",
}
LABELS = {
    "human": "人工", "baseline": "固定基准", "rule": "规则控制", "pid": "PID",
    "mpc": "MPC", "ppo": "PPO", "a2c": "A2C", "dqn": "DQN", "sac": "SAC",
    "ddpg": "DDPG", "td3": "TD3", "trpo": "TRPO", "recurrent_ppo": "RecPPO",
    "multiagent": "多智能体",
}
ORDER = ["human", "baseline", "rule", "pid", "mpc",
         "ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo",
         "multiagent"]


def _label(name: str) -> str:
    return LABELS.get(name, name)


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  已生成 {path}")
    return path


def _ordered_keys(data: dict) -> list[str]:
    return [k for k in ORDER if k in data]


def plot_controller_metrics(benchmark: dict, out: Path | None = None) -> Path:
    """关键指标多面板对比图（6 子图，统一尺度）。"""
    keys = _ordered_keys(benchmark)
    metrics = [
        ("mean_temp", "平均温度", "°C"),
        ("max_temp", "最高温度", "°C"),
        ("comfort", "舒适率", "%"),
        ("total_reward", "累计奖励", ""),
        ("mean_effort", "平均能耗", ""),
        ("fruit_growth_kg_m2", "果实增长", "kg/m²"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (key, title, unit) in zip(axes, metrics):
        vals = [benchmark[k][key] for k in keys]
        colors = [COLORS[k] for k in keys]
        labels = [_label(k) for k in keys]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(f"{title}（{unit}）" if unit else title, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for b, v in zip(bars, vals):
            if abs(v) < 1:
                label = f"{v:.4f}"
            elif abs(v) < 100:
                label = f"{v:.1f}"
            else:
                label = f"{v:.0f}"
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("玻璃温室各控制策略关键指标对比（72 天生长季）",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "controller_metrics_comparison")


def plot_monthly_comfort(benchmark: dict, out: Path | None = None) -> Path:
    """分月舒适率分组柱状图。"""
    keys = _ordered_keys(benchmark)
    months = ["5月", "6月"]
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(months):
        vals = []
        for k in keys:
            mm = {x["month"]: x["comfort"] for x in benchmark[k].get("monthly", [])}
            vals.append(mm.get(m, np.nan))
        ax.bar(x + (i - 1) * width, vals, width, label=m,
               color=["#4c72b0", "#55a868", "#c44e52"][i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(k) for k in keys], fontsize=10)
    ax.set_ylabel("舒适率（%）", fontsize=12)
    ax.set_title("分月舒适率对比", fontsize=14, fontweight="bold")
    ax.legend(title="月份", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)
    return _save(fig, "monthly_comfort_comparison")


def plot_monthly_max_temp(benchmark: dict, out: Path | None = None) -> Path:
    """分月最高温分组柱状图。"""
    keys = _ordered_keys(benchmark)
    months = ["5月", "6月"]
    x = np.arange(len(keys))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, m in enumerate(months):
        vals = []
        for k in keys:
            mm = {x["month"]: x["max_temp"] for x in benchmark[k].get("monthly", [])}
            vals.append(mm.get(m, np.nan))
        ax.bar(x + (i - 1) * width, vals, width, label=m,
               color=["#4c72b0", "#55a868", "#c44e52"][i], edgecolor="white")
    ax.axhline(28, color="red", linestyle="--", linewidth=1, label="舒适上限 28°C")
    ax.set_xticks(x)
    ax.set_xticklabels([_label(k) for k in keys], fontsize=10)
    ax.set_ylabel("最高温度（°C）", fontsize=12)
    ax.set_title("分月最高温度对比", fontsize=14, fontweight="bold")
    ax.legend(title="月份", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    return _save(fig, "monthly_max_temp_comparison")


def plot_temp_trajectory(traj: dict, out: Path | None = None) -> Path:
    """日平均温度轨迹图（72 天，统一 y 轴 10-50°C）。"""
    keys = _ordered_keys(traj)
    fig, ax = plt.subplots(figsize=(14, 6))
    for k in keys:
        t = np.asarray(traj[k]["temp"])
        daily = t.reshape(-1, 24).mean(axis=1)
        ax.plot(daily, label=_label(k), color=COLORS[k], linewidth=1.3)
    ax.axhspan(20, 28, color="green", alpha=0.06, label="昼间适宜 20–28°C")
    ax.axhline(28, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("天数（自 5 月 1 日起）", fontsize=12)
    ax.set_ylabel("日平均温度（°C）", fontsize=12)
    ax.set_title("各控制策略日平均温度轨迹对比", fontsize=14, fontweight="bold")
    ax.set_ylim(10, 50)
    ax.legend(fontsize=9, ncol=4, loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    return _save(fig, "temperature_trajectory_comparison")


def plot_climate_trajectory(traj: dict, out: Path | None = None) -> Path:
    """整季温度 + 湿度变化曲线（逐时，双面板）。"""
    keys = _ordered_keys(traj)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    hours = np.arange(len(next(iter(traj.values()))["temp"])) / 24.0
    for k in keys:
        t = np.asarray(traj[k]["temp"])
        ax1.plot(hours, t, label=_label(k), color=COLORS[k], linewidth=1.0, alpha=0.85)
    ax1.axhspan(20, 28, color="green", alpha=0.06)
    ax1.axhline(28, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("温度（°C）", fontsize=12)
    ax1.set_title("整季温度变化曲线（72 天逐时）", fontsize=14, fontweight="bold")
    ax1.set_ylim(10, 50)
    ax1.legend(fontsize=8, ncol=6, loc="upper right")
    ax1.grid(alpha=0.3, linestyle="--")

    for k in keys:
        r = np.asarray(traj[k]["rh"])
        ax2.plot(hours, r, label=_label(k), color=COLORS[k], linewidth=1.0, alpha=0.85)
    ax2.axhspan(60, 85, color="blue", alpha=0.05)
    ax2.axhline(60, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(85, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("天数（自 5 月 1 日起）", fontsize=12)
    ax2.set_ylabel("相对湿度（%）", fontsize=12)
    ax2.set_title("整季湿度变化曲线（72 天逐时）", fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, ncol=6, loc="upper right")
    ax2.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    return _save(fig, "climate_trajectory_comparison")


# 关键执行器（索引 -> 名称）
ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO₂", "风机", "水泵", "卷膜"]


def plot_control_trajectory(traj: dict, names: list[str] | None = None,
                            out: Path | None = None) -> Path:
    """控制曲线：关键执行器档位随时间（日平均，多策略对比）。"""
    if names is None:
        names = [k for k in ("human", "mpc", "sac") if k in traj]
    # 关键执行器索引: 遮阳0 顶窗3 风机6 水泵7 卷膜8 顶保温1
    act_idx = [0, 3, 6, 7, 8]
    act_labels = [ACT_NAMES[i] for i in act_idx]
    n_acts = len(act_idx)
    fig, axes = plt.subplots(n_acts, 1, figsize=(16, 2.2 * n_acts), sharex=True)
    if n_acts == 1:
        axes = [axes]
    hours = np.arange(len(next(iter(traj.values()))["action"])) / 24.0
    for ax, ai, al in zip(axes, act_idx, act_labels):
        for k in names:
            a = np.asarray(traj[k]["action"])
            daily = a[:, ai].reshape(-1, 24).mean(axis=1)
            ax.plot(hours[::24], daily, label=_label(k), color=COLORS[k],
                    linewidth=1.2, drawstyle="steps-post")
        ax.set_ylabel(f"{al}档位", fontsize=11)
        ax.set_ylim(-0.1, max(3.2, ax.get_ylim()[1]))
        ax.legend(fontsize=8, ncol=len(names), loc="upper right")
        ax.grid(alpha=0.3, linestyle="--")
    axes[-1].set_xlabel("天数（自 5 月 1 日起）", fontsize=12)
    fig.suptitle("关键执行器控制曲线（日平均档位）", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "control_trajectory_comparison")


def plot_oat_sensitivity(csv_path: str | Path, value_col: str, title: str,
                         xlabel: str, unit: str = "") -> Path:
    """OAT 敏感性排序图（tornado 风格）。"""
    df = pd.read_csv(csv_path).sort_values(value_col, ascending=False)
    names = df["param"].tolist()
    vals = df[value_col].tolist()
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.55)))
    colors = ["#c44e52" if v > 0 else "#4c72b0" for v in vals]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel + (f"（{unit}）" if unit else ""), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    for b, v in zip(bars, vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
    return _save(fig, "oat_" + Path(csv_path).stem)


def make_all_plots(benchmark: dict, traj: dict,
                   climate_csv: Path, crop_csv: Path) -> list[Path]:
    paths = []
    paths.append(plot_controller_metrics(benchmark))
    paths.append(plot_monthly_comfort(benchmark))
    paths.append(plot_monthly_max_temp(benchmark))
    if traj:
        paths.append(plot_temp_trajectory(traj))
        paths.append(plot_climate_trajectory(traj))
        paths.append(plot_control_trajectory(traj))
    if climate_csv.exists():
        paths.append(plot_oat_sensitivity(climate_csv, "temp_mae_span",
                                          "气候模型参数敏感性（温度 MAE span）",
                                          "温度 MAE span", "°C"))
    if crop_csv.exists():
        paths.append(plot_oat_sensitivity(crop_csv, "end_span",
                                          "作物模型参数敏感性（产量 span）",
                                          "季末产量 span", "kg/m²"))
    return paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    bm = json.loads((OUT_ROOT / "rl" / "benchmark_all_d30_60.json").read_text(encoding="utf-8"))
    tr = json.loads((OUT_ROOT / "rl" / "benchmark_all_d30_60_trajectories.json").read_text(encoding="utf-8"))
    climate = Path("results/chengdu_agri_greenhouse_001/sensitivity_analysis/oat_climate_params.csv")
    crop = Path("results/chengdu_agri_greenhouse_001/sensitivity_analysis/oat_crop_params.csv")
    make_all_plots(bm, tr, climate, crop)
