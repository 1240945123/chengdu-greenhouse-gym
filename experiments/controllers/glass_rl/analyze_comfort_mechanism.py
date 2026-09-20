# -*- coding: utf-8 -*-
"""A1 延伸分析：舒适带口径的机制分解与相图。

回答两个问题：
  ① 各口径下"主导约束"是温度还是湿度？（用 联合/温度单项 的条件达标率分解）
  ② 为什么 official（N53/N54）口径会让排名反转（ρ = −0.731）？

② 的假设：矩阵带允许 RH 60–85%，官方带要求 RH 50–60%，两者**仅在 60 一点相接**，
   近乎互补；而本温室无除湿设备、成都室外 RH 常年偏高，室内 RH 极少低于 60%，
   故"湿度判定"在两口径下方向相反 → 排名反转。
   本脚本用时序数据直接检验该假设，并输出温度–湿度相图。

用法：
    python -m experiments.controllers.glass_rl.analyze_comfort_mechanism
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from experiments.controllers.glass_rl.eval_algorithm_matrix import (  # noqa: E402
    BUILDERS, FORECAST, NO_WARMUP, RL)
from experiments.controllers.glass_rl.eval_comfort_schemes import (  # noqa: E402
    collect_series)

OUT = RL / "comfort_schemes"
DEFAULT_STRATEGIES = ["mpc", "lookahead_h1", "trpo", "rule", "human", "baseline"]

# 口径边界（用于相图）
MATRIX_T = {"day": (20.0, 28.0), "night": (16.0, 24.0), "rh": (60.0, 85.0)}
OFFICIAL_T = {"day": (25.0, 30.0), "night": (14.0, 16.0), "rh": (50.0, 60.0)}


def analyze(T: np.ndarray, RH: np.ndarray, H: np.ndarray) -> dict:
    day = H >= 6
    # 官方温度带内的时段
    in_official_T = np.where(day, (T >= 25.0) & (T <= 30.0),
                             (T >= 14.0) & (T <= 16.0))
    # 矩阵温度带内的时段
    in_matrix_T = np.where(day, (T >= 20.0) & (T <= 28.0),
                           (T >= 16.0) & (T <= 24.0))

    def frac(mask, sel=None) -> float:
        x = mask if sel is None else mask[sel]
        return float(x.mean() * 100) if x.size else float("nan")

    m_rh = (RH >= 60.0) & (RH <= 85.0)      # 矩阵带湿度区间
    o_rh = (RH >= 50.0) & (RH <= 60.0)      # 官方带湿度区间
    return {
        "rh_lt50": frac(RH < 50.0),
        "rh_50_60": frac((RH >= 50.0) & (RH < 60.0)),
        "rh_60_85": frac(m_rh),
        "rh_gt85": frac(RH > 85.0),
        # 条件达标率：温度已在相应带内时，湿度是否也达标
        "cond_rh_matrix": frac(m_rh, in_matrix_T),
        "cond_rh_official": frac(o_rh, in_official_T),
        "in_official_T": frac(in_official_T),
        "in_matrix_T": frac(in_matrix_T),
        # 两湿度带的重叠程度（理论上仅 RH=60 一点）
        "both_rh_bands": frac((RH >= 60.0) & (RH <= 60.0)),
        "median_rh_day": float(np.median(RH[day])) if day.any() else float("nan"),
        "median_rh_night": float(np.median(RH[~day])) if (~day).any() else float("nan"),
    }


def draw(series: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, (T, RH, H)) in zip(axes, series.items()):
        day = H >= 6
        ax.scatter(T[~day], RH[~day], s=3, c="#7f7f7f", alpha=0.30, label="夜间")
        ax.scatter(T[day], RH[day], s=3, c="#1f77b4", alpha=0.30, label="日间")
        # 矩阵带（实线）
        ax.plot(list(MATRIX_T["day"]), [60, 60], color="#2ca02c", lw=1.2, ls="-")
        ax.plot(list(MATRIX_T["day"]), [85, 85], color="#2ca02c", lw=1.2, ls="-")
        ax.plot([20, 20], [60, 85], color="#2ca02c", lw=1.2, ls="-")
        ax.plot([28, 28], [60, 85], color="#2ca02c", lw=1.2, ls="-")
        # 官方带（虚线）
        ax.plot(list(OFFICIAL_T["day"]), [50, 50], color="#d62728", lw=1.2, ls="--")
        ax.plot(list(OFFICIAL_T["day"]), [60, 60], color="#d62728", lw=1.2, ls="--")
        ax.plot([25, 25], [50, 60], color="#d62728", lw=1.2, ls="--")
        ax.plot([30, 30], [50, 60], color="#d62728", lw=1.2, ls="--")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("室内空气温度 °C")
        ax.set_xlim(10, 45)
        ax.set_ylim(30, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("室内相对湿度 %")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("日间温度–湿度相图：绿框 = 矩阵带（RH 60–85），红虚线框 = 官方带（RH 50–60）",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    p = path.with_suffix(".png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  相图 → {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    ap.add_argument("--out", type=str, default="comfort_mechanism.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    series: dict = {}
    res: dict = {}
    for name in args.strategies:
        if name not in BUILDERS:
            print(f"  {name} 未知，跳过", flush=True)
            continue
        try:
            pred = BUILDERS[name]()
            T, RH, H = collect_series(pred, n_forecast_hours=FORECAST.get(name, 0),
                                      warmup=0 if name in NO_WARMUP else 3)
            series[name] = (T, RH, H)
            res[name] = analyze(T, RH, H)
            print(f"  {name:<14} 已收集 n={T.size}", flush=True)
        except Exception as e:
            print(f"  {name:<14} ❌ {type(e).__name__}: {str(e)[:90]}", flush=True)

    if not res:
        return

    print("\n" + "=" * 100)
    print("湿度分布（% 时间）与条件达标率")
    print("=" * 100)
    print(f"{'策略':<14}{'RH<50':>8}{'RH50-60':>9}{'RH60-85':>9}{'RH>85':>7}"
          f"{'温带内RH达标(矩阵)':>18}{'温带内RH达标(官方)':>18}"
          f"{'日RH中位':>9}{'夜RH中位':>9}")
    for name, r in res.items():
        print(f"{name:<14}{r['rh_lt50']:>8.1f}{r['rh_50_60']:>9.1f}{r['rh_60_85']:>9.1f}"
              f"{r['rh_gt85']:>7.1f}{r['cond_rh_matrix']:>18.1f}"
              f"{r['cond_rh_official']:>18.1f}{r['median_rh_day']:>9.1f}"
              f"{r['median_rh_night']:>9.1f}")

    print()
    print("=" * 100)
    print("两湿度带的重叠：矩阵带 RH∈[60,85] 与官方带 RH∈[50,60] 仅在 RH=60 一点相接")
    print("=" * 100)
    for name, r in res.items():
        print(f"  {name:<14} 矩阵带 {r['rh_60_85']:>5.1f}%   官方带 {r['rh_50_60']:>5.1f}%"
              f"   同时满足 {r['both_rh_bands']:>4.1f}%")

    (OUT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"\n已写入 {OUT / args.out}")
    try:
        draw(series, OUT / "comfort_phase")
        np.savez_compressed(OUT / "comfort_series.npz",
                            **{f"{name}_{i}": vs[i]
                               for name, vs in series.items() for i in range(3)})
    except Exception as e:
        print(f"  绘图失败（数据已保存）：{type(e).__name__}: {str(e)[:90]}")


if __name__ == "__main__":
    main()
