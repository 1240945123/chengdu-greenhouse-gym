"""论文第 3 章（玻璃温室物理数字孪生）插图三件套。

产出（results/.../rl/figures_chapter3/）：
  fig3_1_canopy_overheat.png   冠层-空气温差：修复前(h=300) vs 修复后(h=5760)
  fig3_2_solar_gain_calib.png  逐日最高温：真实 vs 仿真(p209=0.91) vs 仿真(p209=0.42)
  fig3_3_sim_vs_real_temp.png  逐时室内温度轨迹：真实 vs 仿真(标定后)

数据：
  真实：data/processed/chengdu_agri/greenhouse_001/trajectories/hourly_wide_with_controls.csv
  仿真：GlassGreenhouseEnv + 人工操作日志重放（同 eval 冻结协议，102 天 seedling 起点）

用法：
  python -m experiments.controllers.glass_rl.plot_chapter3_physics
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "figures_chapter3"
# 真实室内温度：培养窗口（4/1—7/11）数据集（与 docs/notes/07 同源，逐月峰值
# 4月 37.3 / 5月 42.3 / 6月 38.7 完全一致）
V5_DIR = Path("data/processed/chengdu_agri/greenhouse_001/trajectories/"
              "v5_cultivation_window")
# 清棚后（7/11 以后）原始传感器数据（用于标注误标定所拟合的异常值）
RAW_CSV_DIR = Path("../玻璃温室信息/tomato_2026-04-01_to_2026-07-20_csv")

START_DATE = pd.Timestamp("2026-04-01")
DAYS = 102


# ------------------------------------------------------------------ 真实数据
def load_real() -> pd.DataFrame:
    """培养窗口逐时室内空气温度（4/1—7/11，2 359 h）。"""
    d = pd.concat([pd.read_csv(V5_DIR / f) for f in ("train.csv", "val.csv", "test.csv")])
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    d["day"] = (d["timestamp"].dt.normalize() - START_DATE).dt.days
    out = d[["timestamp", "day", "x_air_temperature"]].rename(
        columns={"x_air_temperature": "temp"})
    return out.reset_index(drop=True)


def raw_post_window_daily_max() -> pd.Series:
    """清棚后（7/12—7/20）原始传感器逐日最高温。

    原始 12 路传感器含故障值（>100°C），先按物理区间 [-10, 60] 过滤，
    再取逐时 12 路最大值、逐日最大值。仅用于标注 7/11 后的异常升温。
    """
    import glob
    if not RAW_CSV_DIR.exists():
        return pd.Series(dtype=float)
    recs = []
    for f in sorted(glob.glob(str(RAW_CSV_DIR / "2026-07" / "*.csv"))):
        try:
            d = pd.read_csv(f, usecols=["timeString", "paramName", "paramVal"],
                            dtype=str)
        except Exception:
            continue
        d = d[d["paramName"] == "空气温度"]
        if d.empty:
            continue
        d["v"] = pd.to_numeric(d["paramVal"], errors="coerce")
        d = d[(d["v"] >= -10) & (d["v"] <= 60)]
        d["t"] = pd.to_datetime(d["timeString"].str.slice(0, 19),
                                format="%Y-%m-%d %H:%M:%S", errors="coerce")
        d = d.dropna(subset=["v", "t"])
        d["hour"] = d["t"].dt.floor("h")
        recs.append(d.groupby("hour")["v"].max())
    if not recs:
        return pd.Series(dtype=float)
    s = pd.concat(recs).sort_index()
    daily = s.groupby(s.index.normalize()).max()
    day = (daily.index - START_DATE).days
    return pd.Series(daily.to_numpy(), index=day)


# ------------------------------------------------------------------ 仿真一次
def run_sim(p209: float = 0.42, canopy_h: float | None = None):
    """用人工操作日志重放 102 天，返回 (air[小时], canopy[小时], day_index)。"""
    from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import (
        load_human_actions, ACT_COLS)

    if canopy_h is not None:                       # 复现修复前冠层对流系数
        from glassgym.models.GlassGreenhouse import ode as ode_mod
        ode_mod.H_AIR_CANOPY_OVERRIDE = float(canopy_h)
    else:
        from glassgym.models.GlassGreenhouse import ode as ode_mod
        ode_mod.H_AIR_CANOPY_OVERRIDE = None

    env = GlassGreenhouseEnv(
        episode_days=DAYS, start_day_index=0, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
        comfort_weight=0.0, smooth_weight=0.0)
    env.params[209] = float(p209)
    env.reset(seed=0)

    ha = load_human_actions().iloc[: DAYS * 24].reset_index(drop=True)
    acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int)
            for i in range(len(ha))]

    air, canopy = [], []
    for i in range(DAYS * 24):
        env.step(acts[i])
        air.append(float(env.x[2]))
        canopy.append(float(env.x[4]))
    return np.array(air), np.array(canopy)


# ------------------------------------------------------------------ 三张图
def fig31(air_fix, can_fix, air_bug, can_bug) -> None:
    """冠层-空气温差：修复前 vs 修复后（选 6 月一段晴好窗口）。"""
    d0, d1 = 60, 74          # 6 月 1—14 日
    sl = slice(d0 * 24, d1 * 24)
    h = np.arange(d1 - d0) / 1.0
    hrs = np.arange((d1 - d0) * 24)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(hrs, (can_bug - air_bug)[sl], color="#d62728", lw=1.4,
            label="修复前 h=300 W/K")
    ax.plot(hrs, (can_fix - air_fix)[sl], color="#2ca02c", lw=1.6,
            label="修复后 h=5760 W/K")
    ax.axhspan(0, 2, color="#2ca02c", alpha=0.10, label="真实合理区间 0—2°C")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("6 月 1 日起小时数"); ax.set_ylabel("冠层 − 空气 温差 °C")
    ax.set_title("图 3-1 冠层过热诊断：冠层-空气温差（人工操作重放，6/1—6/14）",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_1_canopy_overheat.png", dpi=150)
    plt.close(fig)
    print(f"  修复前温差 峰值 {np.nanmax((can_bug - air_bug)[sl]):.1f}°C → "
          f"修复后 {np.nanmax((can_fix - air_fix)[sl]):.1f}°C")


def fig32(real: pd.DataFrame, air_091, air_042) -> None:
    """逐日最高温：真实 vs 仿真(0.91) vs 仿真(0.42)，标注 7/11 清棚。"""
    real_dmax = real.groupby("day")["temp"].max()
    days = np.arange(DAYS)
    simmax_091 = air_091.reshape(DAYS, 24).max(axis=1)
    simmax_042 = air_042.reshape(DAYS, 24).max(axis=1)
    post = raw_post_window_daily_max()

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.plot(real_dmax.index, real_dmax.values, color="#111111", lw=1.8,
            label="真实（培养窗口 4/1—7/11）")
    if len(post):
        ax.plot(post.index, post.values, color="#111111", lw=1.8, ls=":",
                label="真实（清棚后 7/12—7/20，原始传感器）")
    ax.plot(days, simmax_091, color="#d62728", lw=1.5,
            label="仿真 p209=0.91（误标定，生长季虚假 +20°C）")
    ax.plot(days, simmax_042, color="#2ca02c", lw=1.6,
            label="仿真 p209=0.42（标定后）")
    ax.axvline(101, color="#7f7f7f", ls="--", lw=1.2)
    ymax = ax.get_ylim()[1]
    ax.annotate("7/11 清棚\n（59.6°C 异常值，被误当作标定目标）",
                xy=(101, ymax * 0.80), xytext=(60, ymax * 0.93), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#7f7f7f"))
    ax.axvspan(0, 101, color="#2ca02c", alpha=0.06)
    ax.text(8, ymax * 0.90, "标定区间：生长季 4/1—7/10", fontsize=9, color="#2ca02c")
    ax.set_xlabel("定植后天数（0 = 2026-04-01）"); ax.set_ylabel("逐日最高温 °C")
    ax.set_title("图 3-2 辐射增益误标定诊断：逐日室内最高温 真实 vs 仿真",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_2_solar_gain_calib.png", dpi=150)
    plt.close(fig)

    for name, m in (("4月", (0, 30)), ("5月", (30, 61)), ("6月", (61, 91))):
        rr = real_dmax[(real_dmax.index >= m[0]) & (real_dmax.index < m[1])].max()
        print(f"  {name} 峰值  真实 {rr:.1f} | 0.91 {simmax_091[m[0]:m[1]].max():.1f}"
              f" | 0.42 {simmax_042[m[0]:m[1]].max():.1f}")


def fig33(real: pd.DataFrame, air_042) -> None:
    """逐时室内温度轨迹：真实 vs 仿真(标定后)，选 5 月中一周。"""
    d0, d1 = 36, 43
    seg = real[(real["day"] >= d0) & (real["day"] < d1)].sort_values("timestamp")
    real_h = seg["temp"].to_numpy()
    sim_h = air_042[d0 * 24: d1 * 24]
    x = np.arange(len(sim_h)) / 24.0

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(x[: len(real_h)], real_h, color="#111111", lw=1.5, label="真实（传感器）")
    ax.plot(x, sim_h, color="#1f77b4", lw=1.5, label="仿真（标定后 p209=0.42）")
    ax.set_xlabel("5 月 6 日起天数"); ax.set_ylabel("室内空气温度 °C")
    ax.set_title("图 3-3 仿真 vs 真实 逐时室内温度轨迹（示例周：5/6—5/12）",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_3_sim_vs_real_temp.png", dpi=150)
    plt.close(fig)
    n = min(len(real_h), len(sim_h))
    print(f"  示例周 MAE = {np.abs(real_h[:n] - sim_h[:n]).mean():.2f}°C")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real = load_real()
    print(f"真实数据 {real['timestamp'].min()} → {real['timestamp'].max()}，"
          f"{len(real)} 行")

    print("运行仿真：p209=0.42（标定后）…")
    air_042, can_042 = run_sim(0.42, None)
    print("运行仿真：p209=0.91（误标定）…")
    air_091, _ = run_sim(0.91, None)
    print("运行仿真：冠层 h=300（修复前）…")
    air_bug, can_bug = run_sim(0.42, 300.0)

    print("绘图 3-1…"); fig31(air_042, can_042, air_bug, can_bug)
    print("绘图 3-2…"); fig32(real, air_091, air_042)
    print("绘图 3-3…"); fig33(real, air_042)

    meta = {"real_rows": int(len(real)),
            "real_span": [str(real['timestamp'].min()), str(real['timestamp'].max())],
            "days": DAYS,
            "peak_air_canopy_diff_fixed": float(np.nanmax(can_042 - air_042)),
            "peak_air_canopy_diff_bug": float(np.nanmax(can_bug - air_bug))}
    (OUT / "chapter3_figs_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
