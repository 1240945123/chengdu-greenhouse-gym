"""A1：舒适带口径统一 + 多口径稳健性检验。

背景（见 论文/实验设计缺陷分析与补充实验清单_20260919.md §P0-1）：
"舒适率"是本文主指标，但仓库里同时存在**三套不同定义**——
① 矩阵线（论文所用）：昼夜分带 20–28 / 16–24 °C，RH 60–85%，温湿**联合**判定；
② benchmark 线：温湿**分离**报告，安全界 (15,34) °C / (50,85) %；
③ 论文正文引用：N54「昼 25–30 / 夜 14–16 °C」＋ N53「结果期 RH 50–60%」。
→ 主指标的**构念效度**会被质疑。本脚本在同一批仿真轨迹上重算多种口径，检验排名是否稳健。

判据（事先声明）：
  · 各口径与主口径（matrix）之间的 **Spearman 秩相关 ρ**；
  · **ρ ≥ 0.90 → 排名稳健**（主指标选择不影响结论）；
  · **ρ < 0.90 → 必须声明"指标选择本身即结论的一部分"**，并在正文报告多口径结果。

做法：每个策略**只跑一次仿真**（确定性、环境与第 4 章冻结协议一致），
记录 (T, RH, hour) 序列，再离线按各口径重算——**零额外仿真成本**。

用法：
  python -m experiments.controllers.glass_rl.eval_comfort_schemes
  python -m experiments.controllers.glass_rl.eval_comfort_schemes --strategies rule mpc   # 冒烟

输出：
  results/.../rl/comfort_schemes/comfort_schemes.json
  results/.../rl/comfort_schemes/comfort_schemes.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

from experiments.controllers.glass_rl.eval_algorithm_matrix import (  # noqa: E402
    BUILDERS, DAYS, FORECAST, NO_WARMUP, RL, _comfort, make_env)

OUT = RL / "comfort_schemes"

DEFAULT_STRATEGIES = [
    "human", "baseline", "rule", "pid", "mpc", "lookahead_h1",
    "ppo", "ppo_v9", "ppo_v6", "trpo", "sac", "masked_ppo", "tqc", "crossq",
]

DAY_LO, DAY_HI = 6.0, 20.0            # 日间时段（与第 2 章标定口径同源）


# ---------------------------------------------------------------- 各口径
def _vpd_kpa(t, rh):
    """饱和水汽压差（kPa）：es(T) 用 Tetens 式。"""
    es = 0.6108 * np.exp(17.27 * np.asarray(t) / (np.asarray(t) + 237.3))
    return es * (1.0 - np.asarray(rh) / 100.0)


def scheme_matrix(t, rh, hour):
    """① 矩阵线现口径（论文主指标）：昼夜分带 + RH 60–85，联合判定。"""
    return np.array([_comfort(a, b, c) for a, b, c in zip(t, rh, hour)])


def scheme_official(t, rh, hour):
    """② N53/N54 官方口径：昼 25–30 / 夜 14–16 °C，RH 50–60%，联合判定。"""
    day = (hour >= DAY_LO) & (hour < DAY_HI)
    tlo = np.where(day, 25.0, 14.0)
    thi = np.where(day, 30.0, 16.0)
    return (t >= tlo) & (t <= thi) & (rh >= 50.0) & (rh <= 60.0)


def scheme_temp_official(t, rh, hour):
    """③ 温度单项（官方带）——代表"温湿分离、只看温度"的视角。"""
    day = (hour >= DAY_LO) & (hour < DAY_HI)
    tlo = np.where(day, 25.0, 14.0)
    thi = np.where(day, 30.0, 16.0)
    return (t >= tlo) & (t <= thi)


def scheme_temp_matrix(t, rh, hour):
    """④ 温度单项（矩阵带）——用于分离"温度带取值"与"是否看湿度"两个因素。"""
    day = (hour >= DAY_LO) & (hour < DAY_HI)
    tlo = np.where(day, 20.0, 16.0)
    thi = np.where(day, 28.0, 24.0)
    return (t >= tlo) & (t <= thi)


def scheme_vpd(t, rh, hour):
    """⑤ VPD 口径：仅日间要求 0.5–1.2 kPa（夜间过湿才是胁迫），叠加温度上界 35 °C。"""
    vpd = _vpd_kpa(t, rh)
    day = (hour >= DAY_LO) & (hour < DAY_HI)
    ok = np.where(day, (vpd >= 0.5) & (vpd <= 1.2), True)
    return ok & (t <= 35.0)


def scheme_safety(t, rh, hour):
    """⑥ 安全口径（非舒适）：全时不超 35 °C、RH 不超 85%（与 benchmark 安全界一致）。"""
    return (t <= 35.0) & (rh <= 85.0)


SCHEMES = {
    "matrix": ("① 矩阵线现口径（主指标）", scheme_matrix),
    "official": ("② N53/N54 官方口径", scheme_official),
    "temp_official": ("③ 温度单项（官方带）", scheme_temp_official),
    "temp_matrix": ("④ 温度单项（矩阵带）", scheme_temp_matrix),
    "vpd": ("⑤ VPD 口径（日间 0.5–1.2 kPa）", scheme_vpd),
    "safety": ("⑥ 安全口径（T≤35 且 RH≤85）", scheme_safety),
}


# ---------------------------------------------------------------- 仿真
def collect_series(predict, n_forecast_hours: int = 0, warmup: int = 3):
    """跑一遍完整生长季，返回 (T, RH, hour) 序列。"""
    env = make_env(n_forecast_hours)
    env.reset(seed=0)
    for _ in range(warmup):
        predict(env)
    env.reset(seed=0)
    T, RH, H = [], [], []
    for i in range(DAYS * 24):
        a = predict(env)
        _, _, _, _, info = env.step(a)
        T.append(float(info["temperature"])); RH.append(float(info["rh"]))
        H.append(i % 24)
    return np.array(T), np.array(RH), np.array(H)


# ---------------------------------------------------------------- 统计
def spearman(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra @ rb / den) if den > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    ap.add_argument("--out", type=str, default="comfort_schemes.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / args.out
    res: dict = {}
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            res = {}

    for name in args.strategies:
        if name in res:
            print(f"  {name:14} 已存在，跳过", flush=True)
            continue
        if name not in BUILDERS:
            print(f"  {name:14} ⚠️ 未知策略，跳过", flush=True)
            continue
        nf = FORECAST.get(name, 0)
        try:
            pred = BUILDERS[name]()
            T, RH, H = collect_series(pred, n_forecast_hours=nf,
                                      warmup=0 if name in NO_WARMUP else 3)
            row = {k: round(float(fn(T, RH, H).mean() * 100), 1)
                   for k, (_, fn) in SCHEMES.items()}
            row["_n"] = int(T.size)
            res[name] = row
            print(f"  {name:14} " + "  ".join(
                f"{k}={row[k]:5.1f}" for k in SCHEMES), flush=True)
        except Exception as e:
            print(f"  {name:14} ❌ {type(e).__name__}: {str(e)[:90]}", flush=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    # ---------------- 排名稳健性 ----------------
    ok = [s for s in args.strategies if s in res]
    if len(ok) < 3:
        print("\n样本不足，无法比较排名", flush=True)
        return

    print("\n" + "=" * 82, flush=True)
    print("舒适率矩阵（%）", flush=True)
    print("=" * 82, flush=True)
    keys = list(SCHEMES)
    print(f"{'策略':<14}" + "".join(f"{k:>14}" for k in keys), flush=True)
    for s in sorted(ok, key=lambda x: -res[x]["matrix"]):
        print(f"{s:<14}" + "".join(f"{res[s][k]:>14.1f}" for k in keys), flush=True)

    rho = {}
    for k in keys:
        rho[k] = round(spearman([res[s]["matrix"] for s in ok],
                                [res[s][k] for s in ok]), 3)
    print(f"\n排名秩相关（vs {SCHEMES['matrix'][0]}）：", flush=True)
    summary = {}
    for k in keys:
        r = rho[k]
        verdict = "✅ 稳健" if (r != r or r >= 0.90) else "⚠️ 排名实质改变"
        if k == "matrix":
            verdict = "—（基准）"
        print(f"  {SCHEMES[k][0]:<28} ρ = {r:6.3f}   {verdict}", flush=True)
        summary[k] = {"rho_vs_matrix": r, "robust": bool(r >= 0.90) if r == r else None}

    res["_summary"] = {"schemes": {k: SCHEMES[k][0] for k in keys},
                       "rho_vs_matrix": rho, "n_strategies": len(ok)}
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# A1 舒适带口径统一与稳健性检验", "",
          f"评估窗：{DAYS} 天 · 策略数：{len(ok)} · 每策略单次确定性仿真后离线重算", "",
          "## 1. 各口径舒适率（%）", "",
          "| 策略 | " + " | ".join(k for k in keys) + " |",
          "|---" + "|---:" * len(keys) + "|"]
    for s in sorted(ok, key=lambda x: -res[x]["matrix"]):
        md.append(f"| {s} | " + " | ".join(f"{res[s][k]:.1f}" for k in keys) + " |")
    md += ["", "## 2. 排名秩相关（Spearman ρ，vs 主口径）", "",
           "| 口径 | ρ | 判定 |", "|---|---:|---|"]
    for k in keys:
        r = rho[k]
        v = "基准" if k == "matrix" else ("稳健" if r >= 0.90 else "排名实质改变")
        md.append(f"| {SCHEMES[k][0]} | {r:.3f} | {v} |")
    md += ["", "## 3. 口径定义", "",
           "| 键 | 定义 |", "|---|---|",
           "| `matrix` | 昼 06–20 时 20–28 °C、夜 16–24 °C，RH 60–85%，温湿**联合**判定（论文主指标） |",
           "| `official` | 昼 25–30 °C、夜 14–16 °C，RH 50–60%，联合判定（N53/N54） |",
           "| `temp_official` | 仅温度达官方带（温湿分离视角） |",
           "| `temp_matrix` | 仅温度达矩阵带（分离「温度带取值」与「是否看湿度」两因素） |",
           "| `vpd` | 日间 VPD 0.5–1.2 kPa 且 T ≤ 35 °C |",
           "| `safety` | T ≤ 35 °C 且 RH ≤ 85%（安全口径，非舒适） |"]
    (OUT / "comfort_schemes.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已写入 {out_path} 与 comfort_schemes.md", flush=True)


if __name__ == "__main__":
    main()
