"""A5：未修正模型上的控制器对照 —— 给「可信性诊断」一号贡献补实验证据。

背景（见 论文/实验设计缺陷分析与补充实验清单_20260919.md §P0-3）：
第 2 章证明了「四处缺陷使模型失真」，但**未证明「不做诊断会得出错误的控制结论」**——
方法论主张目前是"论证式"的。本脚本把该主张变成可判定的实证。

做法：在四个模型变体上重评同一批控制器，比较**排名是否改变**。

| 变体 | solar_gain_scale p[209] | 冠层对流 h_air_canopy | 含义 |
|---|---|---|---|
| V0 修复后 | 0.42 | None（=5760 W/K） | 本文最终模型（第 4 章基准所用） |
| V1 仅冠层缺陷 | 0.42 | 300.0 | 复现"冠层过热"单项缺陷 |
| V2 仅辐射缺陷 | 0.91 | None | 复现"辐射增益误标定"单项缺陷 |
| V3 未修正 | 0.91 | 300.0 | 两缺陷叠加 = 诊断前的模型 |

判据（事先声明，避免事后挑选）：
  ① 各变体间按舒适率的 **Spearman 排名相关 ρ**；ρ < 0.80 → 排名实质改变；
  ② **人工基线的相对位置**是否变化（人工是唯一"真实发生过"的参照）；
  ③ 性能天花板（最优策略）是否换人。

用法：
  python -m experiments.controllers.glass_rl.eval_uncorrected_model            # 全部变体 × 默认策略
  python -m experiments.controllers.glass_rl.eval_uncorrected_model --variants V0 V3
  python -m experiments.controllers.glass_rl.eval_uncorrected_model --strategies rule mpc --days 12   # 冒烟

输出：
  results/.../rl/uncorrected_model/uncorrected_model.json
  results/.../rl/uncorrected_model/uncorrected_model.md
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

# 只从冻结评估脚本借用「策略构造」与「舒适带定义」，保证口径单一真源
from experiments.controllers.glass_rl.eval_algorithm_matrix import (  # noqa: E402
    BUILDERS, EXEC, FORECAST, NO_WARMUP, OUT as MATRIX_OUT, _comfort)

RL = MATRIX_OUT.parent
OUT = RL / "uncorrected_model"
DAYS = 102

# 变体定义：键为标签，值为 (p[209] solar_gain_scale, 冠层对流覆盖值 or None)
VARIANTS: dict[str, tuple[float, float | None]] = {
    "V0_corrected": (0.42, None),
    "V1_canopy_bug": (0.42, 300.0),
    "V2_radiation_bug": (0.91, None),
    "V3_uncorrected": (0.91, 300.0),
}
VARIANT_CN = {
    "V0_corrected": "V0 修复后",
    "V1_canopy_bug": "V1 仅冠层缺陷",
    "V2_radiation_bug": "V2 仅辐射缺陷",
    "V3_uncorrected": "V3 未修正",
}

DEFAULT_STRATEGIES = [
    "human", "baseline", "rule", "pid", "mpc", "lookahead_h1",
    "ppo", "ppo_v9", "sac", "trpo", "masked_ppo", "tqc", "crossq",
]


# ---------------------------------------------------------------- 变体环境
def make_variant_env(p209: float, canopy: float | None, n_forecast_hours: int, days: int):
    """构建指定变体的环境。

    ⚠️ `h_air_canopy` 在 `define_model` trace 时被烘焙成常量，故覆盖必须在
    **构造 env 之前**写模块全局；而 `params[209]` 是运行时输入，构造后覆盖即可。
    """
    import glassgym.models.GlassGreenhouse.ode as ode_mod
    import glass_env

    ode_mod.H_AIR_CANOPY_OVERRIDE = canopy
    importlib.reload(glass_env)          # 丢弃上一变体 trace 出的 casadi 图缓存

    env = glass_env.GlassGreenhouseEnv(
        n_forecast_hours=n_forecast_hours,
        episode_days=days, start_day_index=0, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
        comfort_weight=0.0, smooth_weight=0.0,
    )
    env.params[209] = p209
    return env


# ---------------------------------------------------------------- 指标
def spearman(a: list[float], b: list[float]) -> float:
    """Spearman 秩相关（平均秩法，不依赖 scipy）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
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
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra @ rb / den) if den > 0 else float("nan")


def variant_metrics(temp, rh, hour, day, rew, f0, env, times, acts, days) -> dict:
    """与冻结脚本同构的指标，唯一差别是 `days` 可调（便于冒烟）。"""
    temp = np.asarray(temp); rh = np.asarray(rh); hour = np.asarray(hour)
    day = np.asarray(day)
    c = np.array([_comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    daily_max = np.array([temp[day == d].max() for d in range(days)])
    per_day = np.array([c[day == d].mean() for d in range(days)])
    A = np.asarray(acts, dtype=int)
    month = np.where(day < 30, 4, np.where(day < 61, 5, np.where(day < 91, 6, 7)))
    return {
        "comfort_pct": round(float(c.mean() * 100), 1),
        "fruit_kg": round(float((float(env.x[25]) - f0) * 1e-6 / 0.081), 3),
        "max_temp": round(float(temp.max()), 2),
        "mean_temp": round(float(temp.mean()), 2),
        "no_overheat_day": round(float((daily_max <= 35.0).mean() * 100), 1),
        "comfort_day": round(float((per_day >= 0.5).mean() * 100), 1),
        "reward_total": round(float(np.sum(rew)), 1),
        "n_unique_actions": int(len(np.unique(A, axis=0))),
        "degenerate": bool(len(np.unique(A, axis=0)) <= 2),
        "month_comfort": {int(m): round(float(c[month == m].mean() * 100), 1)
                          for m in (4, 5, 6, 7) if (month == m).any()},
        "decision_ms_mean": round(float(np.mean(times) * 1000), 3),
    }


def rollout_variant(build_env, predict, n_forecast_hours: int = 0,
                    warmup: int = 3, days: int = DAYS) -> dict:
    env = build_env(n_forecast_hours)
    env.reset(seed=0)
    for _ in range(warmup):              # 预热，排除首次调用开销（human 除外）
        predict(env)
    env.reset(seed=0)
    f0 = float(env.x[25])
    temp, rh, hour, day, rew, times, acts = [], [], [], [], [], [], []
    for i in range(days * 24):
        t0 = time.perf_counter()
        a = predict(env)
        times.append(time.perf_counter() - t0)
        acts.append(np.asarray(a, dtype=int))
        _, r, _, _, info = env.step(a)
        rew.append(float(r)); temp.append(float(info["temperature"]))
        rh.append(float(info["rh"])); hour.append(i % 24); day.append(i // 24)
    m = variant_metrics(temp, rh, hour, day, rew, f0, env, times, acts, days)
    A = np.stack(acts)
    m["action_usage_pct"] = {EXEC[j]: round(float((A[:, j] > 0).mean() * 100), 1)
                             for j in range(A.shape[1])}
    return m


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--out", type=str, default="uncorrected_model.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # 断点续跑：已有结果直接复用，只补缺失的 (variant, strategy)
    out_path = OUT / args.out
    res: dict = {}
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            res = {}
    res.setdefault("meta", {})
    res["meta"] = {"days": args.days, "variants": list(args.variants),
                   "strategies": list(args.strategies),
                   "p209": {k: v[0] for k, v in VARIANTS.items()},
                   "h_air_canopy": {k: v[1] for k, v in VARIANTS.items()}}

    for vname in args.variants:
        if vname not in VARIANTS:
            print(f"  跳过未知变体 {vname}", flush=True)
            continue
        p209, canopy = VARIANTS[vname]
        bucket = res.setdefault(vname, {})
        print(f"\n=== {VARIANT_CN[vname]}  (p[209]={p209}, h_air_canopy={canopy}) ===",
              flush=True)
        for name in args.strategies:
            if name in bucket:
                print(f"  {name:14} 已存在，跳过", flush=True)
                continue
            if name not in BUILDERS:
                print(f"  {name:14} ⚠️ 未知策略，跳过", flush=True)
                continue
            nf = FORECAST.get(name, 0)
            try:
                pred = BUILDERS[name]()
                build = lambda nf, p=p209, c=canopy: make_variant_env(p, c, nf, args.days)
                m = rollout_variant(build, pred, n_forecast_hours=nf,
                                    warmup=0 if name in NO_WARMUP else 3,
                                    days=args.days)
                bucket[name] = m
                flag = " ⚠️退化" if m["degenerate"] else ""
                print(f"  {name:14} 舒适{m['comfort_pct']:6.1f}  果实{m['fruit_kg']:6.2f}"
                      f"  maxT{m['max_temp']:6.1f}  无过热{m['no_overheat_day']:6.1f}"
                      f"{flag}", flush=True)
            except Exception as e:
                print(f"  {name:14} ❌ {type(e).__name__}: {str(e)[:90]}", flush=True)
            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # ---------------- 排名对比 ----------------
    print("\n" + "=" * 78, flush=True)
    print("排名稳定性（按舒适率）", flush=True)
    print("=" * 78, flush=True)
    vs = [v for v in args.variants if res.get(v)]
    common = [s for s in args.strategies
              if all(s in res.get(v, {}) for v in vs)]
    if len(vs) >= 2 and len(common) >= 3:
        base = "V0_corrected" if "V0_corrected" in vs else vs[0]
        print(f"\n{'策略':<16}" + "".join(f"{VARIANT_CN[v]:>16}" for v in vs), flush=True)
        for s in sorted(common, key=lambda x: -res[base][x]["comfort_pct"]):
            line = f"{s:<16}"
            for v in vs:
                line += f"{res[v][s]['comfort_pct']:>16.1f}"
            print(line, flush=True)

        print(f"\n排名秩相关（vs {VARIANT_CN[base]}）：", flush=True)
        summary = {}
        for v in vs:
            rho = spearman([res[base][s]["comfort_pct"] for s in common],
                           [res[v][s]["comfort_pct"] for s in common])
            changed = "⚠️ 排名实质改变" if rho < 0.80 else ("△ 有偏移" if rho < 0.90 else "✅ 稳健")
            summary[v] = round(rho, 3)
            print(f"  {VARIANT_CN[v]:<16} ρ = {rho:6.3f}   {changed}", flush=True)

        # 人工基线相对位置与性能天花板
        print(f"\n{'变体':<16}{'最优策略':<16}{'舒适率':>8}{'人工舒适率':>10}{'人工排名':>9}", flush=True)
        for v in vs:
            ranked = sorted(common, key=lambda x: -res[v][x]["comfort_pct"])
            hr = ranked.index("human") + 1 if "human" in common else None
            print(f"{VARIANT_CN[v]:<16}{ranked[0]:<16}{res[v][ranked[0]]['comfort_pct']:>8.1f}"
                  f"{res[v].get('human', {}).get('comfort_pct', float('nan')):>10.1f}"
                  f"{(str(hr) + '/' + str(len(common))) if hr else '-':>9}", flush=True)
        res["_summary"] = {"spearman_vs_V0": summary, "common": common}

        # Markdown
        md = ["# A5 未修正模型上的控制器对照", "",
              f"评估窗：{args.days} 天 · 变体定义见脚本头", "",
              "## 1. 舒适率矩阵（%）", "",
              "| 策略 | " + " | ".join(VARIANT_CN[v] for v in vs) + " |",
              "|---" + "|---:" * len(vs) + "|"]
        for s in sorted(common, key=lambda x: -res[base][x]["comfort_pct"]):
            md.append(f"| {s} | " + " | ".join(f"{res[v][s]['comfort_pct']:.1f}" for v in vs) + " |")
        md += ["", "## 2. 排名秩相关（Spearman ρ，vs V0 修复后）", "",
               "| 变体 | ρ | 判定 |", "|---|---:|---|"]
        for v in vs:
            rho = summary[v]
            md.append(f"| {VARIANT_CN[v]} | {rho:.3f} | "
                      f"{'排名实质改变' if rho < 0.80 else ('有偏移' if rho < 0.90 else '稳健')} |")
        md += ["", "## 3. 性能天花板与人工基线位置", "",
               "| 变体 | 最优策略 | 舒适率 | 人工舒适率 | 人工排名 |", "|---|---|---:|---:|---:|"]
        for v in vs:
            ranked = sorted(common, key=lambda x: -res[v][x]["comfort_pct"])
            hr = ranked.index("human") + 1 if "human" in common else None
            md.append(f"| {VARIANT_CN[v]} | {ranked[0]} | {res[v][ranked[0]]['comfort_pct']:.1f} | "
                      f"{res[v].get('human', {}).get('comfort_pct', float('nan')):.1f} | "
                      f"{(str(hr) + '/' + str(len(common))) if hr else '-'} |")
        (OUT / "uncorrected_model.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
