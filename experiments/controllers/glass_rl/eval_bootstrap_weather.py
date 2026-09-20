"""A3：块自举天气扰动下的不确定性区间（10 条合成天气 × 关键策略）。

背景（见 论文/实验设计缺陷分析与补充实验清单_20260919.md §P0-2）：
第 4 章矩阵仅在**单一气象年 + 单一随机种子**上评估，全部结论为**点估计**，无不确定性区间。
本脚本用 10 条 `multivariate moving-block bootstrap` 合成天气（block=3 天，110 天）
重放同一批策略，给出**均值 ± 95% 置信区间**。

设计要点（**配对设计**）：10 条天气对**所有策略完全相同**，
故策略间比较使用**配对差值**（同一天气内相减后再取区间）——比独立区间更敏感、更公平。

判据（事先声明）：
  · 各策略舒适率的 **95% CI 宽度**：宽度 < 3 pp → 结论稳定；
  · **策略间配对差值的 95% CI 是否跨 0**：跨 0 → 该名次不可宣称"显著"；
  · 若某策略 CI 跨过另一策略均值 → 该名次不可宣称。

用法：
  python -m experiments.controllers.glass_rl.eval_bootstrap_weather            # 全部天气 × 默认策略
  python -m experiments.controllers.glass_rl.eval_bootstrap_weather --weathers 3000 3001 --strategies rule  # 冒烟

输出：
  results/.../rl/bootstrap_weather/bootstrap_weather.json
  results/.../rl/bootstrap_weather/bootstrap_weather.md
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
    BUILDERS, DAYS, FORECAST, NO_WARMUP, RL, metrics)

WEATHER_DIR = Path("data/processed/chengdu_agri/greenhouse_001/weather/Chengdu_aligned")
OUT = RL / "bootstrap_weather"

DEFAULT_STRATEGIES = ["human", "rule", "mpc", "lookahead_h1", "ppo", "trpo", "masked_ppo"]
# 预先声明的关键两两比较（配对差值 CI）
KEY_COMPARISONS = [("trpo", "ppo"), ("masked_ppo", "ppo"), ("trpo", "human"),
                   ("mpc", "trpo"), ("lookahead_h1", "mpc")]


def make_env_with_weather(weather_path: Path, n_forecast_hours: int = 0):
    from glass_env import GlassGreenhouseEnv
    return GlassGreenhouseEnv(
        weather_path=str(weather_path),
        n_forecast_hours=n_forecast_hours,
        episode_days=DAYS, start_day_index=0, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
        comfort_weight=0.0, smooth_weight=0.0,
    )


def rollout(weather_path: Path, predict, n_forecast_hours: int = 0, warmup: int = 3) -> dict:
    import time as _t
    env = make_env_with_weather(weather_path, n_forecast_hours)
    env.reset(seed=0)
    for _ in range(warmup):
        predict(env)
    env.reset(seed=0)
    f0 = float(env.x[25])
    temp, rh, hour, day, rew, times, acts = [], [], [], [], [], [], []
    for i in range(DAYS * 24):
        t0 = _t.perf_counter()
        a = predict(env)
        times.append(_t.perf_counter() - t0)
        acts.append(np.asarray(a, dtype=int))
        _, r, _, _, info = env.step(a)
        rew.append(float(r)); temp.append(float(info["temperature"]))
        rh.append(float(info["rh"])); hour.append(i % 24); day.append(i // 24)
    m = metrics(temp, rh, hour, day, rew, f0, env, times, acts=acts)
    m.pop("decision_ms_mean", None)      # 并发下不可用，A3 不关注延迟
    return m


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    half = 1.96 * x.std(ddof=1) / np.sqrt(x.size)   # 正态近似；n=10 时与 t 分布差异 <3%
    return float(x.mean() - half), float(x.mean() + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    ap.add_argument("--weathers", nargs="*", default=None, help="默认全部 30xx")
    ap.add_argument("--out", type=str, default="bootstrap_weather.json")
    args = ap.parse_args()

    files = sorted(WEATHER_DIR.glob("30*.csv"))
    if args.weathers:
        files = [f for f in files if f.stem in set(args.weathers)]
    if not files:
        print(f"未找到合成天气：{WEATHER_DIR}（先运行 convert_bootstrap_weather）", flush=True)
        return
    print(f"天气副本 {len(files)} 条：{[f.stem for f in files]}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / args.out
    res: dict = {}
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            res = {}

    for w in files:
        bucket = res.setdefault(w.stem, {})
        print(f"\n=== 天气 {w.stem} ===", flush=True)
        for name in args.strategies:
            if name in bucket:
                continue
            if name not in BUILDERS:
                print(f"  {name:14} ⚠️ 未知策略", flush=True)
                continue
            nf = FORECAST.get(name, 0)
            try:
                pred = BUILDERS[name]()
                m = rollout(w, pred, n_forecast_hours=nf,
                            warmup=0 if name in NO_WARMUP else 3)
                bucket[name] = {k: m[k] for k in
                                ("comfort_pct", "fruit_kg", "max_temp",
                                 "no_overheat_day", "n_unique_actions")}
                print(f"  {name:14} 舒适{bucket[name]['comfort_pct']:6.1f}  "
                      f"果实{bucket[name]['fruit_kg']:6.2f}  "
                      f"maxT{bucket[name]['max_temp']:6.1f}", flush=True)
            except Exception as e:
                print(f"  {name:14} ❌ {type(e).__name__}: {str(e)[:90]}", flush=True)
            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # ---------------- 汇总 ----------------
    strat = [s for s in args.strategies
             if all(s in res.get(w.stem, {}) for w in files)]
    if len(strat) < 2:
        print("\n样本不足，无法汇总", flush=True)
        return

    print("\n" + "=" * 84, flush=True)
    print(f"块自举天气下的舒适率：均值 ± 95% CI（配对设计，n = {len(files)} 条天气）",
          flush=True)
    print("=" * 84, flush=True)
    print(f"{'策略':<14}{'均值%':>8}{'95% CI':>20}{'宽度pp':>9}{'min':>7}{'max':>7}", flush=True)
    agg = {}
    for s in sorted(strat, key=lambda x: -np.mean([res[w.stem][x]["comfort_pct"] for w in files])):
        v = np.array([res[w.stem][s]["comfort_pct"] for w in files])
        lo, hi = ci95(v)
        agg[s] = {"mean": round(float(v.mean()), 2), "ci_lo": round(lo, 2),
                  "ci_hi": round(hi, 2), "width": round(hi - lo, 2),
                  "min": round(float(v.min()), 1), "max": round(float(v.max()), 1)}
        print(f"{s:<14}{v.mean():>8.1f}{f'[{lo:.1f}, {hi:.1f}]':>20}"
              f"{hi - lo:>9.2f}{v.min():>7.1f}{v.max():>7.1f}", flush=True)

    print("\n关键两两比较（配对差值，pp）：", flush=True)
    comps = {}
    for a, b in KEY_COMPARISONS:
        if a not in strat or b not in strat:
            continue
        d = np.array([res[w.stem][a]["comfort_pct"] - res[w.stem][b]["comfort_pct"]
                      for w in files])
        lo, hi = ci95(d)
        sig = "✅ 显著（CI 不跨 0）" if lo > 0 or hi < 0 else "⚠️ 不显著（CI 跨 0）"
        comps[f"{a}-{b}"] = {"mean": round(float(d.mean()), 2),
                             "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                             "significant": bool(lo > 0 or hi < 0)}
        print(f"  {a:>12} − {b:<12} = {d.mean():+6.2f}  [{lo:+.2f}, {hi:+.2f}]   {sig}",
              flush=True)

    res["_summary"] = {"n_weathers": len(files), "strategies": strat,
                       "aggregate": agg, "comparisons": comps}
    out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# A3 块自举天气下的不确定性区间", "",
          f"天气副本 **{len(files)}** 条（multivariate moving-block bootstrap，block=3 d，110 d），"
          f"评估窗 {DAYS} 天；**配对设计**（同一批天气施加于所有策略）。", "",
          "## 1. 舒适率：均值 ± 95% CI", "",
          "| 策略 | 均值 % | 95% CI | 宽度 pp | min | max |", "|---|---:|---|---:|---:|---:|"]
    for s in sorted(strat, key=lambda x: -agg[x]["mean"]):
        a = agg[s]
        md.append(f"| {s} | {a['mean']:.1f} | [{a['ci_lo']:.1f}, {a['ci_hi']:.1f}] | "
                  f"{a['width']:.2f} | {a['min']:.1f} | {a['max']:.1f} |")
    md += ["", "## 2. 关键两两比较（配对差值）", "",
           "| 比较 | 差值 pp | 95% CI | 判定 |", "|---|---:|---|---|"]
    for k, v in comps.items():
        md.append(f"| {k} | {v['mean']:+.2f} | [{v['ci_lo']:+.2f}, {v['ci_hi']:+.2f}] | "
                  f"{'显著' if v['significant'] else '不显著（CI 跨 0）'} |")
    md += ["", "> 判据（事先声明）：CI 宽度 < 3 pp → 结论稳定；"
           "配对差值 CI 跨 0 → 该名次不可宣称。", "",
           "## 3. 气象副本", "",
           "| 副本 | " + " | ".join(strat) + " |", "|---" + "|---:" * len(strat) + "|"]
    for w in files:
        md.append(f"| {w.stem} | " + " | ".join(
            f"{res[w.stem][s]['comfort_pct']:.1f}" for s in strat) + " |")
    (OUT / "bootstrap_weather.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n已写入 {out_path} 与 bootstrap_weather.md", flush=True)


if __name__ == "__main__":
    main()
