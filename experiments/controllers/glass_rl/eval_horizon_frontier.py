"""P0-4：H 步前瞻搜索的成本-性能前沿评估。

对 H ∈ {1, 2, 4, 8} 各跑完整生长季（102 天，4/1 定植起），记录
  性能：舒适率 / 果实 / 最高温 / 无过热日 / 分月舒适率
  成本：决策延迟 ms / 单次决策模型推演步数 / 全季决策总耗时

并产出「前瞻步数 H → 舒适率 → 决策延迟」前沿图。同时把原 `GlassMPC`（H=1，
自定义评分函数、78 ms）作为参照点叠加，说明"同目标(env reward)"与"自定义评分"
两种一步搜索的差别。

输出：
  results/.../rl/horizon_frontier/horizon_frontier.json
  results/.../rl/horizon_frontier/horizon_frontier.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL_DIR = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL_DIR / "horizon_frontier"


def make_env(days: int, start: int):
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=days, start_day_index=start, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
    )


def comfort(t, rh, hour):
    is_day = 6.0 <= hour < 20.0
    t_lo, t_hi = (20.0, 28.0) if is_day else (16.0, 24.0)
    return (t_lo <= t <= t_hi) and (60.0 <= rh <= 85.0)


def evaluate(act_fn, ctrl=None, days: int = 102, start: int = 0) -> dict:
    env = make_env(days, start)
    env.reset(seed=0)
    f0 = float(env.x[25])
    temp, rh, hour, day, rew = [], [], [], [], []
    times, rollouts = [], []
    for i in range(days * 24):
        t0 = time.perf_counter()
        a = act_fn(env)
        times.append(time.perf_counter() - t0)
        if ctrl is not None:
            rollouts.append(ctrl.rollout_count())
        _, r, _, _, info = env.step(a)
        rew.append(float(r))
        temp.append(float(info["temperature"]))
        rh.append(float(info["rh"]))
        hour.append(i % 24)
        day.append(i // 24)

    temp = np.array(temp); rh = np.array(rh); hour = np.array(hour); day = np.array(day)
    c = np.array([comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    daily_max = np.array([temp[day == d].max() for d in range(days)])
    daily_mean = np.array([temp[day == d].mean() for d in range(days)])
    month = np.where(day < 30, 4, np.where(day < 61, 5, np.where(day < 91, 6, 7)))
    return {
        "comfort_pct": float(c.mean() * 100),
        "fruit_kg": float((float(env.x[25]) - f0) * 1e-6 / 0.081),
        "max_temp": float(temp.max()),
        "mean_temp": float(temp.mean()),
        "no_overheat_day": float((daily_max <= 35.0).mean() * 100),
        "comfort_day": float((np.array([c[day == d].mean() for d in range(days)]) >= 0.5).mean() * 100),
        "daily_t_std": float(daily_mean.std()),
        "reward_total": float(np.sum(rew)),
        "decision_ms_mean": float(np.mean(times) * 1000),
        "decide_total_s": float(np.sum(times)),
        "rollouts_per_decision": float(np.mean(rollouts)) if rollouts else None,
        "month_comfort": {int(m): float(c[month == m].mean() * 100) for m in (4, 5, 6, 7)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--budget", type=int, default=192, help="每轮推演预算 N≈budget/H")
    ap.add_argument("--days", type=int, default=102)
    ap.add_argument("--add-mpc", action="store_true", help="额外跑原 GlassMPC 作参照点")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    from experiments.controllers.glass_rl.horizon_search import GlassHorizonSearch

    results: dict = {"config": {"iters": args.iters, "compute_budget": args.budget,
                                "days": args.days, "objective": "env reward（与 RL 同源）"},
                     "horizons": {}}

    for H in args.horizons:
        ctrl = GlassHorizonSearch(horizon=H, iters=args.iters, compute_budget=args.budget)
        print(f"\n===== H={H} mode={ctrl.mode} N={ctrl.population} K={ctrl.iters} =====", flush=True)
        t0 = time.time()
        res = evaluate(lambda e, c=ctrl: c.predict_action(e), ctrl=ctrl, days=args.days)
        res["horizon"] = H
        res["mode"] = ctrl.mode
        res["population"] = ctrl.population
        res["wall_seconds"] = round(time.time() - t0, 1)
        results["horizons"][str(H)] = res
        print(f"  舒适率 {res['comfort_pct']:.1f}% | 果实 {res['fruit_kg']:.2f} | "
              f"最高温 {res['max_temp']:.1f} | 无过热日 {res['no_overheat_day']:.1f}% | "
              f"{res['decision_ms_mean']:.1f} ms/决策 | 推演 {res['rollouts_per_decision']:.0f} 步 | "
              f"总耗时 {res['wall_seconds']:.0f}s", flush=True)
        (OUT / "horizon_frontier.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.add_mpc:
        from experiments.controllers.glass_rl.classical_controllers import GlassMPC
        mpc = GlassMPC()
        print("\n===== 参照点：原 GlassMPC（H=1，自定义评分函数）=====", flush=True)
        t0 = time.time()
        res = evaluate(lambda e: mpc.predict_action(e), days=args.days)
        res["horizon"] = 1
        res["note"] = "原实现：自定义评分函数（humidity_weight=1.0，无 cooling/screen/comfort 项）"
        res["wall_seconds"] = round(time.time() - t0, 1)
        results["glass_mpc_reference"] = res
        print(f"  舒适率 {res['comfort_pct']:.1f}% | 果实 {res['fruit_kg']:.2f} | "
              f"{res['decision_ms_mean']:.1f} ms/决策", flush=True)

    (OUT / "horizon_frontier.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT / 'horizon_frontier.json'}", flush=True)


if __name__ == "__main__":
    main()
