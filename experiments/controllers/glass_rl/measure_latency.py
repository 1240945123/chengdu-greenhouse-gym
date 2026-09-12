"""决策延迟专用测量（空闲状态）——解决 P0-2「延迟受 CPU 竞争污染」。

背景：`eval_algorithm_matrix` 在并发训练下测得延迟绝对值不可用（PPO 2.22ms vs
空闲 0.70ms）。本脚本复用同一批 predictor，但**只做计时**：
  - 不跑完整 2448 步，只跑 N 步（默认 300），足够稳定估计每步决策耗时；
  - 每个策略先 warmup 再计时，取中位数与 P95（对 MPC/前瞻这类重尾更稳健）；
  - 报告单进程独占时的延迟，作为论文核心论点「RL 推理成本约为 MPC 的 1/80」的定稿依据。

**必须在无其它训练/评估任务运行时执行**（否则数字仍被污染）。

用法：
  python -m experiments.controllers.glass_rl.measure_latency
  python -m experiments.controllers.glass_rl.measure_latency --steps 300 --strategies ppo sac mpc lookahead_h1
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

from experiments.controllers.glass_rl.eval_algorithm_matrix import (  # noqa: E402
    BUILDERS, RL, NO_WARMUP)

OUT = RL / "algorithm_matrix"


def measure(pred, steps: int, warmup: int = 5) -> dict:
    from experiments.controllers.glass_rl.eval_algorithm_matrix import make_env
    env = make_env(0)
    env.reset(seed=0)
    for _ in range(warmup):
        pred(env)
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        a = pred(env)
        ts.append((time.perf_counter() - t0) * 1000.0)
        env.step(a)
    ts = np.asarray(ts)
    return {
        "steps": steps,
        "median_ms": round(float(np.median(ts)), 4),
        "mean_ms": round(float(ts.mean()), 4),
        "p95_ms": round(float(np.percentile(ts, 95)), 4),
        "min_ms": round(float(ts.min()), 4),
        "max_ms": round(float(ts.max()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=[
        "human", "baseline", "rule", "pid", "mpc", "lookahead_h1",
        "ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo",
        "tqc", "crossq", "masked_ppo", "multiagent", "dagger_v1"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="latency_idle.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for name in args.strategies:
        if name not in BUILDERS:
            print(f"[skip] 未知 {name}", flush=True)
            continue
        try:
            pred = BUILDERS[name]()
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        r = measure(pred, args.steps, warmup=0 if name in NO_WARMUP else 5)
        res[name] = r
        print(f"[{name:14s}] median {r['median_ms']:9.4f} ms   "
              f"mean {r['mean_ms']:9.4f}   p95 {r['p95_ms']:9.4f}", flush=True)

    (OUT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"\n已写入 {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
