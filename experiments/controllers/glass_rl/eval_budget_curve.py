# -*- coding: utf-8 -*-
"""A4：训练预算—性能曲线（是否已收敛）。

回答 P1-2（「60 万步预算是否充分」）：把同一算法在不同训练步数的检查点
在**同一冻结协议**下评估，看性能是否在 60 万步前已进入平台期。

判据（事先声明）：
    · 450k → 600k 的舒适率提升 < 1.0 pp  → 判「已收敛」
    · 提升 ≥ 1.0 pp                       → 判「预算不足，性能未饱和」

零训练成本：直接复用训练脚本已保存的 `checkpoint_*.zip`（8 算法 × {150k,300k,450k,600k}）。

用法：
    python -m experiments.controllers.glass_rl.eval_budget_curve
    python -m experiments.controllers.glass_rl.eval_budget_curve --algos ppo trpo --steps 150000 600000
"""
from __future__ import annotations

import argparse
import importlib
import json

import numpy as np

from experiments.controllers.glass_rl.eval_algorithm_matrix import (  # noqa: E402
    ALGO, RL, _levels, rollout)

OUT = RL / "budget_curve"

KIND_CLS = {
    "ppo": ("multi", "PPO"), "a2c": ("multi", "A2C"),
    "trpo": ("multi", "TRPO"), "recurrent_ppo": ("multi", "RecurrentPPO"),
    "dqn": ("flat", "DQN"), "sac": ("cont", "SAC"),
    "ddpg": ("cont", "DDPG"), "td3": ("cont", "TD3"),
    "tqc": ("cont", "TQC"), "crossq": ("cont", "CrossQ"),
}
DEFAULT_ALGOS = ["a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo", "ppo"]
DEFAULT_STEPS = [150_000, 300_000, 450_000, 600_000]


def make_pred(alg: str, zpath) -> callable:
    import stable_baselines3 as sb3
    import sb3_contrib
    kind, cls_name = KIND_CLS[alg]
    mod = sb3 if hasattr(sb3, cls_name) else sb3_contrib
    model = getattr(mod, cls_name).load(str(zpath))
    return lambda e: _levels(model.predict(e._get_obs(), deterministic=True)[0], kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="*", default=DEFAULT_ALGOS)
    ap.add_argument("--steps", nargs="*", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--out", type=str, default="budget_curve.json")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / args.out
    res: dict = {}
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            res = {}

    for alg in args.algos:
        if alg not in KIND_CLS:
            print(f"  {alg}: 未知算法，跳过", flush=True)
            continue
        res.setdefault(alg, {})
        for step in args.steps:
            key = str(step)
            if key in res[alg]:
                continue
            z = ALGO / alg / f"checkpoint_{step:07d}.zip"
            if not z.exists():
                print(f"  {alg:<14} {step:>7}  检查点缺失，跳过", flush=True)
                continue
            try:
                r = rollout(make_pred(alg, z))
                res[alg][key] = {k: r[k] for k in
                                 ("comfort_pct", "fruit_kg", "max_temp",
                                  "no_overheat_day", "decision_ms_mean")}
                print(f"  {alg:<14} {step:>7}  舒适 {r['comfort_pct']:6.1f}"
                      f"  果实 {r['fruit_kg']:5.2f}  maxT {r['max_temp']:5.1f}",
                      flush=True)
            except Exception as e:
                print(f"  {alg:<14} {step:>7}  ❌ {type(e).__name__}: {str(e)[:80]}",
                      flush=True)
            out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # ---------------- 收敛判定 ----------------
    # 判据（事先声明，三分——注意必须区分"平台"与"退化"）：
    #   |Δ(450k→600k)| < 1.0 pp  → 平台（已收敛）
    #   Δ ≥ +1.0 pp              → 未饱和（预算不足）
    #   Δ ≤ −1.0 pp              → 性能退化（训练不稳定/崩溃），不得写作"已收敛"
    print()
    print("=" * 104)
    print("预算—性能（舒适率 %）与收敛判定")
    print("  判据：|Δ(450k→600k)|<1.0 → 平台 ｜ Δ≥+1.0 → 未饱和 ｜ Δ≤−1.0 → 性能退化")
    print("=" * 104)
    steps = sorted(args.steps)
    hdr = "".join(f"{s // 1000:>8}k" for s in steps)
    print(f"{'算法':<16}{hdr}{'450k→600k':>11}{'150k→600k':>11}{'判定':>12}")
    verdict = {}
    for alg in args.algos:
        row = res.get(alg, {})
        vals = [row.get(str(s), {}).get("comfort_pct") for s in steps]
        if any(v is None for v in vals):
            print(f"{alg:<16}" + "".join(f"{'—':>9}" for _ in steps) + "      数据不全")
            continue
        d = vals[-1] - vals[-2]
        d_all = vals[-1] - vals[0]
        if abs(d) < 1.0:
            tag = "✅ 平台"
        elif d >= 1.0:
            tag = "⚠️ 未饱和"
        else:
            tag = "❌ 性能退化"
        verdict[alg] = {"delta_450k_600k": round(d, 2),
                        "delta_150k_600k": round(d_all, 2),
                        "verdict": tag}
        print(f"{alg:<16}" + "".join(f"{v:>9.1f}" for v in vals)
              + f"{d:>+11.1f}{d_all:>+11.1f}{tag:>12}")
    (OUT / "budget_verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    unsat = [k for k, v in verdict.items() if "未饱和" in v["verdict"]]
    degen = [k for k, v in verdict.items() if "退化" in v["verdict"]]
    const = []
    for alg in args.algos:
        row = res.get(alg, {})
        vals = [row.get(str(s), {}).get("comfort_pct") for s in steps]
        if vals and all(v is not None for v in vals) and len(set(vals)) == 1:
            const.append(alg)
    print()
    print(f"  未饱和（预算不足）：{unsat or '无'}")
    print(f"  训练不稳定（后期退化）：{degen or '无'}")
    print(f"  全步数完全恒定（策略塌缩，与步数无关）：{const or '无'}")

    # ---------------- 绘图 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        xs = [s / 1000 for s in steps]
        for alg in args.algos:
            row = res.get(alg, {})
            ys = [row.get(str(s), {}).get("comfort_pct") for s in steps]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, marker="o", ms=4, lw=1.4, label=alg)
        ax.set_xlabel("训练步数（千步）")
        ax.set_ylabel("舒适率 %")
        ax.set_title("图 A4-1 训练预算—性能曲线（同冻结协议，102 天）", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUT / "budget_curve.png", dpi=150)
        plt.close(fig)
        print(f"\n  图 → {OUT/'budget_curve.png'}")
    except Exception as e:
        print(f"  绘图失败：{type(e).__name__}: {str(e)[:80]}")
    print(f"  数据 → {out_path}")


if __name__ == "__main__":
    main()
