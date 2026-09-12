"""P0-3：DAgger 蒸馏 MPC —— 把 MPC 的决策能力压进 1 ms 级神经网络。

动机（见 docs/notes/15 §0）：
  现有「MPC」= 一步前瞻枚举搜索（96~384 候选 × 单步积分），性能最强（舒适率 67.4%）
  但决策延迟 58~77 ms。要让「MPC 性能 + RL 成本」这句话真正成立，最直接的办法不是
  在线残差训练（MPC 76 ms × 60 万步 ≈ 12 h），而是**离线蒸馏**：
    ① 用 MPC 在若干场景上生成 (obs, action) 数据集
    ② 监督学习一个多输出 MLP 模仿 MPC 决策
    ③ DAgger 迭代：用学生策略 rollout，在**学生自己访问到的状态**上再问 MPC 要标签，
       聚合回数据集重训（纠正分布漂移）

输出：
  results/.../rl/distill_mpc/
    ├── policy.pt         学生网络
    ├── norm.json         标准化统计量（mean/std）
    ├── dataset.npz       数据集（初始 + DAgger 迭代）
    ├── distill_meta.json 训练元数据（迭代历史、各维一致率）
    └── distill_result.json 三方评估（学生 / MPC / 规则）

用法：
  python -m experiments.controllers.glass_rl.distill_mpc            # 全流程
  python -m experiments.controllers.glass_rl.distill_mpc --daggers 2
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

import torch
import torch.nn as nn

OUT_DIR = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/distill_mpc")

# 数据生成场景：5 个起始日 × 40 天，覆盖 4→7 月各季天气
SCENARIOS = [0, 20, 40, 60, 80]
GEN_DAYS = 40

# 评估口径：与 eval_systematic 完全一致
EVAL_DAYS = 102
EVAL_START = 0

HEAD_SIZES = (2, 2, 2, 3, 2, 2, 4, 2, 2)  # 9 个执行器档位数，和 = 21
ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]


def make_env(days: int, start: int):
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=days,
        start_day_index=start,
        crop_start="seedling",
        disable_supplements=True,
        cooling_weight=0.5,
        humidity_weight=2.0,
        cooling_mode="overheat",
        obs_include_outdoor=True,
        screen_shade_weight=0.5,
    )


def comfort(t: float, rh: float, hour: int) -> bool:
    is_day = 6.0 <= hour < 20.0
    t_lo, t_hi = (20.0, 28.0) if is_day else (16.0, 24.0)
    return (t_lo <= t <= t_hi) and (60.0 <= rh <= 85.0)


# ---------------------------------------------------------------- 数据生成
def collect(expert, scenarios, actor=None):
    """生成数据集。

    actor=None  → 纯专家轨迹（初始数据集）
    actor=fn    → 用学生策略推进环境，但标签仍由专家给出（DAgger 迭代）
    """
    X, Y = [], []
    n_mpc_calls = 0
    t_mpc = 0.0
    for s in scenarios:
        env = make_env(GEN_DAYS, s)
        env.reset(seed=0)
        for _ in range(GEN_DAYS * 24):
            obs = env._get_obs()
            t0 = time.perf_counter()
            a_exp = expert.predict_action(env)
            t_mpc += time.perf_counter() - t0
            n_mpc_calls += 1
            X.append(np.asarray(obs, dtype=np.float32))
            Y.append(np.asarray(a_exp, dtype=np.int64))
            a = a_exp if actor is None else actor(env)
            env.step(a)
        print(f"  [collect] start_day={s} actor={'expert' if actor is None else 'student'} "
              f"累计样本 {len(X)}", flush=True)
    print(f"  [collect] MPC 调用 {n_mpc_calls} 次，累计 {t_mpc:.1f}s "
          f"（均 {1000 * t_mpc / max(n_mpc_calls, 1):.1f} ms/决策）", flush=True)
    return np.stack(X), np.stack(Y)


# ---------------------------------------------------------------- 学生网络
class DistillPolicy(nn.Module):
    """多输出 MLP：共享躯干 + 9 个分类头（每个执行器一个头）。"""

    def __init__(self, n_in: int = 8, hidden=(128, 128), heads=HEAD_SIZES):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, n) for n in heads])

    def forward(self, x):
        z = self.body(x)
        return [h(z) for h in self.heads]

    @torch.no_grad()
    def act(self, obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        x = torch.as_tensor((obs - mean) / std, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x)
        return np.array([int(t.argmax(1).item()) for t in logits], dtype=int)


def train_student(X, Y, mean, std, *, epochs: int = 60, lr: float = 1e-3,
                  batch: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    xn = torch.as_tensor((X - mean) / std, dtype=torch.float32)
    yn = [torch.as_tensor(Y[:, j], dtype=torch.long) for j in range(Y.shape[1])]

    model = DistillPolicy(n_in=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    n = xn.shape[0]
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logits = model(xn[idx])
            loss = sum(lossf(lg, y[idx]) for lg, y in zip(logits, yn))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        hist.append(tot / max(nb, 1))
        if (ep + 1) % 10 == 0:
            print(f"  [train] epoch {ep+1:3d}/{epochs} loss {hist[-1]:.4f}", flush=True)

    # 训练集一致率（逐维）
    with torch.no_grad():
        logits = model(xn)
        acc = [float((lg.argmax(1) == yn[j]).float().mean()) for j, lg in enumerate(logits)]
    return model, hist, acc


# ---------------------------------------------------------------- 评估
def evaluate(act_fn, days: int = EVAL_DAYS, start: int = EVAL_START, label: str = ""):
    env = make_env(days, start)
    env.reset(seed=0)
    f0 = float(env.x[25])
    recs, times = [], []
    for i in range(days * 24):
        t0 = time.perf_counter()
        a = act_fn(env)
        times.append(time.perf_counter() - t0)
        o, r, te, tr, info = env.step(a)
        recs.append((i // 24, i % 24, float(info["temperature"]), float(info["rh"]),
                     float(r), tuple(int(x) for x in a)))
    day = np.array([x[0] for x in recs])
    hour = np.array([x[1] for x in recs])
    temp = np.array([x[2] for x in recs])
    rh = np.array([x[3] for x in recs])
    rew = np.array([x[4] for x in recs])

    c = np.array([comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    per_day = np.array([c[day == d].mean() for d in range(days)])
    daily_max = np.array([temp[day == d].max() for d in range(days)])
    month = np.array([(4 if d < 30 else 5 if d < 61 else 6 if d < 91 else 7) for d in day])
    month_comfort = {int(m): float(c[month == m].mean() * 100) for m in (4, 5, 6, 7)}

    fruit = (float(env.x[25]) - f0) * 1e-6 / 0.081
    res = {
        "label": label,
        "comfort_pct": float(c.mean() * 100),
        "fruit_kg": float(fruit),
        "max_temp": float(temp.max()),
        "mean_temp": float(temp.mean()),
        "no_overheat_day": float((daily_max <= 35.0).mean() * 100),
        "comfort_day": float((per_day >= 0.5).mean() * 100),
        "daily_t_std": float(np.array([temp[day == d].mean() for d in range(days)]).std()),
        "reward_total": float(rew.sum()),
        "decision_ms_mean": float(np.mean(times) * 1000),
        "decide_total_s": float(np.sum(times)),
        "month_comfort": month_comfort,
    }
    return res, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--daggers", type=int, default=1, help="DAgger 迭代轮数（0=仅行为克隆）")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--skip-eval-mpc", action="store_true",
                    help="跳过 MPC 基线评估（复用以有的 eval_systematic.json）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from experiments.controllers.glass_rl.classical_controllers import GlassMPC, GlassRuleBased

    expert = GlassMPC()
    meta: dict = {"expert": "GlassMPC (H=1 一步前瞻枚举搜索)", "scenarios": SCENARIOS,
                  "gen_days": GEN_DAYS, "daggers": args.daggers, "epochs": args.epochs,
                  "iterations": []}

    # ---- ① 初始数据集（专家轨迹）
    print("\n===== ① 初始数据集：MPC 专家轨迹 =====", flush=True)
    t0 = time.time()
    X, Y = collect(expert, SCENARIOS)
    print(f"  数据集 {X.shape} 用时 {time.time()-t0:.1f}s", flush=True)

    mean = X.mean(axis=0).astype(np.float32)
    std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0)).astype(np.float32)
    np.savez_compressed(out / "dataset.npz", X=X, Y=Y, mean=mean, std=std)

    # ---- ② 训练学生
    print("\n===== ② 行为克隆 =====", flush=True)
    model, hist, acc = train_student(X, Y, mean, std, epochs=args.epochs)
    meta["iterations"].append({"stage": "BC", "n_samples": int(X.shape[0]),
                               "final_loss": hist[-1], "per_dim_acc": acc})
    print("  逐维一致率: " + " ".join(f"{ACT_NAMES[j]}={acc[j]:.3f}" for j in range(9)), flush=True)

    # ---- ③ DAgger 迭代
    for it in range(args.daggers):
        print(f"\n===== ③ DAgger 迭代 {it+1}/{args.daggers} =====", flush=True)
        model.eval()

        def student_act(env, _m=model, _mu=mean, _s=std):
            return _m.act(env._get_obs(), _mu, _s)

        Xs, Ys = collect(expert, SCENARIOS, actor=student_act)
        X = np.concatenate([X, Xs], axis=0)
        Y = np.concatenate([Y, Ys], axis=0)
        mean = X.mean(axis=0).astype(np.float32)
        std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0)).astype(np.float32)
        np.savez_compressed(out / "dataset.npz", X=X, Y=Y, mean=mean, std=std)
        model, hist, acc = train_student(X, Y, mean, std, epochs=args.epochs)
        meta["iterations"].append({"stage": f"DAgger-{it+1}", "n_samples": int(X.shape[0]),
                                   "final_loss": hist[-1], "per_dim_acc": acc})
        print("  逐维一致率: " + " ".join(f"{ACT_NAMES[j]}={acc[j]:.3f}" for j in range(9)),
              flush=True)

    torch.save(model.state_dict(), out / "policy.pt")
    (out / "norm.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()},
                                              indent=2), encoding="utf-8")
    meta["final_per_dim_acc"] = acc
    meta["mean_action_agreement"] = float(np.mean(acc))
    meta["n_samples_total"] = int(X.shape[0])

    # ---- ④ 三方评估
    print("\n===== ④ 评估：学生 vs MPC vs 规则 =====", flush=True)
    model.eval()

    def student_fn(env):
        return model.act(env._get_obs(), mean, std)

    res_student, rec_student = evaluate(student_fn, label="DAgger蒸馏")
    print(f"  [学生] 舒适率 {res_student['comfort_pct']:.1f}%  果实 {res_student['fruit_kg']:.2f}  "
          f"最高温 {res_student['max_temp']:.1f}  {res_student['decision_ms_mean']:.2f} ms/决策",
          flush=True)

    res_rule, _ = evaluate(lambda e: GlassRuleBased().predict_action(e), label="规则")
    print(f"  [规则] 舒适率 {res_rule['comfort_pct']:.1f}%  果实 {res_rule['fruit_kg']:.2f}  "
          f"{res_rule['decision_ms_mean']:.3f} ms/决策", flush=True)

    if args.skip_eval_mpc:
        res_mpc = {"note": "跳过（复用 eval_systematic.json 的 MPC 结果）"}
    else:
        res_mpc, rec_mpc = evaluate(lambda e: expert.predict_action(e), label="MPC")
        print(f"  [MPC ] 舒适率 {res_mpc['comfort_pct']:.1f}%  果实 {res_mpc['fruit_kg']:.2f}  "
              f"{res_mpc['decision_ms_mean']:.2f} ms/决策", flush=True)

    # 学生 vs MPC 动作逐时一致率（评估窗口）
    if not args.skip_eval_mpc:
        same = np.mean([a_st == a_mp for a_st, a_mp in
                        zip([r[5] for r in rec_student], [r[5] for r in rec_mpc])]) * 100
        meta["eval_action_agreement_pct"] = float(same)
        print(f"  学生 vs MPC 全季动作完全一致率: {same:.1f}%", flush=True)

    result = {"student": res_student, "rule": res_rule, "mpc": res_mpc,
              "speedup_vs_mpc": (float(res_mpc["decision_ms_mean"]) /
                                 float(res_student["decision_ms_mean"]))
              if not args.skip_eval_mpc else None}
    (out / "distill_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (out / "distill_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n产物已写入 {out}", flush=True)
    print(f"  平均动作一致率 {meta['mean_action_agreement']*100:.1f}% | "
          f"加速比 {result['speedup_vs_mpc']:.1f}x" if result["speedup_vs_mpc"] else "", flush=True)


if __name__ == "__main__":
    main()
