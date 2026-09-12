"""P0-3 v2：蒸馏失败归因 + 增强输入重试。

v1 结果（distill_mpc.py）：逐维动作一致率 94.7%，但**全季动作完全一致率仅 42.5%**，
学生舒适率 44.6%（< 规则 48.8% < MPC 67.4%）。逐维一致率 94.7% 的独立假设下
九维全对期望约 60%，实测 42.5% → **复合误差（compounding error）**主导。

两个待检验假设：
  H1【信息赤字】学生 obs 里的室外温是 **上一小时** 值（glass_env._last_t_out 滞后 1h），
     而专家 MPC 直接用当前时刻天气 d = _weather_at(_w_idx)。
     对策：把「当前天气（辐射/室外温/室外水汽压/风速）+ 时刻 sin-cos + 上一动作(9)」
     拼进学生输入（8 → 23 维）。
  H2【关键维度误差】逐维一致率最低的是 **顶窗 84.3% / 风机 83.9%**，而这恰是消融实验
     里对舒适率最关键的两位（禁用顶窗 −14.0pp）。对策：归因实验——在推理时把学生的
     若干维**替换为专家动作**，看舒适率能否回到 MPC 水平，从而量化"误差出在哪一维"。

输出：results/.../rl/distill_mpc_v2/distill_v2_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

from experiments.controllers.glass_rl.distill_mpc import (  # noqa: E402
    DistillPolicy, comfort, make_env, train_student,
)

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/distill_mpc_v2")
SCENARIOS = [0, 20, 40, 60, 80]
GEN_DAYS = 40
ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]


def features(env, prev_action: np.ndarray) -> np.ndarray:
    """23 维学生输入 = obs(8) + 当前天气(4) + 时刻编码(2) + 上一动作(9)。

    其中 obs(8) 已含 t/rh/co2/t_can/hour/fruit/积温/滞后室外温；
    新增"当前天气"正是专家 MPC 能看到的量，用于检验 H1（信息赤字）。
    """
    obs = env._get_obs()
    d = env._weather_at(env._w_idx)
    hour = float(env._hour)
    extra = np.array([
        float(d[0]),           # 当前辐射 W/m2
        float(d[1]),           # 当前室外温度 °C
        float(d[2]),           # 当前室外水汽压 Pa
        float(d[4]),           # 当前风速 m/s
        float(np.sin(2 * np.pi * hour / 24.0)),
        float(np.cos(2 * np.pi * hour / 24.0)),
    ], dtype=np.float32)
    return np.concatenate([np.asarray(obs, dtype=np.float32), extra,
                           np.asarray(prev_action, dtype=np.float32)]).astype(np.float32)


def collect(expert, scenarios, actor=None):
    X, Y = [], []
    t_mpc, n_mpc = 0.0, 0
    for s in scenarios:
        env = make_env(GEN_DAYS, s)
        env.reset(seed=0)
        prev = np.zeros(9, dtype=np.float32)
        for _ in range(GEN_DAYS * 24):
            feat = features(env, prev)
            t0 = time.perf_counter()
            a_exp = expert.predict_action(env)
            t_mpc += time.perf_counter() - t0
            n_mpc += 1
            X.append(feat)
            Y.append(np.asarray(a_exp, dtype=np.int64))
            a = a_exp if actor is None else actor(env)
            env.step(a)
            prev = np.asarray(a, dtype=np.float32)
        print(f"  [collect] start_day={s} actor={'expert' if actor is None else 'student'} "
              f"样本 {len(X)}", flush=True)
    print(f"  [collect] MPC {n_mpc} 次，{t_mpc:.1f}s（均 {1000*t_mpc/max(n_mpc,1):.1f} ms/决策）",
          flush=True)
    return np.stack(X), np.stack(Y)


def metrics(env, f0, temp, rh, hour, day, rew, times):
    temp = np.asarray(temp); rh = np.asarray(rh)
    hour = np.asarray(hour); day = np.asarray(day)
    c = np.array([comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    daily_max = np.array([temp[day == d].max() for d in range(102)])
    daily_mean = np.array([temp[day == d].mean() for d in range(102)])
    per_day = np.array([c[day == d].mean() for d in range(102)])
    month = np.where(day < 30, 4, np.where(day < 61, 5, np.where(day < 91, 6, 7)))
    return {
        "comfort_pct": float(c.mean() * 100),
        "fruit_kg": float((float(env.x[25]) - f0) * 1e-6 / 0.081),
        "max_temp": float(temp.max()), "mean_temp": float(temp.mean()),
        "no_overheat_day": float((daily_max <= 35.0).mean() * 100),
        "comfort_day": float((per_day >= 0.5).mean() * 100),
        "daily_t_std": float(daily_mean.std()),
        "reward_total": float(np.sum(rew)),
        "decision_ms_mean": float(np.mean(times) * 1000),
        "month_comfort": {int(m): float(c[month == m].mean() * 100) for m in (4, 5, 6, 7)},
    }


def rollout_hybrid(student_fn, expert, expert_dims: tuple[int, ...] = ()):
    """学生策略 + 指定维度替换为专家动作（归因实验）。"""
    env = make_env(102, 0)
    env.reset(seed=0)
    f0 = float(env.x[25])
    temp, rh, hour, day, rew, times = [], [], [], [], [], []
    for i in range(102 * 24):
        t0 = time.perf_counter()
        a = student_fn(env)
        if expert_dims:
            a_exp = expert.predict_action(env)
            a = np.asarray(a, dtype=int).copy()
            for j in expert_dims:
                a[j] = int(a_exp[j])
        times.append(time.perf_counter() - t0)
        _, r, _, _, info = env.step(a)
        rew.append(float(r))
        temp.append(float(info["temperature"]))
        rh.append(float(info["rh"]))
        hour.append(i % 24)
        day.append(i // 24)
    return metrics(env, f0, temp, rh, hour, day, rew, times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--daggers", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    from experiments.controllers.glass_rl.classical_controllers import GlassMPC

    expert = GlassMPC()
    meta: dict = {"student_input_dim": 23, "expert": "GlassMPC(H=1 一步枚举)",
                  "scenarios": SCENARIOS, "gen_days": GEN_DAYS,
                  "hypotheses": ["H1 信息赤字(室外温滞后1h)",
                                 "H2 关键维度误差(顶窗/风机)"]}

    # ---- ① 数据（增强输入）
    print("\n===== ① 数据采集（23 维增强输入）=====", flush=True)
    t0 = time.time()
    X, Y = collect(expert, SCENARIOS)
    print(f"  数据集 {X.shape} 用时 {time.time()-t0:.1f}s", flush=True)
    mean = X.mean(axis=0).astype(np.float32)
    std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0)).astype(np.float32)

    # ---- ② BC
    print("\n===== ② 行为克隆（增强输入）=====", flush=True)
    model, hist, acc = train_student(X, Y, mean, std, epochs=args.epochs)
    print("  逐维一致率: " + " ".join(f"{ACT_NAMES[j]}={acc[j]:.3f}" for j in range(9)), flush=True)
    meta["bc"] = {"n_samples": int(X.shape[0]), "loss": hist[-1], "per_dim_acc": acc,
                  "joint_acc_check": float(np.prod(acc))}

    # ---- ③ DAgger
    for it in range(args.daggers):
        print(f"\n===== ③ DAgger {it+1}/{args.daggers}（增强输入）=====", flush=True)
        model.eval()

        def student_act(env, _m=model, _mu=mean, _s=std):
            return _m.act(features(env, np.zeros(9, dtype=np.float32)), _mu, _s)

        Xs, Ys = collect(expert, SCENARIOS, actor=student_act)
        X = np.concatenate([X, Xs]); Y = np.concatenate([Y, Ys])
        mean = X.mean(axis=0).astype(np.float32)
        std = np.where(X.std(axis=0) < 1e-6, 1.0, X.std(axis=0)).astype(np.float32)
        model, hist, acc = train_student(X, Y, mean, std, epochs=args.epochs)
        print("  逐维一致率: " + " ".join(f"{ACT_NAMES[j]}={acc[j]:.3f}" for j in range(9)),
              flush=True)
        meta[f"dagger_{it+1}"] = {"n_samples": int(X.shape[0]), "loss": hist[-1],
                                  "per_dim_acc": acc, "joint_acc_check": float(np.prod(acc))}

    torch.save(model.state_dict(), OUT / "policy_v2.pt")
    (OUT / "norm_v2.json").write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}),
                                      encoding="utf-8")
    meta["final_per_dim_acc"] = acc
    meta["mean_per_dim_acc"] = float(np.mean(acc))
    meta["prod_per_dim_acc"] = float(np.prod(acc))
    print(f"\n  逐维均值 {np.mean(acc)*100:.1f}% ｜ 独立假设九维全对期望 {np.prod(acc)*100:.1f}%",
          flush=True)

    # ---- ④ 归因诊断 + 评估
    model.eval()
    prev_action = {"v": np.zeros(9, dtype=np.float32)}

    def student_fn(env):
        f = features(env, prev_action["v"])
        a = model.act(f, mean, std)
        prev_action["v"] = np.asarray(a, dtype=np.float32)
        return a

    print("\n===== ④ 归因诊断（把学生的若干维替换为专家动作）=====", flush=True)
    results = {}
    for label, dims in [("v2学生（无替换）", ()),
                        ("v2学生 + 专家接管[顶窗,风机]", (3, 6)),
                        ("v2学生 + 专家接管[外遮阳]", (0,))]:
        prev_action["v"] = np.zeros(9, dtype=np.float32)
        t0 = time.time()
        r = rollout_hybrid(student_fn, expert, dims)
        r["expert_dims"] = list(dims)
        r["wall_seconds"] = round(time.time() - t0, 1)
        results[label] = r
        print(f"  {label}: 舒适 {r['comfort_pct']:.1f}% | 果实 {r['fruit_kg']:.2f} | "
              f"最高温 {r['max_temp']:.1f} | {r['decision_ms_mean']:.2f} ms", flush=True)

    # 读 v1 与 MPC 参照
    v1 = json.loads((Path(str(OUT).replace("_v2", "")) / "distill_result.json")
                    .read_text(encoding="utf-8"))
    out = {"meta": meta, "v2": results,
           "v1_student": v1.get("student"), "mpc_reference": v1.get("mpc"),
           "rule_reference": v1.get("rule")}
    (OUT / "distill_v2_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(f"\n已写入 {OUT / 'distill_v2_result.json'}", flush=True)


if __name__ == "__main__":
    main()
