"""全算法统一矩阵评估（冻结协议 · 102 天完整生长季）。

目的：把「经典控制 / 一步前瞻 / RL 算法矩阵 / P0 候选」全部落到**同一口径**下
的一张横向对比表，供论文第 3、5 章与 P0 结论使用。

冻结协议（与 eval_p0_bundle.make_env 完全一致）：
  episode_days=102, start_day_index=0, crop_start=seedling, disable_supplements=True,
  cooling_weight=0.5, humidity_weight=2.0, cooling_mode="overheat",
  obs_include_outdoor=True, screen_shade_weight=0.5, comfort_weight=0.0, smooth_weight=0.0,
  env.reset(seed=0)，全程 deterministic 推理。

策略分组：
  A 人工基准   human
  B 经典控制   baseline / rule / pid / mpc(H=1 原实现)
  C 前瞻       一/多步由 eval_horizon_frontier 负责，此处补 H=1 统一实现
  D RL 矩阵    ppo / a2c / dqn / sac / ddpg / td3 / trpo / recurrent_ppo （同协议 60 万步）
  E P0/扩展    tqc / crossq / masked_ppo / multiagent / dagger

用法：
  python -m experiments.controllers.glass_rl.eval_algorithm_matrix            # 全部
  python -m experiments.controllers.glass_rl.eval_algorithm_matrix --strategies ppo sac mpc
  python -m experiments.controllers.glass_rl.eval_algorithm_matrix --out latency_idle.json

输出：
  results/.../rl/algorithm_matrix/algorithm_matrix.json
  results/.../rl/algorithm_matrix/algorithm_matrix.md
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

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
ALGO = RL / "algorithms_fullseason"
OUT = RL / "algorithm_matrix"

DAYS = 102
EXEC = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]

# 分组展示顺序（缺模型自动跳过）
GROUPS = {
    "A 人工": ["human"],
    "B 经典控制": ["baseline", "rule", "pid", "mpc"],
    "C 前瞻": ["lookahead_h1"],
    "D RL 矩阵": ["ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo"],
    "E P0/扩展": ["tqc", "crossq", "masked_ppo", "multiagent", "dagger_v1"],
}


# ---------------------------------------------------------------- 环境 / 指标
def make_env(n_forecast_hours: int = 0):
    from glass_env import GlassGreenhouseEnv
    return GlassGreenhouseEnv(
        n_forecast_hours=n_forecast_hours,
        episode_days=DAYS, start_day_index=0, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
        comfort_weight=0.0, smooth_weight=0.0,
    )


def _comfort(t, rh, hour):
    day = 6.0 <= hour < 20.0
    lo, hi = (20.0, 28.0) if day else (16.0, 24.0)
    return (lo <= t <= hi) and (60.0 <= rh <= 85.0)


def metrics(temp, rh, hour, day, rew, f0, env, times) -> dict:
    temp = np.asarray(temp); rh = np.asarray(rh); hour = np.asarray(hour)
    day = np.asarray(day)
    c = np.array([_comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    daily_max = np.array([temp[day == d].max() for d in range(DAYS)])
    daily_mean = np.array([temp[day == d].mean() for d in range(DAYS)])
    per_day = np.array([c[day == d].mean() for d in range(DAYS)])
    month = np.where(day < 30, 4, np.where(day < 61, 5, np.where(day < 91, 6, 7)))
    return {
        "comfort_pct": round(float(c.mean() * 100), 1),
        "fruit_kg": round(float((float(env.x[25]) - f0) * 1e-6 / 0.081), 3),
        "max_temp": round(float(temp.max()), 2),
        "mean_temp": round(float(temp.mean()), 2),
        "no_overheat_day": round(float((daily_max <= 35.0).mean() * 100), 1),
        "comfort_day": round(float((per_day >= 0.5).mean() * 100), 1),
        "daily_t_std": round(float(daily_mean.std()), 3),
        "reward_total": round(float(np.sum(rew)), 1),
        "decision_ms_mean": round(float(np.mean(times) * 1000), 3),
        "month_comfort": {int(m): round(float(c[month == m].mean() * 100), 1)
                          for m in (4, 5, 6, 7)},
    }


def rollout(predict, n_forecast_hours: int = 0, warmup: int = 3):
    env = make_env(n_forecast_hours)
    env.reset(seed=0)
    f0 = float(env.x[25])
    for _ in range(warmup):            # 预热，排除首次调用开销
        predict(env)
    env.reset(seed=0)
    f0 = float(env.x[25])
    temp, rh, hour, day, rew, times, acts = [], [], [], [], [], [], []
    for i in range(DAYS * 24):
        t0 = time.perf_counter()
        a = predict(env)
        times.append(time.perf_counter() - t0)
        acts.append(np.asarray(a, dtype=int))
        _, r, _, _, info = env.step(a)
        rew.append(float(r))
        temp.append(float(info["temperature"]))
        rh.append(float(info["rh"]))
        hour.append(i % 24)
        day.append(i // 24)
    m = metrics(temp, rh, hour, day, rew, f0, env, times)
    A = np.stack(acts)
    m["action_usage_pct"] = {EXEC[j]: round(float((A[:, j] > 0).mean() * 100), 1)
                             for j in range(A.shape[1])}
    m["mean_level_sum"] = round(float(A.sum(axis=1).mean()), 3)
    return m


# ---------------------------------------------------------------- 各策略 predictor
def _levels(a, kind):
    from experiments.controllers.glass_rl.env_variants import (
        continuous_to_levels, decode_levels)
    if kind == "multi":
        return np.asarray(a, dtype=int).reshape(-1)
    if kind == "flat":
        return decode_levels(int(np.asarray(a).reshape(-1)[0]))
    return continuous_to_levels(np.asarray(a, dtype=float).reshape(-1))


def pred_sb3(name: str, subdir: Path | None = None):
    import importlib
    import stable_baselines3 as sb3
    import sb3_contrib

    kind, cls_name = {
        "ppo": ("multi", "PPO"), "a2c": ("multi", "A2C"),
        "trpo": ("multi", "TRPO"), "recurrent_ppo": ("multi", "RecurrentPPO"),
        "dqn": ("flat", "DQN"), "sac": ("cont", "SAC"),
        "ddpg": ("cont", "DDPG"), "td3": ("cont", "TD3"),
        "tqc": ("cont", "TQC"), "crossq": ("cont", "CrossQ"),
    }[name]
    mod = sb3 if hasattr(sb3, cls_name) else sb3_contrib
    path = (subdir or (ALGO / name)) / "model.zip"
    model = getattr(mod, cls_name).load(str(path))
    return lambda e: _levels(model.predict(e._get_obs(), deterministic=True)[0], kind), kind


def pred_maskable():
    from sb3_contrib import MaskablePPO
    from experiments.controllers.glass_rl.env_variants import supplement_action_mask
    model = MaskablePPO.load(str(RL / "masked_ppo" / "model.zip"))
    mask = supplement_action_mask()
    return lambda e: _levels(model.predict(e._get_obs(), deterministic=True,
                                           action_masks=mask)[0], "multi")


def pred_multiagent():
    import stable_baselines3 as sb3
    from experiments.controllers.glass_rl.multiagent_env import AGENT_ORDER, merge_actions
    models = {a: sb3.PPO.load(str(RL / "multiagent" / a / "model.zip")) for a in AGENT_ORDER}

    def f(e):
        obs = e._get_obs()
        acts = {a: np.asarray(models[a].predict(obs, deterministic=True)[0],
                              dtype=int).reshape(-1) for a in AGENT_ORDER}
        return merge_actions(acts["cooling"], acts["pad"], acts["insulation"])
    return f


def pred_dagger_v1():
    import torch
    from experiments.controllers.glass_rl.distill_mpc import DistillPolicy
    norm = json.loads((RL / "distill_mpc" / "norm.json").read_text(encoding="utf-8"))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    model = DistillPolicy(n_in=len(mean))
    model.load_state_dict(torch.load(RL / "distill_mpc" / "policy.pt", map_location="cpu"))
    model.eval()
    return lambda e: model.act(e._get_obs(), mean, std)


def pred_human():
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import (
        load_human_actions, ACT_COLS)
    ha = load_human_actions().iloc[: DAYS * 24].reset_index(drop=True)
    acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
    st = [0]

    def f(_e):
        i = st[0]; st[0] += 1
        return acts[min(i, len(acts) - 1)]   # 护栏：预热/越界时锁定最后一档
    return f


def pred_classical(name: str):
    from experiments.controllers.glass_rl import classical_controllers as cc
    cls = {"baseline": cc.GlassBaseline, "rule": cc.GlassRuleBased,
           "pid": cc.GlassPID, "mpc": cc.GlassMPC}[name]
    return cls().predict_action


def pred_lookahead_h1():
    from experiments.controllers.glass_rl.horizon_search import GlassHorizonSearch
    ctrl = GlassHorizonSearch(horizon=1, compute_budget=192)
    return lambda e: ctrl.predict_action(e)


BUILDERS = {
    "human": pred_human,
    "baseline": lambda: pred_classical("baseline"),
    "rule": lambda: pred_classical("rule"),
    "pid": lambda: pred_classical("pid"),
    "mpc": lambda: pred_classical("mpc"),
    "lookahead_h1": pred_lookahead_h1,
    "masked_ppo": pred_maskable,
    "multiagent": pred_multiagent,
    "dagger_v1": pred_dagger_v1,
}
for _n in ("ppo", "a2c", "dqn", "sac", "ddpg", "td3", "trpo", "recurrent_ppo"):
    BUILDERS[_n] = (lambda n: (lambda: pred_sb3(n)[0]))(_n)
BUILDERS["tqc"] = lambda: pred_sb3("tqc", RL / "tqc")[0]
BUILDERS["crossq"] = lambda: pred_sb3("crossq", RL / "crossq")[0]


def _pred_ppo_at(subdir: str):
    import stable_baselines3 as sb3
    m = sb3.PPO.load(str(RL / subdir / "model.zip"))

    def f(e):
        a, _ = m.predict(e._get_obs(), deterministic=True)
        return np.asarray(a, dtype=int).reshape(-1)
    return f


# 历史 PPO 世代（用于与矩阵内新同协议 PPO 并列对照）
BUILDERS["ppo_v6"] = lambda: _pred_ppo_at("ppo_final_v6")           # obs8
BUILDERS["ppo_v9"] = lambda: _pred_ppo_at("ppo_final_v9")           # obs16（未来 4h 天气）

# 需要额外前瞻观测的策略（观测维度不同，单独标注）
FORECAST = {"dag_v2": 4, "ppo_v9": 4}
# 不可预热策略（预热会消耗有限的动作序列 / 有状态）
NO_WARMUP = {"human"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=None)
    ap.add_argument("--out", default="algorithm_matrix.json")
    args = ap.parse_args()

    todo = args.strategies or [s for g in GROUPS.values() for s in g]
    OUT.mkdir(parents=True, exist_ok=True)

    out = OUT / args.out
    results: dict = {}
    if out.exists():                      # 增量更新：保留已评估的策略，只覆盖本次
        results = json.loads(out.read_text(encoding="utf-8"))
        print(f"增量模式：已载入 {len(results)} 条既有结果", flush=True)

    for name in todo:
        if name not in BUILDERS:
            print(f"[skip] 未知策略 {name}", flush=True)
            continue
        try:
            pred = BUILDERS[name]()
        except Exception as e:
            print(f"[skip] {name}: 模型加载失败 {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        t0 = time.time()
        m = rollout(pred, n_forecast_hours=FORECAST.get(name, 0),
                    warmup=0 if name in NO_WARMUP else 3)
        m["wall_seconds"] = round(time.time() - t0, 1)
        results[name] = m
        print(f"[{name:14s}] 舒适{m['comfort_pct']:6.1f}% 果实{m['fruit_kg']:6.2f} "
              f"maxT{m['max_temp']:5.1f} 无过热{m['no_overheat_day']:5.1f}% "
              f"迟{m['decision_ms_mean']:8.3f}ms ({m['wall_seconds']:.0f}s)", flush=True)

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}  共 {len(results)} 条", flush=True)


if __name__ == "__main__":
    main()
