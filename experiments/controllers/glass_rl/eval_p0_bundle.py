"""P0 汇总评估：把四条 P0 的收益落到同一张对比表 + 一组图上。

覆盖：
  P0-1  MaskablePPO（动作掩码 1536→384） vs PPO v6（同配置无掩码）
  P0-2  TQC / CrossQ vs SAC v3（off-policy 家族，同协议）
  P0-3  DAgger 蒸馏（读 distill_mpc/distill_result.json）vs MPC / 规则
  P0-4  H 步前瞻前沿（读 horizon_frontier/horizon_frontier.json）

评估口径统一：102 天、4/1 定植（start_day_index=0）、seedling 起点、
overheat + humidity2.0 + obs8 + screen0.5 + comfort0 + smooth0。

输出：
  results/.../rl/p0_bundle/p0_bundle.json
  results/.../rl/p0_bundle/p0_masked_ppo.png
  results/.../rl/p0_bundle/p0_offpolicy.png
  results/.../rl/p0_bundle/p0_dagger.png
  results/.../rl/p0_bundle/p0_frontier.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "p0_bundle"

DAYS = 102


def make_env(padded: bool = False):
    from glass_env import GlassGreenhouseEnv

    base = GlassGreenhouseEnv(
        episode_days=DAYS, start_day_index=0, crop_start="seedling",
        disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
        cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
        comfort_weight=0.0, smooth_weight=0.0,
    )
    if not padded:
        return base, None
    from experiments.controllers.glass_rl.env_variants import SupplementMaskEnv
    from sb3_contrib.common.wrappers import ActionMasker

    masked = ActionMasker(SupplementMaskEnv(base), lambda e: e.action_masks())
    return masked, masked


def comfort(t, rh, hour):
    is_day = 6.0 <= hour < 20.0
    t_lo, t_hi = (20.0, 28.0) if is_day else (16.0, 24.0)
    return (t_lo <= t <= t_hi) and (60.0 <= rh <= 85.0)


def metrics(temp, rh, hour, day, rew, f0, env, times) -> dict:
    temp = np.asarray(temp); rh = np.asarray(rh); hour = np.asarray(hour)
    day = np.asarray(day)
    c = np.array([comfort(t, h, hr) for t, h, hr in zip(temp, rh, hour)])
    daily_max = np.array([temp[day == d].max() for d in range(DAYS)])
    daily_mean = np.array([temp[day == d].mean() for d in range(DAYS)])
    per_day = np.array([c[day == d].mean() for d in range(DAYS)])
    month = np.where(day < 30, 4, np.where(day < 61, 5, np.where(day < 91, 6, 7)))
    return {
        "comfort_pct": float(c.mean() * 100),
        "fruit_kg": float((float(env.x[25]) - f0) * 1e-6 / 0.081),
        "max_temp": float(temp.max()),
        "mean_temp": float(temp.mean()),
        "no_overheat_day": float((daily_max <= 35.0).mean() * 100),
        "comfort_day": float((per_day >= 0.5).mean() * 100),
        "daily_t_std": float(daily_mean.std()),
        "reward_total": float(np.sum(rew)),
        "decision_ms_mean": float(np.mean(times) * 1000),
        "month_comfort": {int(m): float(c[month == m].mean() * 100) for m in (4, 5, 6, 7)},
    }


def rollout(predict, padded: bool = False):
    env, wrapped = make_env(padded)
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
    m["action_usage_pct"] = {
        ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"][j]:
            float((A[:, j] > 0).mean() * 100) for j in range(A.shape[1])
    }
    m["mean_level_sum"] = float(A.sum(axis=1).mean())
    return m


# --------------------------------------------------------------- 各模型 predictor
def pred_ppo(path):
    from stable_baselines3 import PPO

    m = PPO.load(str(path))
    return lambda e: m.predict(e._get_obs(), deterministic=True)[0], False


def pred_maskable(path):
    from sb3_contrib import MaskablePPO

    m = MaskablePPO.load(str(path))

    def f(e):
        masks = e.action_masks() if hasattr(e, "action_masks") else None
        return m.predict(e._get_obs(), deterministic=True, action_masks=masks)[0]

    return f, True


def pred_cont(cls_name, path):
    import sb3_contrib
    from stable_baselines3 import SAC
    from experiments.controllers.glass_rl.env_variants import continuous_to_levels

    mod = {"SAC": SAC}
    cls = getattr(sb3_contrib, cls_name) if cls_name != "SAC" else mod["SAC"]
    m = cls.load(str(path))
    return lambda e: continuous_to_levels(m.predict(e._get_obs(), deterministic=True)[0]), False


def pred_dagger():
    import torch

    from experiments.controllers.glass_rl.distill_mpc import DistillPolicy

    norm = json.loads((RL / "distill_mpc" / "norm.json").read_text(encoding="utf-8"))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    model = DistillPolicy(n_in=len(mean))
    model.load_state_dict(torch.load(RL / "distill_mpc" / "policy.pt", map_location="cpu"))
    model.eval()
    return lambda e: model.act(e._get_obs(), mean, std), False


def load_curve(path: Path) -> dict:
    import csv as _csv
    if not path.exists():
        return {}
    rows = list(_csv.DictReader(open(path, encoding="utf-8")))
    ts, rew = [], []
    for r in rows:
        v = r.get("rollout/ep_rew_mean")
        if v in (None, ""):
            continue
        ts.append(float(r["timestep"]))
        rew.append(float(v))
    return {"timestep": ts, "ep_rew_mean": rew}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle: dict = {"env": "overheat + humidity2.0 + obs8 + screen0.5 + comfort0 + smooth0",
                    "days": DAYS, "results": {}}
    curves: dict = {}

    # ---- P0-1 掩码对照
    for label, kind, path in [
        ("PPO v6（无掩码）", "ppo", RL / "ppo_final_v6/model"),
        ("MaskablePPO（掩码384）", "maskable", RL / "masked_ppo/model"),
    ]:
        if not Path(str(path) + ".zip").exists():
            print(f"[skip] {label}: 模型不存在", flush=True)
            continue
        pred, padded = (pred_maskable(path) if kind == "maskable" else pred_ppo(path))
        r = rollout(pred, padded=padded)
        r["group"] = "P0-1 动作掩码"
        bundle["results"][label] = r
        print(f"[P0-1] {label}: 舒适 {r['comfort_pct']:.1f}% 果实 {r['fruit_kg']:.2f} "
              f"{r['decision_ms_mean']:.2f} ms", flush=True)

    # ---- P0-2 off-policy 家族
    for label, cls_name, sub in [
        ("SAC v3", "SAC", "sac_final_v3"),
        ("TQC", "TQC", "tqc"),
        ("CrossQ", "CrossQ", "crossq"),
    ]:
        p = RL / sub / "model"
        if not Path(str(p) + ".zip").exists():
            print(f"[skip] {label}: 模型不存在", flush=True)
            continue
        pred, padded = pred_cont(cls_name, p)
        r = rollout(pred, padded=padded)
        r["group"] = "P0-2 off-policy"
        bundle["results"][label] = r
        curves[label] = load_curve(RL / sub / "training_curve.csv")
        print(f"[P0-2] {label}: 舒适 {r['comfort_pct']:.1f}% 果实 {r['fruit_kg']:.2f} "
              f"{r['decision_ms_mean']:.2f} ms", flush=True)

    # ---- P0-3 DAgger
    dr = RL / "distill_mpc" / "distill_result.json"
    if dr.exists():
        d = json.loads(dr.read_text(encoding="utf-8"))
        bundle["results"]["DAgger蒸馏"] = {**d.get("student", {}), "group": "P0-3 蒸馏"}
        if isinstance(d.get("mpc"), dict) and "comfort_pct" in d["mpc"]:
            bundle["results"]["MPC（原实现）"] = {**d["mpc"], "group": "P0-3 蒸馏"}
        bundle["distill_meta"] = json.loads(
            (RL / "distill_mpc" / "distill_meta.json").read_text(encoding="utf-8"))
        print(f"[P0-3] DAgger: 舒适 {bundle['results']['DAgger蒸馏'].get('comfort_pct', float('nan')):.1f}%", flush=True)
    else:
        print("[skip] P0-3: distill_result.json 不存在", flush=True)

    # ---- P0-4 前沿
    hf = RL / "horizon_frontier" / "horizon_frontier.json"
    if hf.exists():
        bundle["horizon_frontier"] = json.loads(hf.read_text(encoding="utf-8"))
        print("[P0-4] 前沿已并入", flush=True)

    # PPO v6 / SAC v3 的训练曲线也带上（用于样本效率对比）
    curves.setdefault("PPO v6（无掩码）", load_curve(RL / "ppo_final_v6/training_curve.csv"))
    curves.setdefault("MaskablePPO（掩码384）", load_curve(RL / "masked_ppo/training_curve.csv"))
    bundle["curves"] = curves

    (OUT / "p0_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(f"\n已写入 {OUT / 'p0_bundle.json'}", flush=True)
    print("结果行数:", len(bundle["results"]), flush=True)


if __name__ == "__main__":
    main()
