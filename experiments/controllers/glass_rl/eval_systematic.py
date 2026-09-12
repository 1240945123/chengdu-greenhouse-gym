"""五维系统评估：PPO / SAC / MPC / 规则 / 人工。

维度：
1. 成功率   —— 无过热日率（日最高温≤35°C 的天数占比）、舒适达标日率（日舒适≥50% 占比）
2. 稳定性   —— 日均温日间标准差、日内温度标准差、最差日舒适率、最高温极值
3. 样本效率 —— 训练曲线到达"最终收益 80%"所需步数、训练总耗时（非学习类为 N/A）
4. 计算开销 —— 每步决策延迟（ms）、102 天总决策耗时
5. 性能     —— 舒适率、果实、reward

统一 102 天任务与口径（ENV_KW）。输出 eval_systematic.json + 控制台表。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT_JSON = RL / "eval_systematic.json"

ENV_KW = dict(episode_days=102, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5,
              comfort_weight=0.0, smooth_weight=0.0)


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def build_act(spec):
    kind = spec["kind"]
    if kind == "ppo":
        from stable_baselines3 import PPO
        m = PPO.load(str(spec["path"]))
        return lambda e: m.predict(e._get_obs(), deterministic=True)[0]
    if kind == "sac":
        from stable_baselines3 import SAC
        from experiments.controllers.glass_rl.env_variants import continuous_to_levels
        m = SAC.load(str(spec["path"]))
        return lambda e: continuous_to_levels(m.predict(e._get_obs(), deterministic=True)[0])
    if kind == "residual":
        from stable_baselines3 import PPO
        m = PPO.load(str(spec["path"]))
        # 返回原始 Box 残差动作；档位转换由 GlassGreenhouseEnvResidual.step() 内部完成，
        # 切勿在此再转一次（否则会被二次当作残差处理）。
        return lambda e: m.predict(e._get_obs(), deterministic=True)[0]
    if kind == "mpc":
        from experiments.controllers.glass_rl.classical_controllers import GlassMPC
        c = GlassMPC()
        return lambda e: c.predict_action(e)
    if kind == "rule":
        from experiments.controllers.glass_rl.classical_controllers import GlassRuleBased
        c = GlassRuleBased()
        return lambda e: c.predict_action(e)
    raise ValueError(kind)


def run_human(env):
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import load_human_actions, ACT_COLS
    ha = load_human_actions().iloc[:102 * 24].reset_index(drop=True)
    acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
    state = [0]

    def act(_env):
        a = acts[state[0]]
        state[0] += 1
        return a
    _t0 = time.perf_counter()
    recs, times = [], []
    fruit0 = float(env.x[25])
    for i in range(102 * 24):
        s = time.perf_counter()
        a = act(env)
        dt = time.perf_counter() - s
        o, r, te, tr, info = env.step(a)
        times.append(dt)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    return recs, times, fruit0


def run(spec):
    from glass_env import GlassGreenhouseEnv
    if spec["kind"] == "residual":
        from experiments.controllers.glass_rl.residual_env import GlassGreenhouseEnvResidual
        env = GlassGreenhouseEnvResidual(**ENV_KW, base_controller="rule", residual_scale=2)
    else:
        env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    if spec["kind"] == "human":
        recs, times, fruit0 = run_human(env)
        return recs, times, fruit0, env
    act = build_act(spec)
    # 预热一次（排除首次编译/加载开销）
    act(env)
    recs, times = [], []
    fruit0 = float(env.x[25])
    for i in range(102 * 24):
        s = time.perf_counter()
        a = act(env)
        dt = time.perf_counter() - s
        o, r, te, tr, info = env.step(a)
        times.append(dt)
        recs.append({"day": i // 24, "hour": i % 24, "t": info["temperature"], "rh": info["rh"]})
    return recs, times, fruit0, env


def sample_efficiency(csv: Path, label: str) -> dict:
    if not csv.exists():
        return {"steps_to_80pct": None, "note": "无训练曲线（非学习类）"}
    df = pd.read_csv(csv)
    r = pd.to_numeric(df["rollout/ep_rew_mean"], errors="coerce")
    ts = pd.to_numeric(df["timestep"], errors="coerce")
    mask = r.notna()
    r, ts = r[mask].values, ts[mask].values
    if len(r) < 5:
        return {"steps_to_80pct": None, "note": "曲线点过少"}
    k = max(3, len(r) // 20)
    sm = np.convolve(r, np.ones(k) / k, mode="same")
    r0, r1 = sm[:k].mean(), sm[-k:].mean()
    thr = r0 + 0.8 * (r1 - r0)
    idx = np.argmax(sm >= thr) if (sm >= thr).any() else len(sm) - 1
    note = None
    if ts[0] > 10_000:
        note = f"续训曲线（仅覆盖 {int(ts[0])//1000}k–{int(ts[-1])//1000}k 段，步数为该段内计数）"
    return {"steps_to_80pct": int(ts[idx]), "final_reward": float(r1),
            "total_steps": int(ts[-1]), "curve_start": int(ts[0]), "note": note}


SPECS = [
    {"name": "PPO v6", "kind": "ppo", "path": RL / "ppo_final_v6/model", "csv": RL / "ppo_final_v6/training_curve.csv",
     "train_meta": RL / "ppo_final_v6/train_meta.json"},
    {"name": "PPO v8", "kind": "ppo", "path": RL / "ppo_final_v8/model", "csv": RL / "ppo_final_v8/training_curve.csv",
     "train_meta": RL / "ppo_final_v8/train_meta.json"},
    {"name": "SAC v3", "kind": "sac", "path": RL / "sac_final_v3/model", "csv": RL / "sac_final_v3/training_curve.csv",
     "train_meta": RL / "sac_final_v3/train_meta.json"},
    {"name": "SAC v4", "kind": "sac", "path": RL / "sac_final_v4/model", "csv": RL / "sac_final_v4/training_curve.csv",
     "train_meta": RL / "sac_final_v4/train_meta.json"},
    {"name": "残差PPO", "kind": "residual", "path": RL / "residual_ppo/model", "csv": RL / "residual_ppo/training_curve.csv",
     "train_meta": RL / "residual_ppo/train_meta.json"},
    {"name": "MPC", "kind": "mpc"},
    {"name": "规则", "kind": "rule"},
    {"name": "人工", "kind": "human"},
]


def main():
    out = {}
    for spec in SPECS:
        name = spec["name"]
        print(f"评估 {name} ...", flush=True)
        recs, times, fruit0, env = run(spec)
        df = pd.DataFrame(recs)
        df["is_day"] = (df.hour >= 6) & (df.hour < 20)
        df["month"] = pd.cut(df.day, bins=[-1, 30, 61, 91, 102], labels=["4月", "5月", "6月", "7月"])
        daily = df.groupby("day").agg(t_mean=("t", "mean"), t_max=("t", "max")).reset_index()
        daily_comf = np.array([
            np.mean([comfort(x.t, x.rh, x.hour) for x in g.itertuples()]) * 100
            for _, g in df.groupby("day")])
        overall = float(np.mean([comfort(x.t, x.rh, x.hour) for x in df.itertuples()]) * 100)
        fruit = float((float(env.x[25]) - fruit0) * 1e-6 / 0.081)
        mc = {k: float(np.mean([comfort(x.t, x.rh, x.hour) for x in g.itertuples()]) * 100)
              for k, g in df.groupby("month", observed=True)}

        # 稳定性：日内温度标准差（各天小时温度标准差再平均）
        intraday = df.groupby("day")["t"].std().mean()
        res = {
            "comfort_pct": overall,
            "fruit_kg": fruit,
            "max_temp": float(df.t.max()),
            "mean_temp": float(df.t.mean()),
            # 成功率
            "success_no_overheat_day": float((daily.t_max <= 35.0).mean() * 100),
            "success_comfort_day": float((daily_comf >= 50.0).mean() * 100),
            # 稳定性
            "daily_t_std": float(daily.t_mean.std()),
            "intraday_t_std": float(intraday),
            "worst_day_comfort": float(daily_comf.min()),
            "comfort_day_std": float(daily_comf.std()),
            # 计算开销
            "decision_ms_mean": float(np.mean(times) * 1000),
            "decision_ms_p95": float(np.percentile(times, 95) * 1000),
            "decide_total_s": float(np.sum(times)),
            # 样本效率
            "sample_eff": sample_efficiency(spec.get("csv"), name) if spec.get("csv") else
                          {"steps_to_80pct": None, "note": "非学习类（无训练）"},
            "month_comfort": mc,
        }
        if spec.get("train_meta") and Path(spec["train_meta"]).exists():
            tm = json.loads(Path(spec["train_meta"]).read_text(encoding="utf-8"))
            res["train_minutes"] = round(tm.get("elapsed_seconds", 0) / 60, 1)
        out[name] = res
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {name}: 舒适{overall:.1f}% 果实{fruit:.2f} 无过热日{res['success_no_overheat_day']:.0f}% "
              f"决策{res['decision_ms_mean']:.2f}ms", flush=True)

    print(f"\n已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
