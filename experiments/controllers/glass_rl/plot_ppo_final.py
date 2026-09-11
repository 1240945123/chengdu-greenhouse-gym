"""PPO 最终模型 vs 人工 对比图（标定后环境，102 天）。
生成：温度轨迹、果实积累、分月舒适率 三张图。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final/figures")
OUT.mkdir(parents=True, exist_ok=True)

RED = "#c0392b"   # 人工（红）
BLUE = "#2c6fbb"  # RL/PPO（蓝）


def run_traj(act_fn, days: int = 102):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(episode_days=days, start_day_index=0,
                             crop_start="seedling", disable_supplements=True,
                             cooling_weight=0.5)
    obs, _ = env.reset(seed=0)
    fruit0 = float(env.x[25])
    temps, rhs, fruits = [], [], []
    for i in range(days * 24):
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        temps.append(info["temperature"])
        rhs.append(info["rh"])
        fruits.append((float(env.x[25]) - fruit0) * 1e-6 / 0.081)
    return np.array(temps), np.array(rhs), np.array(fruits)


def main():
    from stable_baselines3 import PPO
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import load_human_actions, ACT_COLS

    model = PPO.load("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final/model")
    rl_act = lambda obs: model.predict(obs, deterministic=True)[0]

    ha = load_human_actions().iloc[: 102 * 24].reset_index(drop=True)
    human_acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
    state = {"i": 0}

    def human_act(obs):
        a = human_acts[state["i"]]
        state["i"] += 1
        return a

    print("跑 PPO 轨迹...")
    rt, rrh, rf = run_traj(rl_act)
    print("跑人工轨迹...")
    ht, hrh, hf = run_traj(human_act)

    hours = np.arange(102 * 24)
    days_axis = hours / 24.0

    # 图1：温度轨迹（日均 + 日最高）
    fig, ax = plt.subplots(figsize=(13, 4.8))
    def daily(series, fn):
        s = series.reshape(102, 24)
        return fn(s, axis=1)
    ax.fill_between(days_axis[::24], daily(ht, np.min), daily(ht, np.max),
                    color=RED, alpha=0.15)
    ax.fill_between(days_axis[::24], daily(rt, np.min), daily(rt, np.max),
                    color=BLUE, alpha=0.15)
    ax.plot(days_axis[::24], daily(ht, np.mean), color=RED, lw=1.8, label="人工 日均温")
    ax.plot(days_axis[::24], daily(rt, np.mean), color=BLUE, lw=1.8, label="PPO 日均温")
    ax.plot(days_axis[::24], daily(ht, np.max), color=RED, lw=1.0, ls="--", alpha=0.7, label="人工 日最高")
    ax.plot(days_axis[::24], daily(rt, np.max), color=BLUE, lw=1.0, ls="--", alpha=0.7, label="PPO 日最高")
    ax.axhspan(20, 28, color="green", alpha=0.06)
    ax.set_xlabel("生长季天数（4/1 定植起）")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("完整生长季温度轨迹：人工 vs PPO（标定后环境）")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ppo_temperature.png", dpi=130)
    plt.close(fig)

    # 图2：果实积累
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(days_axis[::24], hf[::24], color=RED, lw=2, label="人工")
    ax.plot(days_axis[::24], rf[::24], color=BLUE, lw=2, label="PPO")
    ax.set_xlabel("生长季天数")
    ax.set_ylabel("果实鲜重 (kg/m²)")
    ax.set_title("果实积累曲线：人工 vs PPO")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ppo_fruit.png", dpi=130)
    plt.close(fig)

    # 图3：分月舒适率
    months = ["4月", "5月", "6月", "7月"]
    bounds = [0, 30, 61, 91, 102]

    def comfort(t, rh, hour):
        day = (6 <= hour < 20)
        tlo, thi = (20, 28) if day else (16, 24)
        return (tlo <= t <= thi) and (60 <= rh <= 85)

    hc, rc = [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        lo_i, hi_i = lo * 24, min(hi * 24, 102 * 24)
        n = hi_i - lo_i
        hs = np.tile(np.arange(24), (n + 23) // 24)[:n]
        day = (hs >= 6) & (hs < 20)
        for t, rh, store in [(ht, hrh, hc), (rt, rrh, rc)]:
            t_seg, rh_seg = t[lo_i:hi_i], rh[lo_i:hi_i]
            ok_t = np.where(day, (t_seg >= 20) & (t_seg <= 28), (t_seg >= 16) & (t_seg <= 24))
            ok_rh = (rh_seg >= 60) & (rh_seg <= 85)
            store.append((ok_t & ok_rh).mean() * 100)
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, hc, 0.38, color=RED, label="人工")
    ax.bar(x + 0.2, rc, 0.38, color=BLUE, label="PPO")
    ax.set_xticks(x); ax.set_xticklabels(months)
    ax.set_ylabel("温湿联合舒适率 %")
    ax.set_title("分月舒适率：人工 vs PPO")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(OUT / "ppo_monthly_comfort.png", dpi=130)
    plt.close(fig)

    print("3 张图已生成到", OUT)


if __name__ == "__main__":
    main()
