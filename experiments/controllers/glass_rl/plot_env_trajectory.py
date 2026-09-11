"""PPO 控制下的温室环境轨迹图（温度/湿度/CO2/控制动作）。

生成 4 张图：
1. 温度轨迹（PPO vs 人工 vs 室外，逐日均温 + 日最高/最低带）
2. 湿度轨迹（PPO vs 人工 vs 室外，逐日）
3. CO2 轨迹（PPO vs 人工，逐日）
4. 控制动作热图（PPO 9 个执行器 × 102 天逐日档位）
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

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v2/figures")
OUT.mkdir(parents=True, exist_ok=True)

RED = "#c0392b"
BLUE = "#2c6fbb"
GRAY = "#888780"
ACT_NAMES = ["外遮阳", "顶保温", "四周保温", "顶窗", "补光", "CO2", "风机", "水泵", "卷膜"]


def run_traj(act_fn, days: int = 102):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(episode_days=days, start_day_index=0,
                             crop_start="seedling", disable_supplements=True,
                             cooling_weight=0.5)
    obs, _ = env.reset(seed=0)
    temps, rhs, co2s, acts, t_outs = [], [], [], [], []
    for i in range(days * 24):
        a = act_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        temps.append(info["temperature"])
        rhs.append(info["rh"])
        co2s.append(float(env.x[0]))
        acts.append(np.asarray(a, dtype=int))
        t_outs.append(float(env._weather_at(env._w_idx - 1)[1]))
    return (np.array(temps), np.array(rhs), np.array(co2s),
            np.array(acts), np.array(t_outs))


def daily(series, fn):
    return fn(series.reshape(-1, 24), axis=1)


def main():
    from stable_baselines3 import PPO
    from experiments.controllers.glass_rl.full_season_benchmark_v3 import load_human_actions, ACT_COLS

    model = PPO.load("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v2/model")
    rl_act = lambda obs: model.predict(obs, deterministic=True)[0]

    ha = load_human_actions().iloc[: 102 * 24].reset_index(drop=True)
    human_acts = [np.array([ha.iloc[i][c] for c in ACT_COLS], dtype=int) for i in range(len(ha))]
    state = {"i": 0}

    def human_act(obs):
        a = human_acts[state["i"]]
        state["i"] += 1
        return a

    print("跑 PPO 轨迹...")
    rt, rrh, rco2, racts, rtout = run_traj(rl_act)
    print("跑人工轨迹...")
    ht, hrh, hco2, hacts, htout = run_traj(human_act)

    days_axis = np.arange(102)

    # ---------- 图1：温度轨迹 ----------
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.fill_between(days_axis, daily(rt, np.min), daily(rt, np.max), color=BLUE, alpha=0.15)
    ax.fill_between(days_axis, daily(ht, np.min), daily(ht, np.max), color=RED, alpha=0.12)
    ax.plot(days_axis, daily(rt, np.mean), color=BLUE, lw=1.8, label="PPO 室内均温")
    ax.plot(days_axis, daily(ht, np.mean), color=RED, lw=1.8, label="人工 室内均温")
    ax.plot(days_axis, daily(rtout, np.mean), color=GRAY, lw=1.2, ls="--", label="室外均温")
    ax.axhspan(20, 28, color="green", alpha=0.05)
    ax.set_xlabel("生长季天数（4/1 定植起）")
    ax.set_ylabel("温度 (°C)")
    ax.set_title("温度环境轨迹：PPO vs 人工 vs 室外（102 天）")
    ax.legend(loc="upper left", ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "env_temperature.png", dpi=130)
    plt.close(fig)

    # ---------- 图2：湿度轨迹 ----------
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.fill_between(days_axis, daily(rrh, np.min), daily(rrh, np.max), color=BLUE, alpha=0.15)
    ax.fill_between(days_axis, daily(hrh, np.min), daily(hrh, np.max), color=RED, alpha=0.12)
    ax.plot(days_axis, daily(rrh, np.mean), color=BLUE, lw=1.8, label="PPO 室内湿度")
    ax.plot(days_axis, daily(hrh, np.mean), color=RED, lw=1.8, label="人工 室内湿度")
    ax.axhspan(60, 85, color="green", alpha=0.05)
    ax.set_xlabel("生长季天数（4/1 定植起）")
    ax.set_ylabel("相对湿度 (%)")
    ax.set_title("湿度环境轨迹：PPO vs 人工（102 天）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "env_humidity.png", dpi=130)
    plt.close(fig)

    # ---------- 图3：CO2 轨迹 ----------
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(days_axis, daily(rco2, np.mean), color=BLUE, lw=1.6, label="PPO 室内 CO2")
    ax.plot(days_axis, daily(hco2, np.mean), color=RED, lw=1.6, label="人工 室内 CO2")
    ax.axhline(420, color=GRAY, lw=1, ls="--", label="室外 420 ppm")
    ax.set_xlabel("生长季天数（4/1 定植起）")
    ax.set_ylabel("CO2 (ppm)")
    ax.set_title("CO2 环境轨迹：PPO vs 人工（102 天）")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "env_co2.png", dpi=130)
    plt.close(fig)

    # ---------- 图4：PPO 控制动作热图 ----------
    # 逐日平均档位（归一化到各执行器最大档位）
    max_lv = np.array([1, 1, 1, 2, 1, 1, 3, 1, 1], dtype=float)
    daily_act = np.zeros((102, 9))
    for d in range(102):
        seg = racts[d * 24:(d + 1) * 24]
        daily_act[d] = seg.mean(axis=0) / max_lv
    fig, ax = plt.subplots(figsize=(13, 5.2))
    im = ax.imshow(daily_act.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(9))
    ax.set_yticklabels(ACT_NAMES)
    ax.set_xlabel("生长季天数（4/1 定植起）")
    ax.set_title("PPO 控制动作热图（9 执行器逐日平均档位，越深=开得越大）")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("归一化档位")
    # 月份分界
    for d in [30, 61, 91]:
        ax.axvline(d - 0.5, color="white", lw=1.2, ls="--")
    ax.set_xticks([15, 45, 76, 96])
    ax.set_xticklabels(["4月", "5月", "6月", "7月"])
    fig.tight_layout()
    fig.savefig(OUT / "env_control_actions.png", dpi=130)
    plt.close(fig)

    print("已生成 4 张环境轨迹图到", OUT)


if __name__ == "__main__":
    main()
