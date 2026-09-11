"""四者完整对比图集：PPO v6 / SAC v3 / 规则 / 固定基准。

一次 rollout（102 天）同时产出三组图：
A. 关键指标 + 分月舒适率（汇总层）
B. 温室状态轨迹：温度 / 湿度 / CO2
C. 控制动作：逐日档位折线（2×3）+ 执行器×策略热图网格
D. 训练过程算法参数：PPO vs SAC 的 reward / loss / 稳定性曲线
   （规则与固定基准为无学习控制器，无训练过程）

统一口径：overheat + humidity_weight=2.0 + obs 8维 + 分温度 screen_pen + 修复后环境。
SAC 的连续动作经 continuous_to_levels 投影为 9 档，保证与其余三者口径一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

RL = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl")
OUT = RL / "figures_four_way"
OUT.mkdir(parents=True, exist_ok=True)

ENV_KW = dict(episode_days=102, start_day_index=0, crop_start="seedling",
              disable_supplements=True, cooling_weight=0.5, humidity_weight=2.0,
              cooling_mode="overheat", obs_include_outdoor=True, screen_shade_weight=0.5)

STRATS = ["PPO v6", "SAC v3", "规则控制", "固定基准"]
COLORS = {"PPO v6": "#2c6fbb", "SAC v3": "#e67e22",
          "规则控制": "#27ae60", "固定基准": "#95a5a6"}
# 关键执行器（绘图用）
KEY_ACT = [("a0", "外遮阳"), ("a1", "顶保温"), ("a3", "顶窗"),
           ("a6", "风机"), ("a7", "水泵"), ("a8", "卷膜")]


def make_act(name):
    if name == "PPO v6":
        from stable_baselines3 import PPO
        m = PPO.load(str(RL / "ppo_final_v6/model"))
        return lambda e: m.predict(e._get_obs(), deterministic=True)[0]
    if name == "SAC v3":
        from stable_baselines3 import SAC
        from experiments.controllers.glass_rl.env_variants import continuous_to_levels
        m = SAC.load(str(RL / "sac_final_v3/model"))
        return lambda e: continuous_to_levels(m.predict(e._get_obs(), deterministic=True)[0])
    if name == "规则控制":
        from experiments.controllers.glass_rl.classical_controllers import GlassRuleBased
        c = GlassRuleBased()
        return lambda e: c.predict_action(e)
    from experiments.controllers.glass_rl.classical_controllers import GlassBaseline
    c = GlassBaseline()
    return lambda e: c.predict_action(e)


def rollout(name):
    from glass_env import GlassGreenhouseEnv
    env = GlassGreenhouseEnv(**ENV_KW)
    env.reset(seed=0)
    fruit0 = float(env.x[25])
    act = make_act(name)
    rows = []
    for i in range(102 * 24):
        w = env.weather.iloc[min(env._w_idx, len(env.weather) - 1)]
        out_t = float(w["temp"]); out_rh = float(w["rh"])
        a = act(env)
        lv = np.asarray(a).reshape(-1)
        obs, r, term, tr, info = env.step(lv)
        row = {"day": i // 24, "hour": i % 24, "t": info["temperature"],
               "rh": info["rh"], "co2": float(env.x[0]), "out_t": out_t, "out_rh": out_rh}
        for j in range(9):
            row[f"a{j}"] = int(lv[j])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["is_day"] = (df.hour >= 6) & (df.hour < 20)
    fruit = (float(env.x[25]) - fruit0) * 1e-6 / 0.081
    return df, fruit


def comfort(t, rh, h):
    dd = 6.0 <= h < 20.0
    tlo, thi = (20.0, 28.0) if dd else (16.0, 24.0)
    return (tlo <= t <= thi) and (60.0 <= rh <= 85.0)


def summarize(df, fruit):
    df = df.copy()
    df["month"] = pd.cut(df.day, bins=[-1, 30, 61, 91, 102], labels=["4月", "5月", "6月", "7月"])
    day, night = df[df.is_day], df[~df.is_day]
    mc = {m: np.mean([comfort(r.t, r.rh, r.hour) for r in g.itertuples()]) * 100
          for m, g in df.groupby("month", observed=True)}
    return {
        "comfort_pct": float(np.mean([comfort(r.t, r.rh, r.hour) for r in df.itertuples()]) * 100),
        "mean_temp": float(df.t.mean()), "max_temp": float(df.t.max()),
        "rh_over85_day": float((day.rh > 85).mean() * 100),
        "rh_over85_night": float((night.rh > 85).mean() * 100),
        "fruit_kg": float(fruit),
        "month_comfort": {k: float(v) for k, v in mc.items()},
    }


def smooth(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def main():
    data, dfs = {}, {}
    for s in STRATS:
        print(f"rollout {s} ...", flush=True)
        df, fruit = rollout(s)
        dfs[s] = df
        data[s] = summarize(df, fruit)
    (RL / "final_comparison_light.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    order = [s for s in STRATS]
    days = np.arange(102)

    # ================= A. 关键指标 + 分月舒适率 =================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [("comfort_pct", "舒适率 %", "越高越好"), ("fruit_kg", "果实 kg/m²", "越高越好"),
               ("max_temp", "最高温 °C", "越低越好"), ("mean_temp", "均温 °C", "接近舒适区"),
               ("rh_over85_day", "白天湿度>85% %", "越低越好"),
               ("rh_over85_night", "夜间湿度>85% %", "越低越好")]
    for ax, (k, nm, note) in zip(axes.ravel(), metrics):
        vals = [data[s][k] for s in order]
        bars = ax.bar(order, vals, color=[COLORS[s] for s in order])
        ax.set_title(f"{nm}（{note}）"); ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("四者对比：关键指标（102天，修复后环境）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "four_way_metrics.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    months = ["4月", "5月", "6月", "7月"]; x = np.arange(4); w = 0.2
    for i, s in enumerate(order):
        vals = [data[s]["month_comfort"][m] for m in months]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=s, color=COLORS[s])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(months); ax.set_ylabel("舒适率 %")
    ax.set_title("分月舒适率对比"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "four_way_month_comfort.png", dpi=130); plt.close(fig)

    # ================= B. 温室状态轨迹 =================
    def daily(col):
        return {s: dfs[s].groupby("day")[col].mean().values for s in order}

    # 温度
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(20, 28, color="#e8f5e9", alpha=0.6, label="白天舒适区 20-28°C")
    ax.plot(days, dfs[order[0]].groupby("day")["out_t"].mean().values,
            color="#888780", ls="--", lw=1.6, label="室外气温")
    for s in order:
        ax.plot(days, daily("t")[s], color=COLORS[s], lw=2, label=s)
    ax.set_xlabel("定植后天数（4/1 起）"); ax.set_ylabel("逐日均温 °C")
    ax.set_title("温室温度轨迹：PPO v6 vs SAC v3 vs 规则控制 vs 固定基准")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "four_way_state_temperature.png", dpi=130); plt.close(fig)

    # 湿度
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhspan(60, 85, color="#e8f5e9", alpha=0.6, label="舒适湿度 60-85%")
    ax.plot(days, dfs[order[0]].groupby("day")["out_rh"].mean().values,
            color="#888780", ls="--", lw=1.6, label="室外湿度")
    for s in order:
        ax.plot(days, daily("rh")[s], color=COLORS[s], lw=2, label=s)
    ax.set_xlabel("定植后天数（4/1 起）"); ax.set_ylabel("逐日均湿 %")
    ax.set_title("温室湿度轨迹：四者对比")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "four_way_state_humidity.png", dpi=130); plt.close(fig)

    # CO2
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for s in order:
        ax.plot(days, daily("co2")[s], color=COLORS[s], lw=2, label=s)
    ax.axhline(420, color="#888780", ls="--", lw=1.2, label="室外 420 ppm")
    ax.set_xlabel("定植后天数（4/1 起）"); ax.set_ylabel("逐日均 CO2 ppm")
    ax.set_title("温室 CO2 轨迹：四者对比")
    ax.legend(ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "four_way_state_co2.png", dpi=130); plt.close(fig)

    # ================= C. 控制动作 =================
    # C1 折线（2×3：每个执行器一张，4 条策略线）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, nm) in zip(axes.ravel(), KEY_ACT):
        for s in order:
            v = dfs[s].groupby("day")[col].mean().values
            ax.plot(days, v, color=COLORS[s], lw=1.8, label=s)
        ax.set_title(nm); ax.grid(alpha=0.3); ax.set_ylim(-0.05, None)
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("天数")
    fig.suptitle("控制动作逐日档位（日均）：PPO v6 vs SAC v3 vs 规则控制 vs 固定基准", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "four_way_control_lines.png", dpi=130); plt.close(fig)

    # C2 热图网格（行=执行器，列=策略）
    fig, axes = plt.subplots(len(KEY_ACT), len(order), figsize=(13, 9),
                             sharex=True, sharey="row")
    for r, (col, nm) in enumerate(KEY_ACT):
        for c, s in enumerate(order):
            v = dfs[s].groupby("day")[col].mean().values[None, :]
            ax = axes[r, c]
            im = ax.imshow(v, aspect="auto", cmap="YlOrRd", vmin=0, vmax=3,
                           extent=[0, 102, 0, 1])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(s, fontsize=10, color=COLORS[s], fontweight="bold")
            if c == 0:
                ax.set_ylabel(nm, fontsize=10)
            if r == len(KEY_ACT) - 1:
                ax.set_xlabel("天数", fontsize=8)
    fig.suptitle("控制动作热图（行=执行器，列=策略；颜色深浅=档位高低）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "four_way_control_heatmap.png", dpi=130); plt.close(fig)

    # ================= D. 训练过程算法参数 =================
    ppo_curve = pd.read_csv(RL / "ppo_final_v6/training_curve.csv")
    sac_curve = pd.read_csv(RL / "sac_final_v3/training_curve.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) reward 收敛
    ax = axes[0, 0]
    ax.plot(ppo_curve["timestep"] / 1000,
            smooth(ppo_curve["rollout/ep_rew_mean"].interpolate(), 7),
            color=COLORS["PPO v6"], lw=2, label="PPO v6")
    ax.plot(sac_curve["timestep"] / 1000,
            smooth(sac_curve["rollout/ep_rew_mean"].interpolate(), 7),
            color=COLORS["SAC v3"], lw=2, label="SAC v3")
    ax.set_xlabel("训练步数 (×1000)"); ax.set_ylabel("平均 episode reward")
    ax.set_title("(a) 收敛曲线：奖励提升过程"); ax.legend(); ax.grid(alpha=0.3)

    # (b) PPO 损失
    ax = axes[0, 1]
    for k, c in [("train/loss", "#2c6fbb"), ("train/value_loss", "#e67e22"),
                 ("train/policy_gradient_loss", "#27ae60")]:
        ax.plot(ppo_curve["timestep"] / 1000, smooth(ppo_curve[k].interpolate(), 7),
                color=c, lw=1.8, label=k.replace("train/", ""))
    ax.set_xlabel("训练步数 (×1000)"); ax.set_ylabel("损失")
    ax.set_title("(b) PPO 损失项"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) SAC 损失
    ax = axes[1, 0]
    for k, c in [("train/actor_loss", "#e67e22"), ("train/critic_loss", "#2c6fbb")]:
        ax.plot(sac_curve["timestep"] / 1000, smooth(sac_curve[k].interpolate(), 7),
                color=c, lw=1.8, label=k.replace("train/", ""))
    ax.set_xlabel("训练步数 (×1000)"); ax.set_ylabel("损失")
    ax.set_title("(c) SAC 损失项（actor / critic）"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) 探索与稳定性
    ax = axes[1, 1]
    ax.plot(ppo_curve["timestep"] / 1000, smooth(ppo_curve["train/entropy_loss"].interpolate(), 7),
            color="#2c6fbb", lw=1.8, label="PPO 熵损失")
    ax.set_xlabel("训练步数 (×1000)"); ax.set_ylabel("熵损失")
    ax2 = ax.twinx()
    ax2.plot(ppo_curve["timestep"] / 1000, smooth(ppo_curve["train/approx_kl"].interpolate(), 7),
             color="#a32d2d", lw=1.5, ls="--", label="PPO approx_kl")
    ax2.set_ylabel("approx_kl")
    ln1, lb1 = ax.get_legend_handles_labels(); ln2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="upper right")
    ax.set_title("(d) 探索（熵）与更新稳定性（KL）"); ax.grid(alpha=0.3)

    fig.suptitle("训练过程算法参数：PPO v6 vs SAC v3"
                 "（规则控制 / 固定基准为无学习控制器，无训练过程）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "four_way_training.png", dpi=130); plt.close(fig)

    print(f"\n完成，图集输出到: {OUT}")
    for p in sorted(OUT.glob("*.png")):
        print("  -", p.name)


if __name__ == "__main__":
    main()
