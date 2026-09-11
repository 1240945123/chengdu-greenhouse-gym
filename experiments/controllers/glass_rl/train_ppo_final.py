"""PPO 最终优化训练脚本（郫都作物参数标定后环境）。

训练目标
--------
在标定后的郫都玻璃温室数字孪生（GlassGreenhouse，作物参数已用 2026 郫都基地
实测生长曲线标定，误差 40%→8.2%）中，训练一个 PPO 控制器，使它在完整生长季
（4/1 定植 → 7/12 清棚，102 天）内最大化综合 reward：

    reward = 产量项 − 温度惩罚 − 湿度惩罚 − 能耗 + 降温奖励

即同时兼顾「作物产量、温湿度舒适度、能耗」三个目标，并针对成都 7 月极端高温
（室外最高 37.5°C、无制冷设备）额外引入降温奖励，鼓励前瞻式降温联动。

奖励函数设计（与 glass_env.py 一致，权重为环境默认）
------------------------------------------------------
    yield_term     = yield_weight(1.0) × 果实鲜重增量[kg/m²] × 100
                     果实鲜重增量 = ΔcFruit[mg/m²] × 1e-6 / 0.081（干物质含量换算）
    temp_pen       = temp_weight(1.0) × dist(T, [20,28]日 / [16,24]夜) / 10
    rh_pen         = humidity_weight(1.0) × dist(RH, [60,85]) / 40
    effort         = effort_weight(0.2) × (0.15·风机 + 0.10·水泵 + 0.05·遮阳
                                           + 0.20·补光 + 0.15·保温 + 0.05·顶窗)
    cooling_bonus  = cooling_weight(0.5) × max(0, 室外温 − 室内温) / 10

设计要点：
- 产量项量级主导（1 kg/m²≈100 reward），引导 RL 不牺牲生长换极端控温；
- 温度惩罚用「离舒适带的距离」而非二值，梯度连续、利于学习；
- 降温奖励解决 7 月「怎么做都是负」导致的放弃降温问题。

PPO 超参数（本脚本显式设定）
----------------------------
见下方 PPO_CONFIG，每个参数的含义与取值理由写在注释中。

训练流程
--------
1. 环境：seedling 起点（发育积温从 0 起）、禁用补光/CO2、标定参数自动生效；
2. 训练场景 day_indices [0,20,40,60,80] × 40 天（覆盖 4-7 月全季天气，含 7 月高温）；
3. 每 150k 步保存 checkpoint，最终保存 model；
4. CurveCallback 每 1000 步把 rollout/ep_rew_mean 与 train/*_loss 写入 CSV（收敛判断用）；
5. tensorboard_log 写 TensorBoard（可用 tensorboard --logdir 看 loss 曲线）。

用法
----
  python experiments/controllers/glass_rl/train_ppo_final.py --timesteps 600000
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v2")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

# ---------------------------------------------------------------------------
# PPO 关键超参数
# ---------------------------------------------------------------------------
PPO_CONFIG = {
    # 学习率：Adam 步长。3e-4 是离散控制任务的标准起点；过大易震荡、过小收敛慢。
    "learning_rate": 3e-4,
    # 每轮 rollout 收集的环境步数（1 个 episode = 40 天 × 24h = 960 步，
    # 2048 ≈ 2.1 个 episode，能覆盖跨昼夜/跨天气模式的样本）。
    "n_steps": 2048,
    # mini-batch 大小（2048 / 128 = 16 个 mini-batch，整除，梯度更新稳定）。
    "batch_size": 128,
    # 每批数据用于梯度更新的轮数（epoch 数）。10 是 PPO 经验默认。
    "n_epochs": 10,
    # 折扣因子：温室控制是长时程任务（40 天 horizon），0.99 让未来奖励仍有可观权重。
    "gamma": 0.99,
    # GAE 参数：λ→1 偏差小方差大，λ→0 偏差大方差小；0.95 是通用折中。
    "gae_lambda": 0.95,
    # 裁剪范围（PPO 核心）：限制新旧策略概率比在 [1−ε, 1+ε]，ε=0.2 是标准值。
    "clip_range": 0.2,
    # 熵系数：鼓励探索。动作空间 MultiDiscrete(1536 组合)，保留少量熵避免过早
    # 陷入局部最优（此前 PPO 用 0.0，此处小幅提升到 0.01）。
    "ent_coef": 0.01,
    # 价值函数损失权重（critic 相对 actor 的相对权重）。
    "vf_coef": 0.5,
    # 梯度裁剪阈值：防止单步梯度爆炸，稳定训练。
    "max_grad_norm": 0.5,
}


def build_env():
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=1.0,
        effort_weight=0.2,
        cooling_weight=0.5,
        crop_start="seedling",
        disable_supplements=True,
    )


def build_curve_callback(out_dir: Path):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    # SB3 PPO 实际记录的 train 指标 key（注意是 policy_gradient_loss 而非 policy_loss）
    TRAIN_KEYS = (
        "train/loss", "train/value_loss", "train/policy_gradient_loss",
        "train/entropy_loss", "train/approx_kl", "train/clip_fraction",
        "train/clip_range", "train/learning_rate", "train/explained_variance",
        "train/n_updates",
    )

    class CurveCallback(BaseCallback):
        """每个 rollout（n_steps=2048）结束记录一次完整训练指标到 CSV。

        - rollout/ep_rew_mean / ep_len_mean 从 model.ep_info_buffer 计算（该值
          只在 rollout 结束时可用，on_step 型回调采样不到）；
        - train/* 指标从 logger.name_to_value 读取（为上一轮 train 的结果，
          滞后一个 rollout，但对画曲线无影响）。
        """

        def __init__(self, out_dir: Path, verbose: int = 0):
            super().__init__(verbose)
            self.out_dir = out_dir
            self.rows: list[dict] = []

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            row = {"timestep": int(self.num_timesteps)}
            buf = getattr(self.model, "ep_info_buffer", None)
            if buf is not None and len(buf) > 0:
                ep_rews = [e["r"] for e in buf if "r" in e]
                ep_lens = [e["l"] for e in buf if "l" in e]
                if ep_rews:
                    row["rollout/ep_rew_mean"] = round(float(np.mean(ep_rews)), 4)
                if ep_lens:
                    row["rollout/ep_len_mean"] = round(float(np.mean(ep_lens)), 2)
            lg = self.model.logger.name_to_value
            for k in TRAIN_KEYS:
                if k in lg:
                    row[k] = round(float(lg[k]), 6)
            self.rows.append(row)

        def _on_training_end(self) -> None:
            with open(str(self.out_dir / "training_curve.csv"), "w",
                      newline="", encoding="utf-8") as f:
                fields = ["timestep", "rollout/ep_rew_mean", "rollout/ep_len_mean"] + list(TRAIN_KEYS)
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)

    return CurveCallback(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = build_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(OUT_ROOT)

    model = PPO(
        "MlpPolicy", env,
        seed=0, verbose=1,
        tensorboard_log=str(OUT_ROOT / "tb_log"),
        **PPO_CONFIG,
    )

    print(f"\n===== PPO 最终训练：{args.timesteps} 步 =====", flush=True)
    print("超参数:", json.dumps(PPO_CONFIG, ensure_ascii=False), flush=True)
    t0 = time.time()
    done = 0
    while done < args.timesteps:
        step = min(args.checkpoint_interval, args.timesteps - done)
        print(f"\n=== 训练段 {done} -> {done + step} ===", flush=True)
        model.learn(total_timesteps=step, reset_num_timesteps=(done == 0),
                    progress_bar=False, callback=callback)
        done += step
        ckpt = OUT_ROOT / f"checkpoint_{done:07d}"
        model.save(ckpt)
        print(f"checkpoint 已保存: {ckpt} ({time.time()-t0:.0f}s)", flush=True)

    model.save(OUT_ROOT / "model")
    elapsed = time.time() - t0
    meta = {
        "algorithm": "ppo", "sb3_class": "PPO", "env_kind": "multi",
        "timesteps": args.timesteps, "seed": 0,
        "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "cooling_weight": 0.5, "yield_weight": 1.0,
        "temperature_weight": 1.0, "humidity_weight": 1.0, "effort_weight": 0.2,
        "calibration": "郫都2026实测标定(p68/69=0.85,p137x1.7,p152x0.5,p154x1.15,p156x2)",
        "hyperparameters": PPO_CONFIG,
    }
    (OUT_ROOT / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPPO 训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {OUT_ROOT / 'model'}", flush=True)
    print(f"训练曲线: {OUT_ROOT / 'training_curve.csv'}", flush=True)


if __name__ == "__main__":
    main()
