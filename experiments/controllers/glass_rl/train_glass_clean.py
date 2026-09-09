"""玻璃温室 RL 干净重训（CO2 bug 修复后）。

口径（方案 A/C 收口）：
- CO2 单位修复后环境（作物在 400ppm 工作，见 ode.py x_crop 转换）
- 禁补光/CO2（disable_supplements，真实人工 4-7 月不用）
- 作物起点 early_fruiting（5 月初坐果初株），与季节匹配
- 训练场景 day [30,45,60,70] × 30 天（终点不越 day100）
- 综合最优 reward（舒适+产量-能耗）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, ".tmp")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/glass_ppo_clean")

TRAIN_DAYS = [30, 45, 60, 70]
EPISODE_DAYS = 30


def build_env(seed: int = 0):
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=1.0,
        effort_weight=0.2,
        crop_start="early_fruiting",
        disable_supplements=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--checkpoint-interval", type=int, default=150_000)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = build_env()
    model = PPO(
        "MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=128,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, verbose=1, seed=0,
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = args.timesteps
    done = 0
    while done < total:
        step = min(args.checkpoint_interval, total - done)
        print(f"\n=== 训练段 {done} -> {done + step} ===")
        model.learn(total_timesteps=step, reset_num_timesteps=(done == 0), progress_bar=False)
        done += step
        ckpt = OUT_ROOT / f"checkpoint_{done:07d}"
        model.save(ckpt)
        print(f"checkpoint 已保存: {ckpt}")
    model.save(OUT_ROOT / "model")
    print(f"最终模型: {OUT_ROOT / 'model'}")


if __name__ == "__main__":
    main()
