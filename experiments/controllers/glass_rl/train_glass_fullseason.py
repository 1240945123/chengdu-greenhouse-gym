"""玻璃温室全季 RL 重训（冠层过热 + 遮阳透光 bug 修复后）。

口径：
- 完整生长季：作物起点 seedling（4/1 定植苗），发育积温从 0 起算
- 禁补光/CO2（disable_supplements，真实人工 4-7 月不用）
- 训练场景 day [0, 20, 40, 60] × 40 天（覆盖 4-7 月全季天气）
- 综合最优 reward（舒适+产量-能耗）

模型修复背景见 docs/notes/04-完整生长季_冠层过热bug修复.md：
1. 冠层对流 h_air_canopy 300->5760（冠层过热 bug）
2. 外遮阳透光 p[89]=p[90] 0.01->0.25（遮光幕误当遮阳网）
3. 分配门平滑 + 半饱和点 cBufMax/2（缓冲池储备）
4. 果实分配 ×canopy_frac（营养生长优先）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, ".tmp")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/glass_ppo_fullseason")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40


def build_env(seed: int = 0):
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=1.0,
        effort_weight=0.2,
        crop_start="seedling",
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
