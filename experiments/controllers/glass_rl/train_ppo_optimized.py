"""PPO 优化版训练（奖励函数重构 + 状态表示增强）。

针对基线版 ppo_final_v2（舒适率仅 23%，主因湿度超标 85-100%）的优化：

优化 1（奖励函数）：cooling_bonus 由「室外温差」改为「过热缓解」
  - 旧(outdoor)：max(0, t_out - t_in)/10，室内越低于室外奖励越多 → 诱导过度降温+湿帘加湿
  - 新(overheat)：max(0, t_in - t_hi)/10，只在温度超舒适上限(白天28/夜间24)时奖励降温
  - 温度已舒适即不再奖励，抑制"过度降温 + 湿帘 85-100% 加湿"

优化 2（奖励权重）：humidity_weight 1.0 → 2.0，加强对湿度超标(>85%)的惩罚

优化 3（状态表示）：obs 7 维 → 8 维，加入室外温度 t_out，让 PPO 前瞻式控制
  （不再只能从室内温度"反应式"推断）

目标：在不牺牲产量（基线 7.443 kg/m²）的前提下，把舒适率从 23% 提升到 40%+，
      降低湿度超标率。

用法：
  python experiments/controllers/glass_rl/train_ppo_optimized.py --timesteps 600000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v6")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

# 与基线版 ppo_final_v2 完全相同的 PPO 超参数（仅改变环境 reward/obs）
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def build_env():
    from glass_env import GlassGreenhouseEnv

    return GlassGreenhouseEnv(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=2.0,          # 优化2：湿度惩罚加倍
        effort_weight=0.2,
        cooling_weight=0.5,
        cooling_mode="overheat",      # 优化1：过热缓解（非室外温差）
        obs_include_outdoor=True,     # 优化3：obs 加入室外温度
        screen_shade_weight=0.5,      # 优化4：白天顶保温挡光惩罚
        crop_start="seedling",
        disable_supplements=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from train_ppo_final import build_curve_callback

    env = build_env()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(OUT_ROOT)

    model = PPO(
        "MlpPolicy", env,
        seed=0, verbose=1,
        tensorboard_log=str(OUT_ROOT / "tb_log"),
        **PPO_CONFIG,
    )

    print(f"\n===== PPO 优化版训练：{args.timesteps} 步 =====", flush=True)
    print("超参数:", json.dumps(PPO_CONFIG, ensure_ascii=False), flush=True)
    print("优化: cooling_mode=overheat, humidity_weight=2.0, obs=8维(含室外温度)", flush=True)
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
        "algorithm": "ppo_optimized", "sb3_class": "PPO", "env_kind": "multi",
        "timesteps": args.timesteps, "seed": 0,
        "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "yield_weight": 1.0, "temperature_weight": 1.0, "humidity_weight": 2.0,
        "effort_weight": 0.2, "cooling_weight": 0.5,
        "cooling_mode": "overheat", "obs_include_outdoor": True, "obs_dim": 8,
        "hyperparameters": PPO_CONFIG,
    }
    (OUT_ROOT / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPPO 优化版训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {OUT_ROOT / 'model'}", flush=True)


if __name__ == "__main__":
    main()
