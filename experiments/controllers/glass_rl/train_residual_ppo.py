"""残差 PPO 训练：RL 学习「规则控制器 + 修正量」。

配置：base_controller=rule、residual_scale=2（±2 档）、comfort_weight=0.3、smooth_weight=0。
输出：rl/residual_ppo/{model, training_curve.csv, train_meta.json, tb_log/}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/residual_ppo")
TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

PPO_CONFIG = {
    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 128, "n_epochs": 10,
    "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
    "vf_coef": 0.5, "max_grad_norm": 0.5,
}


def build_env(base_controller: str, residual_scale: int, comfort_weight: float, smooth_weight: float):
    from experiments.controllers.glass_rl.residual_env import GlassGreenhouseEnvResidual
    return GlassGreenhouseEnvResidual(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0, temperature_weight=1.0, humidity_weight=2.0, effort_weight=0.2,
        cooling_weight=0.5, cooling_mode="overheat", obs_include_outdoor=True,
        screen_shade_weight=0.5, comfort_weight=comfort_weight, smooth_weight=smooth_weight,
        crop_start="seedling", disable_supplements=True,
        base_controller=base_controller, residual_scale=residual_scale,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--out", type=str, default=str(OUT_ROOT))
    p.add_argument("--base-controller", type=str, default="rule")
    p.add_argument("--residual-scale", type=int, default=2)
    p.add_argument("--comfort-weight", type=float, default=0.3)
    p.add_argument("--smooth-weight", type=float, default=0.0)
    args = p.parse_args()
    out_root = Path(args.out)

    from stable_baselines3 import PPO
    from train_ppo_final import build_curve_callback

    env = build_env(args.base_controller, args.residual_scale, args.comfort_weight, args.smooth_weight)
    out_root.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(out_root)

    model = PPO("MlpPolicy", env, seed=0, verbose=1,
                tensorboard_log=str(out_root / "tb_log"), **PPO_CONFIG)

    print(f"\n===== 残差 PPO 训练：{args.timesteps} 步（base={args.base_controller}, "
          f"scale={args.residual_scale}）=====", flush=True)
    t0 = time.time()
    done = 0
    while done < args.timesteps:
        step = min(CHECKPOINT_INTERVAL, args.timesteps - done)
        print(f"\n=== 训练段 {done} -> {done + step} ===", flush=True)
        model.learn(total_timesteps=step, reset_num_timesteps=(done == 0),
                    progress_bar=False, callback=callback)
        done += step
        ckpt = out_root / f"checkpoint_{done:07d}"
        model.save(ckpt)
        print(f"checkpoint 已保存: {ckpt} ({time.time()-t0:.0f}s)", flush=True)

    model.save(out_root / "model")
    elapsed = time.time() - t0
    meta = {
        "algorithm": "residual_ppo", "sb3_class": "PPO", "env_kind": "residual",
        "base_controller": args.base_controller, "residual_scale": args.residual_scale,
        "timesteps": args.timesteps, "seed": 0, "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "yield_weight": 1.0, "temperature_weight": 1.0, "humidity_weight": 2.0,
        "effort_weight": 0.2, "cooling_weight": 0.5, "cooling_mode": "overheat",
        "obs_include_outdoor": True, "obs_dim": 8,
        "comfort_weight": args.comfort_weight, "smooth_weight": args.smooth_weight,
        "hyperparameters": PPO_CONFIG,
    }
    (out_root / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n残差 PPO 训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {out_root / 'model'}", flush=True)


if __name__ == "__main__":
    main()
