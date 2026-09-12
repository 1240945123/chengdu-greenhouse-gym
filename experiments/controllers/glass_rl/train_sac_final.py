"""SAC 最终训练（修复后环境 + 优化版奖励/状态）。

与 PPO 优化版（ppo_final_v4）完全一致的奖励与状态配置，仅算法不同：
- cooling_mode=overheat（过热缓解降温奖励）
- humidity_weight=2.0（加强湿度惩罚）
- obs_include_outdoor=True（8 维 obs，含室外温度）
- 环境：修复保温幕长波反射后的 GlassGreenhouse

SAC 用连续动作空间（Box(-1,1,9)），step 时投影到 9 离散档位。
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

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/sac_final_v4")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

SAC_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
}

# SAC 记录的 train 指标 key（与 PPO 不同）
TRAIN_KEYS = ("train/actor_loss", "train/critic_loss", "train/ent_coef",
              "train/learning_rate", "train/ent_coef_loss")


def build_env():
    from experiments.controllers.glass_rl.env_variants import GlassGreenhouseEnvContinuous

    return GlassGreenhouseEnvContinuous(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=2.0,
        effort_weight=0.2,
        cooling_weight=0.5,
        cooling_mode="overheat",
        obs_include_outdoor=True,
        screen_shade_weight=0.5,
        comfort_weight=0.3,
        smooth_weight=0.1,
        crop_start="seedling",
        disable_supplements=True,
    )


def build_curve_callback(out_dir: Path):
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        def __init__(self, out_dir: Path, verbose: int = 0):
            super().__init__(verbose)
            self.out_dir = out_dir
            self.rows: list[dict] = []

        def _on_step(self) -> bool:
            # SAC 是 off-policy，无 PPO 式 rollout 边界，改用每 2000 步采样
            if self.n_calls % 2000 == 0:
                row = {"timestep": int(self.num_timesteps)}
                buf = getattr(self.model, "ep_info_buffer", None)
                if buf is not None and len(buf) > 0:
                    ep_rews = [e["r"] for e in buf if "r" in e]
                    if ep_rews:
                        row["rollout/ep_rew_mean"] = round(float(np.mean(ep_rews)), 4)
                lg = self.model.logger.name_to_value
                for k in TRAIN_KEYS:
                    if k in lg:
                        row[k] = round(float(lg[k]), 6)
                self.rows.append(row)
            return True

        def _on_training_end(self) -> None:
            with open(str(self.out_dir / "training_curve.csv"), "w",
                      newline="", encoding="utf-8") as f:
                fields = ["timestep", "rollout/ep_rew_mean"] + list(TRAIN_KEYS)
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)

    return CurveCallback(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--out", type=str, default=str(OUT_ROOT))
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 续训（路径不含 .zip）")
    parser.add_argument("--resume-steps", type=int, default=0, help="该 checkpoint 已完成的步数")
    args = parser.parse_args()
    out_root = Path(args.out)

    from stable_baselines3 import SAC

    env = build_env()
    out_root.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(out_root)

    if args.resume:
        model = SAC.load(args.resume, env=env, tensorboard_log=str(out_root / "tb_log"))
        print(f"从 {args.resume} 续训（已完成 {args.resume_steps} 步）", flush=True)
    else:
        model = SAC(
            "MlpPolicy", env,
            seed=0, verbose=1,
            tensorboard_log=str(out_root / "tb_log"),
            **SAC_CONFIG,
        )

    print(f"\n===== SAC 最终训练：{args.timesteps} 步（修复后环境）=====", flush=True)
    print("超参数:", json.dumps(SAC_CONFIG, ensure_ascii=False), flush=True)
    t0 = time.time()
    done = int(args.resume_steps)
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
        "algorithm": "sac", "sb3_class": "SAC", "env_kind": "cont",
        "timesteps": args.timesteps, "seed": 0,
        "resumed_from": args.resume, "resume_steps": args.resume_steps,
        "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "yield_weight": 1.0, "temperature_weight": 1.0, "humidity_weight": 2.0,
        "effort_weight": 0.2, "cooling_weight": 0.5,
        "cooling_mode": "overheat", "obs_include_outdoor": True, "obs_dim": 8,
        "comfort_weight": 0.3, "smooth_weight": 0.1,
        "hyperparameters": SAC_CONFIG,
    }
    (out_root / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSAC 训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {out_root / 'model'}", flush=True)


if __name__ == "__main__":
    main()
