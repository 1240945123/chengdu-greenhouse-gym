"""P0-1：MaskablePPO + 动作掩码训练（动作组合 1536 → 384）。

动机（见 docs/notes/15 §5 P0-1）：
  `disable_supplements=True` 时补光（u[4]）与 CO₂（u[5]）在环境中被强制置 0，
  但动作空间仍保留 2×2 个组合位——策略会把探索预算浪费在无效档位上
  （1536 组合里只有 384 个是"有效不同"的动作）。MaskablePPO 在采样时把无效
  档位的 logits 掩掉，等价于告诉策略"这两位不用管"。

单变量对照设计：
  组 A（基线）= PPO，环境参数与 `ppo_final_v6` 完全一致（overheat + humidity2.0 +
    obs8 + screen0.5 分温度，comfort/smooth 均为 0），PPO 超参数一致
  组 B（实验）= **MaskablePPO，其余完全相同**，仅额外施加动作掩码
  → 两组唯一差异是掩码，可干净归因。

输出：results/.../rl/masked_ppo/
  model.zip / checkpoint_*.zip / training_curve.csv / train_meta.json / tb_log/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/masked_ppo")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

# 与 ppo_final_v6 完全相同的超参数
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


MASKED_DIMS = (4, 5)  # 补光、CO₂（disable_supplements 下无效）


def effective_combinations(masked: bool) -> int:
    """掩码前后的有效动作组合数（不依赖 env 属性透传，直接按档位维度算）。"""
    import numpy as np

    from experiments.controllers.glass_rl.env_variants import _LEVEL_DIMS

    dims = [int(d) if (not masked or j not in MASKED_DIMS) else 1
            for j, d in enumerate(_LEVEL_DIMS)]
    return int(np.prod(dims))


def build_env(masked: bool = True):
    """构建环境；masked=True 时包一层动作掩码。

    环境参数与 ppo_final_v6 一致（comfort_weight=0, smooth_weight=0）。
    """
    from glass_env import GlassGreenhouseEnv
    from experiments.controllers.glass_rl.env_variants import SupplementMaskEnv

    base = GlassGreenhouseEnv(
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
        comfort_weight=0.0,
        smooth_weight=0.0,
        crop_start="seedling",
        disable_supplements=True,
    )
    if not masked:
        return base
    from sb3_contrib.common.wrappers import ActionMasker

    return ActionMasker(SupplementMaskEnv(base), lambda env: env.action_masks())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--out", type=str, default=str(OUT_ROOT))
    ap.add_argument("--no-mask", action="store_true", help="对照组：不施加掩码（应等价于 PPO v6）")
    args = ap.parse_args()
    out_root = Path(args.out)
    masked = not args.no_mask

    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    from train_ppo_final import build_curve_callback

    env = build_env(masked)
    out_root.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(out_root)

    ModelCls = MaskablePPO if masked else PPO
    model = ModelCls(
        "MlpPolicy", env,
        seed=0, verbose=1,
        tensorboard_log=str(out_root / "tb_log"),
        **PPO_CONFIG,
    )

    n_comb = effective_combinations(masked)
    print(f"\n===== MaskablePPO 训练：{args.timesteps} 步（masked={masked}）=====", flush=True)
    print("超参数:", json.dumps(PPO_CONFIG, ensure_ascii=False), flush=True)
    print(f"动作掩码: {'开启' if masked else '关闭'} | 有效组合数 {n_comb}"
          + ("（补光/CO₂ 通道仅允许档位 0）" if masked else ""), flush=True)

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
        "algorithm": "maskable_ppo" if masked else "ppo_unmasked_control",
        "sb3_class": ModelCls.__name__,
        "env_kind": "multi_discrete",
        "action_masking": bool(masked),
        "n_combinations": n_comb,
        "masked_dims": list(MASKED_DIMS) if masked else [],
        "timesteps": args.timesteps, "seed": 0,
        "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "yield_weight": 1.0, "temperature_weight": 1.0, "humidity_weight": 2.0,
        "effort_weight": 0.2, "cooling_weight": 0.5,
        "cooling_mode": "overheat", "obs_include_outdoor": True, "obs_dim": 8,
        "screen_shade_weight": 0.5, "comfort_weight": 0.0, "smooth_weight": 0.0,
        "hyperparameters": PPO_CONFIG,
        "note": "环境与超参数对齐 ppo_final_v6，唯一差异为动作掩码",
    }
    (out_root / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {out_root / 'model'}", flush=True)


if __name__ == "__main__":
    main()
