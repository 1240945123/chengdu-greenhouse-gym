"""P0-2：off-policy 家族补强 —— TQC 与 CrossQ。

动机（见 docs/notes/15 §5 P0-2）：
  当前 off-policy 只有 SAC，且是我们的短板（SAC v3 舒适率 35.9%、最高温 41.9°C）。
  sb3-contrib 已装，两个更强的 off-policy 算法可直接用：
    - **TQC**（Truncated Quantile Critics）：SAC + 分布式截断分位数 critic，
      缓解过估计；默认带优先回放。
    - **CrossQ**（2024）：用 BatchNorm 取代目标网络，去掉 target 更新的滞后，
      显著提升样本效率。

协议一致性：环境配置与 SAC（sac_final_v3：overheat + humidity2.0 + obs8 +
screen0.5，comfort/smooth=0）、训练超参（lr 3e-4 / buffer 100k / batch 256 /
gamma 0.99 / train_freq 1 / learning_starts 1000）完全一致；仅保留算法必需项：
  TQC    → tau=0.005, top_quantiles_to_drop_per_net=2
  CrossQ → policy_delay=3（其核心机制，若去掉即失去 CrossQ 的意义）
注：CrossQ 原论文推荐 lr=1e-3，此处为协议一致性仍用 3e-4，已在 meta 中声明。

输出：results/.../rl/{tqc,crossq}/
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

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
CHECKPOINT_INTERVAL = 150_000

# 与 SAC 统一的 off-policy 协议
BASE_CONFIG = {
    "learning_rate": 3e-4,
    "buffer_size": 100_000,
    "learning_starts": 1000,
    "batch_size": 256,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
}

ALGO_CONFIG = {
    "tqc": {**BASE_CONFIG, "tau": 0.005, "top_quantiles_to_drop_per_net": 2},
    "crossq": {**BASE_CONFIG, "policy_delay": 3},
}

ALGO_CLASS = {"tqc": "TQC", "crossq": "CrossQ"}


def build_env(comfort_weight: float = 0.0, smooth_weight: float = 0.0):
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
        comfort_weight=comfort_weight,
        smooth_weight=smooth_weight,
        crop_start="seedling",
        disable_supplements=True,
    )


def build_curve_callback(out_dir: Path):
    """off-policy 曲线采集：每 2000 步采样；表头用「观测到的 key 并集」，
    以兼容 TQC / CrossQ / SAC 三者不同的 logger key。"""
    import numpy as np
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        def __init__(self, out_dir: Path, verbose: int = 0):
            super().__init__(verbose)
            self.out_dir = out_dir
            self.rows: list[dict] = []

        def _on_step(self) -> bool:
            if self.n_calls % 2000 == 0:
                row: dict = {"timestep": int(self.num_timesteps)}
                buf = getattr(self.model, "ep_info_buffer", None)
                if buf:
                    ep_rews = [e["r"] for e in buf if "r" in e]
                    if ep_rews:
                        row["rollout/ep_rew_mean"] = round(float(np.mean(ep_rews)), 4)
                for k, v in self.model.logger.name_to_value.items():
                    if k.startswith("train/") or k.startswith("rollout/"):
                        try:
                            row[k] = round(float(v), 6)
                        except (TypeError, ValueError):
                            pass
                self.rows.append(row)
            return True

        def _on_training_end(self) -> None:
            fields: list[str] = ["timestep"]
            for r in self.rows:
                for k in r:
                    if k not in fields:
                        fields.append(k)
            with open(str(self.out_dir / "training_curve.csv"), "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)

    return CurveCallback(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=sorted(ALGO_CLASS), required=True)
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--comfort-weight", type=float, default=0.0)
    ap.add_argument("--smooth-weight", type=float, default=0.0)
    args = ap.parse_args()

    out_root = Path(args.out or
                    f"results/chengdu_agri_greenhouse_001/real_greenhouse/rl/{args.algo}")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = ALGO_CONFIG[args.algo]

    import sb3_contrib

    ModelCls = getattr(sb3_contrib, ALGO_CLASS[args.algo])
    env = build_env(args.comfort_weight, args.smooth_weight)
    callback = build_curve_callback(out_root)

    model = ModelCls("MlpPolicy", env, seed=0, verbose=1,
                     tensorboard_log=str(out_root / "tb_log"), **cfg)

    print(f"\n===== {ALGO_CLASS[args.algo]} 训练：{args.timesteps} 步 =====", flush=True)
    print("超参数:", json.dumps(cfg, ensure_ascii=False), flush=True)
    print(f"环境: overheat + humidity2.0 + obs8 + screen0.5 + "
          f"comfort{args.comfort_weight} + smooth{args.smooth_weight}（与 SAC 协议一致）", flush=True)

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
        "algorithm": args.algo,
        "sb3_class": ALGO_CLASS[args.algo],
        "env_kind": "cont",
        "timesteps": args.timesteps, "seed": 0,
        "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "crop_start": "seedling", "disable_supplements": True,
        "yield_weight": 1.0, "temperature_weight": 1.0, "humidity_weight": 2.0,
        "effort_weight": 0.2, "cooling_weight": 0.5,
        "cooling_mode": "overheat", "obs_include_outdoor": True, "obs_dim": 8,
        "screen_shade_weight": 0.5,
        "comfort_weight": args.comfort_weight, "smooth_weight": args.smooth_weight,
        "hyperparameters": cfg,
        "protocol_note": "lr/buffer/batch/gamma/train_freq 与 SAC 一致；"
                         "CrossQ 原论文推荐 lr=1e-3，此处为协议一致性保留 3e-4",
    }
    (out_root / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ALGO_CLASS[args.algo]} 训练完成：{args.timesteps} 步, 用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {out_root / 'model'}", flush=True)


if __name__ == "__main__":
    main()
