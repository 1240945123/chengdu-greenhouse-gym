"""从检查点续训 off-policy 算法（TQC / CrossQ）——用于补齐被中断的训练段。

背景：CrossQ 600k 训练在第 4 段（450k→600k）被中断，仅存 checkpoint_0450000。
本脚本加载该检查点、按原协议（train_offpolicy_extra 的 BASE_CONFIG / ALGO_CONFIG）
续训剩余步数，落盘 model.zip 与 train_meta.json，使全算法矩阵预算对齐到 600k。

用法：
  python -m experiments.controllers.glass_rl.resume_offpolicy \
      --algo crossq --ckpt results/.../rl/crossq/checkpoint_0450000 --timesteps 150000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

from experiments.controllers.glass_rl.train_offpolicy_extra import (  # noqa: E402
    ALGO_CLASS, ALGO_CONFIG, build_curve_callback, build_env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=sorted(ALGO_CLASS), required=True)
    ap.add_argument("--ckpt", required=True, help="检查点路径（不含 .zip）")
    ap.add_argument("--timesteps", type=int, required=True, help="本次续训的步数")
    ap.add_argument("--warmup", type=int, default=20_000,
                    help="续训开始前纯采集的步数（预热空回放池，防止发散）")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_root = Path(args.out or
                    f"results/chengdu_agri_greenhouse_001/real_greenhouse/rl/{args.algo}")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = ALGO_CONFIG[args.algo]

    import sb3_contrib
    ModelCls = getattr(sb3_contrib, ALGO_CLASS[args.algo])
    env = build_env(0.0, 0.0)
    callback = build_curve_callback(out_root)

    ckpt = Path(args.ckpt)
    model = ModelCls.load(str(ckpt), env=env,
                          tensorboard_log=str(out_root / "tb_log"))
    start_steps = int(model.num_timesteps)
    # 关键：续训时回放池为空，而 num_timesteps 已远超 learning_starts，若立即训练
    # 会拿极少量样本做梯度更新 → CrossQ 的 BatchNorm actor 直接发散成 NaN。
    # 因此把 learning_starts 前推到「续训后先纯采集 WARMUP 步」再开始学习。
    WARMUP = int(args.warmup)
    if hasattr(model, "learning_starts"):
        model.learning_starts = start_steps + WARMUP
        print(f"已加载 {ckpt}.zip（num_timesteps={start_steps}），"
              f"learning_starts 前推至 {model.learning_starts}（先采集 {WARMUP} 步预热回放池）",
              flush=True)
    else:
        print(f"已加载 {ckpt}.zip（num_timesteps={start_steps}），续训 {args.timesteps} 步",
              flush=True)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False,
                progress_bar=False, callback=callback)
    model.save(out_root / "model")
    elapsed = time.time() - t0
    total = int(model.num_timesteps)
    meta = {
        "algorithm": args.algo, "sb3_class": ALGO_CLASS[args.algo], "env_kind": "cont",
        "timesteps": total, "resumed_from": str(ckpt), "resumed_steps": args.timesteps,
        "resume_elapsed_seconds": round(elapsed, 1), "seed": 0,
        "humidity_weight": 2.0, "cooling_weight": 0.5, "cooling_mode": "overheat",
        "obs_include_outdoor": True, "obs_dim": 8, "screen_shade_weight": 0.5,
        "crop_start": "seedling", "disable_supplements": True,
        "hyperparameters": cfg,
    }
    (out_root / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{ALGO_CLASS[args.algo]} 续训完成：累计 {total} 步, 用时 {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
