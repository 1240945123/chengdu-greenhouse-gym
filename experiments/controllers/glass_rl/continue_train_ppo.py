"""从现有 checkpoint 继续训练 PPO（优化版），延长到 120 万步。

从 ppo_final_v3/checkpoint_0600000 继续，再训 60 万步 → 共 120 万步。
新曲线写入 continue_log/，最终模型保存为 model_1200k。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/controllers/glass_rl")

OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/ppo_final_v3")
CHECKPOINT_INTERVAL = 150_000


def main() -> None:
    from stable_baselines3 import PPO
    from train_ppo_optimized import build_env
    from train_ppo_final import build_curve_callback

    ckpt = OUT / "checkpoint_0600000"
    model = PPO.load(str(ckpt))
    env = build_env()
    model.set_env(env)

    log_dir = OUT / "continue_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    callback = build_curve_callback(log_dir)

    total = 600_000
    t0 = time.time()
    done = 0
    while done < total:
        step = min(CHECKPOINT_INTERVAL, total - done)
        print(f"\n=== 继续训练段 {done} -> {done + step}（总步数 {600000+done} -> {600000+done+step}）===", flush=True)
        model.learn(total_timesteps=step, reset_num_timesteps=False,
                    progress_bar=False, callback=callback)
        done += step
        ckpt_out = OUT / f"checkpoint_{600000+done:07d}"
        model.save(ckpt_out)
        print(f"checkpoint 已保存: {ckpt_out} ({time.time()-t0:.0f}s)", flush=True)

    model.save(OUT / "model_1200k")
    elapsed = time.time() - t0
    print(f"\n继续训练完成：额外 {total} 步，共 120 万步，用时 {elapsed:.0f}s", flush=True)
    print(f"模型: {OUT / 'model_1200k'}", flush=True)
    print(f"新曲线: {log_dir / 'training_curve.csv'}", flush=True)


if __name__ == "__main__":
    main()
