"""玻璃温室全算法 RL 训练（完整生长季 + 优化 reward）。

口径（102 天完整生长季 + 优化）：
- 作物起点 seedling（4/1 定植苗，发育积温从 0 起算）
- 禁补光/CO2（disable_supplements）
- 训练场景 day [0,20,40,60,80] × 40 天（覆盖 4-7 月全季，含 7 月高温）
- reward 加 cooling_bonus（降温奖励，7 月高温有正反馈）
- 统一 60 万步、seed 0

算法 -> 环境映射：
- MultiDiscrete:  A2C / TRPO / RecurrentPPO / PPO
- Discrete(1536): DQN
- Box(-1,1,9):    SAC / DDPG / TD3

用法：
  python -m experiments.controllers.glass_rl.train_all_algorithms_fullseason \
      --algorithms a2c dqn --timesteps 600000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/algorithms_fullseason")

TRAIN_DAYS = [0, 20, 40, 60, 80]
EPISODE_DAYS = 40
TOTAL_STEPS = 600_000
CHECKPOINT_INTERVAL = 150_000

ALG_SPECS = {
    "a2c": ("multi", "A2C", dict(learning_rate=7e-4, n_steps=5, gamma=0.99, ent_coef=0.01)),
    "dqn": ("flat", "DQN", dict(learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
                                target_update_interval=1000, train_freq=4,
                                exploration_fraction=0.15, batch_size=128, gamma=0.99)),
    "sac": ("cont", "SAC", dict(learning_rate=3e-4, buffer_size=100_000, learning_starts=1000,
                                batch_size=256, tau=0.005, gamma=0.99, train_freq=1)),
    "ddpg": ("cont", "DDPG", dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=1000,
                                  batch_size=256, tau=0.005, gamma=0.99, train_freq=1)),
    "td3": ("cont", "TD3", dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=1000,
                                batch_size=256, tau=0.005, gamma=0.99, train_freq=1)),
    "trpo": ("multi", "TRPO", dict(n_steps=1024, gamma=0.99, learning_rate=1e-3)),
    "recurrent_ppo": ("multi", "RecurrentPPO", dict(n_steps=256, gamma=0.99,
                                                    learning_rate=3e-4, batch_size=64)),
    "ppo": ("multi", "PPO", dict(learning_rate=3e-4, n_steps=2048, batch_size=128,
                                 n_epochs=10, gamma=0.99)),
}


def build_env(env_kind: str):
    from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv
    from experiments.controllers.glass_rl.env_variants import (
        GlassGreenhouseEnvContinuous, GlassGreenhouseEnvFlatDiscrete)

    kwargs = dict(
        episode_days=EPISODE_DAYS,
        day_indices=TRAIN_DAYS,
        yield_weight=1.0,
        temperature_weight=1.0,
        humidity_weight=2.0,          # 统一为标定后协议（原 1.0）
        effort_weight=0.2,
        cooling_weight=0.5,
        cooling_mode="overheat",
        obs_include_outdoor=True,
        screen_shade_weight=0.5,
        crop_start="seedling",
        disable_supplements=True,
    )
    if env_kind == "multi":
        return GlassGreenhouseEnv(**kwargs)
    if env_kind == "cont":
        return GlassGreenhouseEnvContinuous(**kwargs)
    if env_kind == "flat":
        return GlassGreenhouseEnvFlatDiscrete(**kwargs)
    raise ValueError(env_kind)


def train_one(algorithm: str, timesteps: int) -> None:
    env_kind, sb3_name, hyper = ALG_SPECS[algorithm]
    out_dir = OUT_ROOT / algorithm
    out_dir.mkdir(parents=True, exist_ok=True)

    import importlib
    mod = (importlib.import_module("stable_baselines3") if sb3_name in
           ("PPO", "A2C", "DQN", "SAC", "DDPG", "TD3")
           else importlib.import_module("sb3_contrib"))
    ModelCls = getattr(mod, sb3_name)

    env = build_env(env_kind)
    policy = "MlpLstmPolicy" if sb3_name == "RecurrentPPO" else "MlpPolicy"
    kwargs = dict(hyper)
    kwargs.update(dict(verbose=1, seed=0, tensorboard_log=str(out_dir / "tb_log")))
    model = ModelCls(policy, env, **kwargs)

    # 训练曲线记录 callback：每 1000 步记录 reward/loss 到 CSV（可画收敛曲线）
    from stable_baselines3.common.callbacks import BaseCallback

    class CurveCallback(BaseCallback):
        LOG_KEYS = ("rollout/ep_rew_mean", "train/loss", "train/value_loss",
                    "train/policy_loss", "train/entropy_loss")

        def __init__(self, out_dir, verbose=0):
            super().__init__(verbose)
            self.out_dir = out_dir
            self.rows = []

        def _on_step(self):
            if self.n_calls % 1000 == 0:
                lg = self.model.logger.name_to_value
                row = {"timestep": int(self.num_timesteps)}
                for k in self.LOG_KEYS:
                    if k in lg:
                        row[k] = round(float(lg[k]), 5)
                self.rows.append(row)
            return True

        def _on_training_end(self):
            import csv
            # 用固定完整字段列表作表头（而非第一行 keys）：n_steps 大的算法（如 PPO
            # 2048）第一次梯度更新晚于第一个记录点，早期行缺 train/* 字段，用「第一行
            # keys」作表头会让后续行报错。
            with open(str(self.out_dir / "training_curve.csv"), "w", newline="", encoding="utf-8") as f:
                fields = ["timestep"] + list(self.LOG_KEYS)
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(self.rows)

    callback = CurveCallback(out_dir)

    print(f"\n===== [{algorithm}] {sb3_name} @ {env_kind} 训练 {timesteps} 步 =====", flush=True)
    t0 = time.time()
    done = 0
    while done < timesteps:
        step = min(CHECKPOINT_INTERVAL, timesteps - done)
        model.learn(total_timesteps=step, reset_num_timesteps=(done == 0),
                    progress_bar=False, callback=callback)
        done += step
        ckpt = out_dir / f"checkpoint_{done:07d}"
        model.save(ckpt)
        print(f"[{algorithm}] checkpoint {done} 已保存 ({time.time()-t0:.0f}s)", flush=True)
    model.save(out_dir / "model")
    elapsed = time.time() - t0
    meta = {
        "algorithm": algorithm, "sb3_class": sb3_name, "env_kind": env_kind,
        "timesteps": timesteps, "seed": 0, "elapsed_seconds": round(elapsed, 1),
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
        "cooling_weight": 0.5, "crop_start": "seedling",
        "hyperparameters": {k: v for k, v in hyper.items()},
    }
    (out_dir / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{algorithm}] 完成: {timesteps} 步, 用时 {elapsed:.0f}s, 模型 {out_dir / 'model'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithms", nargs="+", default=list(ALG_SPECS.keys()),
                        help="要训练的算法（默认全部）")
    parser.add_argument("--timesteps", type=int, default=TOTAL_STEPS)
    args = parser.parse_args()

    for alg in args.algorithms:
        if alg not in ALG_SPECS:
            print(f"未知算法 {alg}，可选: {list(ALG_SPECS)}")
            continue
        try:
            train_one(alg, args.timesteps)
        except Exception as e:
            print(f"\n[{alg}] 训练失败: {type(e).__name__}: {str(e)[:200]}", flush=True)
            continue


if __name__ == "__main__":
    main()
