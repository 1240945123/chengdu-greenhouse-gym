"""多智能体独立学习（Independent IPPO）迭代训练。

3 个功能智能体（cooling / pad / insulation）轮流训练：
- 每个智能体用 SB3 PPO 独立训练自己的子动作空间
- 训练某智能体时，其他智能体用「当前已学策略」（初始为全关）
- 所有智能体共享全局综合 reward（非平稳环境下的独立学习）
- 每个智能体累计训练 60 万步（默认 3 轮 × 20 万步/轮）

用法：
  python -m experiments.controllers.glass_rl.train_multiagent \
      --iterations 3 --steps-per-iter 200000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from experiments.controllers.glass_rl.multiagent_env import (
    AGENTS, AGENT_ORDER, SingleAgentView)

OUT_ROOT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/rl/multiagent")
TRAIN_DAYS = [30, 45, 60, 70]
EPISODE_DAYS = 30


def zero_policy(agent_name: str):
    dims = AGENTS[agent_name]["dims"]
    return lambda obs: np.zeros(len(dims), dtype=int)


def train(iterations: int, steps_per_iter: int) -> None:
    from stable_baselines3 import PPO

    policies: dict[str, object] = {}  # agent_name -> predict 函数

    for it in range(iterations):
        for agent_name in AGENT_ORDER:
            fixed = {}
            for other in AGENT_ORDER:
                if other == agent_name:
                    continue
                fixed[other] = policies.get(other) or zero_policy(other)

            env = SingleAgentView(
                agent_name, fixed, episode_days=EPISODE_DAYS,
                day_indices=TRAIN_DAYS, yield_weight=1.0)

            # 每轮从头训，或载入上一轮的模型继续
            out_dir = OUT_ROOT / agent_name
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path = out_dir / "model.zip"
            if model_path.exists():
                model = PPO.load(str(model_path), env=env)
            else:
                model = PPO(
                    "MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                    batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.0, verbose=0, seed=0)

            t0 = time.time()
            model.learn(total_timesteps=steps_per_iter, progress_bar=False)
            model.save(out_dir / "model")
            policies[agent_name] = (
                lambda obs, m=model: m.predict(obs, deterministic=True)[0])
            print(f"[iter {it}] {agent_name}: +{steps_per_iter} 步 "
                  f"({time.time()-t0:.0f}s), 保存 {out_dir / 'model'}",
                  flush=True)

    # 保存元数据
    meta = {
        "algorithm": "independent_ippo", "agents": list(AGENT_ORDER),
        "agent_specs": {a: {"indices": AGENTS[a]["indices"], "dims": AGENTS[a]["dims"],
                            "names": AGENTS[a]["names"]} for a in AGENT_ORDER},
        "iterations": iterations, "steps_per_iter": steps_per_iter,
        "total_steps_per_agent": iterations * steps_per_iter,
        "train_days": TRAIN_DAYS, "episode_days": EPISODE_DAYS,
    }
    (OUT_ROOT / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("训练完成，元数据已保存", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--steps-per-iter", type=int, default=200_000)
    args = parser.parse_args()
    train(args.iterations, args.steps_per_iter)
