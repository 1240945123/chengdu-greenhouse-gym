"""残差强化学习环境：action = 基线控制器档位 + RL 修正量。

动机（来自系统评估诊断）：
- MPC 靠逐步访问真模型把白天温湿控得很准（白天干<60 仅 1.9%），PPO 却双向摆动
  （白天冷 19.5% + 热 7.7% 并存）——纯 RL 难以学到精细的前瞻调节。
- 残差 RL 让 RL 只学「在既有控制器基础上的修正量」，直接继承基线的先验精度，
  通常能以更少样本达到或超过基线（本项目已有 residual_ppo 概念）。

设计：
- 基线控制器输出 9 维离散档位（如 GlassRuleBased）。
- RL 输出 Box(-1,1,9) 残差，×residual_scale 后四舍五入为整数增量（默认 ±2 档）。
- 最终档位 = clip(base_levels + delta, 0, LEVEL_DIMS-1)，仍落在合法离散动作空间。
- 基线为无学习启发式，计算开销≈0，因此训练代价与普通 RL 相当。
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .glass_env import GlassGreenhouseEnv, LEVEL_DIMS

_LEVEL_MAX = np.asarray(LEVEL_DIMS, dtype=int) - 1


class GlassGreenhouseEnvResidual(GlassGreenhouseEnv):
    """连续残差动作版：Box(-1,1,9) 叠加在基线控制器之上。"""

    def __init__(self, *, base_controller: str = "rule", residual_scale: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        from experiments.controllers.glass_rl.classical_controllers import (
            GlassRuleBased, GlassBaseline)
        base_reg = {"rule": GlassRuleBased, "baseline": GlassBaseline}
        self._base = base_reg[base_controller]()
        self.base_controller_name = base_controller
        self.residual_scale = int(residual_scale)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def _residual_to_levels(self, action) -> np.ndarray:
        r = np.asarray(action, dtype=float).reshape(9)
        delta = np.round(np.clip(r, -1.0, 1.0) * self.residual_scale).astype(int)
        base_levels = np.asarray(self._base.predict_action(self), dtype=int)
        return np.clip(base_levels + delta, 0, _LEVEL_MAX)

    def step(self, action):
        return self._integrate(self._action_to_u(self._residual_to_levels(action)))


if __name__ == "__main__":
    env = GlassGreenhouseEnvResidual(episode_days=2, start_day_index=60,
                                     crop_start="early_fruiting", disable_supplements=True,
                                     base_controller="rule", residual_scale=2)
    obs, _ = env.reset(seed=0)
    print("obs shape:", obs.shape, "action_space:", env.action_space)
    for a in [np.zeros(9), np.ones(9), -np.ones(9)]:
        o, r, te, tr, info = env.step(a)
        print(f"residual={a[0]:+.0f}: t={info['temperature']:.1f} rh={info['rh']:.0f} r={r:.3f}")
    print("残差环境 OK")
