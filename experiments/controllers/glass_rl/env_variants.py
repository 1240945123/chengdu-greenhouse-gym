"""玻璃温室环境变体：为不同动作空间约束的 RL 算法提供适配。

基础环境 GlassGreenhouseEnv 的动作空间是 MultiDiscrete([2,2,2,3,2,2,4,2,2])。
不同算法的支持矩阵：
- MultiDiscrete: PPO / A2C / TRPO / RecurrentPPO（直接用基础环境）
- Box(连续):     SAC / DDPG / TD3 → GlassGreenhouseEnvContinuous
                 （连续输出投影到离散档位，符合真实执行器物理）
- Discrete(单维): DQN → GlassGreenhouseEnvFlatDiscrete
                 （1536 种档位组合枚举成单索引）

投影规则（公平：均匀随机策略下各档位等概率）：
- 2 档执行器: v <= 0 -> 0, v > 0 -> 1
- 3 档顶窗:   [-1,-1/3)->0, [-1/3,1/3)->1, [1/3,1]->2
- 4 档风机:   [-1,-1/2)->0, [-1/2,0)->1, [0,1/2)->2, [1/2,1]->3
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .glass_env import GlassGreenhouseEnv, LEVEL_DIMS

_LEVEL_DIMS = tuple(LEVEL_DIMS)


def continuous_to_levels(v: np.ndarray) -> np.ndarray:
    """连续 [-1,1]^9 -> 9 维离散档位。"""
    v = np.asarray(v, dtype=float).reshape(9)
    a = np.zeros(9, dtype=int)
    for i, dim in enumerate(_LEVEL_DIMS):
        x = float(np.clip(v[i], -1.0, 1.0))
        if dim == 2:
            a[i] = 1 if x > 0.0 else 0
        elif dim == 3:
            if x < -1.0 / 3.0:
                a[i] = 0
            elif x < 1.0 / 3.0:
                a[i] = 1
            else:
                a[i] = 2
        elif dim == 4:
            if x < -0.5:
                a[i] = 0
            elif x < 0.0:
                a[i] = 1
            elif x < 0.5:
                a[i] = 2
            else:
                a[i] = 3
        else:  # 理论上不会出现
            a[i] = int(round((x + 1.0) / 2.0 * (dim - 1)))
    return a


def encode_levels(levels: np.ndarray) -> int:
    """9 维档位 -> 混合基数单索引。"""
    idx = 0
    for v, dim in zip(np.asarray(levels, dtype=int), _LEVEL_DIMS):
        idx = idx * int(dim) + int(v)
    return idx


def decode_levels(idx: int) -> np.ndarray:
    """单索引 -> 9 维档位。"""
    levels = np.zeros(9, dtype=int)
    i = int(idx)
    for k in range(8, -1, -1):
        levels[k] = i % _LEVEL_DIMS[k]
        i //= _LEVEL_DIMS[k]
    return levels


TOTAL_COMBINATIONS = int(np.prod(_LEVEL_DIMS))  # 1536


class GlassGreenhouseEnvContinuous(GlassGreenhouseEnv):
    """连续动作版：Box(-1,1,(9,))，step 时投影到离散档位。

    供 SAC / DDPG / TD3 使用（它们只支持连续 Box 动作空间）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )

    def step(self, action):
        return self._integrate(self._action_to_u(continuous_to_levels(action)))


class GlassGreenhouseEnvFlatDiscrete(GlassGreenhouseEnv):
    """单维 Discrete 版：Discrete(1536)，索引解码到 9 维档位。

    供 DQN 使用（DQN 只支持单维 Discrete 动作空间）。
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(TOTAL_COMBINATIONS)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(action.reshape(-1)[0])
        return self._integrate(self._action_to_u(decode_levels(int(action))))


if __name__ == "__main__":
    print(f"档位维度: {_LEVEL_DIMS}, 总组合数: {TOTAL_COMBINATIONS}")
    # 编码往返测试
    for levels in [np.zeros(9, int), np.ones(9, int),
                   np.array([1, 1, 1, 2, 0, 0, 3, 1, 1]),
                   np.array([0, 0, 0, 1, 0, 0, 2, 0, 1])]:
        idx = encode_levels(levels)
        back = decode_levels(idx)
        assert np.array_equal(levels, back), f"编码失败: {levels} -> {idx} -> {back}"
    print("编码往返 OK")
    # 投影测试
    v = np.array([0.5, -0.3, 0.0, 0.8, -0.9, 0.2, 0.6, 0.0, -0.5])
    print(f"连续 {v} -> 档位 {continuous_to_levels(v)}")
    # 三个环境 smoke
    for cls in (GlassGreenhouseEnv, GlassGreenhouseEnvContinuous,
                GlassGreenhouseEnvFlatDiscrete):
        env = cls(episode_days=1, start_day_index=60, crop_start="early_fruiting",
                  disable_supplements=True)
        obs, _ = env.reset(seed=0)
        a = env.action_space.sample()
        obs2, r, term, trunc, info = env.step(a)
        print(f"{cls.__name__}: action_space={env.action_space}, "
              f"sample={np.asarray(a).shape}, step OK, r={r:.3f}, "
              f"t={info['temperature']:.1f}°C")
