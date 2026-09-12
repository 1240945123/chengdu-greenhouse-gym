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


MASKED_DIMS_DEFAULT = (4, 5)  # 补光、CO₂（disable_supplements 下无效）


def supplement_action_mask(disabled_dims: tuple[int, ...] = MASKED_DIMS_DEFAULT,
                           nvec: tuple[int, ...] = _LEVEL_DIMS) -> np.ndarray:
    """构造静态动作掩码（展平的布尔向量，MaskablePPO 约定）。

    屏蔽 `disabled_dims` 指定通道的除"档位 0"以外的所有档位。
    提取为模块级函数，便于**评估时直接传给 `model.predict(action_masks=...)`**，
    无需再包一层 gym Wrapper（Wrapper 不透传自定义属性，取不到 env.x）。
    """
    nvec = np.asarray(nvec, dtype=int)
    offsets = np.concatenate([[0], np.cumsum(nvec)])[:-1]
    mask = np.ones(int(nvec.sum()), dtype=bool)
    for dim in disabled_dims:
        off = int(offsets[dim])
        mask[off + 1: off + int(nvec[dim])] = False
    return mask


def effective_combinations(disabled_dims: tuple[int, ...] = MASKED_DIMS_DEFAULT,
                           nvec: tuple[int, ...] = _LEVEL_DIMS) -> int:
    """掩码后的有效动作组合数（默认 2·2·2·3·1·1·4·2·2 = 384）。"""
    nvec = np.asarray(nvec, dtype=int)
    dims = [int(nvec[j]) if j not in disabled_dims else 1 for j in range(len(nvec))]
    return int(np.prod(dims))


class SupplementMaskEnv(gym.ActionWrapper):
    """动作掩码包装：屏蔽无效执行器通道，供 MaskablePPO 使用。

    `disable_supplements=True` 时，补光（u[4]）与 CO₂（u[5]）在环境里被强制置 0，
    但动作空间仍保留 2×2 个组合位 → 策略会白白把探索预算花在"无效但仍被采样"的
    档位上（1536 组合中有 1152 个是重复/无效的）。

    本包装提供 `action_masks()`（MaskablePPO 约定的接口），返回展平后的布尔掩码：
    补光与 CO₂ 两个通道只允许档位 0，其余通道全允许。
    有效组合数 2×2×2×3×1×1×4×2×2 = **384**。

    掩码是静态的（不随状态变化），因此实现极简；若日后加入状态相关的物理约束
    （如"湿帘需三件套齐备"），只需把 `_mask` 改成每次按状态重算。
    """

    def __init__(self, env: GlassGreenhouseEnv, disabled_dims: tuple[int, ...] = (4, 5)):
        super().__init__(env)
        self._mask = supplement_action_mask(disabled_dims)
        self.nvec = np.asarray(env.action_space.nvec, dtype=int)
        self.n_combinations = effective_combinations(disabled_dims, tuple(self.nvec))

    def action_masks(self) -> np.ndarray:
        return self._mask

    def action(self, action):
        return action


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
    # 掩码包装 smoke
    base = GlassGreenhouseEnv(episode_days=1, start_day_index=60,
                              crop_start="seedling", disable_supplements=True)
    masked = SupplementMaskEnv(base)
    m = masked.action_masks()
    print(f"SupplementMaskEnv: 掩码长度 {m.shape[0]}，允许 {int(m.sum())} 项，"
          f"有效组合数 {masked.n_combinations}（原 {TOTAL_COMBINATIONS}）")
    assert m.sum() == 19 and masked.n_combinations == 384
    print("掩码 OK（补光/CO₂ 通道仅允许档位 0）")
