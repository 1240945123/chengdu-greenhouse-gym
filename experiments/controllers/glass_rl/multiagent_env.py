"""玻璃温室多智能体环境（3 个功能智能体 + 联合动作）。

智能体划分（7 个可控执行器，补光/CO2 禁用）：
- cooling    降温组：外遮阳[0] + 顶窗[3] + 风机[6]      -> MultiDiscrete([2,3,4])
- pad        湿帘组：水泵[7] + 卷膜[8]                 -> MultiDiscrete([2,2])
- insulation 保温组：顶保温[1] + 四周保温[2]           -> MultiDiscrete([2,2])

所有智能体共享全局观测（7 维：温度/湿度/CO2/冠层/时间/果实/积温）与全局综合 reward。
独立学习（IQL/IPPO）：每个智能体各自用 SB3 PPO 训练，其他智能体策略固定。
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv

# 智能体定义：执行器在 9 维档位中的索引 + 档位维度 + 名称
AGENTS = {
    "cooling":    {"indices": [0, 3, 6], "dims": [2, 3, 4],
                   "names": ["外遮阳", "顶窗", "风机"]},
    "pad":        {"indices": [7, 8], "dims": [2, 2],
                   "names": ["水泵", "卷膜"]},
    "insulation": {"indices": [1, 2], "dims": [2, 2],
                   "names": ["顶保温", "四周保温"]},
}
AGENT_ORDER = ["cooling", "pad", "insulation"]


def merge_actions(cooling: np.ndarray, pad: np.ndarray,
                  insulation: np.ndarray) -> np.ndarray:
    """3 个智能体的动作 -> 9 维档位联合动作（补光[4]/CO2[5] 固定 0）。"""
    a = np.zeros(9, dtype=int)
    for i, idx in enumerate(AGENTS["cooling"]["indices"]):
        a[idx] = int(cooling[i])
    for i, idx in enumerate(AGENTS["pad"]["indices"]):
        a[idx] = int(pad[i])
    for i, idx in enumerate(AGENTS["insulation"]["indices"]):
        a[idx] = int(insulation[i])
    return a


def split_action(agent_name: str, joint: np.ndarray) -> np.ndarray:
    """9 维档位 -> 指定智能体的子动作。"""
    idx = AGENTS[agent_name]["indices"]
    return np.asarray([joint[i] for i in idx], dtype=int)


class SingleAgentView(gym.Env):
    """多智能体环境的单智能体视角（固定其他智能体策略），供 SB3 训练。

    当前训练 `agent_name` 智能体；其他智能体用 `fixed_policies` 提供的
    predict 函数（返回各自子动作），基于共享全局观测决策。
    奖励为全局综合 reward（所有智能体共享）。
    """

    def __init__(self, agent_name: str, fixed_policies: dict,
                 episode_days: int = 30, day_indices: list[int] | None = None,
                 **env_kwargs) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.fixed_policies = fixed_policies
        self.env = GlassGreenhouseEnv(
            episode_days=episode_days, day_indices=day_indices,
            crop_start="early_fruiting", disable_supplements=True, **env_kwargs)
        self.action_space = spaces.MultiDiscrete(AGENTS[agent_name]["dims"])
        self.observation_space = self.env.observation_space

    def _joint_action(self, action: np.ndarray) -> np.ndarray:
        joint = np.zeros(9, dtype=int)
        for i, idx in enumerate(AGENTS[self.agent_name]["indices"]):
            joint[idx] = int(action[i])
        for other, policy in self.fixed_policies.items():
            a_other = policy(self.env._get_obs())
            for i, idx in enumerate(AGENTS[other]["indices"]):
                joint[idx] = int(a_other[i])
        return joint

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        self.env._prev_fruit = float(self.env.x[25])
        return obs, info

    def step(self, action):
        joint = self._joint_action(np.asarray(action, dtype=int))
        return self.env.step(joint)


if __name__ == "__main__":
    # 冒烟：3 个 agent 联合动作 + 单 agent 视图
    env = GlassGreenhouseEnv(episode_days=1, start_day_index=60,
                             crop_start="early_fruiting", disable_supplements=True)
    env.reset(seed=0)
    # 联合动作
    a = merge_actions([1, 2, 3], [1, 1], [1, 1])
    print("联合动作:", a, "(应 [1,1,1,2,0,0,3,1,1])")
    obs, r, term, trunc, info = env.step(a)
    print(f"step OK: r={r:.2f}, t={info['temperature']:.1f}°C")

    # 单 agent 视图（固定其他为规则策略）
    rule = {"pad": lambda obs: np.array([0, 0]),
            "insulation": lambda obs: np.array([0, 0])}
    view = SingleAgentView("cooling", rule, episode_days=1,
                           day_indices=[60])
    obs, _ = view.reset(seed=0)
    print(f"SingleAgentView(cooling): action_space={view.action_space}")
    a_cooling = view.action_space.sample()
    obs2, r2, term2, trunc2, info2 = view.step(a_cooling)
    print(f"  采样动作 {a_cooling} -> r={r2:.2f}, t={info2['temperature']:.1f}°C")
