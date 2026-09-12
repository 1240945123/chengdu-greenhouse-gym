"""P0-4：H 步前瞻搜索（H-step receding-horizon search）。

背景（见 docs/notes/15 §0）：
  现有 `GlassMPC` 实为 **H=1 一步前瞻枚举搜索**：96~384 个候选各预测一步，
  按「产量增益 − 温湿惩罚 − 能耗」取最优。它有两个可改进点：
    ① 只看一步 —— 无法评估"现在开湿帘，几小时后的湿度代价"
    ② 评分函数是自定义的（humidity_weight=1.0，无 cooling_bonus / screen_pen /
       comfort_bonus），与环境 reward 不同源 → 与 RL 的对比不够公平

本模块给出统一实现：**目标函数 = 环境 reward 本身**（同源），把前瞻步数 H 作为唯一
自变量，从而画出「H → 性能 → 决策延迟」的成本-性能前沿曲线。

  H = 1，mode='enum' —— 完整枚举（96~384 候选，各推演 1 步）
  H >= 2，mode='cem' —— 交叉熵方法（CEM）：对 H×9 维动作序列的每个分量维护一个分类
                        分布，采样 N 条序列、按累计 reward 选精英、更新分布，K 轮后取
                        分布的逐维 argmax 作为首个动作（滚动时域只执行第一步）。

  等算力约束：令 N = compute_budget / H，则每轮推演 N×H ≈ 常数 → 不同 H 的延迟可比，
  于是 H 的收益纯粹来自"看得更远"，而不是"算得更多"。

  7 个可变状态字段需回滚（读 glass_env._integrate 确认）：
    x, _w_idx, _step_in_episode, _hour, _prev_fruit, _last_t_out, _prev_u

用法：
  from experiments.controllers.glass_rl.horizon_search import GlassHorizonSearch
  ctrl = GlassHorizonSearch(horizon=4, compute_budget=192, iters=2)
  a = ctrl.predict_action(env)
"""
from __future__ import annotations

import itertools

import numpy as np

N_VEC = (2, 2, 2, 3, 2, 2, 4, 2, 2)  # 9 个执行器档位数
N_DIM = len(N_VEC)

_STATE_FIELDS = ("x", "_w_idx", "_step_in_episode", "_hour",
                 "_prev_fruit", "_last_t_out", "_prev_u")


class GlassHorizonSearch:
    """H 步前瞻搜索控制器（H=1 枚举 / H>=2 CEM 采样）。

    Args:
        horizon: 前瞻步数 H。
        iters: CEM 迭代轮数 K。
        elite_frac: 精英比例。
        seed: 随机种子（可复现）。
        compute_budget: 每轮推演预算，用于自动定 N = max(24, budget // H)，使
            N×H 近似恒定（等算力前沿）。若为 None 则用 population。
        population: 显式指定 N（compute_budget 为 None 时生效）。
    """

    def __init__(self, *, horizon: int = 1, iters: int = 2, elite_frac: float = 0.25,
                 seed: int = 0, compute_budget: int | None = 192,
                 population: int | None = None):
        self.horizon = max(1, int(horizon))
        self.iters = max(1, int(iters))
        self.elite_frac = float(elite_frac)
        self.rng = np.random.default_rng(seed)
        if compute_budget is not None:
            self.population = max(24, int(compute_budget) // self.horizon)
        else:
            self.population = max(8, int(population or 48))
        self.mode = "enum" if self.horizon == 1 else "cem"
        self._n_rollouts = 0  # 上次决策消耗的模型推演步数（算力指标）

    # ------------------------------------------------------------ 状态回滚
    @staticmethod
    def _save(env) -> dict:
        snap = {}
        for f in _STATE_FIELDS:
            v = getattr(env, f)
            snap[f] = v.copy() if isinstance(v, np.ndarray) else v
        return snap

    @staticmethod
    def _restore(env, snap: dict) -> None:
        for f, v in snap.items():
            setattr(env, f, v)

    # ------------------------------------------------------------ 内推演
    def _rollout(self, env, seq) -> float:
        """在环境上推演 len(seq) 步，返回累计 reward，随后回滚状态。

        seq: 形状 (H, 9) 的档位序列。
        """
        snap = self._save(env)
        total = 0.0
        try:
            for a in np.asarray(seq):
                _, r, term, _, _ = env.step(a)
                self._n_rollouts += 1
                total += float(r)
                if term:
                    break
        finally:
            self._restore(env, snap)
        return total

    # ------------------------------------------------------------ H=1 枚举
    def _build_enum_candidates(self, env) -> list[np.ndarray]:
        t = float(env.x[2])
        hour = env._hour
        cold = t < 16.0 or (t < 19.0 and not (6.0 <= hour < 20.0))
        insulations = ((0, 0), (1, 1)) if cold else ((0, 0),)
        cands = []
        for s, rv, f, p, cur in itertools.product((0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1), (0, 1)):
            for ti, si in insulations:
                a = np.zeros(N_DIM, dtype=int)
                a[0] = s
                a[3] = rv
                a[6] = f
                a[7] = p
                a[8] = cur
                a[1] = ti
                a[2] = si
                cands.append(a)
        return cands

    # ------------------------------------------------------------ CEM 搜索
    def _cem(self, env) -> np.ndarray:
        H, N = self.horizon, self.population
        # probs[k][j] = 第 k 步、第 j 个执行器的档位分布
        probs = [[np.full(n, 1.0 / n) for n in N_VEC] for _ in range(H)]
        for _ in range(self.iters):
            # 采样 N 条长度 H 的序列
            seqs = np.empty((N, H, N_DIM), dtype=int)
            for k in range(H):
                for j, n in enumerate(N_VEC):
                    seqs[:, k, j] = self.rng.choice(n, size=N, p=probs[k][j])
            returns = np.array([self._rollout(env, s) for s in seqs])
            n_elite = max(4, int(N * self.elite_frac))
            elite = seqs[np.argsort(-returns)[:n_elite]]
            for k in range(H):
                for j, n in enumerate(N_VEC):
                    counts = np.bincount(elite[:, k, j], minlength=n).astype(float)
                    probs[k][j] = (counts + 1.0) / (counts.sum() + n)  # 拉普拉斯平滑
        # 滚动时域：只取最终分布下第 0 步的逐维 argmax
        return np.array([int(np.argmax(probs[0][j])) for j in range(N_DIM)], dtype=int)

    # ------------------------------------------------------------ 对外接口
    def predict_action(self, env) -> np.ndarray:
        self._n_rollouts = 0
        if self.mode == "enum":
            best, best_score = np.zeros(N_DIM, dtype=int), -np.inf
            for a in self._build_enum_candidates(env):
                s = self._rollout(env, [a])
                if s > best_score:
                    best_score, best = s, a
            return best
        return self._cem(env)

    def rollout_count(self) -> int:
        """上一次决策消耗的模型推演步数（算力指标）。"""
        return self._n_rollouts


def _smoke() -> None:
    import sys
    import time

    sys.path.insert(0, ".")
    sys.path.insert(0, "experiments/controllers/glass_rl")
    from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv

    def mk():
        e = GlassGreenhouseEnv(episode_days=1, start_day_index=60, crop_start="seedling",
                               disable_supplements=True, cooling_mode="overheat",
                               obs_include_outdoor=True, humidity_weight=2.0,
                               screen_shade_weight=0.5)
        e.reset(seed=0)
        return e

    for H in (1, 2, 4, 8):
        ctrl = GlassHorizonSearch(horizon=H, compute_budget=192, iters=2)
        env = mk()
        t0 = time.perf_counter()
        a = ctrl.predict_action(env)
        dt = time.perf_counter() - t0
        n1 = ctrl.rollout_count()
        before = (float(env.x[2]), env._w_idx, env._hour, env._step_in_episode,
                  None if env._prev_u is None else env._prev_u.tolist())
        _ = ctrl.predict_action(env)
        after = (float(env.x[2]), env._w_idx, env._hour, env._step_in_episode,
                 None if env._prev_u is None else env._prev_u.tolist())
        print(f"  H={H} mode={ctrl.mode} N={ctrl.population} K={ctrl.iters} | "
              f"{dt*1000:7.1f} ms | 推演 {n1:4d} 步 | 动作 {list(a)} | "
              f"回滚{'OK' if before == after else 'FAIL'}")


if __name__ == "__main__":
    print("=== H 步前瞻搜索 冒烟 ===")
    _smoke()
