"""玻璃温室经典控制器（离散档位动作，9 维）。

输出统一为 9 维离散档位（与 GlassGreenhouseEnv 动作空间一致）：
  [外遮阳, 顶保温, 四周保温, 顶窗, 补光, CO2, 风机, 水泵, 卷膜]

控制器列表：
- GlassBaseline    固定基准（白天顶窗 1 档，夜间全关）
- GlassRuleBased   温度分层规则（模拟人工启发式）
- GlassPID         分段 PID（温度误差 → 顶窗/风机档位 + 遮阳/水泵/保温联动）
- GlassMPC         模型预测控制（GlassGreenhouse 单步预测 + 候选搜索）
"""
from __future__ import annotations

import itertools

import numpy as np
import casadi as ca

# 动作空间顺序: [外遮阳, 顶保温, 四周保温, 顶窗, 补光, CO2, 风机, 水泵, 卷膜]
# 档位: 遮阳/保温/补光/CO2/水泵/卷膜 {0,1}，顶窗 {0,1,2}，风机 {0,1,2,3}

ROOF_VENT_LEVEL = {0: 0.0, 1: 0.5, 2: 1.0}   # 顶窗档位 -> u 连续值
FAN_LEVEL = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0}  # 风机档位 -> u 连续值


def levels_to_u(levels: np.ndarray, disable_supplements: bool = True) -> np.ndarray:
    """9 维档位 -> GlassGreenhouse 10 维连续控制 u。"""
    a = np.asarray(levels, dtype=int)
    u = np.zeros(10)
    u[0] = 0.0
    u[1] = 0.0 if disable_supplements else float(a[5])
    u[2] = float(a[1])
    u[3] = ROOF_VENT_LEVEL[int(a[3])]
    u[4] = 0.0 if disable_supplements else float(a[4])
    u[5] = float(a[0])
    u[6] = FAN_LEVEL[int(a[6])]
    u[7] = float(a[7])
    u[8] = float(a[2])
    u[9] = float(a[8])
    return u


class GlassBaseline:
    """固定基准：白天顶窗 1 档，夜间全关（最低干预）。"""

    def predict_action(self, env) -> np.ndarray:
        hour = env._hour
        a = np.zeros(9, dtype=int)
        if 6.0 <= hour < 20.0:
            a[3] = 1
        return a


class GlassRuleBased:
    """温度分层规则（人工启发式风格）。"""

    def predict_action(self, env) -> np.ndarray:
        t = float(env.x[2])
        hour = env._hour
        a = np.zeros(9, dtype=int)
        if 6.0 <= hour < 20.0:
            if t > 30.0:
                a = [1, 0, 0, 2, 0, 0, 3, 1, 1]
            elif t > 27.0:
                a = [1, 0, 0, 2, 0, 0, 2, 1, 1]
            elif t > 25.0:
                a = [1, 0, 0, 1, 0, 0, 1, 0, 0]
            elif t < 15.0:
                a = [0, 1, 1, 0, 0, 0, 0, 0, 0]
            else:
                a = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        else:
            if t > 23.0:
                a = [0, 0, 0, 1, 0, 0, 1, 0, 0]
            elif t < 19.0:
                a = [0, 1, 1, 0, 0, 0, 0, 0, 0]
        return np.asarray(a, dtype=int)


class GlassPID:
    """分段 PID：温度误差驱动顶窗/风机档位，高温联动遮阳/湿帘，低温联动保温。"""

    def __init__(
        self,
        *,
        day_setpoint: float = 26.0,
        night_setpoint: float = 18.0,
    ) -> None:
        self.day_setpoint = day_setpoint
        self.night_setpoint = night_setpoint

    def predict_action(self, env) -> np.ndarray:
        t = float(env.x[2])
        rh = float(env.x[15] / sat_vp(t) * 100.0) if False else _rh_of(env)
        hour = env._hour
        is_day = 6.0 <= hour < 20.0
        setpoint = self.day_setpoint if is_day else self.night_setpoint
        err = t - setpoint  # 正 = 过热
        a = np.zeros(9, dtype=int)

        # 遮阳：过热开
        a[0] = 1 if t > 27.0 else 0
        # 顶窗：按误差分档
        if err > 3.0:
            a[3] = 2
        elif err > 0.5:
            a[3] = 1
        else:
            a[3] = 0
        # 风机：温度分档
        if t > 29.0:
            a[6] = 3
        elif t > 27.0:
            a[6] = 2
        elif t > 25.0:
            a[6] = 1
        else:
            a[6] = 0
        # 湿帘（水泵+卷膜）：高温且不太湿时蒸发降温
        if t > 29.0 and rh < 85.0:
            a[7] = 1
            a[8] = 1
        # 保温：冷夜
        if not is_day and t < 14.0:
            a[1] = 1
            a[2] = 1
        return a


def _rh_of(env) -> float:
    from glassgym.environments.utils import vaporPres2rh
    return float(vaporPres2rh(env.x[2], env.x[15]))


def sat_vp(t):
    return 610.78 * np.exp(17.2694 * t / (t + 237.3))


class GlassMPC:
    """模型预测控制：GlassGreenhouse 单步预测 + 候选搜索。

    候选集 = 遮阳{0,1} × 顶窗{0,1,2} × 风机{0,1,2,3} × 水泵{0,1} × 卷膜{0,1}
           = 96 组合；保温{0,1}² 依据当前温度动态附加（冷时加保温候选）。
    评分 = 预测一步后的 (产量增益 - 温度惩罚 - 湿度惩罚 - 能耗)。
    """

    def __init__(self, *, yield_weight: float = 1.0, effort_weight: float = 0.2,
                 temperature_weight: float = 1.0, humidity_weight: float = 1.0):
        self.yield_weight = yield_weight
        self.effort_weight = effort_weight
        self.temperature_weight = temperature_weight
        self.humidity_weight = humidity_weight
        self.base_candidates = tuple(
            (int(s), int(rv), int(f), int(p), int(cur))
            for s, rv, f, p, cur in itertools.product(
                (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1), (0, 1))
        )
        self._fail_score = -1e12

    def _build_candidates(self, t: float, hour: float) -> list[np.ndarray]:
        cands = []
        cold = t < 16.0 or (t < 19.0 and not (6.0 <= hour < 20.0))
        insulations = ((0, 0), (1, 1)) if cold else ((0, 0),)
        for s, rv, f, p, cur in self.base_candidates:
            for ti, si in insulations:
                a = np.zeros(9, dtype=int)
                a[0] = s; a[3] = rv; a[6] = f; a[7] = p; a[8] = cur
                a[1] = ti; a[2] = si
                cands.append(a)
        return cands

    def _score(self, env, a: np.ndarray) -> float:
        u = levels_to_u(a, disable_supplements=env.disable_supplements)
        d = env._weather_at(env._w_idx)
        try:
            r = env.F(x0=ca.DM(env.x), u=ca.DM(u),
                      p=ca.vertcat(ca.DM(d), ca.DM(env.params)))
            xf = np.asarray(r["xf"]).flatten()
        except Exception:
            return self._fail_score
        if not np.isfinite(xf).all():
            return self._fail_score
        t_next = float(xf[2])
        rh_next = float(_rh_from_state(xf))
        hour = env._hour
        is_day = 6.0 <= hour < 20.0
        t_lo, t_hi = (20.0, 28.0) if is_day else (16.0, 24.0)
        t_dist = max(t_lo - t_next, 0.0) + max(t_next - t_hi, 0.0)
        rh_dist = max(60.0 - rh_next, 0.0) + max(rh_next - 85.0, 0.0)
        temp_pen = self.temperature_weight * t_dist / 10.0
        rh_pen = self.humidity_weight * rh_dist / 40.0
        fruit_gain = max(0.0, float(xf[25]) - float(env.x[25]))
        yield_term = self.yield_weight * fruit_gain * 1e-6 / 0.081 * 100.0
        effort = self.effort_weight * (
            0.15 * u[6] + 0.10 * u[7] + 0.05 * u[5] + 0.20 * u[4]
            + 0.15 * u[1] + 0.05 * u[3]
        )
        return yield_term - temp_pen - rh_pen - effort

    def predict_action(self, env) -> np.ndarray:
        t = float(env.x[2])
        hour = env._hour
        best = np.zeros(9, dtype=int)
        best_score = -np.inf
        for a in self._build_candidates(t, hour):
            score = self._score(env, a)
            if score > best_score:
                best_score = score
                best = a
        return best


def _rh_from_state(x: np.ndarray) -> float:
    t = float(x[2])
    vp = float(x[15])
    svp = 610.78 * np.exp(17.2694 * t / (t + 237.3))
    return float(vp / svp * 100.0)


CONTROLLERS = {
    "baseline": GlassBaseline,
    "rule": GlassRuleBased,
    "pid": GlassPID,
    "mpc": GlassMPC,
}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv

    env = GlassGreenhouseEnv(episode_days=1, start_day_index=60,
                             crop_start="early_fruiting", disable_supplements=True)
    env.reset(seed=0)
    print("=== 经典控制器 24h 冒烟 ===")
    for name, cls in CONTROLLERS.items():
        ctrl = cls()
        env.reset(seed=0)
        tot = 0.0
        temps = []
        for _ in range(24):
            a = ctrl.predict_action(env)
            obs, r, term, trunc, info = env.step(a)
            tot += r
            temps.append(info["temperature"])
        print(f"{name:10s}: reward {tot:8.1f} | 均温 {np.mean(temps):5.1f}°C "
              f"max {max(temps):5.1f}°C")
