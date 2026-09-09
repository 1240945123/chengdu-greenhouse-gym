from __future__ import annotations

import itertools

import casadi as ca
import numpy as np

from glassgym.components.pid import PIDLoop
from glassgym.environments.utils import vaporPres2rh


CONTROL_INDICES = np.array([3, 5, 6, 7], dtype=int)


def target_to_action(env, target: np.ndarray) -> np.ndarray:
    target_values = np.asarray(target, dtype=float)
    if target_values.shape != (4,) or not np.isfinite(target_values).all():
        raise ValueError("V5 controller target must contain four finite controls")
    current = np.asarray(env.u[CONTROL_INDICES], dtype=float)
    delta = np.asarray(env.delta_u_max[CONTROL_INDICES], dtype=float)
    return np.clip((target_values - current) / delta, -1.0, 1.0).astype(np.float32)


class V5FixedBaseline:
    def __init__(self, target: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)) -> None:
        self.target = np.asarray(target, dtype=float)
        if self.target.shape != (4,) or np.any((self.target < 0.0) | (self.target > 1.0)):
            raise ValueError("Baseline target must be inside [0, 1]")

    def reset(self) -> None:
        pass

    def predict_action(self, env) -> np.ndarray:
        return target_to_action(env, self.target)


class V5PIDController:
    def __init__(self, fan_rh_lockout: float = 90.0) -> None:
        self.fan_rh_lockout = float(fan_rh_lockout)
        if not 0.0 <= self.fan_rh_lockout <= 100.0:
            raise ValueError("PID fan RH lockout must be inside [0, 100]")
        self.temperature_loop = PIDLoop(
            kp=0.25,
            ki=0.02,
            kd=0.02,
            output_min=0.0,
            output_max=1.0,
            integral_min=-20.0,
            integral_max=20.0,
        )
        self.humidity_loop = PIDLoop(
            kp=0.08,
            ki=0.005,
            kd=0.0,
            output_min=0.0,
            output_max=1.0,
            integral_min=-50.0,
            integral_max=50.0,
        )

    def reset(self) -> None:
        self.temperature_loop.reset()
        self.humidity_loop.reset()

    def predict_action(self, env) -> np.ndarray:
        temperature = float(env.x[2])
        rh = float(vaporPres2rh(env.x[2], env.x[15]))
        day = 6.0 <= float(env.hour_of_day) < 20.0
        upper_temperature = 28.0 if day else 24.0
        dt_hours = float(env.dt) / 3600.0
        temperature_cooling = self.temperature_loop.compute(
            setpoint=temperature,
            measurement=upper_temperature,
            dt=dt_hours,
        )
        humidity_ventilation = self.humidity_loop.compute(
            setpoint=rh,
            measurement=85.0,
            dt=dt_hours,
        )
        # 遮阳幕：温度 > 26°C 线性打开（26→32°C 对应 0→1），削减辐射得热
        shading = float(np.clip((temperature - 26.0) / 6.0, 0.0, 1.0))
        # 湿帘水泵：温度 > 28°C 线性打开（28→34°C 对应 0→1），蒸发降温；
        # 高湿时蒸发无效，直接关（安全层另有互锁：水泵 ≤ 风机）
        pump = 0.0
        if rh < self.fan_rh_lockout:
            pump = float(np.clip((temperature - 28.0) / 6.0, 0.0, 1.0))
        target = np.array(
            [
                max(temperature_cooling, humidity_ventilation),
                shading,
                0.0 if rh >= self.fan_rh_lockout else temperature_cooling,
                pump,
            ],
            dtype=float,
        )
        return target_to_action(env, target)


class V5HybridMPC:
    def __init__(
        self,
        horizon_steps: int = 2,
        *,
        levels: tuple[float, ...] = (0.0, 0.5, 1.0),
        temperature_weight: float = 1.0,
        humidity_weight: float = 0.25,
        effort_weight: float = 0.2,
        variation_weight: float = 0.3,
    ) -> None:
        if int(horizon_steps) < 1:
            raise ValueError("MPC horizon must be positive")
        level_values = tuple(float(value) for value in levels)
        if not level_values or any(not 0.0 <= value <= 1.0 for value in level_values):
            raise ValueError("MPC levels must be a nonempty sequence inside [0, 1]")
        weights = (
            float(temperature_weight),
            float(humidity_weight),
            float(effort_weight),
            float(variation_weight),
        )
        if not np.isfinite(weights).all() or any(value < 0.0 for value in weights):
            raise ValueError("MPC cost weights must be finite and nonnegative")
        self.horizon_steps = int(horizon_steps)
        self.levels = level_values
        (
            self.temperature_weight,
            self.humidity_weight,
            self.effort_weight,
            self.variation_weight,
        ) = weights
        self.candidates = tuple(
            np.asarray(values, dtype=float)
            # uVent/uPadFan 用细档（连续调节），uBlScr/uPadPump 用开关档
            for values in itertools.product(
                self.levels, (0.0, 1.0), self.levels, (0.0, 1.0)
            )
        )

    def reset(self) -> None:
        pass

    def predict_action(self, env) -> np.ndarray:
        forecast = env.issue_forecast()
        best_target = self.candidates[0]
        best_score = -np.inf
        for target in self.candidates:
            score = self._score_candidate(env, target, forecast)
            if score > best_score:
                best_score = score
                best_target = target
        if not np.isfinite(best_score):
            # 所有候选预测均数值失败：保持当前控制（no-change），由环境安全层兜底
            best_target = np.asarray(env.u[CONTROL_INDICES], dtype=float)
        return target_to_action(env, best_target)

    def _score_candidate(self, env, target: np.ndarray, forecast) -> float:
        state = np.asarray(env.x, dtype=float).copy()
        controls = np.asarray(env.u, dtype=float).copy()
        total = 0.0
        for step in range(self.horizon_steps):
            disturbance = np.asarray(
                forecast.values[min(step, len(forecast.values) - 1)],
                dtype=float,
            )
            proposed = controls.copy()
            delta = np.asarray(env.delta_u_max[CONTROL_INDICES], dtype=float)
            proposed[CONTROL_INDICES] = controls[CONTROL_INDICES] + np.clip(
                target - controls[CONTROL_INDICES], -delta, delta
            )
            hour = (float(env.hour_of_day) + step * float(env.dt) / 3600.0) % 24.0
            projection = env.safety_projector.project(
                proposed,
                previous=controls,
                disturbance=disturbance,
                hour_of_day=hour,
                indoor_relative_humidity=float(vaporPres2rh(state[2], state[15])),
                indoor_temperature=float(state[2]),
            )
            executed = projection.executed
            try:
                result = env.F(
                    x0=ca.DM(state),
                    u=ca.DM(executed),
                    p=ca.vertcat(ca.DM(disturbance), ca.DM(env.p)),
                )
                raw_state = result["xf"].full().flatten()
            except RuntimeError:
                # CVode 数值失败（如 CV_TOO_MUCH_WORK）：该候选的预测不可行，
                # 返回极差分数使其不被选中，交由剩余候选或环境安全层兜底。
                return -np.inf
            state, _info = env.state_postprocessor.correct(
                previous_state=state,
                raw_state=raw_state,
                controls=executed,
                disturbance=disturbance,
                dt_seconds=float(env.dt),
                hour_of_day=hour,
                day_of_year=float(env.day_of_year) + step * float(env.dt) / 86400.0,
            )
            temperature = float(state[2])
            rh = float(vaporPres2rh(state[2], state[15]))
            is_day = 6.0 <= (hour + float(env.dt) / 3600.0) % 24.0 < 20.0
            low, high = (20.0, 28.0) if is_day else (16.0, 24.0)
            temperature_error = max(low - temperature, 0.0) + max(temperature - high, 0.0)
            humidity_error = max(60.0 - rh, 0.0) + max(rh - 85.0, 0.0)
            action_change = np.mean(np.abs(executed[CONTROL_INDICES] - controls[CONTROL_INDICES]))
            effort = np.mean(np.abs(executed[CONTROL_INDICES]))
            total -= self.temperature_weight * temperature_error / 10.0
            total -= self.humidity_weight * humidity_error / 10.0
            total -= self.effort_weight * effort
            total -= self.variation_weight * action_change
            controls = executed
        return float(total)
