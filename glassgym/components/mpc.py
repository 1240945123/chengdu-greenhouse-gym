from __future__ import annotations

import numpy as np

from glassgym.components.controller_model import ControllerModelConfigV2, ControllerModelV2
from glassgym.core.types import ControllerStepContextV2, StepContext
from glassgym.environments.utils import co2dens2ppm, satVp


CONTROL_INDEX = {
    "uBoil": 0,
    "uCO2": 1,
    "uThScr": 2,
    "uVent": 3,
    "uLamp": 4,
    "uBlScr": 5,
}


class CandidateActionGenerator:
    def __init__(
        self,
        levels: dict[str, list[float]],
        include_current: bool = True,
        include_conservative: bool = True,
    ):
        self.levels = levels
        self.include_current = include_current
        self.include_conservative = include_conservative

    def generate(self, current_u: np.ndarray) -> np.ndarray:
        current = np.clip(np.asarray(current_u, dtype=np.float32), 0.0, 1.0)
        candidates: list[np.ndarray] = []

        if self.include_current:
            candidates.append(current.copy())

        base = current.copy()
        for name, values in self.levels.items():
            idx = CONTROL_INDEX[name]
            for value in values:
                candidate = base.copy()
                candidate[idx] = float(value)
                candidates.append(candidate)

        if self.include_conservative:
            conservative = np.zeros(6, dtype=np.float32)
            conservative[0] = 0.4
            conservative[2] = 0.5
            conservative[3] = 0.1
            candidates.append(conservative)

        unique: list[np.ndarray] = []
        seen: set[tuple[float, ...]] = set()
        for candidate in candidates:
            clipped = np.clip(candidate, 0.0, 1.0).astype(np.float32)
            key = tuple(np.round(clipped, 4))
            if key not in seen:
                seen.add(key)
                unique.append(clipped)

        return np.asarray(unique, dtype=np.float32)


class LightweightMPCController:
    def __init__(
        self,
        horizon_steps: int,
        candidate_levels: dict[str, list[float]],
        objective_weights: dict[str, float],
        temp_day_setpoint: float,
        temp_night_setpoint: float,
        co2_day_setpoint: float,
        rh_max: float,
        include_current: bool = True,
        include_conservative: bool = True,
        fallback_control: list[float] | None = None,
        max_action_change: float | list[float] | None = None,
        comfort_objective: dict[str, float] | None = None,
        controller_model: dict[str, float | str] | None = None,
    ):
        self.horizon_steps = int(horizon_steps)
        self.objective_weights = objective_weights
        self.temp_day_setpoint = temp_day_setpoint
        self.temp_night_setpoint = temp_night_setpoint
        self.co2_day_setpoint = co2_day_setpoint
        self.rh_max = rh_max
        self.max_action_change = max_action_change
        self.comfort_objective = comfort_objective
        self.generator = CandidateActionGenerator(
            levels=candidate_levels,
            include_current=include_current,
            include_conservative=include_conservative,
        )
        self.fallback_control = np.asarray(
            fallback_control if fallback_control is not None else [0.5, 0.0, 0.5, 0.1, 0.0, 0.0],
            dtype=np.float32,
        )
        self.controller_model = ControllerModelV2(
            ControllerModelConfigV2(**(controller_model or {}))
        )

    def reset(self):
        pass

    def predict(self, ctx: ControllerStepContextV2) -> np.ndarray:
        if not isinstance(ctx, ControllerStepContextV2):
            raise TypeError("LightweightMPCController requires ControllerStepContextV2")

        candidates = self.generator.generate(ctx.u)
        best_score = -np.inf
        best_u = candidates[0] if len(candidates) else self.fallback_control

        for candidate in candidates:
            score = self._rollout_score(ctx, candidate)
            if score > best_score:
                best_score = score
                best_u = candidate

        return np.clip(best_u, 0.0, 1.0).astype(np.float32)

    def _rollout_score(
        self, ctx: ControllerStepContextV2, candidate_u: np.ndarray
    ) -> float:
        climate = np.array(
            [ctx.indoor_temperature, ctx.indoor_relative_humidity],
            dtype=np.float64,
        )
        previous_u = np.asarray(ctx.u, dtype=np.float64).copy()
        total = 0.0

        for k in range(self.horizon_steps):
            d = self._forecast_disturbance(ctx, horizon_index=k)
            applied_u = self._apply_slew_rate(previous_u, candidate_u)
            try:
                climate = self.controller_model.predict_next(
                    climate,
                    applied_u,
                    d,
                    dt_seconds=ctx.dt,
                )
                if not np.all(np.isfinite(climate)):
                    return -np.inf
            except Exception:
                return -np.inf
            total += self._controller_stage_score(
                temperature=float(climate[0]),
                relative_humidity=float(climate[1]),
                u=applied_u,
                prev_u=previous_u,
                d=d,
                hour_of_day=self._prediction_hour(ctx, horizon_index=k),
            )
            previous_u = applied_u

        return float(total)

    @staticmethod
    def _forecast_disturbance(
        ctx: ControllerStepContextV2 | StepContext, horizon_index: int
    ) -> np.ndarray:
        index = int(horizon_index)
        if index < 0:
            raise ValueError("horizon_index must be non-negative")
        if ctx.forecast is None:
            if isinstance(ctx, StepContext):
                return np.asarray(ctx.d[ctx.t], dtype=np.float64)
            raise ValueError("MPC requires an issue-time forecast")
        if ctx.forecast.issue_timestep != ctx.t:
            raise ValueError("MPC forecast issue timestep does not match controller timestep")
        if index >= len(ctx.forecast.values):
            return np.asarray(ctx.forecast.values[-1], dtype=np.float64)
        return np.asarray(ctx.forecast.values[index], dtype=np.float64)

    @staticmethod
    def _prediction_hour(
        ctx: ControllerStepContextV2 | StepContext, horizon_index: int
    ) -> float:
        return (
            ctx.hour_of_day + (int(horizon_index) + 1) * ctx.dt / 3600.0
        ) % 24.0

    def _apply_slew_rate(self, previous: np.ndarray, target: np.ndarray) -> np.ndarray:
        previous_values = np.asarray(previous, dtype=np.float64)
        target_values = np.asarray(target, dtype=np.float64)
        if self.max_action_change is None:
            return np.clip(target_values, 0.0, 1.0)
        limit = np.asarray(self.max_action_change, dtype=np.float64)
        if limit.ndim == 0:
            limit = np.full(previous_values.shape, float(limit))
        if limit.shape != previous_values.shape or np.any(limit <= 0.0):
            raise ValueError("max_action_change must be positive and scalar or match controls")
        change = np.clip(target_values - previous_values, -limit, limit)
        return np.clip(previous_values + change, 0.0, 1.0)

    def _controller_stage_score(
        self,
        *,
        temperature: float,
        relative_humidity: float,
        u: np.ndarray,
        prev_u: np.ndarray,
        d: np.ndarray,
        hour_of_day: float,
    ) -> float:
        if self.comfort_objective is not None:
            objective = self.comfort_objective
            is_day = objective["day_start"] <= hour_of_day < objective["day_end"]
            temp_low = objective["temp_day_low"] if is_day else objective["temp_night_low"]
            temp_high = objective["temp_day_high"] if is_day else objective["temp_night_high"]
            controlled = np.array([2, 3, 4, 5], dtype=int)
            penalty = objective["temperature_weight"] * (
                self._distance_outside(temperature, temp_low, temp_high) / 10.0
            )
            penalty += objective["humidity_weight"] * (
                self._distance_outside(
                    relative_humidity, objective["rh_low"], objective["rh_high"]
                ) / 40.0
            )
            penalty += objective["lamp_weight"] * float(u[4])
            penalty += objective["effort_weight"] * float(np.mean(np.abs(u[controlled])))
            penalty += objective["action_change_weight"] * float(
                np.mean(np.abs(u[controlled] - prev_u[controlled]))
            )
            return -float(penalty)

        weights = self.objective_weights
        is_day = float(d[8])
        setpoint = (
            is_day * self.temp_day_setpoint
            + (1.0 - is_day) * self.temp_night_setpoint
        )
        score = -weights["temperature_error"] * abs(temperature - setpoint)
        score -= weights["rh_violation"] * max(0.0, relative_humidity - self.rh_max) / 100.0
        score -= weights["heat_cost"] * float(u[0])
        score -= weights["lamp_cost"] * float(u[4])
        score -= weights["action_change"] * float(np.mean(np.abs(u - prev_u)))
        return float(score)

    def _stage_score(
        self,
        x: np.ndarray,
        prev_x: np.ndarray,
        u: np.ndarray,
        prev_u: np.ndarray,
        d: np.ndarray,
        hour_of_day: float | None = None,
    ) -> float:
        if self.comfort_objective is not None:
            return self._comfort_stage_score(
                x=x,
                u=u,
                prev_u=prev_u,
                d=d,
                hour_of_day=hour_of_day,
            )
        weights = self.objective_weights
        t_air = float(x[2])
        co2_ppm = float(co2dens2ppm(x[2], 1e-6 * x[0]))
        rh = float(100.0 * x[15] / satVp(x[2]))
        is_day = float(d[8])
        temp_setpoint = is_day * self.temp_day_setpoint + (1.0 - is_day) * self.temp_night_setpoint
        co2_setpoint = is_day * self.co2_day_setpoint

        fruit_growth = max(0.0, float(x[25] - prev_x[25]))
        score = 0.0
        score -= weights["temperature_error"] * abs(t_air - temp_setpoint)
        score -= weights["co2_error"] * abs(co2_ppm - co2_setpoint) / 1000.0
        score -= weights["rh_violation"] * max(0.0, rh - self.rh_max) / 100.0
        score -= weights["heat_cost"] * float(u[0])
        score -= weights["co2_cost"] * float(u[1])
        score -= weights["lamp_cost"] * float(u[4])
        score -= weights["action_change"] * float(np.mean(np.abs(u - prev_u)))
        score += weights["fruit_growth"] * fruit_growth
        return float(score)

    @staticmethod
    def _distance_outside(value: float, low: float, high: float) -> float:
        return max(low - value, 0.0) + max(value - high, 0.0)

    def _comfort_stage_score(
        self,
        *,
        x: np.ndarray,
        u: np.ndarray,
        prev_u: np.ndarray,
        d: np.ndarray,
        hour_of_day: float | None,
    ) -> float:
        objective = self.comfort_objective
        t_air = float(x[2])
        rh = float(100.0 * x[15] / satVp(x[2]))
        if hour_of_day is None:
            is_day = bool(d[8])
        else:
            is_day = objective["day_start"] <= hour_of_day < objective["day_end"]
        temp_low = objective["temp_day_low"] if is_day else objective["temp_night_low"]
        temp_high = objective["temp_day_high"] if is_day else objective["temp_night_high"]
        controlled = np.array([2, 3, 4, 5], dtype=int)

        penalty = 0.0
        penalty += objective["temperature_weight"] * (
            self._distance_outside(t_air, temp_low, temp_high) / 10.0
        )
        penalty += objective["humidity_weight"] * (
            self._distance_outside(rh, objective["rh_low"], objective["rh_high"]) / 40.0
        )
        penalty += objective["lamp_weight"] * float(u[4])
        penalty += objective["effort_weight"] * float(np.mean(np.abs(u[controlled])))
        penalty += objective["action_change_weight"] * float(
            np.mean(np.abs(u[controlled] - prev_u[controlled]))
        )
        return -float(penalty)
