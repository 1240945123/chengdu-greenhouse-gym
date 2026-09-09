from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glassgym.core.types import ActionSchemaV2


@dataclass(frozen=True)
class SafetyConfigV2:
    schema_version: str = "safety-projection-v2"
    normal_slew_limit: float = 0.1
    emergency_vent_slew_limit: float = 1.0
    rain_close_threshold_mm_h: float = 0.0
    wind_close_threshold_m_s: float = 10.0
    lamp_start_hour: float = 6.0
    lamp_end_hour: float = 20.0
    wet_pad_rh_lockout_percent: float = 85.0
    high_temperature_emergency_c: float = 34.0
    independent_fan_control: bool = False

    def __post_init__(self) -> None:
        numeric = np.asarray(
            [
                self.normal_slew_limit,
                self.emergency_vent_slew_limit,
                self.rain_close_threshold_mm_h,
                self.wind_close_threshold_m_s,
                self.lamp_start_hour,
                self.lamp_end_hour,
                self.wet_pad_rh_lockout_percent,
                self.high_temperature_emergency_c,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)):
            raise ValueError("SafetyConfigV2 values must be finite")
        if self.normal_slew_limit <= 0.0 or self.emergency_vent_slew_limit <= 0.0:
            raise ValueError("Safety slew limits must be positive")
        if self.rain_close_threshold_mm_h < 0.0 or self.wind_close_threshold_m_s <= 0.0:
            raise ValueError("Safety weather thresholds are invalid")
        if not 0.0 <= self.lamp_start_hour < self.lamp_end_hour <= 24.0:
            raise ValueError("Safety lamp schedule must be inside one day")
        if not 0.0 <= self.wet_pad_rh_lockout_percent <= 100.0:
            raise ValueError("Wet-pad humidity lockout must be within 0-100%")


@dataclass(frozen=True)
class SafetyProjectionResultV2:
    proposed: np.ndarray
    executed: np.ndarray
    interventions: tuple[str, ...]
    fallback_used: bool
    invalid_proposal_indices: tuple[int, ...]


class SafetyProjectorV2:
    def __init__(self, schema: ActionSchemaV2, config: SafetyConfigV2) -> None:
        self.schema = schema
        self.config = config
        self.low = np.asarray([field.low for field in schema.fields], dtype=np.float64)
        self.high = np.asarray([field.high for field in schema.fields], dtype=np.float64)

    def project(
        self,
        proposed: np.ndarray,
        *,
        previous: np.ndarray,
        disturbance: np.ndarray,
        hour_of_day: float,
        indoor_relative_humidity: float | None = None,
        indoor_temperature: float | None = None,
    ) -> SafetyProjectionResultV2:
        proposed_array = np.asarray(proposed, dtype=np.float64)
        previous_array = np.asarray(previous, dtype=np.float64)
        disturbance_array = np.asarray(disturbance, dtype=np.float64)
        expected_shape = (self.schema.count,)
        if proposed_array.shape != expected_shape or previous_array.shape != expected_shape:
            raise ValueError(f"Safety actions must have shape {expected_shape}")
        if disturbance_array.ndim != 1 or len(disturbance_array) < 11:
            raise ValueError("SafetyProjectionV2 requires 11 disturbance values")
        if not np.all(np.isfinite(previous_array)):
            raise ValueError("Previous executed action must be finite")

        interventions: list[str] = []
        invalid_proposal_indices = tuple(
            int(index) for index in np.flatnonzero(~np.isfinite(proposed_array))
        )
        fallback_used = bool(invalid_proposal_indices)
        candidate = previous_array.copy() if fallback_used else proposed_array.copy()
        logged_proposal = candidate.copy()
        if fallback_used:
            interventions.append("invalid_proposal_fallback")
        bounded = np.clip(candidate, self.low, self.high)
        if not np.array_equal(bounded, candidate):
            interventions.append("equipment_bounds")
        candidate = bounded
        hard = np.zeros(self.schema.count, dtype=bool)

        if candidate[0] != 0.0:
            interventions.append("heating_disabled")
        if candidate[1] != 0.0:
            interventions.append("co2_disabled")
        candidate[0] = 0.0
        candidate[1] = 0.0
        hard[[0, 1]] = True

        rain_active = float(disturbance_array[10]) > self.config.rain_close_threshold_mm_h
        wind_active = float(disturbance_array[4]) >= self.config.wind_close_threshold_m_s
        if rain_active or wind_active:
            candidate[3] = max(
                0.0,
                previous_array[3] - self.config.emergency_vent_slew_limit,
            )
            hard[3] = True
        if rain_active:
            interventions.append("rain_vent_closure")
        if wind_active:
            interventions.append("wind_vent_closure")

        if (
            indoor_temperature is not None
            and float(indoor_temperature) >= self.config.high_temperature_emergency_c
        ):
            candidate[2] = 0.0
            candidate[4] = 0.0
            candidate[5] = 0.0
            hard[[2, 4, 5]] = True
            if not wind_active:
                candidate[3] = 1.0
                hard[3] = True
                if rain_active:
                    interventions.append("rain_vent_override_high_temperature")
            interventions.append("high_temperature_emergency")

        if not self.config.lamp_start_hour <= float(hour_of_day) < self.config.lamp_end_hour:
            candidate[4] = 0.0
            hard[4] = True
            interventions.append("lamp_schedule")

        if self.schema.count == 8:
            if (
                indoor_relative_humidity is not None
                and float(indoor_relative_humidity)
                >= self.config.wet_pad_rh_lockout_percent
            ):
                if self.config.independent_fan_control:
                    candidate[7] = 0.0
                    hard[7] = True
                else:
                    candidate[6:8] = 0.0
                    hard[6:8] = True
                interventions.append("high_humidity_wet_pad_lockout")
            if candidate[7] > candidate[6]:
                candidate[7] = candidate[6]
                interventions.append("wet_pad_fan_pump_interlock")

        normal_indices = np.flatnonzero(~hard)
        delta = candidate[normal_indices] - previous_array[normal_indices]
        limited_delta = np.clip(
            delta,
            -self.config.normal_slew_limit,
            self.config.normal_slew_limit,
        )
        if not np.array_equal(limited_delta, delta):
            interventions.append("normal_slew_limit")
        candidate[normal_indices] = previous_array[normal_indices] + limited_delta
        candidate = np.clip(candidate, self.low, self.high)
        if self.schema.count == 8 and candidate[7] > candidate[6]:
            candidate[7] = candidate[6]
            if "wet_pad_fan_pump_interlock" not in interventions:
                interventions.append("wet_pad_fan_pump_interlock")

        return SafetyProjectionResultV2(
            proposed=logged_proposal,
            executed=candidate,
            interventions=tuple(interventions),
            fallback_used=fallback_used,
            invalid_proposal_indices=invalid_proposal_indices,
        )
