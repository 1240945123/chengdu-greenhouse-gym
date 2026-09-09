from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glassgym.core.types import StepContext
from glassgym.environments.utils import co2dens2ppm, satVp


@dataclass
class PIDLoop:
    kp: float
    ki: float
    kd: float
    output_min: float = 0.0
    output_max: float = 1.0
    integral_min: float = -10.0
    integral_max: float = 10.0

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute(self, setpoint: float, measurement: float, dt: float) -> float:
        dt = max(float(dt), 1e-9)
        error = float(setpoint) - float(measurement)
        self.integral = float(np.clip(
            self.integral + error * dt,
            self.integral_min,
            self.integral_max,
        ))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, self.output_min, self.output_max))


class PIDController:
    def __init__(
        self,
        temp_day_setpoint: float,
        temp_night_setpoint: float,
        temp_deadzone: float,
        co2_day_setpoint: float,
        rh_max: float,
        heat_pid: dict,
        vent_temp_pid: dict,
        co2_pid: dict,
        vent_rh_pid: dict,
        lamps_on: float,
        lamps_off: float,
        lamps_off_sun: float,
        lamp_rad_sum_limit: float,
        thscr_temp_threshold_day: float,
        thscr_temp_threshold_night: float,
        thscr_value: float,
        use_bl_scr: float,
    ):
        self.temp_day_setpoint = temp_day_setpoint
        self.temp_night_setpoint = temp_night_setpoint
        self.temp_deadzone = temp_deadzone
        self.co2_day_setpoint = co2_day_setpoint
        self.rh_max = rh_max
        self.lamps_on = lamps_on
        self.lamps_off = lamps_off
        self.lamps_off_sun = lamps_off_sun
        self.lamp_rad_sum_limit = lamp_rad_sum_limit
        self.thscr_temp_threshold_day = thscr_temp_threshold_day
        self.thscr_temp_threshold_night = thscr_temp_threshold_night
        self.thscr_value = thscr_value
        self.use_bl_scr = use_bl_scr

        self.heat_loop = PIDLoop(**heat_pid)
        self.vent_temp_loop = PIDLoop(**vent_temp_pid)
        self.co2_loop = PIDLoop(**co2_pid)
        self.vent_rh_loop = PIDLoop(**vent_rh_pid)

    def reset(self):
        self.heat_loop.reset()
        self.vent_temp_loop.reset()
        self.co2_loop.reset()
        self.vent_rh_loop.reset()

    def predict(self, ctx: StepContext):
        d = ctx.d[ctx.t]
        t_air = float(ctx.x[2])
        co2_ppm = float(co2dens2ppm(ctx.x[2], 1e-6 * ctx.x[0]))
        rh = float(100.0 * ctx.x[15] / satVp(ctx.x[2]))
        dt_hours = float(ctx.dt) / 3600.0

        is_day_inside = max(float(d[8]), self._lamp_time_active(ctx.hour_of_day))
        temp_setpoint = (
            is_day_inside * self.temp_day_setpoint
            + (1.0 - is_day_inside) * self.temp_night_setpoint
        )

        u = np.zeros(6, dtype=np.float32)
        u[0] = self.heat_loop.compute(temp_setpoint, t_air, dt_hours)
        u[1] = self._co2_control(co2_ppm, is_day_inside, dt_hours)

        vent_temp = self.vent_temp_loop.compute(
            measurement=temp_setpoint + self.temp_deadzone,
            setpoint=t_air,
            dt=dt_hours,
        )
        vent_rh = self.vent_rh_loop.compute(
            measurement=self.rh_max,
            setpoint=rh,
            dt=dt_hours,
        )
        u[3] = max(vent_temp, vent_rh)

        u[4] = self._lamp_control(ctx, d, t_air, rh)
        u[2] = self._thermal_screen_control(d)
        u[5] = self.use_bl_scr * (1.0 - float(d[9])) * u[4]
        return np.clip(u, 0.0, 1.0).astype(np.float32)

    def _co2_control(self, co2_ppm: float, is_day_inside: float, dt_hours: float) -> float:
        if is_day_inside <= 0.0:
            return 0.0
        return self.co2_loop.compute(self.co2_day_setpoint, co2_ppm, dt_hours)

    def _lamp_time_active(self, hour_of_day: float) -> float:
        if self.lamps_on == self.lamps_off:
            return 0.0
        if self.lamps_on < self.lamps_off:
            return float(self.lamps_on < hour_of_day < self.lamps_off)
        return float(hour_of_day > self.lamps_on or hour_of_day < self.lamps_off)

    def _lamp_control(self, ctx: StepContext, d: np.ndarray, t_air: float, rh: float) -> float:
        time_ok = self._lamp_time_active(ctx.hour_of_day)
        radiation_ok = float(d[0] < self.lamps_off_sun)
        dli_ok = float(d[7] < self.lamp_rad_sum_limit)
        temp_ok = float(t_air < self.temp_day_setpoint + self.temp_deadzone)
        rh_ok = float(rh < self.rh_max + 10.0)
        return float(time_ok * radiation_ok * dli_ok * temp_ok * rh_ok)

    def _thermal_screen_control(self, d: np.ndarray) -> float:
        is_day = float(d[8])
        threshold = (
            is_day * self.thscr_temp_threshold_day
            + (1.0 - is_day) * self.thscr_temp_threshold_night
        )
        return float(self.thscr_value if d[1] < threshold else 0.0)
