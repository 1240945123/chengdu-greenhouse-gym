from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from glassgym.environments.utils import satVp


@dataclass(frozen=True)
class ControllerModelConfigV2:
    schema_version: str = "controller-model-v2"
    outdoor_exchange_per_hour: float = 0.25
    ventilation_exchange_per_hour: float = 1.50
    wind_exchange_per_hour_per_m_s: float = 0.03
    screen_exchange_reduction: float = 0.50
    solar_gain_per_w_m2_hour: float = 0.0010
    lamp_gain_per_hour: float = 1.00
    boiler_gain_per_hour: float = 8.00
    vapor_exchange_per_hour: float = 0.20
    vapor_ventilation_exchange_per_hour: float = 1.50
    vapor_source_pa_per_hour: float = 20.00
    lamp_drying_pa_per_hour: float = 20.00
    boiler_drying_pa_per_hour: float = 40.00

    def __post_init__(self) -> None:
        if self.schema_version != "controller-model-v2":
            raise ValueError("Unsupported controller model schema version")
        values = [
            value for name, value in asdict(self).items() if name != "schema_version"
        ]
        if not np.all(np.isfinite(values)) or np.any(np.asarray(values) < 0.0):
            raise ValueError("Controller model coefficients must be finite and non-negative")
        if self.screen_exchange_reduction > 1.0:
            raise ValueError("screen_exchange_reduction must not exceed one")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ControllerModelV2:
    def __init__(self, config: ControllerModelConfigV2) -> None:
        self.config = config

    def predict_next(
        self,
        state: np.ndarray,
        controls: np.ndarray,
        disturbance: np.ndarray,
        *,
        dt_seconds: float,
    ) -> np.ndarray:
        climate = np.asarray(state, dtype=np.float64)
        u = np.asarray(controls, dtype=np.float64)
        d = np.asarray(disturbance, dtype=np.float64)
        if climate.shape != (2,) or u.shape != (6,) or d.ndim != 1 or len(d) < 5:
            raise ValueError("Invalid ControllerModelV2 transition dimensions")
        if not np.all(np.isfinite(climate)) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(d)):
            raise ValueError("ControllerModelV2 inputs must be finite")
        dt_hours = float(dt_seconds) / 3600.0
        if not np.isfinite(dt_hours) or dt_hours <= 0.0:
            raise ValueError("dt_seconds must be finite and positive")

        cfg = self.config
        t_air, rh_air = climate
        radiation = max(float(d[0]), 0.0)
        t_out = float(d[1])
        vp_out = max(float(d[2]), 0.0)
        wind = max(float(d[4]), 0.0)
        screen_factor = max(0.0, 1.0 - cfg.screen_exchange_reduction * float(u[2]))
        exchange = (
            cfg.outdoor_exchange_per_hour * screen_factor
            + cfg.ventilation_exchange_per_hour * float(u[3])
            + cfg.wind_exchange_per_hour_per_m_s * wind * float(u[3])
        )
        d_temperature = exchange * (t_out - t_air)
        d_temperature += cfg.solar_gain_per_w_m2_hour * radiation * (1.0 - float(u[5]))
        d_temperature += cfg.lamp_gain_per_hour * float(u[4])
        d_temperature += cfg.boiler_gain_per_hour * float(u[0])

        predicted_temperature = float(
            np.clip(t_air + dt_hours * d_temperature, -30.0, 60.0)
        )
        vp_air = float(np.clip(rh_air, 0.0, 100.0) / 100.0 * satVp(t_air))
        vapor_exchange = (
            cfg.vapor_exchange_per_hour * screen_factor
            + cfg.vapor_ventilation_exchange_per_hour * float(u[3])
        )
        d_vapor = vapor_exchange * (vp_out - vp_air)
        d_vapor += cfg.vapor_source_pa_per_hour
        d_vapor -= cfg.lamp_drying_pa_per_hour * float(u[4])
        d_vapor -= cfg.boiler_drying_pa_per_hour * float(u[0])
        predicted_vapor = max(0.0, vp_air + dt_hours * d_vapor)
        predicted_rh = float(
            np.clip(100.0 * predicted_vapor / satVp(predicted_temperature), 0.0, 100.0)
        )

        return np.array([
            predicted_temperature,
            predicted_rh,
        ], dtype=np.float64)
