from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.reports.evaluate_chengdu_physics import load_parameter_vector
from experiments.controllers.benchmark_protocol import PROJECT_ROOT, _load_environment_kwargs
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres, vaporPres2rh


DEFAULT_PARAMS_PATH = PROJECT_ROOT / "results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json"
DEFAULT_RESIDUAL_PATH = PROJECT_ROOT / "results/chengdu_agri_greenhouse_001/physics_v5_cultivation_greybox_residual/selected_model.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V5ResidualClimatePostprocessor:
    def __init__(self, residual_model: dict[str, Any]) -> None:
        if residual_model.get("schema_version") != "ridge_residual_v1":
            raise ValueError("V5 hybrid environment requires ridge_residual_v1")
        required_targets = {"air_temperature", "relative_humidity"}
        if set(residual_model.get("target_names", [])) != required_targets:
            raise ValueError("V5 residual model must correct temperature and relative humidity")
        self.residual_model = deepcopy(residual_model)
        self.feature_columns = tuple(str(value) for value in residual_model["feature_columns"])
        self.feature_means = np.asarray(residual_model["feature_means"], dtype=float)
        self.feature_scales = np.asarray(residual_model["feature_scales"], dtype=float)
        if (
            self.feature_means.shape != (len(self.feature_columns),)
            or self.feature_scales.shape != self.feature_means.shape
            or not np.isfinite(self.feature_means).all()
            or not np.isfinite(self.feature_scales).all()
            or np.any(self.feature_scales <= 0.0)
        ):
            raise ValueError("V5 residual feature statistics are invalid")
        self.coefficients = {
            target: np.asarray(residual_model["coefficients"][target], dtype=float)
            for target in residual_model["target_names"]
        }
        expected_coefficient_shape = (len(self.feature_columns) + 1,)
        if any(
            values.shape != expected_coefficient_shape or not np.isfinite(values).all()
            for values in self.coefficients.values()
        ):
            raise ValueError("V5 residual coefficients are invalid")
        self.target_gains = {
            str(target): float(value)
            for target, value in residual_model.get("target_gains", {}).items()
        }
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.target_gains.values()
        ):
            raise ValueError("Residual target gain must be between zero and one")

    def predict_corrected_climate(
        self,
        *,
        raw_temperature: float,
        raw_relative_humidity: float,
        outdoor_temperature: float,
        outdoor_relative_humidity: float,
        radiation: float,
        wind_speed: float,
        controls: np.ndarray,
        hour_of_day: float,
        applied_gain: float,
    ) -> tuple[float, float, bool]:
        u = np.asarray(controls, dtype=float)
        source_values = np.asarray(
            [
                raw_temperature,
                raw_relative_humidity,
                outdoor_temperature,
                outdoor_relative_humidity,
                radiation,
                wind_speed,
                hour_of_day,
                applied_gain,
            ],
            dtype=float,
        )
        if u.shape != (8,) or not np.isfinite(u).all() or not np.isfinite(source_values).all():
            raise ValueError("V5 residual prediction inputs must be finite")
        if not 0.0 <= float(applied_gain) <= 1.0:
            raise ValueError("Residual feedback gain must be between zero and one")

        roof = float(np.clip(u[3], 0.0, 1.0))
        fan = float(np.clip(u[6], 0.0, 1.0))
        pad = float(np.clip(u[7], 0.0, 1.0))
        nonnegative_radiation = max(float(radiation), 0.0)
        nonnegative_wind = max(float(wind_speed), 0.0)
        temperature_gap = float(raw_temperature) - float(outdoor_temperature)
        positive_gap = max(temperature_gap, 0.0)
        humidity_deficit = float(np.clip(1.0 - float(outdoor_relative_humidity) / 100.0, 0.0, 1.0))
        angle = 2.0 * np.pi * float(hour_of_day) / 24.0
        hour_sin = float(np.sin(angle))
        hour_cos = float(np.cos(angle))
        feature_sources = {
            "pred_air_temperature": float(raw_temperature),
            "pred_relative_humidity": float(raw_relative_humidity),
            "d_air_temperature": float(outdoor_temperature),
            "d_relative_humidity": float(outdoor_relative_humidity),
            "d_global_radiation": float(radiation),
            "d_wind_speed": float(wind_speed),
            "uRoofVent": float(u[3]),
            "uFan": float(u[6]),
            "uBlScr": float(u[5]),
            "uThScr": float(u[2]),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "temperature_outdoor_gap": temperature_gap,
            "humidity_outdoor_gap": float(raw_relative_humidity) - float(outdoor_relative_humidity),
            "radiation_hour_sin": float(radiation) * hour_sin,
            "radiation_hour_cos": float(radiation) * hour_cos,
            "uPad": float(u[7]),
            "solar_trapping": nonnegative_radiation * (1.0 - roof) * (1.0 - fan) * (1.0 - pad),
            "ventilation_heat_exchange": roof * nonnegative_wind * temperature_gap,
            "fan_cooling_demand": fan * positive_gap,
            "pad_evaporative_potential": pad * positive_gap * humidity_deficit,
            "hot_solar_load": nonnegative_radiation * max(float(outdoor_temperature) - 25.0, 0.0),
        }
        missing = [name for name in self.feature_columns if name not in feature_sources]
        if missing:
            raise ValueError(f"Unsupported V5 residual features: {missing}")
        features = np.asarray(
            [feature_sources[name] for name in self.feature_columns], dtype=float
        )
        design = np.concatenate(
            ([1.0], (features - self.feature_means) / self.feature_scales)
        )
        fallback = False
        corrected: dict[str, float] = {}
        raw_targets = {
            "air_temperature": float(raw_temperature),
            "relative_humidity": float(raw_relative_humidity),
        }
        for target in self.residual_model["target_names"]:
            residual = float(design @ self.coefficients[target])
            if not np.isfinite(residual):
                residual = 0.0
                fallback = True
            lower, upper = (-10.0, 60.0) if target == "air_temperature" else (0.0, 100.0)
            corrected[target] = float(
                np.clip(
                    raw_targets[target]
                    + float(applied_gain) * self.target_gains.get(target, 1.0) * residual,
                    lower,
                    upper,
                )
            )
        return corrected["air_temperature"], corrected["relative_humidity"], fallback

    def correct(
        self,
        *,
        previous_state: np.ndarray,
        raw_state: np.ndarray,
        controls: np.ndarray,
        disturbance: np.ndarray,
        dt_seconds: float,
        hour_of_day: float,
        day_of_year: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        previous = np.asarray(previous_state, dtype=float)
        raw = np.asarray(raw_state, dtype=float)
        u = np.asarray(controls, dtype=float)
        d = np.asarray(disturbance, dtype=float)
        if previous.shape != raw.shape or raw.ndim != 1 or len(raw) < 16:
            raise ValueError("V5 climate correction requires matching greenhouse states")
        if u.shape != (8,):
            raise ValueError("V5 climate correction requires eight controls")
        if d.ndim != 1 or len(d) < 5:
            raise ValueError("V5 climate correction requires greenhouse disturbances")
        values = np.concatenate([previous, raw, u, d, [dt_seconds, hour_of_day, day_of_year]])
        if not np.isfinite(values).all() or not 0.0 < float(dt_seconds) <= 3600.0:
            raise ValueError("V5 climate correction inputs must be finite with dt inside (0, 3600]")

        raw_temperature = float(raw[2])
        raw_rh = float(vaporPres2rh(raw_temperature, raw[15]))
        outdoor_rh = float(vaporPres2rh(float(d[1]), float(d[2])))
        gain_scale = min(1.0, float(dt_seconds) / 3600.0)
        applied_gain = gain_scale * float(self.residual_model.get("gain", 1.0))
        corrected_temperature, corrected_rh, fallback = self.predict_corrected_climate(
            raw_temperature=raw_temperature,
            raw_relative_humidity=raw_rh,
            outdoor_temperature=float(d[1]),
            outdoor_relative_humidity=outdoor_rh,
            radiation=float(d[0]),
            wind_speed=float(d[4]),
            controls=u,
            hour_of_day=float(hour_of_day),
            applied_gain=applied_gain,
        )
        corrected = raw.copy()
        corrected[2] = corrected_temperature
        corrected[15] = vaporDens2pres(
            corrected_temperature,
            rh2vaporDens(corrected_temperature, corrected_rh),
        )
        return corrected, {
            "climate_postprocessed": True,
            "residual_gain_scale": gain_scale,
            "residual_applied_gain": applied_gain,
            "residual_fallback": fallback,
            "raw_air_temperature": raw_temperature,
            "raw_relative_humidity": raw_rh,
            "corrected_air_temperature": corrected_temperature,
            "corrected_relative_humidity": corrected_rh,
        }


def build_v5_hybrid_environment(
    *,
    growth_year: int,
    start_day: int,
    episode_days: int = 1,
    dt_seconds: int = 900,
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    residual_model_path: str | Path = DEFAULT_RESIDUAL_PATH,
    reward_fn: str | None = None,
    reward_kwargs: dict[str, Any] | None = None,
) -> GreenLightEnv:
    params_path = Path(params_path)
    residual_model_path = Path(residual_model_path)
    params = load_parameter_vector(params_path)
    if params is None or len(params) != 225 or not np.isfinite(params).all():
        raise ValueError("V5 hybrid environment requires 225 finite parameters")
    residual_model = json.loads(residual_model_path.read_text(encoding="utf-8"))
    postprocessor = V5ResidualClimatePostprocessor(residual_model)

    kwargs = _load_environment_kwargs()
    safety_config = dict(kwargs["safety_config"])
    safety_config["independent_fan_control"] = True
    kwargs.update(
        {
            "num_params": 225,
            "nx": 28,
            "nu": 8,
            "nd": 11,
            "model_backend": "ChengduPhysicsV4",
            "u_min": [0.0] * 8,
            "u_max": [1.0] * 8,
            "controlled_inputs": ["uVent", "uBlScr", "uPadFan", "uPadPump"],
            "season_length": int(episode_days),
            "dt": int(dt_seconds),
            "weather_scenario_sampler": "fixed",
            "weather_scenario_sampler_kwargs": {
                "location": "Chengdu",
                "growth_year": int(growth_year),
                "start_day": int(start_day),
            },
            "state_postprocessor": postprocessor,
            "safety_config": safety_config,
        }
    )
    if reward_fn is not None:
        kwargs["reward_fn"] = reward_fn
    if reward_kwargs is not None:
        kwargs["reward_kwargs"] = reward_kwargs
    env = GreenLightEnv(**kwargs)
    env.base_p = np.asarray(params, dtype=np.float64).copy()
    env.parameter_provider.base_p = env.base_p.copy()
    env.p = env.base_p.copy()
    env.v5_hybrid_manifest = {
        "schema_version": "chengdu-v5-hybrid-environment-v1",
        "model_backend": "ChengduPhysicsV4",
        "num_params": 225,
        "controls": 8,
        "optimized_controls": ["uRoofVent", "uBlScr", "uFan", "uPad"],
        "params_sha256": _sha256(params_path),
        "residual_model_sha256": _sha256(residual_model_path),
        "residual_training_scope": "next_timestamp_before_2026-07-12",
    }
    return env
