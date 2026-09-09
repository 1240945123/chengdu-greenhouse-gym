from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def derive_initial_thermal_sum_scenarios(
    baseline_observations: pd.DataFrame,
    climate: pd.DataFrame,
    *,
    baseline_timestamp: str | pd.Timestamp,
    temperature_window_days: int = 7,
    timezone: str = "Asia/Shanghai",
) -> dict[str, Any]:
    required_observations = {"observation_date", "flower_truss_count"}
    required_climate = {"timestamp", "air_temperature"}
    if missing := sorted(required_observations - set(baseline_observations.columns)):
        raise ValueError(f"thermal-sum observations missing columns: {', '.join(missing)}")
    if missing := sorted(required_climate - set(climate.columns)):
        raise ValueError(f"thermal-sum climate missing columns: {', '.join(missing)}")
    window_days = int(temperature_window_days)
    if window_days <= 0:
        raise ValueError("temperature_window_days must be positive")

    baseline = _timestamp(baseline_timestamp, timezone).normalize()
    samples = baseline_observations[list(required_observations)].copy()
    samples["observation_date"] = samples["observation_date"].map(
        lambda value: _timestamp(value, timezone).normalize()
    )
    samples = samples.loc[samples["observation_date"].eq(baseline)]
    trusses = pd.to_numeric(samples["flower_truss_count"], errors="coerce").to_numpy(
        dtype=float
    )
    if len(trusses) == 0 or not np.isfinite(trusses).all() or (trusses <= 0.0).any():
        raise ValueError("baseline flower truss counts must be positive and finite")

    temperature = climate[list(required_climate)].copy()
    temperature["timestamp"] = temperature["timestamp"].map(
        lambda value: _timestamp(value, timezone)
    )
    temperature["air_temperature"] = pd.to_numeric(
        temperature["air_temperature"], errors="coerce"
    )
    temperature = temperature.sort_values("timestamp", kind="stable")
    if temperature.empty or temperature["timestamp"].isna().any():
        raise ValueError("thermal-sum climate timestamps must be valid")
    window_start = temperature["timestamp"].min()
    window_end = window_start + pd.Timedelta(days=window_days)
    temperature = temperature.loc[
        temperature["timestamp"].between(window_start, window_end)
    ]
    temperatures = temperature["air_temperature"].to_numpy(dtype=float)
    if not np.isfinite(temperatures).all() or (temperatures <= 0.0).any():
        raise ValueError("thermal-sum proxy temperatures must be positive and finite")
    rates = -0.2903 + 0.1454 * np.log(temperatures)
    if (rates <= 0.0).any():
        raise ValueError("thermal-sum inference requires a positive truss appearance rate")

    inferred = (
        trusses[:, np.newaxis]
        * temperatures[np.newaxis, :]
        / rates[np.newaxis, :]
    ).reshape(-1)
    quantiles = (("low", 0.10), ("median", 0.50), ("high", 0.90))
    scenarios = [
        {
            "name": name,
            "quantile": quantile,
            "initial_t_can_sum_deg_day": float(np.quantile(inferred, quantile)),
        }
        for name, quantile in quantiles
    ]
    return {
        "baseline_timestamp": baseline.isoformat(),
        "baseline_replicate_count": int(len(trusses)),
        "baseline_flower_truss_mean": float(np.mean(trusses)),
        "temperature_window_start": window_start.isoformat(),
        "temperature_window_end": window_end.isoformat(),
        "temperature_window_days": window_days,
        "temperature_sample_count": int(len(temperatures)),
        "temperature_proxy_scope": "first_postbaseline_week_not_prebaseline_measurement",
        "equation": "thermal_sum=flower_trusses*T_air/(-0.2903+0.1454*ln(T_air))",
        "scenarios": scenarios,
        "held_out_mass_used": False,
        "uncertainty_type": "structural_sensitivity_not_predictive_interval",
    }


def evaluate_temperature_driven_trusses(
    climate: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    baseline_timestamp: str | pd.Timestamp,
    timezone: str = "Asia/Shanghai",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate De Koning's temperature-driven tomato truss appearance model."""
    climate_required = {"timestamp", "air_temperature"}
    observation_required = {"observation_date", "fruit_truss_count"}
    if missing := sorted(climate_required - set(climate.columns)):
        raise ValueError(f"phenology climate missing columns: {', '.join(missing)}")
    if missing := sorted(observation_required - set(observations.columns)):
        raise ValueError(f"phenology observations missing columns: {', '.join(missing)}")

    weather = climate[list(climate_required)].copy()
    weather["timestamp"] = weather["timestamp"].map(
        lambda value: _timestamp(value, timezone)
    )
    weather["air_temperature"] = pd.to_numeric(
        weather["air_temperature"], errors="coerce"
    )
    weather = weather.sort_values("timestamp", kind="stable").reset_index(drop=True)
    values = weather["air_temperature"].to_numpy(dtype=float)
    if len(weather) < 2 or not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("phenology climate requires at least two positive finite temperatures")
    if weather["timestamp"].isna().any() or weather["timestamp"].duplicated().any():
        raise ValueError("phenology climate timestamps must be valid and unique")

    baseline = _timestamp(baseline_timestamp, timezone)
    samples = observations[list(observation_required)].copy()
    samples["observation_date"] = samples["observation_date"].map(
        lambda value: _timestamp(value, timezone).normalize()
    )
    samples["fruit_truss_count"] = pd.to_numeric(
        samples["fruit_truss_count"], errors="coerce"
    )
    sample_values = samples["fruit_truss_count"].to_numpy(dtype=float)
    if not np.isfinite(sample_values).all() or (sample_values < 0.0).any():
        raise ValueError("fruit truss observations must be finite and non-negative")
    grouped = samples.groupby("observation_date", sort=True)["fruit_truss_count"]
    observed = grouped.agg(["mean", "count"]).reset_index()
    if observed.empty:
        raise ValueError("at least one fruit truss observation is required")
    if (
        observed["observation_date"].min() < weather["timestamp"].min()
        or observed["observation_date"].max() > weather["timestamp"].max()
        or baseline < weather["timestamp"].min()
        or baseline > weather["timestamp"].max()
    ):
        raise ValueError("phenology observations or baseline are outside the climate interval")

    interval_days = (
        weather["timestamp"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        / 86400.0
    )
    temperature_24h = (
        weather["air_temperature"].rolling(24, min_periods=1).mean().to_numpy(dtype=float)
    )
    interval_temperature = 0.5 * (temperature_24h[:-1] + temperature_24h[1:])
    daily_rate = np.maximum(0.0, -0.2903 + 0.1454 * np.log(interval_temperature))
    cumulative = np.concatenate([[0.0], np.cumsum(daily_rate * interval_days)])
    climate_seconds = weather["timestamp"].map(lambda value: value.timestamp()).to_numpy()
    baseline_total = float(np.interp(baseline.timestamp(), climate_seconds, cumulative))
    predicted = np.interp(
        observed["observation_date"].map(lambda value: value.timestamp()),
        climate_seconds,
        cumulative,
    ) - baseline_total

    aligned = pd.DataFrame(
        {
            "observation_date": observed["observation_date"],
            "replicate_count": observed["count"].astype(int),
            "observed_mean_fruit_truss_count": observed["mean"].astype(float),
            "predicted_fruit_truss_count": np.maximum(0.0, predicted),
        }
    )
    residual = (
        aligned["predicted_fruit_truss_count"]
        - aligned["observed_mean_fruit_truss_count"]
    ).to_numpy(dtype=float)
    denominator = float(aligned["observed_mean_fruit_truss_count"].sum())
    metrics: dict[str, Any] = {
        "model": "de_koning_1994_temperature_truss_rate",
        "equation": "trusses_per_day=max(0,-0.2903+0.1454*ln(T_air_24h_C))",
        "baseline_timestamp": baseline.isoformat(),
        "observation_date_count": int(len(aligned)),
        "replicate_count": int(aligned["replicate_count"].sum()),
        "mae_trusses": float(np.mean(np.abs(residual))),
        "rmse_trusses": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_bias_trusses": float(np.mean(residual)),
        "wmape": float(np.abs(residual).sum() / denominator) if denominator > 0.0 else None,
        "source": "De Koning 1994, as reproduced in Heuvelink, Tomato growth and yield: quantitative analysis and synthesis",
        "scope": "independent_temperature_driven_phenology_check_not_yield_calibration",
    }
    return metrics, aligned


def _timestamp(value: object, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)
