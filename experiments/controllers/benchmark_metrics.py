from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


ALLOWED_CONTROL_COLUMNS = ("uThScr", "uVent", "uLamp", "uBlScr")
TEMPERATURE_SAFETY_BOUNDS_C = (15.0, 34.0)
HUMIDITY_SAFETY_BOUNDS_PERCENT = (50.0, 85.0)


def _require_columns(rows: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(rows.columns))
    if missing:
        raise ValueError(f"Missing benchmark columns: {', '.join(missing)}")


def summarize_episode(rows: pd.DataFrame, dt_seconds: float) -> dict[str, float | int | bool]:
    required = (
        "reward",
        "air_temperature",
        "temperature_target",
        "temperature_low",
        "temperature_high",
        "relative_humidity",
        "humidity_target",
        "humidity_low",
        "humidity_high",
        "uBoil",
        "uCO2",
        *ALLOWED_CONTROL_COLUMNS,
        *(f"proposed_{column}" for column in ALLOWED_CONTROL_COLUMNS),
        "safety_intervened",
        "safety_interventions",
        "safety_fallback_used",
        "safety_fallback_duration_steps",
        "wall_time_seconds",
        "controller_inference_seconds",
        "controller_fallback_used",
        "controller_failure_kind",
        "controller_fallback_duration_steps",
        "harvested_dry_matter_mg_m2",
        "allocated_fruit_dry_matter_mg_m2",
        "dry_matter_fraction",
        "c_buffer_mg_m2",
        "c_leaf_mg_m2",
        "c_stem_mg_m2",
        "c_fruit_mg_m2",
        "c_fruit_previous_mg_m2",
        "terminated",
        "truncated",
    )
    _require_columns(rows, required)
    if rows.empty:
        raise ValueError("Cannot summarize an empty episode")
    numeric = rows[[
        "reward",
        "air_temperature",
        "temperature_target",
        "temperature_low",
        "temperature_high",
        "relative_humidity",
        "humidity_target",
        "humidity_low",
        "humidity_high",
        "uBoil",
        "uCO2",
        *ALLOWED_CONTROL_COLUMNS,
        *(f"proposed_{column}" for column in ALLOWED_CONTROL_COLUMNS),
        "safety_intervened",
        "safety_fallback_used",
        "safety_fallback_duration_steps",
        "wall_time_seconds",
        "controller_inference_seconds",
        "controller_fallback_used",
        "controller_fallback_duration_steps",
        "harvested_dry_matter_mg_m2",
        "allocated_fruit_dry_matter_mg_m2",
        "dry_matter_fraction",
        "c_buffer_mg_m2",
        "c_leaf_mg_m2",
        "c_stem_mg_m2",
        "c_fruit_mg_m2",
        "c_fruit_previous_mg_m2",
    ]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Episode contains non-finite values")
    if not np.allclose(rows[["uBoil", "uCO2"]].to_numpy(dtype=float), 0.0, atol=1e-9):
        raise ValueError("Heating and CO2 controls must remain zero")

    dt_hours = float(dt_seconds) / 3600.0
    temperature = rows["air_temperature"].to_numpy(dtype=float)
    humidity = rows["relative_humidity"].to_numpy(dtype=float)
    temperature_error = temperature - rows["temperature_target"].to_numpy(dtype=float)
    humidity_error = humidity - rows["humidity_target"].to_numpy(dtype=float)
    controls = rows[list(ALLOWED_CONTROL_COLUMNS)].to_numpy(dtype=float)
    proposed_controls = rows[
        [f"proposed_{column}" for column in ALLOWED_CONTROL_COLUMNS]
    ].to_numpy(dtype=float)
    projection_l1 = np.abs(proposed_controls - controls).sum(axis=1)
    safety_intervened = rows["safety_intervened"].astype(bool).to_numpy()
    safety_fallback = rows["safety_fallback_used"].astype(bool).to_numpy()
    intervention_text = rows["safety_interventions"].fillna("").astype(str)
    controller_failure = rows["controller_failure_kind"].fillna("").astype(str)
    controller_fallback = rows["controller_fallback_used"].astype(bool).to_numpy()
    changes = np.abs(np.diff(controls, axis=0))
    temp_comfort = (
        (temperature >= rows["temperature_low"].to_numpy(dtype=float))
        & (temperature <= rows["temperature_high"].to_numpy(dtype=float))
    )
    humidity_comfort = (
        (humidity >= rows["humidity_low"].to_numpy(dtype=float))
        & (humidity <= rows["humidity_high"].to_numpy(dtype=float))
    )
    harvested_dm = rows["harvested_dry_matter_mg_m2"].to_numpy(dtype=float)
    allocated_fruit_dm = rows["allocated_fruit_dry_matter_mg_m2"].to_numpy(dtype=float)
    dmfm_values = rows["dry_matter_fraction"].to_numpy(dtype=float)
    if np.any(harvested_dm < 0.0):
        raise ValueError("Harvested dry matter must be non-negative")
    if np.any(allocated_fruit_dm < 0.0):
        raise ValueError("Allocated fruit dry matter must be non-negative")
    if np.any(dmfm_values <= 0.0) or not np.allclose(dmfm_values, dmfm_values[0]):
        raise ValueError("Dry matter fraction must be positive and constant within an episode")
    dmfm = float(dmfm_values[0])
    fresh_yield_steps = harvested_dm * 1e-6 / dmfm
    day_index = np.floor(np.arange(len(rows)) * float(dt_seconds) / 86400.0).astype(int)
    daily_fresh_yield = pd.Series(fresh_yield_steps).groupby(day_index).sum()
    positive_harvest = np.flatnonzero(harvested_dm > 0.0)
    initial_fruit_dm = float(rows["c_fruit_previous_mg_m2"].iloc[0])
    final_fruit_dm = float(rows["c_fruit_mg_m2"].iloc[-1])
    net_fruit_production_dm = float(harvested_dm.sum() + final_fruit_dm - initial_fruit_dm)

    return {
        "cumulative_reward": float(rows["reward"].sum()),
        "mean_reward": float(rows["reward"].mean()),
        "temperature_mae": float(np.mean(np.abs(temperature_error))),
        "temperature_rmse": float(np.sqrt(np.mean(np.square(temperature_error)))),
        "temperature_min": float(np.min(temperature)),
        "temperature_max": float(np.max(temperature)),
        "temperature_below_safety_hours": float(
            np.count_nonzero(temperature < TEMPERATURE_SAFETY_BOUNDS_C[0]) * dt_hours
        ),
        "temperature_above_safety_hours": float(
            np.count_nonzero(temperature > TEMPERATURE_SAFETY_BOUNDS_C[1]) * dt_hours
        ),
        "humidity_mae": float(np.mean(np.abs(humidity_error))),
        "humidity_rmse": float(np.sqrt(np.mean(np.square(humidity_error)))),
        "humidity_min": float(np.min(humidity)),
        "humidity_max": float(np.max(humidity)),
        "humidity_below_safety_hours": float(
            np.count_nonzero(humidity < HUMIDITY_SAFETY_BOUNDS_PERCENT[0]) * dt_hours
        ),
        "humidity_above_safety_hours": float(
            np.count_nonzero(humidity > HUMIDITY_SAFETY_BOUNDS_PERCENT[1]) * dt_hours
        ),
        "temperature_violation_hours": float(np.count_nonzero(~temp_comfort) * dt_hours),
        "humidity_violation_hours": float(np.count_nonzero(~humidity_comfort) * dt_hours),
        "temperature_comfort_fraction": float(np.mean(temp_comfort)),
        "humidity_comfort_fraction": float(np.mean(humidity_comfort)),
        "joint_comfort_fraction": float(np.mean(temp_comfort & humidity_comfort)),
        "mean_actuator_effort": float(np.mean(np.abs(controls))),
        "total_actuator_effort": float(np.sum(np.abs(controls)) * dt_hours),
        "total_action_variation": float(changes.sum()) if len(changes) else 0.0,
        "switching_count": int(np.count_nonzero(changes > 0.05)) if len(changes) else 0,
        "safety_intervention_fraction": float(np.mean(safety_intervened)),
        "safety_intervention_steps": int(np.count_nonzero(safety_intervened)),
        "safety_fallback_steps": int(np.count_nonzero(safety_fallback)),
        "max_safety_fallback_duration_steps": int(
            rows["safety_fallback_duration_steps"].max()
        ),
        "rain_vent_closure_steps": int(
            intervention_text.str.contains("rain_vent_closure", regex=False).sum()
        ),
        "wind_vent_closure_steps": int(
            intervention_text.str.contains("wind_vent_closure", regex=False).sum()
        ),
        "high_temperature_emergency_steps": int(
            intervention_text.str.contains(
                "high_temperature_emergency", regex=False
            ).sum()
        ),
        "rain_vent_override_high_temperature_steps": int(
            intervention_text.str.contains(
                "rain_vent_override_high_temperature", regex=False
            ).sum()
        ),
        "mean_safety_projection_l1": float(np.mean(projection_l1)),
        "total_safety_projection_l1": float(np.sum(projection_l1)),
        "lamp_use_hours": float(rows["uLamp"].sum() * dt_hours),
        "mean_wall_time_seconds": float(rows["wall_time_seconds"].mean()),
        "mean_wall_time_ms": float(rows["wall_time_seconds"].mean() * 1000.0),
        "mean_controller_inference_ms": float(
            rows["controller_inference_seconds"].mean() * 1000.0
        ),
        "controller_fallback_fraction": float(np.mean(controller_fallback)),
        "controller_fallback_steps": int(np.count_nonzero(controller_fallback)),
        "controller_timeout_fraction": float(
            controller_failure.str.startswith("timeout").mean()
        ),
        "max_controller_fallback_duration_steps": int(
            rows["controller_fallback_duration_steps"].max()
        ),
        "harvested_dry_matter_kg_m2": float(harvested_dm.sum() * 1e-6),
        "fresh_yield_kg_m2": float(fresh_yield_steps.sum()),
        "allocated_fruit_dry_matter_kg_m2": float(allocated_fruit_dm.sum() * 1e-6),
        "net_fruit_production_dry_matter_kg_m2": net_fruit_production_dm * 1e-6,
        "simulated_fresh_fruit_production_kg_m2": net_fruit_production_dm * 1e-6 / dmfm,
        "final_unharvested_fresh_fruit_kg_m2": final_fruit_dm * 1e-6 / dmfm,
        "harvest_onset_day": (
            float(positive_harvest[0] * float(dt_seconds) / 86400.0)
            if len(positive_harvest) else -1.0
        ),
        "peak_daily_fresh_yield_kg_m2": float(daily_fresh_yield.max()),
        "final_buffer_dry_matter_kg_m2": float(rows["c_buffer_mg_m2"].iloc[-1] * 1e-6),
        "final_leaf_dry_matter_kg_m2": float(rows["c_leaf_mg_m2"].iloc[-1] * 1e-6),
        "final_stem_dry_matter_kg_m2": float(rows["c_stem_mg_m2"].iloc[-1] * 1e-6),
        "final_fruit_dry_matter_kg_m2": float(rows["c_fruit_mg_m2"].iloc[-1] * 1e-6),
        "episode_steps": int(len(rows)),
        "episode_complete": bool(rows["terminated"].iloc[-1] and not rows["truncated"].any()),
        **{
            f"{column}_mean": float(np.mean(np.abs(rows[column].to_numpy(dtype=float))))
            for column in ALLOWED_CONTROL_COLUMNS
        },
    }


def _bootstrap_interval(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    indices = rng.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def aggregate_algorithms(
    episodes: pd.DataFrame,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    _require_columns(episodes, ("algorithm", "cumulative_reward"))
    if episodes.empty:
        raise ValueError("Cannot aggregate empty episode metrics")
    rng = np.random.default_rng(bootstrap_seed)
    records: list[dict[str, float | int | str]] = []
    for algorithm, group in episodes.groupby("algorithm", sort=True):
        rewards = group["cumulative_reward"].to_numpy(dtype=float)
        if not np.all(np.isfinite(rewards)):
            raise ValueError(f"Non-finite reward for algorithm {algorithm}")
        ci_low, ci_high = _bootstrap_interval(rewards, bootstrap_samples, rng)
        record: dict[str, float | int | str] = {
            "algorithm": str(algorithm),
            "episodes": int(len(group)),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            "reward_median": float(np.median(rewards)),
            "reward_ci95_low": ci_low,
            "reward_ci95_high": ci_high,
        }
        for column in group.select_dtypes(include=[np.number]).columns:
            if column in {"cumulative_reward", "seed"}:
                continue
            values = group[column].to_numpy(dtype=float)
            if np.all(np.isfinite(values)):
                record[f"{column}_mean"] = float(np.mean(values))
                record[f"{column}_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                )
        records.append(record)
    return pd.DataFrame(records).set_index("algorithm")


def benchmark_completeness(
    episodes: pd.DataFrame,
    *,
    required_algorithms: Iterable[str],
    required_seeds: Iterable[int],
    deterministic_algorithms: Iterable[str] = (),
) -> dict[str, object]:
    _require_columns(episodes, ("algorithm", "seed", "episode_complete"))
    algorithms = tuple(required_algorithms)
    seeds = {int(seed) for seed in required_seeds}
    deterministic = set(deterministic_algorithms)
    present_algorithms = set(episodes["algorithm"].astype(str))
    missing_algorithms = sorted(set(algorithms) - present_algorithms)
    missing_seeds: dict[str, list[int]] = {}
    incomplete: dict[str, list[int]] = {}
    for algorithm in algorithms:
        group = episodes[episodes["algorithm"] == algorithm]
        present_seeds = set(group["seed"].astype(int))
        expected_seeds = {0} if algorithm in deterministic else seeds
        missing = sorted(expected_seeds - present_seeds)
        if missing:
            missing_seeds[algorithm] = missing
        failed = sorted(set(group.loc[~group["episode_complete"].astype(bool), "seed"].astype(int)))
        if failed:
            incomplete[algorithm] = failed
    return {
        "paper_ready": not missing_algorithms and not missing_seeds and not incomplete,
        "missing_algorithms": missing_algorithms,
        "missing_seeds": missing_seeds,
        "incomplete_seeds": incomplete,
    }
