from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import casadi as ca

from common.standing_crop_evaluation import evaluate_standing_crop_predictions
from common.tomato_phenology import (
    derive_initial_thermal_sum_scenarios,
    evaluate_temperature_driven_trusses,
)
from glassgym.components.weather import WeatherRepository
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import load_weather_data
from glassgym.environments.utils import (
    co2ppm2dens,
    init_state,
    rh2vaporDens,
    vaporDens2pres,
)
from glassgym.configs.default_params import init_default_params
from glassgym.models.GreenLight.crop import crop_derivatives, crop_fluxes
from processing.chengdu_crop_initialization import build_target_crop_initial_state


CONTROL_NAMES = ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")
FLUX_NAMES = (
    "photosynthesis",
    "buffer_to_leaf",
    "buffer_to_stem",
    "buffer_to_fruit",
    "growth_respiration",
    "leaf_maintenance",
    "stem_maintenance",
    "fruit_maintenance",
    "leaf_pruning",
    "fruit_harvest",
)
FLUX_OUTPUT_COLUMNS = {
    "photosynthesis": "gross_photosynthesis_kg_m2",
    "buffer_to_leaf": "buffer_to_leaf_kg_m2",
    "buffer_to_stem": "buffer_to_stem_kg_m2",
    "buffer_to_fruit": "buffer_to_fruit_kg_m2",
    "growth_respiration": "growth_respiration_kg_m2",
    "leaf_maintenance": "leaf_maintenance_kg_m2",
    "stem_maintenance": "stem_maintenance_kg_m2",
    "fruit_maintenance": "fruit_maintenance_kg_m2",
    "leaf_pruning": "leaf_pruning_kg_m2",
    "fruit_harvest": "native_harvested_dry_kg_m2",
}
FLUX_DIAGNOSTIC_NAMES = {
    **{name: name for name in FLUX_NAMES},
    "photosynthesis": "gross_photosynthesis",
    "fruit_harvest": "native_fruit_harvest",
}


def prepare_observed_climate(
    aligned: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timezone: str = "Asia/Shanghai",
    maximum_interpolation_gap_hours: int = 8,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "timestamp",
        "air_temperature",
        "canopy_temperature",
        "relative_humidity",
    }
    missing = sorted(required - set(aligned.columns))
    if missing:
        raise ValueError(f"aligned target climate missing columns: {', '.join(missing)}")
    co2_candidates = [
        column
        for column in aligned.columns
        if column == "co2_concentration"
        or (
            column.startswith("co2_concentration__")
            and not column.endswith(("_observed_fraction", "_interpolated_fraction"))
        )
    ]
    if not co2_candidates:
        raise ValueError("aligned target climate has no CO2 concentration column")

    source = aligned.copy()
    source["timestamp"] = source["timestamp"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    start_time = _local_timestamp(start, timezone)
    end_time = _local_timestamp(end, timezone)
    source = source.loc[source["timestamp"].between(start_time, end_time)].copy()
    if source.empty:
        raise ValueError("aligned target climate does not cover the requested interval")
    co2_source = max(
        co2_candidates,
        key=lambda column: pd.to_numeric(source[column], errors="coerce").notna().mean(),
    )
    climate = source[
        [
            "timestamp",
            "air_temperature",
            "canopy_temperature",
            "relative_humidity",
            co2_source,
        ]
    ].rename(columns={co2_source: "co2_concentration"})
    climate = climate.sort_values("timestamp", kind="stable")
    if climate["timestamp"].duplicated().any():
        raise ValueError("aligned target climate timestamps must be unique")
    climate = climate.set_index("timestamp").reindex(
        pd.date_range(start_time, end_time, freq="1h", inclusive="both")
    )
    ranges = {
        "air_temperature": (0.0, 50.0),
        "canopy_temperature": (0.0, 50.0),
        "relative_humidity": (10.0, 100.0),
        "co2_concentration": (250.0, 2000.0),
    }
    audit: dict[str, Any] = {
        "co2_source_column": co2_source,
        "requested_start": start_time.isoformat(),
        "requested_end": end_time.isoformat(),
        "maximum_interpolation_gap_hours": int(maximum_interpolation_gap_hours),
    }
    before_missing: dict[str, int] = {}
    for column, (low, high) in ranges.items():
        climate[column] = pd.to_numeric(climate[column], errors="coerce")
        invalid = climate[column].notna() & ~climate[column].between(low, high)
        audit[f"invalid_{column}_rows"] = int(invalid.sum())
        climate.loc[invalid, column] = np.nan
        before_missing[column] = int(climate[column].isna().sum())
    climate = climate.interpolate(
        method="time",
        limit=int(maximum_interpolation_gap_hours),
        limit_direction="both",
    )
    for column in ranges:
        remaining = int(climate[column].isna().sum())
        audit[f"interpolated_{column}_rows"] = before_missing[column] - remaining
        audit[f"remaining_missing_{column}_rows"] = remaining
        if remaining:
            raise ValueError(
                f"target climate {column} has a gap longer than allowed interpolation"
            )
    climate.index.name = "timestamp"
    return climate.reset_index(), audit


def prepare_observed_control_targets(
    controls: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    dt_seconds: int,
    timezone: str = "Asia/Shanghai",
    maximum_source_gap_hours: float = 1.5,
) -> pd.DataFrame:
    required = {"timestamp", *CONTROL_NAMES}
    missing = sorted(required - set(controls.columns))
    if missing:
        raise ValueError(f"observed controls missing columns: {', '.join(missing)}")
    if int(dt_seconds) <= 0:
        raise ValueError("dt_seconds must be positive")
    start_time = _local_timestamp(start, timezone)
    end_time = _local_timestamp(end, timezone)
    if end_time <= start_time:
        raise ValueError("control target end must be after start")

    source = controls[list(required)].copy()
    source["timestamp"] = source["timestamp"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    source = source.sort_values("timestamp", kind="stable")
    if source["timestamp"].isna().any() or source["timestamp"].duplicated().any():
        raise ValueError("observed control timestamps must be valid and unique")
    for column in CONTROL_NAMES:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    values = source[list(CONTROL_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("observed control targets must be finite fractions in [0, 1]")

    relevant = source.loc[
        source["timestamp"].between(
            start_time - pd.Timedelta(hours=maximum_source_gap_hours), end_time
        )
    ]
    gaps = relevant["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    if not gaps.empty and (gaps > float(maximum_source_gap_hours) + 1e-9).any():
        raise ValueError("observed control source contains a gap larger than allowed")

    targets = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start_time,
                end_time,
                freq=pd.Timedelta(seconds=int(dt_seconds)),
                inclusive="left",
            )
        }
    )
    expanded = pd.merge_asof(
        targets,
        source,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(hours=maximum_source_gap_hours),
    )
    if expanded[list(CONTROL_NAMES)].isna().any().any():
        raise ValueError("observed control source does not cover the requested interval")
    return expanded


def action_toward_control_target(
    *, current: np.ndarray, target: np.ndarray, delta_u_max: float
) -> np.ndarray:
    current_values = np.asarray(current, dtype=float)
    target_values = np.asarray(target, dtype=float)
    delta = float(delta_u_max)
    if current_values.shape != target_values.shape:
        raise ValueError("current and target controls must have the same shape")
    if not np.isfinite(current_values).all() or not np.isfinite(target_values).all():
        raise ValueError("control values must be finite")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta_u_max must be finite and positive")
    return np.clip((target_values - current_values) / delta, -1.0, 1.0).astype(
        np.float32
    )


def stage_sink_parameter_multipliers(
    t_can_sum_deg_day: float, schedule: dict[str, Any] | None
) -> dict[str, float]:
    organs = ("fruit", "leaf", "stem")
    thermal_sum = float(t_can_sum_deg_day)
    if not np.isfinite(thermal_sum) or thermal_sum < 0.0:
        raise ValueError("canopy thermal sum must be finite and non-negative")
    if schedule is None:
        return {organ: 1.0 for organ in organs}

    required = {
        "reference_t_can_sum_deg_day",
        "exponent",
        "maximum_multiplier",
        "organs",
    }
    if missing := sorted(required - set(schedule)):
        raise ValueError(f"stage sink schedule missing fields: {', '.join(missing)}")
    reference = float(schedule["reference_t_can_sum_deg_day"])
    exponent = float(schedule["exponent"])
    maximum = float(schedule["maximum_multiplier"])
    selected_organs = list(schedule["organs"])
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("stage sink reference thermal sum must be finite and positive")
    if not np.isfinite(exponent) or exponent < 0.0:
        raise ValueError("stage sink exponent must be finite and non-negative")
    if not np.isfinite(maximum) or maximum < 1.0:
        raise ValueError("stage sink maximum multiplier must be finite and at least one")
    if (
        not selected_organs
        or len(selected_organs) != len(set(selected_organs))
        or not set(selected_organs).issubset(organs)
    ):
        raise ValueError("stage sink organs must be unique fruit, leaf, or stem names")

    factor = min(maximum, max(1.0, (thermal_sum / reference) ** exponent))
    return {
        organ: float(factor if organ in selected_organs else 1.0)
        for organ in organs
    }


def simulate_forced_crop_trajectory(
    *,
    climate: pd.DataFrame,
    controls: pd.DataFrame,
    weather: np.ndarray,
    initial_crop_state: dict[str, float],
    fruit_dry_matter_fraction: float,
    substep_seconds: int = 300,
    parameters: np.ndarray | None = None,
    include_flux_diagnostics: bool = True,
    stage_sink_schedule: dict[str, Any] | None = None,
) -> pd.DataFrame:
    required_climate = {
        "timestamp",
        "air_temperature",
        "canopy_temperature",
        "relative_humidity",
        "co2_concentration",
    }
    required_controls = {"timestamp", *CONTROL_NAMES}
    missing_climate = sorted(required_climate - set(climate.columns))
    missing_controls = sorted(required_controls - set(controls.columns))
    if missing_climate:
        raise ValueError(f"forced climate missing columns: {', '.join(missing_climate)}")
    if missing_controls:
        raise ValueError(f"forced controls missing columns: {', '.join(missing_controls)}")
    if len(climate) < 2 or len(controls) != len(climate) - 1:
        raise ValueError("forced climate must contain one more row than controls")
    if int(substep_seconds) <= 0:
        raise ValueError("substep_seconds must be positive")
    dry_fraction = float(fruit_dry_matter_fraction)
    if not np.isfinite(dry_fraction) or not 0.0 < dry_fraction < 1.0:
        raise ValueError("fruit_dry_matter_fraction must be in (0, 1)")

    climate_frame = climate.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    control_frame = controls.copy().sort_values("timestamp", kind="stable").reset_index(drop=True)
    climate_frame["timestamp"] = climate_frame["timestamp"].map(_local_timestamp)
    control_frame["timestamp"] = control_frame["timestamp"].map(_local_timestamp)
    if climate_frame["timestamp"].duplicated().any() or control_frame["timestamp"].duplicated().any():
        raise ValueError("forced input timestamps must be unique")
    if not control_frame["timestamp"].reset_index(drop=True).equals(
        climate_frame["timestamp"].iloc[:-1].reset_index(drop=True)
    ):
        raise ValueError("each forced control row must match an interval start")

    climate_columns = [
        "air_temperature",
        "canopy_temperature",
        "relative_humidity",
        "co2_concentration",
    ]
    for column in climate_columns:
        climate_frame[column] = pd.to_numeric(climate_frame[column], errors="coerce")
    climate_values = climate_frame[climate_columns].to_numpy(dtype=float)
    if not np.isfinite(climate_values).all():
        raise ValueError("forced climate values must be finite")
    if not climate_frame["relative_humidity"].between(0.0, 100.0).all():
        raise ValueError("forced relative humidity must be in [0, 100]")
    if not climate_frame["co2_concentration"].gt(0.0).all():
        raise ValueError("forced CO2 concentration must be positive")
    for column in CONTROL_NAMES:
        control_frame[column] = pd.to_numeric(control_frame[column], errors="coerce")
    control_values = control_frame[list(CONTROL_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(control_values).all() or (control_values < 0.0).any() or (control_values > 1.0).any():
        raise ValueError("forced controls must be finite fractions in [0, 1]")

    weather_values = np.asarray(weather, dtype=float)
    if weather_values.ndim != 2 or weather_values.shape[0] < len(climate_frame) or weather_values.shape[1] != 10:
        raise ValueError("weather must provide at least one 10-value row per climate timestamp")
    weather_values = weather_values[: len(climate_frame)]
    if not np.isfinite(weather_values).all():
        raise ValueError("forced weather values must be finite")

    p = np.asarray(
        init_default_params(216) if parameters is None else parameters, dtype=float
    )
    if p.shape != (216,) or not np.isfinite(p).all():
        raise ValueError("crop parameter vector must contain 216 finite values")
    stage_sink_parameter_multipliers(
        float(initial_crop_state["tCanSum"]), stage_sink_schedule
    )

    templates = []
    canopy_24h = climate_frame["canopy_temperature"].rolling(24, min_periods=1).mean()
    for index, row in climate_frame.iterrows():
        state = init_state(weather_values[index])
        air_temperature = float(row["air_temperature"])
        canopy_temperature = float(row["canopy_temperature"])
        co2_density = float(
            co2ppm2dens(air_temperature, float(row["co2_concentration"])) * 1e6
        )
        vapor_density = rh2vaporDens(
            air_temperature, float(row["relative_humidity"])
        )
        vapor_pressure = float(vaporDens2pres(air_temperature, vapor_density))
        state[0:2] = co2_density
        state[2:4] = air_temperature
        state[4] = canopy_temperature
        state[15:17] = vapor_pressure
        state[21] = float(canopy_24h.iloc[index])
        templates.append(state)
    templates_array = np.asarray(templates, dtype=float)

    crop = np.asarray(
        [
            initial_crop_state[name]
            for name in ("cBuf", "cLeaf", "cStem", "cFruit", "tCanSum")
        ]
        + [0.0],
        dtype=float,
    )
    if not np.isfinite(crop).all() or (crop < 0.0).any():
        raise ValueError("initial forced crop states must be finite and non-negative")

    x_symbol = ca.SX.sym("forced_crop_x", 28)
    u_symbol = ca.SX.sym("forced_crop_u", 6)
    d_symbol = ca.SX.sym("forced_crop_d", 10)
    p_symbol = ca.SX.sym("forced_crop_p", 216)
    rhs_function = ca.Function(
        "forced_crop_rhs",
        [x_symbol, u_symbol, d_symbol, p_symbol],
        [crop_derivatives(x_symbol, u_symbol, d_symbol, p_symbol)],
    )
    flux = crop_fluxes(x_symbol, u_symbol, d_symbol, p_symbol)
    if include_flux_diagnostics:
        flux_function = ca.Function(
            "forced_crop_fluxes",
            [x_symbol, u_symbol, d_symbol, p_symbol],
            [ca.vertcat(*(flux[name] for name in FLUX_NAMES))],
        )
    else:
        flux_function = ca.Function(
            "forced_crop_harvest_only",
            [x_symbol, u_symbol, d_symbol, p_symbol],
            [flux["fruit_harvest"]],
        )

    rows = [
        _forced_trajectory_row(
            timestamp=climate_frame.loc[0, "timestamp"],
            crop=crop,
            climate_row=climate_frame.loc[0],
            dry_fraction=dry_fraction,
            harvested_dry_kg_m2=0.0,
            net_fruit_change_kg_m2=0.0,
            floor_correction_mg_m2=0.0,
            interval_fluxes_kg_m2=(
                {name: 0.0 for name in FLUX_NAMES}
                if include_flux_diagnostics
                else None
            ),
            crop_mass_balance_residual_kg_m2=(
                0.0 if include_flux_diagnostics else None
            ),
            stage_sink_multipliers=stage_sink_parameter_multipliers(
                float(crop[4]), stage_sink_schedule
            ),
        )
    ]
    for interval in range(len(control_frame)):
        interval_seconds = float(
            (
                climate_frame.loc[interval + 1, "timestamp"]
                - climate_frame.loc[interval, "timestamp"]
            ).total_seconds()
        )
        steps = int(round(interval_seconds / int(substep_seconds)))
        if steps <= 0 or not np.isclose(steps * int(substep_seconds), interval_seconds):
            raise ValueError("forced climate interval must be divisible by substep_seconds")
        u = control_values[interval]
        previous_fruit = float(crop[3])
        previous_crop_mass = float(crop[:4].sum())
        integrated_flux_mg_m2 = np.zeros(len(FLUX_NAMES), dtype=float)
        floor_correction = 0.0
        for substep in range(steps):
            alpha0 = substep / steps
            alpha_mid = (substep + 0.5) / steps
            alpha1 = (substep + 1.0) / steps

            def evaluate_stage(
                candidate: np.ndarray, alpha: float
            ) -> tuple[np.ndarray, np.ndarray]:
                template = (
                    (1.0 - alpha) * templates_array[interval]
                    + alpha * templates_array[interval + 1]
                )
                template[22:28] = candidate
                disturbance = (
                    (1.0 - alpha) * weather_values[interval]
                    + alpha * weather_values[interval + 1]
                )
                stage_parameters = p
                if stage_sink_schedule is not None:
                    multipliers = stage_sink_parameter_multipliers(
                        float(candidate[4]), stage_sink_schedule
                    )
                    stage_parameters = p.copy()
                    for organ, parameter_index in {
                        "fruit": 154,
                        "leaf": 155,
                        "stem": 156,
                    }.items():
                        stage_parameters[parameter_index] *= multipliers[organ]
                derivative = np.asarray(
                    rhs_function(template, u, disturbance, stage_parameters), dtype=float
                ).reshape(-1)
                evaluated_flux = np.asarray(
                    flux_function(template, u, disturbance, stage_parameters), dtype=float
                ).reshape(-1)
                if include_flux_diagnostics:
                    flux_values = evaluated_flux
                else:
                    flux_values = np.zeros(len(FLUX_NAMES), dtype=float)
                    flux_values[FLUX_NAMES.index("fruit_harvest")] = float(
                        evaluated_flux[0]
                    )
                return derivative, flux_values

            h = float(substep_seconds)
            k1, flux1 = evaluate_stage(crop, alpha0)
            k2, flux2 = evaluate_stage(crop + 0.5 * h * k1, alpha_mid)
            k3, flux3 = evaluate_stage(crop + 0.5 * h * k2, alpha_mid)
            k4, flux4 = evaluate_stage(crop + h * k3, alpha1)

            end_candidate = crop + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            integrated_flux_mg_m2 += h * (
                flux1 + 2.0 * flux2 + 2.0 * flux3 + flux4
            ) / 6.0
            negative = np.minimum(end_candidate[:4], 0.0)
            floor_correction += float(-negative.sum())
            crop = end_candidate
            crop[:4] = np.maximum(crop[:4], 0.0)
            if not np.isfinite(crop).all():
                raise RuntimeError(
                    f"forced crop integration became non-finite at interval {interval}"
                )

        flux_kg_m2 = (
            dict(zip(FLUX_NAMES, (integrated_flux_mg_m2 * 1e-6).tolist()))
            if include_flux_diagnostics
            else None
        )
        harvested_mg_m2 = float(
            integrated_flux_mg_m2[FLUX_NAMES.index("fruit_harvest")]
        )
        harvested_dry = harvested_mg_m2 * 1e-6
        net_change = (float(crop[3]) - previous_fruit + harvested_mg_m2) * 1e-6
        if include_flux_diagnostics:
            external_net_mg_m2 = (
                integrated_flux_mg_m2[FLUX_NAMES.index("photosynthesis")]
                - integrated_flux_mg_m2[FLUX_NAMES.index("growth_respiration")]
                - integrated_flux_mg_m2[FLUX_NAMES.index("leaf_maintenance")]
                - integrated_flux_mg_m2[FLUX_NAMES.index("stem_maintenance")]
                - integrated_flux_mg_m2[FLUX_NAMES.index("fruit_maintenance")]
                - integrated_flux_mg_m2[FLUX_NAMES.index("leaf_pruning")]
                - harvested_mg_m2
            )
            mass_balance_residual = (
                float(crop[:4].sum())
                - previous_crop_mass
                - external_net_mg_m2
                - floor_correction
            ) * 1e-6
        else:
            mass_balance_residual = None
        rows.append(
            _forced_trajectory_row(
                timestamp=climate_frame.loc[interval + 1, "timestamp"],
                crop=crop,
                climate_row=climate_frame.loc[interval + 1],
                dry_fraction=dry_fraction,
                harvested_dry_kg_m2=harvested_dry,
                net_fruit_change_kg_m2=net_change,
                floor_correction_mg_m2=floor_correction,
                interval_fluxes_kg_m2=flux_kg_m2,
                crop_mass_balance_residual_kg_m2=mass_balance_residual,
                stage_sink_multipliers=stage_sink_parameter_multipliers(
                    float(crop[4]), stage_sink_schedule
                ),
            )
        )
    return pd.DataFrame(rows)


def calibrate_fruit_allocation_multiplier(
    *,
    climate: pd.DataFrame,
    controls: pd.DataFrame,
    weather: np.ndarray,
    initial_crop_state: dict[str, float],
    fruit_dry_matter_fraction: float,
    calibration_timestamp: str | pd.Timestamp,
    observed_fresh_kg_m2: float,
    candidate_multipliers: list[float] | np.ndarray,
    substep_seconds: int = 300,
) -> dict[str, Any]:
    calibration_time = _local_timestamp(calibration_timestamp)
    observed = float(observed_fresh_kg_m2)
    candidates = np.asarray(candidate_multipliers, dtype=float)
    if not np.isfinite(observed) or observed < 0.0:
        raise ValueError("calibration standing fresh mass must be finite and non-negative")
    if (
        candidates.ndim != 1
        or len(candidates) == 0
        or not np.isfinite(candidates).all()
        or (candidates <= 0.0).any()
        or len(np.unique(candidates)) != len(candidates)
    ):
        raise ValueError("candidate fruit-allocation multipliers must be unique and positive")

    climate_times = climate["timestamp"].map(_local_timestamp)
    if not climate_times.eq(calibration_time).any():
        raise ValueError("calibration timestamp must occur in the forced climate trajectory")
    climate_cut = climate.loc[climate_times.le(calibration_time)].copy()
    control_times = controls["timestamp"].map(_local_timestamp)
    controls_cut = controls.loc[control_times.lt(calibration_time)].copy()
    weather_cut = np.asarray(weather)[: len(climate_cut)]

    candidate_scores: list[dict[str, float]] = []
    best_trajectory: pd.DataFrame | None = None
    best_key: tuple[float, float] | None = None
    best_multiplier = float("nan")
    for multiplier in candidates:
        parameters = np.asarray(init_default_params(216), dtype=float)
        parameters[154] *= float(multiplier)
        trajectory = simulate_forced_crop_trajectory(
            climate=climate_cut,
            controls=controls_cut,
            weather=weather_cut,
            initial_crop_state=initial_crop_state,
            fruit_dry_matter_fraction=fruit_dry_matter_fraction,
            substep_seconds=substep_seconds,
            parameters=parameters,
            include_flux_diagnostics=False,
        )
        prediction = float(trajectory.iloc[-1]["standing_fresh_kg_m2"])
        error = prediction - observed
        candidate_scores.append(
            {
                "multiplier": float(multiplier),
                "predicted_fresh_kg_m2": prediction,
                "residual_kg_m2": error,
                "absolute_error_kg_m2": abs(error),
            }
        )
        key = (abs(error), float(multiplier))
        if best_key is None or key < best_key:
            best_key = key
            best_multiplier = float(multiplier)
            best_trajectory = trajectory
    assert best_key is not None and best_trajectory is not None
    return {
        "best_multiplier": best_multiplier,
        "base_rg_fruit": float(init_default_params(216)[154]),
        "calibrated_rg_fruit": float(init_default_params(216)[154]) * best_multiplier,
        "calibration_timestamp": calibration_time.isoformat(),
        "observed_fresh_kg_m2": observed,
        "predicted_fresh_kg_m2": float(
            best_trajectory.iloc[-1]["standing_fresh_kg_m2"]
        ),
        "absolute_error_kg_m2": float(best_key[0]),
        "candidate_scores": candidate_scores,
        "best_trajectory": best_trajectory,
    }


def calibrate_organ_allocation_multipliers(
    *,
    climate: pd.DataFrame,
    controls: pd.DataFrame,
    weather: np.ndarray,
    initial_crop_state: dict[str, float],
    fruit_dry_matter_fraction: float,
    calibration_timestamp: str | pd.Timestamp,
    observed_dry_kg_m2: dict[str, float],
    candidate_multipliers: dict[str, list[float] | np.ndarray],
    substep_seconds: int = 300,
) -> dict[str, Any]:
    organ_parameter_indices = {"fruit": 154, "leaf": 155, "stem": 156}
    if set(observed_dry_kg_m2) != set(organ_parameter_indices):
        raise ValueError("observed dry mass must contain exactly fruit, leaf, and stem")
    if set(candidate_multipliers) != set(organ_parameter_indices):
        raise ValueError("candidate multipliers must contain exactly fruit, leaf, and stem")
    observed = np.asarray(
        [observed_dry_kg_m2[organ] for organ in organ_parameter_indices], dtype=float
    )
    if not np.isfinite(observed).all() or (observed <= 0.0).any():
        raise ValueError("calibration organ dry masses must be finite and positive")

    candidates: dict[str, np.ndarray] = {}
    for organ, raw_values in candidate_multipliers.items():
        values = np.asarray(raw_values, dtype=float)
        if (
            values.ndim != 1
            or len(values) == 0
            or not np.isfinite(values).all()
            or (values <= 0.0).any()
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError(f"{organ} allocation candidates must be unique and positive")
        candidates[organ] = values

    calibration_time = _local_timestamp(calibration_timestamp)
    climate_times = climate["timestamp"].map(_local_timestamp)
    if not climate_times.eq(calibration_time).any():
        raise ValueError("calibration timestamp must occur in the forced climate trajectory")
    climate_cut = climate.loc[climate_times.le(calibration_time)].copy()
    control_times = controls["timestamp"].map(_local_timestamp)
    controls_cut = controls.loc[control_times.lt(calibration_time)].copy()
    weather_cut = np.asarray(weather)[: len(climate_cut)]

    candidate_scores: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_multipliers: dict[str, float] | None = None
    best_trajectory: pd.DataFrame | None = None
    organ_order = tuple(organ_parameter_indices)
    for combination in itertools.product(*(candidates[name] for name in organ_order)):
        multipliers = dict(zip(organ_order, map(float, combination)))
        parameters = np.asarray(init_default_params(216), dtype=float)
        for organ, parameter_index in organ_parameter_indices.items():
            parameters[parameter_index] *= multipliers[organ]
        trajectory = simulate_forced_crop_trajectory(
            climate=climate_cut,
            controls=controls_cut,
            weather=weather_cut,
            initial_crop_state=initial_crop_state,
            fruit_dry_matter_fraction=fruit_dry_matter_fraction,
            substep_seconds=substep_seconds,
            parameters=parameters,
            include_flux_diagnostics=False,
        )
        final = trajectory.iloc[-1]
        predicted = np.asarray(
            [
                final["standing_dry_kg_m2"],
                float(final["c_leaf_mg_m2"]) * 1e-6,
                float(final["c_stem_mg_m2"]) * 1e-6,
            ],
            dtype=float,
        )
        normalized_residual = (predicted - observed) / observed
        objective = float(np.sqrt(np.mean(np.square(normalized_residual))))
        score = {
            "multipliers": multipliers,
            "predicted_dry_kg_m2": dict(zip(organ_order, map(float, predicted))),
            "normalized_residual": dict(
                zip(organ_order, map(float, normalized_residual))
            ),
            "objective_nrmse": objective,
        }
        candidate_scores.append(score)
        key = (objective, *combination)
        if best_key is None or key < best_key:
            best_key = key
            best_multipliers = multipliers
            best_trajectory = trajectory
    assert best_key is not None and best_multipliers is not None and best_trajectory is not None
    return {
        "best_multipliers": best_multipliers,
        "parameter_indices": organ_parameter_indices,
        "calibration_timestamp": calibration_time.isoformat(),
        "observed_dry_kg_m2": dict(zip(organ_order, map(float, observed))),
        "predicted_dry_kg_m2": candidate_scores[
            min(range(len(candidate_scores)), key=lambda index: candidate_scores[index]["objective_nrmse"])
        ]["predicted_dry_kg_m2"],
        "objective_nrmse": float(best_key[0]),
        "candidate_scores": candidate_scores,
        "best_trajectory": best_trajectory,
    }


def calibrate_organ_allocation_proportional(
    *,
    climate: pd.DataFrame,
    controls: pd.DataFrame,
    weather: np.ndarray,
    initial_crop_state: dict[str, float],
    fruit_dry_matter_fraction: float,
    calibration_timestamp: str | pd.Timestamp,
    observed_dry_kg_m2: dict[str, float],
    base_parameters: np.ndarray,
    initial_multipliers: dict[str, float],
    iterations: int = 3,
    multiplier_bounds: tuple[float, float] = (0.05, 2.0),
    substep_seconds: int = 300,
) -> dict[str, Any]:
    organ_parameter_indices = {"fruit": 154, "leaf": 155, "stem": 156}
    organ_order = tuple(organ_parameter_indices)
    if set(observed_dry_kg_m2) != set(organ_order):
        raise ValueError("observed dry mass must contain exactly fruit, leaf, and stem")
    if set(initial_multipliers) != set(organ_order):
        raise ValueError("initial multipliers must contain exactly fruit, leaf, and stem")
    observed = np.asarray([observed_dry_kg_m2[name] for name in organ_order], dtype=float)
    initial_mass = np.asarray(
        [
            float(initial_crop_state["cFruit"]) * 1e-6,
            float(initial_crop_state["cLeaf"]) * 1e-6,
            float(initial_crop_state["cStem"]) * 1e-6,
        ],
        dtype=float,
    )
    if not np.isfinite(observed).all() or (observed <= initial_mass).any():
        raise ValueError("calibration organ dry masses must exceed finite initial masses")
    parameters_template = np.asarray(base_parameters, dtype=float)
    if parameters_template.shape != (216,) or not np.isfinite(parameters_template).all():
        raise ValueError("base_parameters must contain 216 finite values")
    count = int(iterations)
    low, high = map(float, multiplier_bounds)
    if count <= 0:
        raise ValueError("iterations must be positive")
    if not np.isfinite([low, high]).all() or not 0.0 < low < high:
        raise ValueError("multiplier bounds must be finite, positive, and ordered")
    multipliers = np.asarray(
        [initial_multipliers[name] for name in organ_order], dtype=float
    )
    if not np.isfinite(multipliers).all() or (multipliers <= 0.0).any():
        raise ValueError("initial organ multipliers must be finite and positive")
    multipliers = np.clip(multipliers, low, high)

    calibration_time = _local_timestamp(calibration_timestamp)
    climate_times = climate["timestamp"].map(_local_timestamp)
    if not climate_times.eq(calibration_time).any():
        raise ValueError("calibration timestamp must occur in the forced climate trajectory")
    climate_cut = climate.loc[climate_times.le(calibration_time)].copy()
    control_times = controls["timestamp"].map(_local_timestamp)
    controls_cut = controls.loc[control_times.lt(calibration_time)].copy()
    weather_cut = np.asarray(weather)[: len(climate_cut)]

    history: list[dict[str, Any]] = []
    selected_trajectory: pd.DataFrame | None = None
    selected_multipliers: dict[str, float] | None = None
    selected_iteration = 0
    selected_key: tuple[float, int] | None = None
    for iteration in range(count):
        parameters = parameters_template.copy()
        for index, organ in enumerate(organ_order):
            parameters[organ_parameter_indices[organ]] *= float(multipliers[index])
        trajectory = simulate_forced_crop_trajectory(
            climate=climate_cut,
            controls=controls_cut,
            weather=weather_cut,
            initial_crop_state=initial_crop_state,
            fruit_dry_matter_fraction=fruit_dry_matter_fraction,
            substep_seconds=substep_seconds,
            parameters=parameters,
            include_flux_diagnostics=False,
        )
        final = trajectory.iloc[-1]
        predicted = np.asarray(
            [
                float(final["standing_dry_kg_m2"]),
                float(final["c_leaf_mg_m2"]) * 1e-6,
                float(final["c_stem_mg_m2"]) * 1e-6,
            ],
            dtype=float,
        )
        normalized_residual = (predicted - observed) / observed
        objective = float(np.sqrt(np.mean(np.square(normalized_residual))))
        history.append(
            {
                "iteration": iteration + 1,
                "multipliers": dict(zip(organ_order, map(float, multipliers))),
                "predicted_dry_kg_m2": dict(zip(organ_order, map(float, predicted))),
                "objective_nrmse": objective,
            }
        )
        selection_key = (objective, iteration + 1)
        if selected_key is None or selection_key < selected_key:
            selected_key = selection_key
            selected_iteration = iteration + 1
            selected_multipliers = dict(zip(organ_order, map(float, multipliers)))
            selected_trajectory = trajectory
        if iteration < count - 1:
            predicted_increment = predicted - initial_mass
            observed_increment = observed - initial_mass
            if (predicted_increment <= 0.0).any():
                raise RuntimeError("proportional organ calibration requires positive modeled increments")
            multipliers = np.clip(
                multipliers * observed_increment / predicted_increment,
                low,
                high,
            )
    assert (
        selected_key is not None
        and selected_multipliers is not None
        and selected_trajectory is not None
    )
    return {
        "calibration_timestamp": calibration_time.isoformat(),
        "iterations": count,
        "multiplier_bounds": [low, high],
        "initial_multipliers": history[0]["multipliers"],
        "final_multipliers": selected_multipliers,
        "initial_objective_nrmse": float(history[0]["objective_nrmse"]),
        "final_objective_nrmse": float(selected_key[0]),
        "selected_iteration_from_calibration": selected_iteration,
        "selection_metric": "calibration_organ_nrmse",
        "history": history,
        "final_trajectory": selected_trajectory,
        "held_out_data_used": False,
    }


def evaluate_organ_dry_predictions(
    observations: pd.DataFrame,
    trajectory: pd.DataFrame,
    *,
    observation_timestamp: str | pd.Timestamp,
    dry_matter_fractions: dict[str, float],
) -> dict[str, Any]:
    fresh_columns = {
        "fruit": "ripe_fruit_fresh_kg_m2",
        "leaf": "leaf_fresh_kg_m2",
        "stem": "stem_fresh_kg_m2",
    }
    trajectory_columns = {
        "fruit": "standing_dry_kg_m2",
        "leaf": "c_leaf_mg_m2",
        "stem": "c_stem_mg_m2",
    }
    if set(dry_matter_fractions) != set(fresh_columns):
        raise ValueError("dry matter fractions must contain exactly fruit, leaf, and stem")
    missing = sorted({"observation_date", *fresh_columns.values()} - set(observations.columns))
    if missing:
        raise ValueError(f"organ observations missing columns: {', '.join(missing)}")
    missing = sorted({"timestamp", *trajectory_columns.values()} - set(trajectory.columns))
    if missing:
        raise ValueError(f"organ trajectory missing columns: {', '.join(missing)}")

    timestamp = _local_timestamp(observation_timestamp)
    sample_times = observations["observation_date"].map(_local_timestamp)
    samples = observations.loc[sample_times.eq(timestamp)].copy()
    trajectory_times = trajectory["timestamp"].map(_local_timestamp)
    prediction_rows = trajectory.loc[trajectory_times.eq(timestamp)]
    if samples.empty or len(prediction_rows) != 1:
        raise ValueError("organ evaluation requires samples and one trajectory row at the requested time")
    prediction_row = prediction_rows.iloc[0]

    organ_results: dict[str, dict[str, float]] = {}
    squared_relative_errors: list[float] = []
    for organ in fresh_columns:
        fraction = float(dry_matter_fractions[organ])
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("organ dry matter fractions must be finite values in (0, 1)")
        fresh = pd.to_numeric(samples[fresh_columns[organ]], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(fresh).all() or (fresh < 0.0).any():
            raise ValueError("organ fresh observations must be finite and non-negative")
        observed_dry = float(fresh.mean() * fraction)
        predicted_dry = float(prediction_row[trajectory_columns[organ]])
        if organ != "fruit":
            predicted_dry *= 1e-6
        residual = predicted_dry - observed_dry
        relative_error = residual / observed_dry if observed_dry > 0.0 else None
        if relative_error is not None:
            squared_relative_errors.append(relative_error**2)
        organ_results[organ] = {
            "observed_dry_kg_m2": observed_dry,
            "predicted_dry_kg_m2": predicted_dry,
            "residual_dry_kg_m2": residual,
            "absolute_error_dry_kg_m2": abs(residual),
            "relative_error": relative_error,
        }
    return {
        "observation_timestamp": timestamp.isoformat(),
        "replicate_count": int(len(samples)),
        "organs": organ_results,
        "organ_normalized_rmse": (
            float(np.sqrt(np.mean(squared_relative_errors)))
            if squared_relative_errors
            else None
        ),
        "dry_matter_fraction_scope": "xindu_to_pidu_transfer_not_target_measurement",
    }


def summarize_initial_thermal_sum_sensitivity(
    scenario_rows: list[dict[str, Any]] | pd.DataFrame,
    *,
    validation_observed_fresh_kg_m2: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "scenario",
        "initial_t_can_sum_deg_day",
        "rg_fruit_multiplier",
        "calibration_observed_fresh_kg_m2",
        "calibration_predicted_fresh_kg_m2",
        "validation_predicted_fresh_kg_m2",
    }
    aligned = pd.DataFrame(scenario_rows).copy()
    if missing := sorted(required - set(aligned.columns)):
        raise ValueError(f"thermal-sum scenario rows missing columns: {', '.join(missing)}")
    if aligned.empty or aligned["scenario"].astype(str).duplicated().any():
        raise ValueError("thermal-sum scenarios must be non-empty and uniquely named")
    numeric_columns = sorted(required - {"scenario"})
    for column in numeric_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    values = aligned[numeric_columns].to_numpy(dtype=float)
    observed = float(validation_observed_fresh_kg_m2)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("thermal-sum scenario values must be finite and non-negative")
    if not np.isfinite(observed) or observed < 0.0:
        raise ValueError("validation observed fresh mass must be finite and non-negative")

    aligned = aligned.sort_values("initial_t_can_sum_deg_day", kind="stable").reset_index(
        drop=True
    )
    aligned["calibration_residual_fresh_kg_m2"] = (
        aligned["calibration_predicted_fresh_kg_m2"]
        - aligned["calibration_observed_fresh_kg_m2"]
    )
    aligned["calibration_absolute_error_fresh_kg_m2"] = aligned[
        "calibration_residual_fresh_kg_m2"
    ].abs()
    aligned["validation_observed_fresh_kg_m2"] = observed
    aligned["validation_residual_fresh_kg_m2"] = (
        aligned["validation_predicted_fresh_kg_m2"] - observed
    )
    aligned["validation_absolute_error_fresh_kg_m2"] = aligned[
        "validation_residual_fresh_kg_m2"
    ].abs()
    envelope_low = float(aligned["validation_predicted_fresh_kg_m2"].min())
    envelope_high = float(aligned["validation_predicted_fresh_kg_m2"].max())
    absolute_validation_errors = aligned[
        "validation_absolute_error_fresh_kg_m2"
    ].to_numpy(dtype=float)
    summary = {
        "scenario_count": int(len(aligned)),
        "validation_observed_fresh_kg_m2": observed,
        "validation_envelope_low_fresh_kg_m2": envelope_low,
        "validation_envelope_high_fresh_kg_m2": envelope_high,
        "validation_envelope_width_fresh_kg_m2": envelope_high - envelope_low,
        "validation_envelope_contains_observed": bool(
            envelope_low <= observed <= envelope_high
        ),
        "best_case_absolute_relative_error": (
            float(absolute_validation_errors.min() / observed)
            if observed > 0.0
            else None
        ),
        "worst_case_absolute_relative_error": (
            float(absolute_validation_errors.max() / observed)
            if observed > 0.0
            else None
        ),
        "scenario_selected_from_validation": False,
        "uncertainty_type": "structural_sensitivity_not_predictive_interval",
        "adoption_status": "diagnostic_not_final_target_parameterization",
    }
    return summary, aligned


def summarize_biomass_flux_period(
    trajectory: pd.DataFrame,
    *,
    start_timestamp: str | pd.Timestamp,
    end_timestamp: str | pd.Timestamp,
    observed_start_dry_kg_m2: dict[str, float],
    observed_end_dry_kg_m2: dict[str, float],
    specific_leaf_area_m2_mg: float,
) -> dict[str, Any]:
    state_columns = {
        "buffer": "c_buffer_mg_m2",
        "leaf": "c_leaf_mg_m2",
        "stem": "c_stem_mg_m2",
        "fruit": "standing_dry_kg_m2",
    }
    required = {"timestamp", "crop_mass_balance_residual_kg_m2"}
    required.update(state_columns.values())
    required.update(FLUX_OUTPUT_COLUMNS.values())
    if missing := sorted(required - set(trajectory.columns)):
        raise ValueError(f"biomass flux trajectory missing columns: {', '.join(missing)}")
    organs = {"leaf", "stem", "fruit"}
    if set(observed_start_dry_kg_m2) != organs or set(observed_end_dry_kg_m2) != organs:
        raise ValueError("observed dry masses must contain exactly leaf, stem, and fruit")
    observed_start = {
        organ: float(observed_start_dry_kg_m2[organ]) for organ in sorted(organs)
    }
    observed_end = {
        organ: float(observed_end_dry_kg_m2[organ]) for organ in sorted(organs)
    }
    observed_values = np.asarray(
        [*observed_start.values(), *observed_end.values()], dtype=float
    )
    sla = float(specific_leaf_area_m2_mg)
    if not np.isfinite(observed_values).all() or (observed_values < 0.0).any():
        raise ValueError("observed organ dry masses must be finite and non-negative")
    if not np.isfinite(sla) or sla <= 0.0:
        raise ValueError("specific leaf area must be finite and positive")

    frame = trajectory.copy()
    frame["timestamp"] = frame["timestamp"].map(_local_timestamp)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("biomass flux trajectory timestamps must be valid and unique")
    start = _local_timestamp(start_timestamp)
    end = _local_timestamp(end_timestamp)
    if end <= start:
        raise ValueError("biomass flux period end must be after start")
    start_rows = frame.loc[frame["timestamp"].eq(start)]
    end_rows = frame.loc[frame["timestamp"].eq(end)]
    if len(start_rows) != 1 or len(end_rows) != 1:
        raise ValueError("biomass flux period boundaries must occur in the trajectory")
    period = frame.loc[frame["timestamp"].gt(start) & frame["timestamp"].le(end)]
    if period.empty:
        raise ValueError("biomass flux period contains no intervals")
    start_row = start_rows.iloc[0]
    end_row = end_rows.iloc[0]

    flux_totals = {
        FLUX_DIAGNOSTIC_NAMES[flux_name]: float(period[column].sum())
        for flux_name, column in FLUX_OUTPUT_COLUMNS.items()
    }
    model_change = {
        "leaf": (float(end_row["c_leaf_mg_m2"]) - float(start_row["c_leaf_mg_m2"]))
        * 1e-6,
        "stem": (float(end_row["c_stem_mg_m2"]) - float(start_row["c_stem_mg_m2"]))
        * 1e-6,
        "fruit": float(end_row["standing_dry_kg_m2"])
        - float(start_row["standing_dry_kg_m2"]),
    }
    observed_change = {
        organ: observed_end[organ] - observed_start[organ] for organ in sorted(organs)
    }
    model_aboveground_change = float(sum(model_change.values()))
    observed_aboveground_change = float(sum(observed_change.values()))
    buffer_change = (
        float(end_row["c_buffer_mg_m2"]) - float(start_row["c_buffer_mg_m2"])
    ) * 1e-6
    external_losses = float(
        flux_totals["growth_respiration"]
        + flux_totals["leaf_maintenance"]
        + flux_totals["stem_maintenance"]
        + flux_totals["fruit_maintenance"]
        + flux_totals["leaf_pruning"]
        + flux_totals["native_fruit_harvest"]
    )
    gross_photosynthesis = float(flux_totals["gross_photosynthesis"])
    model_retained_change = model_aboveground_change + buffer_change
    required_gross = observed_aboveground_change + buffer_change + external_losses
    return {
        "start_timestamp": start.isoformat(),
        "end_timestamp": end.isoformat(),
        "period_hours": float((end - start).total_seconds() / 3600.0),
        "flux_totals_kg_m2": flux_totals,
        "external_loss_total_kg_m2": external_losses,
        "model_organ_change_kg_m2": model_change,
        "observed_organ_change_kg_m2": observed_change,
        "model_buffer_change_kg_m2": buffer_change,
        "model_retained_crop_change_kg_m2": model_retained_change,
        "model_aboveground_change_kg_m2": model_aboveground_change,
        "observed_aboveground_change_kg_m2": observed_aboveground_change,
        "aboveground_change_deficit_kg_m2": (
            observed_aboveground_change - model_aboveground_change
        ),
        "model_fraction_of_observed_change": (
            model_aboveground_change / observed_aboveground_change
            if observed_aboveground_change != 0.0
            else None
        ),
        "external_loss_fraction_of_gross_photosynthesis": (
            external_losses / gross_photosynthesis
            if gross_photosynthesis != 0.0
            else None
        ),
        "required_gross_if_losses_and_buffer_unchanged_kg_m2": required_gross,
        "required_to_modeled_gross_ratio": (
            required_gross / gross_photosynthesis
            if gross_photosynthesis != 0.0
            else None
        ),
        "modeled_lai_start_m2_m2": float(start_row["c_leaf_mg_m2"]) * sla,
        "modeled_lai_end_m2_m2": float(end_row["c_leaf_mg_m2"]) * sla,
        "mass_balance_residual_sum_kg_m2": float(
            period["crop_mass_balance_residual_kg_m2"].sum()
        ),
        "mass_balance_residual_absolute_sum_kg_m2": float(
            period["crop_mass_balance_residual_kg_m2"].abs().sum()
        ),
        "mass_semantics": "greenlight_carbohydrate_equivalent_vs_inferred_observed_dry_matter",
        "holdout_used_for_parameter_selection": False,
        "status": "diagnostic_not_parameter_calibration",
    }


def summarize_assimilation_capacity_sensitivity(
    scenario_rows: list[dict[str, Any]] | pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "capacity_multiplier",
        "calibration_organ_nrmse",
        "validation_organ_nrmse",
        "validation_predicted_fresh_kg_m2",
        "validation_observed_fresh_kg_m2",
    }
    aligned = pd.DataFrame(scenario_rows).copy()
    if missing := sorted(required - set(aligned.columns)):
        raise ValueError(f"assimilation scenario rows missing columns: {', '.join(missing)}")
    if aligned.empty:
        raise ValueError("at least one assimilation capacity scenario is required")
    for column in required:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    values = aligned[list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("assimilation scenario values must be finite and non-negative")
    if aligned["capacity_multiplier"].le(0.0).any() or aligned[
        "capacity_multiplier"
    ].duplicated().any():
        raise ValueError("assimilation capacity multipliers must be positive and unique")
    observed_values = aligned["validation_observed_fresh_kg_m2"].to_numpy(dtype=float)
    if not np.allclose(observed_values, observed_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("assimilation scenarios must share one validation observation")
    observed = float(observed_values[0])
    if observed <= 0.0:
        raise ValueError("validation observed fresh mass must be positive")

    aligned = aligned.sort_values("capacity_multiplier", kind="stable").reset_index(
        drop=True
    )
    aligned["validation_fresh_residual_kg_m2"] = (
        aligned["validation_predicted_fresh_kg_m2"] - observed
    )
    aligned["validation_fresh_relative_error"] = (
        aligned["validation_fresh_residual_kg_m2"] / observed
    )
    predictions = aligned["validation_predicted_fresh_kg_m2"].to_numpy(dtype=float)
    low = float(predictions.min())
    high = float(predictions.max())
    summary = {
        "scenario_count": int(len(aligned)),
        "registered_capacity_multipliers": aligned["capacity_multiplier"].astype(float).tolist(),
        "calibration_organ_nrmse_range": [
            float(aligned["calibration_organ_nrmse"].min()),
            float(aligned["calibration_organ_nrmse"].max()),
        ],
        "validation_organ_nrmse_range": [
            float(aligned["validation_organ_nrmse"].min()),
            float(aligned["validation_organ_nrmse"].max()),
        ],
        "validation_observed_fresh_kg_m2": observed,
        "validation_fresh_envelope_kg_m2": [low, high],
        "validation_fresh_envelope_contains_observed": bool(low <= observed <= high),
        "best_case_validation_fresh_absolute_relative_error": float(
            aligned["validation_fresh_relative_error"].abs().min()
        ),
        "scenario_selected_from_validation": False,
        "held_out_data_used_for_calibration": False,
        "uncertainty_type": "structural_sensitivity_not_predictive_interval",
        "adoption_status": "diagnostic_not_final_target_parameterization",
    }
    return summary, aligned


def summarize_stage_sink_structure_comparison(
    scenario_rows: list[dict[str, Any]] | pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "scenario",
        "calibration_organ_nrmse",
        "development_organ_nrmse",
        "development_predicted_fresh_kg_m2",
        "development_observed_fresh_kg_m2",
    }
    aligned = pd.DataFrame(scenario_rows).copy()
    if missing := sorted(required - set(aligned.columns)):
        raise ValueError(f"stage sink scenario rows missing columns: {', '.join(missing)}")
    if aligned.empty:
        raise ValueError("at least one stage sink scenario is required")
    aligned["scenario"] = aligned["scenario"].astype(str)
    if aligned["scenario"].eq("").any() or aligned["scenario"].duplicated().any():
        raise ValueError("stage sink scenario names must be non-empty and unique")
    numeric_columns = sorted(required - {"scenario"})
    for column in numeric_columns:
        aligned[column] = pd.to_numeric(aligned[column], errors="coerce")
    values = aligned[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("stage sink scenario values must be finite and non-negative")
    observed_values = aligned["development_observed_fresh_kg_m2"].to_numpy(
        dtype=float
    )
    if not np.allclose(observed_values, observed_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("stage sink scenarios must share one development observation")
    observed = float(observed_values[0])
    if observed <= 0.0:
        raise ValueError("development observed fresh mass must be positive")

    aligned["development_fresh_residual_kg_m2"] = (
        aligned["development_predicted_fresh_kg_m2"] - observed
    )
    aligned["development_fresh_relative_error"] = (
        aligned["development_fresh_residual_kg_m2"] / observed
    )
    predictions = aligned["development_predicted_fresh_kg_m2"].to_numpy(
        dtype=float
    )
    low = float(predictions.min())
    high = float(predictions.max())
    summary = {
        "scenario_count": int(len(aligned)),
        "registered_scenarios": aligned["scenario"].tolist(),
        "calibration_organ_nrmse_range": [
            float(aligned["calibration_organ_nrmse"].min()),
            float(aligned["calibration_organ_nrmse"].max()),
        ],
        "development_organ_nrmse_range": [
            float(aligned["development_organ_nrmse"].min()),
            float(aligned["development_organ_nrmse"].max()),
        ],
        "development_observed_fresh_kg_m2": observed,
        "development_fresh_envelope_kg_m2": [low, high],
        "development_fresh_envelope_contains_observed": bool(
            low <= observed <= high
        ),
        "best_case_development_fresh_absolute_relative_error": float(
            aligned["development_fresh_relative_error"].abs().min()
        ),
        "scenario_selected_from_development": False,
        "posthoc_structure_development": True,
        "independent_validation": False,
        "uncertainty_type": "posthoc_structural_sensitivity_not_predictive_interval",
        "adoption_status": "diagnostic_not_final_target_parameterization",
    }
    return summary, aligned


def run_target_crop_validation(
    *,
    crop_observations_path: str | Path,
    controls_path: str | Path,
    aligned_climate_path: str | Path = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv",
    env_config_path: str | Path = "configs/envs/ChengduSingleGreenhouseEnv.yml",
    harvest_config_path: str | Path = "configs/crops/chengdu_tomato_harvest.yml",
    output_root: str | Path = "results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation",
    baseline_date: str = "2026-03-30",
    simulation_start: str = "2026-04-01 00:00:00",
    seed: int = 20260728,
) -> dict[str, Any]:
    observations = pd.read_csv(crop_observations_path)
    harvest_config = yaml.safe_load(Path(harvest_config_path).read_text(encoding="utf-8"))
    target = harvest_config["target_site"]
    target_id = int(target["greenhouse_id"])
    target_code = str(target["greenhouse_code"])
    initial_state, initialization_audit = build_target_crop_initial_state(
        observations,
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
        baseline_date=baseline_date,
    )

    observation_dates = observations.loc[
        pd.to_numeric(observations["greenhouse_id"], errors="coerce").eq(target_id)
        & observations["greenhouse_code"].astype(str).eq(target_code),
        "observation_date",
    ].map(_local_timestamp)
    start = _local_timestamp(simulation_start)
    scored_dates = observation_dates.loc[observation_dates.ge(start)]
    if scored_dates.empty:
        raise ValueError("no target standing-crop observations occur after simulation start")
    end = scored_dates.max().normalize()

    env_kwargs = _load_environment_kwargs(env_config_path)
    dt_seconds = 3600
    duration_days = (end - start).total_seconds() / 86400.0
    climate, climate_audit = prepare_observed_climate(
        pd.read_csv(aligned_climate_path), start=start, end=end
    )
    controls = prepare_observed_control_targets(
        pd.read_csv(controls_path),
        start=start,
        end=end,
        dt_seconds=dt_seconds,
    )
    fruit_dry_fraction = float(
        initialization_audit["dry_matter_fraction_transfer"]["ripe_fruit"]["estimate"]
    )
    weather = env_kwargs["weather_repository"].load(
        location="Chengdu",
        growth_year=2026,
        start_day=0,
        season_length=int(np.ceil(duration_days)) + 1,
        pred_horizon=0,
        dt=dt_seconds,
        nd=10,
    )
    trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=fruit_dry_fraction,
        substep_seconds=300,
        include_flux_diagnostics=False,
    )
    scoring_observations = observations.loc[
        observations["observation_date"].map(_local_timestamp).ge(start)
    ].copy()
    prediction_columns = [
        "timestamp",
        "greenhouse_id",
        "greenhouse_code",
        "standing_fresh_kg_m2",
    ]
    predictions = trajectory.copy()
    predictions["greenhouse_id"] = target_id
    predictions["greenhouse_code"] = target_code
    metrics, aligned = evaluate_standing_crop_predictions(
        scoring_observations,
        predictions[prediction_columns],
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
    )
    metrics["baseline_date_used_for_initialization"] = _local_timestamp(
        baseline_date
    ).isoformat()
    metrics["baseline_date_excluded_from_prediction_metrics"] = True
    metrics["fruit_dry_matter_fraction"] = fruit_dry_fraction
    metrics["fruit_dry_matter_fraction_scope"] = "xindu_to_pidu_transfer_not_calibrated"
    metrics["nonnegative_floor_correction_total_mg_m2"] = float(
        trajectory["nonnegative_floor_correction_mg_m2"].sum()
    )

    target_scored = scoring_observations.loc[
        pd.to_numeric(scoring_observations["greenhouse_id"], errors="coerce").eq(
            target_id
        )
        & scoring_observations["greenhouse_code"].astype(str).eq(target_code)
    ].copy()
    target_scored["observation_date"] = target_scored["observation_date"].map(
        _local_timestamp
    )
    observed_by_date = target_scored.groupby("observation_date")[
        "ripe_fruit_fresh_kg_m2"
    ].mean()
    if len(observed_by_date) < 2:
        raise ValueError(
            "at least two post-baseline target dates are required for chronological crop calibration"
        )
    calibration_date = observed_by_date.index.min()
    validation_date = observed_by_date.index.max()
    calibration = calibrate_fruit_allocation_multiplier(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=fruit_dry_fraction,
        calibration_timestamp=calibration_date,
        observed_fresh_kg_m2=float(observed_by_date.loc[calibration_date]),
        candidate_multipliers=np.round(np.arange(0.30, 0.5001, 0.025), 3),
        substep_seconds=300,
    )
    calibrated_parameters = np.asarray(init_default_params(216), dtype=float)
    calibrated_parameters[154] *= float(calibration["best_multiplier"])
    calibrated_trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=fruit_dry_fraction,
        substep_seconds=300,
        parameters=calibrated_parameters,
        include_flux_diagnostics=False,
    )
    fruit_fraction_ci95 = harvest_config["source_precalibration"][
        "fruit_dry_matter_fraction"
    ]["ci95"]
    fraction_low, fraction_high = map(float, fruit_fraction_ci95)
    calibrated_trajectory["standing_fresh_dmf95_low_kg_m2"] = (
        calibrated_trajectory["standing_dry_kg_m2"] / fraction_high
    )
    calibrated_trajectory["standing_fresh_dmf95_high_kg_m2"] = (
        calibrated_trajectory["standing_dry_kg_m2"] / fraction_low
    )
    calibrated_predictions = calibrated_trajectory.copy()
    calibrated_predictions["greenhouse_id"] = target_id
    calibrated_predictions["greenhouse_code"] = target_code
    calibration_observations = target_scored.loc[
        target_scored["observation_date"].eq(calibration_date)
    ]
    validation_observations = target_scored.loc[
        target_scored["observation_date"].eq(validation_date)
    ]
    calibration_metrics, calibration_aligned = evaluate_standing_crop_predictions(
        calibration_observations,
        calibrated_predictions[prediction_columns],
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
    )
    validation_metrics, validation_aligned = evaluate_standing_crop_predictions(
        validation_observations,
        calibrated_predictions[prediction_columns],
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
    )
    validation_prediction = calibrated_trajectory.loc[
        calibrated_trajectory["timestamp"].map(_local_timestamp).eq(validation_date)
    ].iloc[0]
    observed_validation_mean = float(
        validation_aligned.iloc[0]["observed_mean_fresh_kg_m2"]
    )
    validation_metrics["dmf95_sensitivity_low_kg_m2"] = float(
        validation_prediction["standing_fresh_dmf95_low_kg_m2"]
    )
    validation_metrics["dmf95_sensitivity_high_kg_m2"] = float(
        validation_prediction["standing_fresh_dmf95_high_kg_m2"]
    )
    validation_metrics["dmf95_sensitivity_contains_observed_mean"] = bool(
        validation_metrics["dmf95_sensitivity_low_kg_m2"]
        <= observed_validation_mean
        <= validation_metrics["dmf95_sensitivity_high_kg_m2"]
    )

    baseline_time = _local_timestamp(baseline_date).normalize()
    observation_times = observations["observation_date"].map(_local_timestamp)
    baseline_target_observations = observations.loc[
        pd.to_numeric(observations["greenhouse_id"], errors="coerce").eq(target_id)
        & observations["greenhouse_code"].astype(str).eq(target_code)
        & observation_times.map(lambda value: value.normalize()).eq(baseline_time)
    ].copy()
    thermal_sum_prior = derive_initial_thermal_sum_scenarios(
        baseline_target_observations[["observation_date", "flower_truss_count"]],
        climate[["timestamp", "air_temperature"]],
        baseline_timestamp=baseline_time,
        temperature_window_days=7,
    )
    thermal_sum_rows: list[dict[str, Any]] = []
    for scenario in thermal_sum_prior["scenarios"]:
        scenario_initial_state = dict(initial_state)
        scenario_initial_state["tCanSum"] = float(
            scenario["initial_t_can_sum_deg_day"]
        )
        scenario_calibration = calibrate_fruit_allocation_multiplier(
            climate=climate,
            controls=controls,
            weather=weather,
            initial_crop_state=scenario_initial_state,
            fruit_dry_matter_fraction=fruit_dry_fraction,
            calibration_timestamp=calibration_date,
            observed_fresh_kg_m2=float(observed_by_date.loc[calibration_date]),
            candidate_multipliers=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            substep_seconds=300,
        )
        scenario_parameters = np.asarray(init_default_params(216), dtype=float)
        scenario_parameters[154] *= float(scenario_calibration["best_multiplier"])
        scenario_trajectory = simulate_forced_crop_trajectory(
            climate=climate,
            controls=controls,
            weather=weather,
            initial_crop_state=scenario_initial_state,
            fruit_dry_matter_fraction=fruit_dry_fraction,
            substep_seconds=300,
            parameters=scenario_parameters,
            include_flux_diagnostics=False,
        )
        scenario_validation = scenario_trajectory.loc[
            scenario_trajectory["timestamp"].map(_local_timestamp).eq(validation_date)
        ]
        if len(scenario_validation) != 1:
            raise RuntimeError("thermal-sum scenario trajectory missed the validation date")
        thermal_sum_rows.append(
            {
                "scenario": str(scenario["name"]),
                "scenario_quantile": float(scenario["quantile"]),
                "initial_t_can_sum_deg_day": float(
                    scenario["initial_t_can_sum_deg_day"]
                ),
                "rg_fruit_multiplier": float(
                    scenario_calibration["best_multiplier"]
                ),
                "calibration_observed_fresh_kg_m2": float(
                    observed_by_date.loc[calibration_date]
                ),
                "calibration_predicted_fresh_kg_m2": float(
                    scenario_calibration["predicted_fresh_kg_m2"]
                ),
                "validation_predicted_fresh_kg_m2": float(
                    scenario_validation.iloc[0]["standing_fresh_kg_m2"]
                ),
            }
        )
    thermal_sum_summary, thermal_sum_aligned = (
        summarize_initial_thermal_sum_sensitivity(
            thermal_sum_rows,
            validation_observed_fresh_kg_m2=observed_validation_mean,
        )
    )
    thermal_sum_summary["prior"] = thermal_sum_prior
    thermal_sum_summary["calibration_timestamp"] = calibration_date.isoformat()
    thermal_sum_summary["validation_timestamp"] = validation_date.isoformat()
    thermal_sum_summary["rg_fruit_candidate_multipliers"] = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    ]
    thermal_sum_summary["zero_placeholder_reference"] = {
        "initial_t_can_sum_deg_day": 0.0,
        "rg_fruit_multiplier": float(calibration["best_multiplier"]),
        "validation_predicted_fresh_kg_m2": float(
            validation_prediction["standing_fresh_kg_m2"]
        ),
    }
    calibration_summary = {
        key: value for key, value in calibration.items() if key != "best_trajectory"
    }
    calibration_summary["split_policy"] = (
        "first_post_baseline_date_fit_last_post_baseline_date_validation"
    )
    calibration_summary["calibration_metrics"] = calibration_metrics
    calibration_summary["validation_date"] = validation_date.isoformat()
    calibration_summary["validation_metrics"] = validation_metrics
    calibration_summary["parameter_scope"] = "preliminary_single_date_target_crop_fit"

    dry_matter_fractions = {
        "fruit": fruit_dry_fraction,
        "leaf": float(
            initialization_audit["dry_matter_fraction_transfer"]["leaf"]["estimate"]
        ),
        "stem": float(
            initialization_audit["dry_matter_fraction_transfer"]["stem"]["estimate"]
        ),
    }
    calibration_organ_targets = evaluate_organ_dry_predictions(
        calibration_observations,
        trajectory,
        observation_timestamp=calibration_date,
        dry_matter_fractions=dry_matter_fractions,
    )
    observed_organ_dry = {
        organ: values["observed_dry_kg_m2"]
        for organ, values in calibration_organ_targets["organs"].items()
    }
    organ_calibration = calibrate_organ_allocation_multipliers(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=fruit_dry_fraction,
        calibration_timestamp=calibration_date,
        observed_dry_kg_m2=observed_organ_dry,
        candidate_multipliers={
            "fruit": [0.3, 0.4, 0.5],
            "leaf": [0.3, 0.5],
            "stem": [0.5, 0.65, 1.0],
        },
        substep_seconds=300,
    )
    organ_parameters = np.asarray(init_default_params(216), dtype=float)
    for organ, parameter_index in organ_calibration["parameter_indices"].items():
        organ_parameters[parameter_index] *= float(
            organ_calibration["best_multipliers"][organ]
        )
    organ_trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=fruit_dry_fraction,
        substep_seconds=300,
        parameters=organ_parameters,
    )
    organ_calibration_metrics = evaluate_organ_dry_predictions(
        calibration_observations,
        organ_trajectory,
        observation_timestamp=calibration_date,
        dry_matter_fractions=dry_matter_fractions,
    )
    organ_validation_metrics = evaluate_organ_dry_predictions(
        validation_observations,
        organ_trajectory,
        observation_timestamp=validation_date,
        dry_matter_fractions=dry_matter_fractions,
    )
    observed_organ_start = {
        organ: values["observed_dry_kg_m2"]
        for organ, values in organ_calibration_metrics["organs"].items()
    }
    observed_organ_end = {
        organ: values["observed_dry_kg_m2"]
        for organ, values in organ_validation_metrics["organs"].items()
    }
    late_biomass_flux_diagnostics = summarize_biomass_flux_period(
        organ_trajectory,
        start_timestamp=calibration_date,
        end_timestamp=validation_date,
        observed_start_dry_kg_m2=observed_organ_start,
        observed_end_dry_kg_m2=observed_organ_end,
        specific_leaf_area_m2_mg=float(init_default_params(216)[142]),
    )
    if late_biomass_flux_diagnostics[
        "mass_balance_residual_absolute_sum_kg_m2"
    ] > 1e-8:
        raise RuntimeError("forced crop flux diagnostics failed mass-balance tolerance")

    reference_row = organ_trajectory.loc[
        organ_trajectory["timestamp"].map(_local_timestamp).eq(calibration_date)
    ]
    if len(reference_row) != 1:
        raise RuntimeError("organ trajectory missed stage sink reference date")
    stage_reference = float(
        reference_row.iloc[0]["canopy_thermal_sum_deg_day"]
    )
    registered_stage_structures = [
        {"scenario": "constant", "exponent": 0.0, "organs": []},
        {
            "scenario": "sqrt_all",
            "exponent": 0.5,
            "organs": ["fruit", "leaf", "stem"],
        },
        {
            "scenario": "linear_all",
            "exponent": 1.0,
            "organs": ["fruit", "leaf", "stem"],
        },
        {"scenario": "sqrt_fruit", "exponent": 0.5, "organs": ["fruit"]},
        {"scenario": "linear_fruit", "exponent": 1.0, "organs": ["fruit"]},
    ]
    stage_rows: list[dict[str, Any]] = []
    stage_details: list[dict[str, Any]] = []
    for structure in registered_stage_structures:
        schedule = None
        if structure["scenario"] != "constant":
            schedule = {
                "reference_t_can_sum_deg_day": stage_reference,
                "exponent": float(structure["exponent"]),
                "maximum_multiplier": 2.0,
                "organs": list(structure["organs"]),
            }
        stage_trajectory = (
            organ_trajectory
            if schedule is None
            else simulate_forced_crop_trajectory(
                climate=climate,
                controls=controls,
                weather=weather,
                initial_crop_state=initial_state,
                fruit_dry_matter_fraction=fruit_dry_fraction,
                substep_seconds=300,
                parameters=organ_parameters,
                include_flux_diagnostics=False,
                stage_sink_schedule=schedule,
            )
        )
        stage_calibration_metrics = evaluate_organ_dry_predictions(
            calibration_observations,
            stage_trajectory,
            observation_timestamp=calibration_date,
            dry_matter_fractions=dry_matter_fractions,
        )
        stage_development_metrics = evaluate_organ_dry_predictions(
            validation_observations,
            stage_trajectory,
            observation_timestamp=validation_date,
            dry_matter_fractions=dry_matter_fractions,
        )
        development_row = stage_trajectory.loc[
            stage_trajectory["timestamp"].map(_local_timestamp).eq(validation_date)
        ]
        if len(development_row) != 1:
            raise RuntimeError("stage sink trajectory missed development date")
        stage_rows.append(
            {
                "scenario": str(structure["scenario"]),
                "exponent": float(structure["exponent"]),
                "organs": ",".join(structure["organs"]) or "none",
                "reference_t_can_sum_deg_day": stage_reference,
                "maximum_multiplier": 1.0 if schedule is None else 2.0,
                "calibration_organ_nrmse": float(
                    stage_calibration_metrics["organ_normalized_rmse"]
                ),
                "development_organ_nrmse": float(
                    stage_development_metrics["organ_normalized_rmse"]
                ),
                "development_predicted_fresh_kg_m2": float(
                    development_row.iloc[0]["standing_fresh_kg_m2"]
                ),
                "development_observed_fresh_kg_m2": observed_validation_mean,
                "development_fruit_relative_error": float(
                    stage_development_metrics["organs"]["fruit"]["relative_error"]
                ),
                "development_leaf_relative_error": float(
                    stage_development_metrics["organs"]["leaf"]["relative_error"]
                ),
                "development_stem_relative_error": float(
                    stage_development_metrics["organs"]["stem"]["relative_error"]
                ),
                "development_stage_multiplier_fruit": float(
                    development_row.iloc[0]["stage_sink_multiplier_fruit"]
                ),
                "development_stage_multiplier_leaf": float(
                    development_row.iloc[0]["stage_sink_multiplier_leaf"]
                ),
                "development_stage_multiplier_stem": float(
                    development_row.iloc[0]["stage_sink_multiplier_stem"]
                ),
            }
        )
        stage_details.append(
            {
                "scenario": str(structure["scenario"]),
                "schedule": schedule,
                "calibration_metrics": stage_calibration_metrics,
                "development_metrics": stage_development_metrics,
            }
        )
    stage_summary, stage_aligned = summarize_stage_sink_structure_comparison(
        stage_rows
    )
    stage_summary.update(
        {
            "reference_timestamp": calibration_date.isoformat(),
            "reference_t_can_sum_deg_day": stage_reference,
            "development_timestamp": validation_date.isoformat(),
            "maximum_multiplier": 2.0,
            "scenario_details": stage_details,
            "hypothesis_timing": "formed_after_inspecting_development_date_discrepancy",
        }
    )

    capacity_rows: list[dict[str, Any]] = []
    capacity_details: list[dict[str, Any]] = []
    registered_capacity_multipliers = [0.75, 1.0, 1.25, 1.5, 2.0]
    for capacity_multiplier in registered_capacity_multipliers:
        capacity_base_parameters = np.asarray(init_default_params(216), dtype=float)
        capacity_base_parameters[129] *= float(capacity_multiplier)
        capacity_calibration = calibrate_organ_allocation_proportional(
            climate=climate,
            controls=controls,
            weather=weather,
            initial_crop_state=initial_state,
            fruit_dry_matter_fraction=fruit_dry_fraction,
            calibration_timestamp=calibration_date,
            observed_dry_kg_m2=observed_organ_dry,
            base_parameters=capacity_base_parameters,
            initial_multipliers={
                organ: float(value)
                for organ, value in organ_calibration["best_multipliers"].items()
            },
            iterations=3,
            multiplier_bounds=(0.05, 2.0),
            substep_seconds=300,
        )
        final_multipliers = capacity_calibration["final_multipliers"]
        capacity_parameters = capacity_base_parameters.copy()
        for organ, parameter_index in {"fruit": 154, "leaf": 155, "stem": 156}.items():
            capacity_parameters[parameter_index] *= float(final_multipliers[organ])
        capacity_trajectory = simulate_forced_crop_trajectory(
            climate=climate,
            controls=controls,
            weather=weather,
            initial_crop_state=initial_state,
            fruit_dry_matter_fraction=fruit_dry_fraction,
            substep_seconds=300,
            parameters=capacity_parameters,
            include_flux_diagnostics=False,
        )
        capacity_calibration_metrics = evaluate_organ_dry_predictions(
            calibration_observations,
            capacity_trajectory,
            observation_timestamp=calibration_date,
            dry_matter_fractions=dry_matter_fractions,
        )
        capacity_validation_metrics = evaluate_organ_dry_predictions(
            validation_observations,
            capacity_trajectory,
            observation_timestamp=validation_date,
            dry_matter_fractions=dry_matter_fractions,
        )
        validation_row = capacity_trajectory.loc[
            capacity_trajectory["timestamp"].map(_local_timestamp).eq(validation_date)
        ]
        if len(validation_row) != 1:
            raise RuntimeError("assimilation capacity trajectory missed validation date")
        capacity_rows.append(
            {
                "capacity_multiplier": float(capacity_multiplier),
                "j25_leaf_max": float(capacity_parameters[129]),
                "rg_fruit_multiplier": float(final_multipliers["fruit"]),
                "rg_leaf_multiplier": float(final_multipliers["leaf"]),
                "rg_stem_multiplier": float(final_multipliers["stem"]),
                "calibration_organ_nrmse": float(
                    capacity_calibration_metrics["organ_normalized_rmse"]
                ),
                "validation_organ_nrmse": float(
                    capacity_validation_metrics["organ_normalized_rmse"]
                ),
                "validation_predicted_fresh_kg_m2": float(
                    validation_row.iloc[0]["standing_fresh_kg_m2"]
                ),
                "validation_observed_fresh_kg_m2": observed_validation_mean,
                "validation_fruit_relative_error": float(
                    capacity_validation_metrics["organs"]["fruit"]["relative_error"]
                ),
                "validation_leaf_relative_error": float(
                    capacity_validation_metrics["organs"]["leaf"]["relative_error"]
                ),
                "validation_stem_relative_error": float(
                    capacity_validation_metrics["organs"]["stem"]["relative_error"]
                ),
            }
        )
        capacity_details.append(
            {
                "capacity_multiplier": float(capacity_multiplier),
                "calibration": {
                    key: value
                    for key, value in capacity_calibration.items()
                    if key != "final_trajectory"
                },
                "calibration_metrics": capacity_calibration_metrics,
                "validation_metrics": capacity_validation_metrics,
            }
        )
    capacity_summary, capacity_aligned = summarize_assimilation_capacity_sensitivity(
        capacity_rows
    )
    capacity_summary.update(
        {
            "parameter": "j25LeafMax",
            "parameter_index": 129,
            "base_value": float(init_default_params(216)[129]),
            "organ_calibration_iterations": 3,
            "organ_multiplier_bounds": [0.05, 2.0],
            "calibration_timestamp": calibration_date.isoformat(),
            "validation_timestamp": validation_date.isoformat(),
            "scenario_details": capacity_details,
            "interpretation_scope": "capacity_sensitivity_after_calibration_date_only_sink_refit",
        }
    )
    organ_calibration_summary = {
        key: value for key, value in organ_calibration.items() if key != "best_trajectory"
    }
    organ_calibration_summary.update(
        {
            "split_policy": "first_post_baseline_date_fit_last_post_baseline_date_validation",
            "calibration_metrics": organ_calibration_metrics,
            "validation_metrics": organ_validation_metrics,
            "parameter_scope": "diagnostic_fixed_multi_organ_single_date_fit",
            "adoption_status": "diagnostic_not_final_target_parameterization",
        }
    )

    phenology_metrics, phenology_aligned = evaluate_temperature_driven_trusses(
        climate[["timestamp", "air_temperature"]],
        target_scored[["observation_date", "fruit_truss_count"]],
        baseline_timestamp=start,
    )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output / "target_crop_trajectory.csv", index=False)
    calibrated_trajectory.to_csv(
        output / "target_crop_trajectory_calibrated.csv", index=False
    )
    organ_trajectory.to_csv(
        output / "target_crop_trajectory_organ_calibrated.csv", index=False
    )
    aligned.to_csv(output / "standing_crop_aligned.csv", index=False)
    calibration_aligned.to_csv(
        output / "standing_crop_calibration_aligned.csv", index=False
    )
    validation_aligned.to_csv(
        output / "standing_crop_validation_aligned.csv", index=False
    )
    phenology_aligned.to_csv(output / "phenology_truss_aligned.csv", index=False)
    thermal_sum_aligned.to_csv(
        output / "initial_thermal_sum_sensitivity.csv", index=False
    )
    capacity_aligned.to_csv(
        output / "assimilation_capacity_sensitivity.csv", index=False
    )
    stage_aligned.to_csv(
        output / "stage_sink_structure_comparison.csv", index=False
    )
    (output / "target_crop_state_initialization.json").write_text(
        json.dumps(initialization_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "standing_crop_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "standing_crop_calibration.json").write_text(
        json.dumps(calibration_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "organ_allocation_calibration.json").write_text(
        json.dumps(organ_calibration_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "phenology_truss_metrics.json").write_text(
        json.dumps(phenology_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "initial_thermal_sum_sensitivity.json").write_text(
        json.dumps(thermal_sum_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "late_biomass_flux_diagnostics.json").write_text(
        json.dumps(late_biomass_flux_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "assimilation_capacity_sensitivity.json").write_text(
        json.dumps(capacity_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "stage_sink_structure_comparison.json").write_text(
        json.dumps(stage_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = {
        "target_greenhouse_id": target_id,
        "target_greenhouse_code": target_code,
        "simulation_start": start.isoformat(),
        "simulation_end": end.isoformat(),
        "trajectory_rows": int(len(trajectory)),
        "scored_observation_dates": int(metrics["observation_date_count"]),
        "standing_crop_metrics": metrics,
        "standing_crop_precalibration": calibration_summary,
        "fixed_multi_organ_calibration": organ_calibration_summary,
        "temperature_driven_phenology_validation": phenology_metrics,
        "initial_thermal_sum_sensitivity": thermal_sum_summary,
        "late_biomass_flux_diagnostics": late_biomass_flux_diagnostics,
        "assimilation_capacity_sensitivity": capacity_summary,
        "stage_sink_structure_comparison": stage_summary,
        "weather_source": "target_2026_observed_outdoor_weather",
        "control_source": "target_recorded_device_states_hourly_exact_replay",
        "climate_state_source": "target_observed_indoor_climate_forced_crop_model",
        "climate_input_audit": climate_audit,
        "coupled_climate_model_used_for_crop_scoring": False,
        "coupled_climate_exclusion_reason": (
            "chengdu_physics_multistep_trajectory_exhibited_nonphysical_temperature_and_integrator_failure"
        ),
        "uncertainty_scope": {
            "measurement_replicate_uncertainty_reported": True,
            "source_to_target_dry_fraction_transfer_reported": True,
            "dry_matter_fraction_conversion_sensitivity_reported": True,
            "initial_thermal_sum_structural_sensitivity_reported": True,
            "assimilation_capacity_structural_sensitivity_reported": True,
            "stage_sink_posthoc_structure_development_reported": True,
            "predictive_interval_available": False,
            "reason": "only_one_partial_target_season_and_no_target_harvest_residuals",
        },
        "initial_thermal_sum_limitation": {
            "current_value_deg_day": float(initial_state["tCanSum"]),
            "semantics_correction": "tCanSum_should_accumulate_canopy_temperature_since_transplant_not_since_first_fruit",
            "prebaseline_indoor_climate_available": False,
            "not_back_calibrated_from_held_out_mass": True,
        },
        "season_complete": False,
        "eligible_for_harvest_calibration": False,
        "target_harvest_validated": False,
        "status": "preliminary_target_standing_crop_validation",
    }
    (output / "target_crop_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _load_environment_kwargs(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    kwargs = dict(raw.get("GreenLightEnv", raw))
    repository_kwargs = kwargs.pop("weather_repository_kwargs")
    kwargs.pop("eval_scenarios", None)
    kwargs["weather_repository"] = WeatherRepository(
        weather_data_dir=repository_kwargs["weather_data_dir"],
        load_weather_data_fn=load_weather_data,
    )
    return kwargs


def _forced_trajectory_row(
    *,
    timestamp: pd.Timestamp,
    crop: np.ndarray,
    climate_row: pd.Series,
    dry_fraction: float,
    harvested_dry_kg_m2: float,
    net_fruit_change_kg_m2: float,
    floor_correction_mg_m2: float,
    interval_fluxes_kg_m2: dict[str, float] | None,
    crop_mass_balance_residual_kg_m2: float | None,
    stage_sink_multipliers: dict[str, float],
) -> dict[str, Any]:
    standing_dry = float(crop[3]) * 1e-6
    row = {
        "timestamp": timestamp,
        "standing_dry_kg_m2": standing_dry,
        "standing_fresh_kg_m2": standing_dry / float(dry_fraction),
        "native_harvested_dry_kg_m2": float(harvested_dry_kg_m2),
        "net_fruit_dry_matter_change_kg_m2": float(net_fruit_change_kg_m2),
        "c_buffer_mg_m2": float(crop[0]),
        "c_leaf_mg_m2": float(crop[1]),
        "c_stem_mg_m2": float(crop[2]),
        "canopy_thermal_sum_deg_day": float(crop[4]),
        "crop_time_days": float(crop[5]),
        "air_temperature_c": float(climate_row["air_temperature"]),
        "canopy_temperature_c": float(climate_row["canopy_temperature"]),
        "relative_humidity_percent": float(climate_row["relative_humidity"]),
        "co2_concentration_ppm": float(climate_row["co2_concentration"]),
        "nonnegative_floor_correction_mg_m2": float(floor_correction_mg_m2),
        **{
            f"stage_sink_multiplier_{organ}": float(multiplier)
            for organ, multiplier in stage_sink_multipliers.items()
        },
    }
    if interval_fluxes_kg_m2 is not None:
        row["crop_mass_balance_residual_kg_m2"] = float(
            crop_mass_balance_residual_kg_m2
        )
        for flux_name, column in FLUX_OUTPUT_COLUMNS.items():
            row[column] = float(interval_fluxes_kg_m2[flux_name])
    return row


def _trajectory_row(
    *,
    timestamp: pd.Timestamp,
    env: GreenLightEnv,
    fruit_dry_fraction: float,
    harvested_dry_matter_mg_m2: float,
    previous_fruit_mg_m2: float,
) -> dict[str, Any]:
    standing_dry = max(0.0, float(env.x[25])) * 1e-6
    harvested_dry = max(0.0, float(harvested_dry_matter_mg_m2)) * 1e-6
    net_change = (float(env.x[25]) - float(previous_fruit_mg_m2)) * 1e-6 + harvested_dry
    return {
        "timestamp": timestamp,
        "standing_dry_kg_m2": standing_dry,
        "standing_fresh_kg_m2": standing_dry / float(fruit_dry_fraction),
        "native_harvested_dry_kg_m2": harvested_dry,
        "net_fruit_dry_matter_change_kg_m2": net_change,
        "c_buffer_mg_m2": float(env.x[22]),
        "c_leaf_mg_m2": float(env.x[23]),
        "c_stem_mg_m2": float(env.x[24]),
        "indoor_air_temperature_c": float(env.x[2]),
        "canopy_temperature_c": float(env.x[4]),
        **{name: float(env.u[index]) for index, name in enumerate(CONTROL_NAMES)},
    }


def _local_timestamp(value: object, timezone: str = "Asia/Shanghai") -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the measured Pidu crop state against later standing-fruit samples"
    )
    parser.add_argument(
        "--crop-observations",
        default="results/chengdu_agri_greenhouse_001/harvest_model/processed/standing_crop_samples.csv",
    )
    parser.add_argument(
        "--controls",
        default="data/processed/chengdu_agri/greenhouse_001/controls/controls_1h.csv",
    )
    parser.add_argument(
        "--aligned-climate",
        default="data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv",
    )
    parser.add_argument(
        "--env-config", default="configs/envs/ChengduSingleGreenhouseEnv.yml"
    )
    parser.add_argument(
        "--harvest-config", default="configs/crops/chengdu_tomato_harvest.yml"
    )
    parser.add_argument(
        "--output-root",
        default="results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation",
    )
    parser.add_argument("--baseline-date", default="2026-03-30")
    parser.add_argument("--simulation-start", default="2026-04-01 00:00:00")
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()
    report = run_target_crop_validation(
        crop_observations_path=args.crop_observations,
        controls_path=args.controls,
        aligned_climate_path=args.aligned_climate,
        env_config_path=args.env_config,
        harvest_config_path=args.harvest_config,
        output_root=args.output_root,
        baseline_date=args.baseline_date,
        simulation_start=args.simulation_start,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
