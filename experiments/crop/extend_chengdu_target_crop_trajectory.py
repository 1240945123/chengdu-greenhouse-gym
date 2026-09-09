from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.crop.run_chengdu_target_crop_validation import (
    _load_environment_kwargs,
    _local_timestamp,
    prepare_observed_climate,
    prepare_observed_control_targets,
    simulate_forced_crop_trajectory,
)
from glassgym.configs.default_params import init_default_params
from glassgym.environments.utils import (
    co2ppm2dens,
    computeisDay,
    dailLightSum,
    rh2vaporDens,
    soilTempNl,
    vaporDens2pres,
)
from processing.chengdu_crop_initialization import build_target_crop_initial_state


DEFAULT_OUTPUT_ROOT = Path(
    "results/chengdu_agri_greenhouse_001/harvest_model/target_crop_validation"
)
TRAJECTORY_FILENAME = "target_crop_trajectory_driver_horizon.csv"
AUDIT_FILENAME = "target_crop_trajectory_driver_horizon_audit.json"


def run_driver_horizon_extension(
    *,
    crop_observations_path: str | Path,
    controls_path: str | Path,
    aligned_climate_path: str | Path,
    env_config_path: str | Path,
    harvest_config_path: str | Path,
    organ_calibration_path: str | Path,
    output_root: str | Path,
    simulation_start: str = "2026-04-01 00:00:00",
    simulation_end: str = "2026-07-20 09:00:00",
    baseline_date: str = "2026-03-30",
    substep_seconds: int = 300,
) -> dict[str, Any]:
    start = _local_timestamp(simulation_start)
    end = _local_timestamp(simulation_end)
    if end <= start:
        raise ValueError("simulation end must be after simulation start")
    if int(substep_seconds) <= 0 or 3600 % int(substep_seconds) != 0:
        raise ValueError("substep_seconds must be a positive divisor of one hour")

    observations = pd.read_csv(crop_observations_path)
    harvest_config = _load_mapping(harvest_config_path)
    target = harvest_config["target_site"]
    target_id = int(target["greenhouse_id"])
    target_code = str(target["greenhouse_code"])
    initial_state, initialization_audit = build_target_crop_initial_state(
        observations,
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
        baseline_date=baseline_date,
    )
    dry_fraction = float(
        initialization_audit["dry_matter_fraction_transfer"]["ripe_fruit"][
            "estimate"
        ]
    )

    aligned_climate, canopy_proxy_audit = fill_long_canopy_temperature_gaps(
        pd.read_csv(aligned_climate_path)
    )
    climate, climate_audit = prepare_observed_climate(
        aligned_climate, start=start, end=end
    )
    controls = prepare_observed_control_targets(
        pd.read_csv(controls_path),
        start=start,
        end=end,
        dt_seconds=3600,
    )
    calibration = _load_mapping(organ_calibration_path)
    multipliers, parameter_indices, parameters = _diagnostic_parameters(calibration)

    duration_days = float((end - start).total_seconds() / 86400.0)
    environment = _load_environment_kwargs(env_config_path)
    weather_path = (
        environment["weather_repository"].weather_data_dir
        / "Chengdu"
        / "2026.csv"
    )
    weather, weather_audit = load_observed_weather_horizon(
        weather_path,
        simulation_start=start,
        simulation_end=end,
    )
    trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial_state,
        fruit_dry_matter_fraction=dry_fraction,
        substep_seconds=int(substep_seconds),
        parameters=parameters,
        include_flux_diagnostics=True,
    )
    trajectory_start = _local_timestamp(trajectory["timestamp"].iloc[0])
    trajectory_end = _local_timestamp(trajectory["timestamp"].iloc[-1])
    if trajectory_start != start or trajectory_end != end:
        raise RuntimeError("extended crop trajectory did not cover the requested horizon")
    numeric = trajectory.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("extended crop trajectory contains non-finite values")

    residual = trajectory["crop_mass_balance_residual_kg_m2"].to_numpy(dtype=float)
    report: dict[str, Any] = {
        "target_greenhouse_id": target_id,
        "target_greenhouse_code": target_code,
        "simulation_start": start.isoformat(),
        "simulation_end": end.isoformat(),
        "trajectory_rows": int(len(trajectory)),
        "span_days": duration_days,
        "substep_seconds": int(substep_seconds),
        "organ_allocation_multipliers": multipliers,
        "organ_parameter_indices": parameter_indices,
        "parameter_source": str(Path(organ_calibration_path).resolve()),
        "parameter_calibration_timestamp": calibration.get(
            "calibration_timestamp"
        ),
        "parameter_validation_metrics": calibration.get("validation_metrics"),
        "initial_state_source": str(Path(crop_observations_path).resolve()),
        "fruit_dry_matter_fraction": dry_fraction,
        "climate_source": str(Path(aligned_climate_path).resolve()),
        "control_source": str(Path(controls_path).resolve()),
        "weather_source": str(weather_path.resolve()),
        "weather_input_audit": weather_audit,
        "climate_input_audit": climate_audit,
        "canopy_temperature_proxy_audit": canopy_proxy_audit,
        "nonnegative_floor_correction_total_mg_m2": float(
            trajectory["nonnegative_floor_correction_mg_m2"].sum()
        ),
        "mass_balance_residual_absolute_sum_kg_m2": float(
            np.abs(residual).sum()
        ),
        "mass_balance_residual_max_abs_kg_m2": float(np.abs(residual).max()),
        "driver_model_status": "diagnostic_not_final",
        "season_complete": False,
        "target_harvest_validated": False,
        "use": "driver_preparation_only_not_calibration_evidence",
        "limitations": [
            "single_date_multi_organ_parameter_fit",
            "held_out_standing_crop_acceptance_failed",
            "post_validation_period_is_unvalidated_extrapolation",
            "no_target_harvest_events_used_or_available",
            "canopy_temperature_proxy_uncertainty_not_propagated",
        ],
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / TRAJECTORY_FILENAME
    audit_path = output / AUDIT_FILENAME
    report["trajectory_artifact"] = str(trajectory_path.resolve())
    report["audit_artifact"] = str(audit_path.resolve())
    trajectory.to_csv(trajectory_path, index=False)
    audit_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def load_observed_weather_horizon(
    weather_path: str | Path,
    *,
    simulation_start: str | pd.Timestamp,
    simulation_end: str | pd.Timestamp,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Transform an exact partial-year observed weather slice for GreenLight."""
    start = _local_timestamp(simulation_start)
    end = _local_timestamp(simulation_end)
    if end < start:
        raise ValueError("weather horizon end must not precede start")
    raw = pd.read_csv(weather_path)
    required = {
        "time",
        "global radiation",
        "wind speed",
        "air temperature",
        "sky temperature",
        "CO2 concentration",
        "RH",
    }
    if missing := sorted(required - set(raw.columns)):
        raise ValueError(f"observed weather missing columns: {', '.join(missing)}")
    for column in required:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if not np.isfinite(raw[list(required)].to_numpy(dtype=float)).all():
        raise ValueError("observed weather values must be finite")
    time = raw["time"].to_numpy(dtype=float)
    if len(time) < 2:
        raise ValueError("observed weather must contain at least two rows")
    intervals = np.diff(time)
    if not np.allclose(intervals, 3600.0, rtol=0.0, atol=1e-6):
        raise ValueError("observed weather must use fixed hourly intervals")

    def seconds_since_year_start(timestamp: pd.Timestamp) -> float:
        return float(
            (timestamp.dayofyear - 1) * 86400
            + timestamp.hour * 3600
            + timestamp.minute * 60
            + timestamp.second
        )

    start_seconds = seconds_since_year_start(start)
    end_seconds = seconds_since_year_start(end)
    selected = raw[raw["time"].between(start_seconds, end_seconds)].copy()
    expected_rows = int(round((end - start).total_seconds() / 3600.0)) + 1
    if len(selected) != expected_rows:
        raise ValueError("observed weather does not exactly cover the requested hours")
    selected_time = selected["time"].to_numpy(dtype=float)
    if not np.isclose(selected_time[0], start_seconds) or not np.isclose(
        selected_time[-1], end_seconds
    ):
        raise ValueError("observed weather endpoints do not match the requested horizon")

    air = selected["air temperature"].to_numpy(dtype=float)
    radiation = selected["global radiation"].to_numpy(dtype=float)
    rh = selected["RH"].to_numpy(dtype=float)
    co2_ppm = selected["CO2 concentration"].to_numpy(dtype=float)
    weather = np.empty((len(selected), 10), dtype=float)
    weather[:, 0] = radiation
    weather[:, 1] = air
    weather[:, 2] = vaporDens2pres(air, rh2vaporDens(air, rh))
    weather[:, 3] = co2ppm2dens(air, co2_ppm) * 1e6
    weather[:, 4] = selected["wind speed"].to_numpy(dtype=float)
    weather[:, 5] = selected["sky temperature"].to_numpy(dtype=float)
    weather[:, 6] = soilTempNl(selected_time)
    weather[:, 7] = dailLightSum(selected_time, radiation, 86400)
    weather[:, 8], weather[:, 9] = computeisDay(radiation, 3600.0)
    if not np.isfinite(weather).all():
        raise ValueError("transformed observed weather contains non-finite values")
    return weather, {
        "weather_rows": int(len(weather)),
        "source_start_time_seconds": float(selected_time[0]),
        "source_end_time_seconds": float(selected_time[-1]),
        "source_start_day_number": int(np.floor(selected_time[0] / 86400.0)) + 1,
        "source_end_day_number": int(np.floor(selected_time[-1] / 86400.0)) + 1,
        "fixed_interval_seconds": 3600,
        "partial_final_day_supported": bool(end_seconds % 86400 != 0),
        "transformation": "greenlight_weather_disturbance_equations_exact_rows",
    }


def fill_long_canopy_temperature_gaps(
    aligned: pd.DataFrame,
    *,
    maximum_interpolation_gap_hours: int = 8,
    local_reference_window_days: int = 14,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill only long canopy gaps from local hour-specific canopy-air deltas."""
    required = {"timestamp", "air_temperature", "canopy_temperature"}
    if missing := sorted(required - set(aligned.columns)):
        raise ValueError(f"canopy proxy input missing columns: {', '.join(missing)}")
    if int(maximum_interpolation_gap_hours) < 1:
        raise ValueError("maximum_interpolation_gap_hours must be positive")
    if int(local_reference_window_days) < 1:
        raise ValueError("local_reference_window_days must be positive")
    frame = aligned.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any() or frame["timestamp"].duplicated().any():
        raise ValueError("canopy proxy timestamps must be valid and unique")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    air = pd.to_numeric(frame["air_temperature"], errors="coerce")
    canopy = pd.to_numeric(frame["canopy_temperature"], errors="coerce")
    missing_canopy = canopy.isna()
    group_ids = missing_canopy.ne(missing_canopy.shift()).cumsum()
    long_runs: list[np.ndarray] = []
    for _, run in frame.loc[missing_canopy].groupby(group_ids[missing_canopy]):
        positions = run.index.to_numpy(dtype=int)
        if len(positions) > int(maximum_interpolation_gap_hours):
            long_runs.append(positions)

    filled_positions: list[int] = []
    reference_errors: list[float] = []
    gap_audit: list[dict[str, Any]] = []
    window = pd.Timedelta(days=int(local_reference_window_days))
    for positions in long_runs:
        gap_start = frame.loc[positions[0], "timestamp"]
        gap_end = frame.loc[positions[-1], "timestamp"]
        reference = (
            frame["timestamp"].between(gap_start - window, gap_end + window)
            & ~frame.index.isin(positions)
            & air.notna()
            & canopy.notna()
        )
        if not reference.any():
            raise ValueError("long canopy gap has no local canopy-air reference data")
        reference_delta = canopy[reference] - air[reference]
        reference_hours = frame.loc[reference, "timestamp"].dt.hour
        hourly_delta = reference_delta.groupby(reference_hours).median()
        global_delta = float(reference_delta.median())
        gap_hours = frame.loc[positions, "timestamp"].dt.hour
        proxy_delta = gap_hours.map(hourly_delta).fillna(global_delta).to_numpy(dtype=float)
        if air.iloc[positions].isna().any() or not np.isfinite(proxy_delta).all():
            raise ValueError("long canopy gap proxy requires finite indoor air temperature")
        canopy.iloc[positions] = air.iloc[positions].to_numpy(dtype=float) + proxy_delta
        reference_prediction = (
            air[reference]
            + reference_hours.map(hourly_delta).fillna(global_delta).to_numpy(dtype=float)
        )
        reference_errors.extend(
            (reference_prediction - canopy[reference]).to_numpy(dtype=float).tolist()
        )
        filled_positions.extend(positions.tolist())
        gap_audit.append(
            {
                "start": pd.Timestamp(gap_start).isoformat(),
                "end": pd.Timestamp(gap_end).isoformat(),
                "rows": int(len(positions)),
                "reference_rows": int(reference.sum()),
                "reference_window_days": int(local_reference_window_days),
            }
        )
    frame["canopy_temperature"] = canopy
    errors = np.asarray(reference_errors, dtype=float)
    return frame, {
        "method": "local_hourly_median_canopy_minus_air_delta",
        "maximum_short_gap_interpolation_hours": int(
            maximum_interpolation_gap_hours
        ),
        "local_reference_window_days": int(local_reference_window_days),
        "long_gap_count": int(len(long_runs)),
        "proxy_filled_rows": int(len(filled_positions)),
        "proxy_reference_mae_c": (
            float(np.mean(np.abs(errors))) if len(errors) else 0.0
        ),
        "proxy_reference_rmse_c": (
            float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0
        ),
        "uncertainty_propagated": False,
        "gaps": gap_audit,
    }


def _diagnostic_parameters(
    calibration: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int], np.ndarray]:
    organs = {"fruit", "leaf", "stem"}
    raw_multipliers = calibration.get("best_multipliers")
    raw_indices = calibration.get("parameter_indices")
    if not isinstance(raw_multipliers, dict) or set(raw_multipliers) != organs:
        raise ValueError("organ calibration must contain fruit, leaf, and stem multipliers")
    if not isinstance(raw_indices, dict) or set(raw_indices) != organs:
        raise ValueError("organ calibration must contain fruit, leaf, and stem indices")
    multipliers = {organ: float(raw_multipliers[organ]) for organ in sorted(organs)}
    indices = {organ: int(raw_indices[organ]) for organ in sorted(organs)}
    values = np.asarray(list(multipliers.values()), dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("organ allocation multipliers must be finite and positive")
    if len(set(indices.values())) != len(organs) or any(
        index < 0 or index >= 216 for index in indices.values()
    ):
        raise ValueError("organ parameter indices must be unique and within the model")
    parameters = np.asarray(init_default_params(216), dtype=float)
    for organ in organs:
        parameters[indices[organ]] *= multipliers[organ]
    return multipliers, indices, parameters


def _load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        content = json.loads(source.read_text(encoding="utf-8"))
    else:
        content = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extend the diagnostic Pidu crop trajectory over observed inputs."
    )
    parser.add_argument(
        "--crop-observations",
        default=(
            "results/chengdu_agri_greenhouse_001/harvest_model/processed/"
            "standing_crop_samples.csv"
        ),
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
        "--organ-calibration",
        default=(
            "results/chengdu_agri_greenhouse_001/harvest_model/"
            "target_crop_validation/organ_allocation_calibration.json"
        ),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--simulation-start", default="2026-04-01 00:00:00")
    parser.add_argument("--simulation-end", default="2026-07-20 09:00:00")
    parser.add_argument("--baseline-date", default="2026-03-30")
    parser.add_argument("--substep-seconds", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_driver_horizon_extension(
        crop_observations_path=args.crop_observations,
        controls_path=args.controls,
        aligned_climate_path=args.aligned_climate,
        env_config_path=args.env_config,
        harvest_config_path=args.harvest_config,
        organ_calibration_path=args.organ_calibration,
        output_root=args.output_root,
        simulation_start=args.simulation_start,
        simulation_end=args.simulation_end,
        baseline_date=args.baseline_date,
        substep_seconds=args.substep_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
