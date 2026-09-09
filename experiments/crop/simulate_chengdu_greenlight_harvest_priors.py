from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.stats import qmc

from experiments.controllers.benchmark_protocol import (
    build_environment,
    full_control_target_to_action,
    load_benchmark_config,
)
from experiments.controllers.tune_benchmark_classical import (
    _agent_params,
    _step_context,
    build_classical_controller,
    resolve_classical_params,
)
from experiments.crop.calibrate_harvest_model import (
    build_harvest_drivers_from_greenlight,
)
from glassgym.models.harvest import (
    HarvestCohortModel,
    HarvestCohortParameters,
    ReproductiveHeatStressParameters,
    trailing_heat_retention,
)


CLIMATE_SCENARIOS = {
    "calibrated_low": 0.75,
    "calibrated": 1.0,
    "calibrated_high": 1.25,
}
DEFAULT_PROCESS_PRIOR_PATH = Path(
    "data/processed/external/wur_agc2/wur_process_priors.json"
)
TARGET_DRY_MATTER_FRACTION_BOUNDS = (
    0.07582388370007698,
    0.08461793001033059,
)


def load_transfer_priors(path: str | Path = DEFAULT_PROCESS_PRIOR_PATH) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "external_observed":
        raise ValueError("WUR process prior must be external_observed evidence")
    if bool(payload.get("target_eligible", True)):
        raise ValueError("WUR process prior must remain target-ineligible")
    if payload.get("transfer_role") != "external_process_prior_not_target_calibration":
        raise ValueError("WUR process prior has an invalid transfer role")
    metrics = payload.get("metrics", {})
    maturity = metrics.get("maturity_thermal_time_deg_day", {})
    pick = metrics.get("median_interpick_days", {})
    maturity_bounds = (float(maturity["ci95_low"]), float(maturity["ci95_high"]))
    if not 0.0 < maturity_bounds[0] < maturity_bounds[1]:
        raise ValueError("WUR maturity thermal-time interval is invalid")
    pick_days = int(round(float(pick["estimate"])))
    if pick_days <= 0:
        raise ValueError("WUR picking interval is invalid")
    return {
        "maturity_thermal_time_deg_day": maturity_bounds,
        "pick_interval_days": (pick_days, pick_days),
        "source_dois": list(payload.get("source_dois", [])),
        "source_path": str(Path(path).as_posix()),
        "transfer_calibration_status": (
            "external_transfer_calibrated_not_target_validated"
        ),
    }


def _run_greenlight_task(
    season_data: dict[str, object],
    climate_scenario: str,
    strength: float,
    seed: int,
) -> tuple[tuple[str, str], pd.DataFrame]:
    season = pd.Series(season_data)
    frame = run_greenlight_season(
        season,
        climate_scenario=climate_scenario,
        climate_calibration_strength=strength,
        seed=seed,
    )
    return (str(season["season_id"]), climate_scenario), frame


def collect_greenlight_trajectories(
    seasons: pd.DataFrame,
    *,
    seed: int,
    max_workers: int,
    trajectory_runner: Callable[..., pd.DataFrame] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Collect all season/climate trajectories, optionally in worker processes."""
    if int(max_workers) <= 0:
        raise ValueError("max_workers must be positive")
    tasks = [
        (pd.Series(row._asdict()), climate_name, strength)
        for row in seasons.itertuples(index=False)
        for climate_name, strength in CLIMATE_SCENARIOS.items()
    ]
    if trajectory_runner is not None or int(max_workers) == 1:
        runner = run_greenlight_season if trajectory_runner is None else trajectory_runner
        return {
            (str(season["season_id"]), climate_name): runner(
                season,
                climate_scenario=climate_name,
                climate_calibration_strength=strength,
                seed=int(seed),
            )
            for season, climate_name, strength in tasks
        }
    collected: dict[tuple[str, str], pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
        futures = {
            executor.submit(
                _run_greenlight_task,
                season.to_dict(),
                climate_name,
                strength,
                int(seed),
            ): (str(season["season_id"]), climate_name)
            for season, climate_name, strength in tasks
        }
        for future in as_completed(futures):
            key, frame = future.result()
            collected[key] = frame
    return collected


def write_ensemble_artifacts(
    output_dir: str | Path,
    *,
    frames: dict[str, pd.DataFrame],
    audit: dict[str, object],
) -> list[Path]:
    """Atomically write the separated diagnostic ensemble artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, frame in frames.items():
        path = output / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        written.append(path)
    audit_path = output / "simulation_audit.json"
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    temporary_audit.replace(audit_path)
    written.append(audit_path)
    return written


def load_greenlight_trajectory_cache(
    path: str | Path,
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load a completed combined trajectory artifact for cohort-only reruns."""
    frame = pd.read_csv(path)
    required = {"season_id", "climate_scenario", "timestep"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trajectory cache missing columns: {missing}")
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for key, group in frame.groupby(["season_id", "climate_scenario"], sort=False):
        cache[(str(key[0]), str(key[1]))] = group.sort_values(
            "timestep", kind="stable"
        ).reset_index(drop=True)
    return cache


def assemble_high_fidelity_ensemble(
    seasons: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    trajectory_runner: Callable[..., pd.DataFrame] | None = None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run each climate trajectory once and expand cohort parameter scenarios."""
    if trajectory_runner is None:
        trajectory_runner = run_greenlight_season
    required_seasons = {"season_id", "growth_year", "start_day", "season_days"}
    missing_seasons = sorted(required_seasons - set(seasons.columns))
    if missing_seasons:
        raise ValueError(f"season definitions missing columns: {missing_seasons}")
    required_samples = {
        "scenario_id",
        "season_id",
        "climate_scenario",
        "climate_calibration_strength",
    }
    missing_samples = sorted(required_samples - set(samples.columns))
    if missing_samples:
        raise ValueError(f"sampled parameters missing columns: {missing_samples}")
    season_lookup = {
        str(row.season_id): pd.Series(row._asdict())
        for row in seasons.itertuples(index=False)
    }
    unknown = sorted(set(samples["season_id"].astype(str)) - set(season_lookup))
    if unknown:
        raise ValueError(f"sampled parameters contain unknown seasons: {unknown}")

    trajectory_frames: list[pd.DataFrame] = []
    driver_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    trajectory_completeness: dict[str, bool] = {}
    for sample in samples.itertuples(index=False):
        sample_series = pd.Series(sample._asdict())
        season_id = str(sample.season_id)
        climate_name = str(sample.climate_scenario)
        cache_key = (season_id, climate_name)
        if cache_key not in cache:
            season = season_lookup[season_id]
            trajectory = trajectory_runner(
                season,
                climate_scenario=climate_name,
                climate_calibration_strength=float(sample.climate_calibration_strength),
                seed=int(seed),
            ).copy()
            expected_rows = int(season["season_days"]) * 24
            trajectory_id = f"{season_id}__{climate_name}"
            trajectory_completeness[trajectory_id] = len(trajectory) == expected_rows
            if len(trajectory) != expected_rows:
                raise ValueError(
                    f"trajectory {trajectory_id} has {len(trajectory)} rows; "
                    f"expected {expected_rows}"
                )
            trajectory["trajectory_id"] = trajectory_id
            trajectory["evidence_class"] = "simulated_prior"
            trajectory["target_eligible"] = False
            cache[cache_key] = trajectory
            trajectory_frames.append(trajectory)
        drivers, events, summary = simulate_cohort_scenario(
            cache[cache_key], sample_series
        )
        drivers["scenario_id"] = str(sample.scenario_id)
        drivers["climate_scenario"] = climate_name
        drivers["evidence_class"] = "simulated_prior"
        drivers["target_eligible"] = False
        driver_frames.append(drivers)
        event_frames.append(events)
        summaries.append(summary)

    trajectories = pd.concat(trajectory_frames, ignore_index=True)
    drivers = pd.concat(driver_frames, ignore_index=True)
    non_empty_events = [frame for frame in event_frames if not frame.empty]
    events = (
        pd.concat(non_empty_events, ignore_index=True)
        if non_empty_events
        else pd.DataFrame(
            columns=(
                "synthetic_event_id",
                "scenario_id",
                "evidence_class",
                "target_eligible",
            )
        )
    )
    summary_frame = pd.DataFrame(summaries)
    max_balance_error = float(
        summary_frame["dry_matter_balance_error_kg_m2"].abs().max()
    )
    audit: dict[str, object] = {
        "method": "chengdu_physics_greenlight_cohort_ensemble",
        "seed": int(seed),
        "season_count": int(seasons["season_id"].nunique()),
        "trajectory_count": int(len(cache)),
        "scenario_count": int(len(summary_frame)),
        "event_count": int(len(events)),
        "pipeline": {
            "weather_source": "ERA5_reanalysis_not_site_observation",
            "climate_backend": "ChengduPhysics_calibrated_sensitivity",
            "crop_backend": "Vanthoor2011_GreenLight",
            "harvest_backend": "dry_matter_age_cohort",
            "controller": "baseline_rule_based_heating_and_co2_disabled",
            "timestep_seconds": 3600,
        },
        "sampling_design": {
            "method": "latin_hypercube_for_cohort_parameters",
            "paired_across_climate_scenarios": bool(
                "cohort_draw" in samples.columns
                and samples.groupby(["season_id", "cohort_draw"])[
                    "climate_scenario"
                ].nunique().eq(len(CLIMATE_SCENARIOS)).all()
            ),
            "climate_sensitivity_interpretation": "sensitivity_levels_not_confidence_intervals",
        },
        "quality_gates": {
            "all_trajectories_complete": bool(all(trajectory_completeness.values())),
            "all_mass_balances_closed": bool(max_balance_error <= 1e-8),
            "maximum_abs_dry_matter_balance_error_kg_m2": max_balance_error,
            "all_synthetic_identifiers": bool(
                events.empty or events["synthetic_event_id"].astype(str).str.startswith("SIM_").all()
            ),
            "all_target_ineligible": bool(
                events.empty or not events["target_eligible"].astype(bool).any()
            ),
        },
        "evidence_class": "simulated_prior",
        "target_eligible": False,
        "limitations": [
            "diagnostic_model_ensemble_not_target_harvest_validation",
            "era5_weather_not_target_greenhouse_weather_observation",
            "climate_sensitivity_levels_are_not_statistical_confidence_intervals",
            "harvest_parameters_transfer_from_external_wur_prior",
            "partial_120_day_seasons",
        ],
    }
    return trajectories, drivers, events, summary_frame, audit


def run_greenlight_season(
    season: pd.Series,
    *,
    climate_scenario: str,
    climate_calibration_strength: float,
    seed: int,
    max_steps: int | None = None,
) -> pd.DataFrame:
    """Run one ChengduPhysics/GreenLight season under the baseline controller."""
    if climate_scenario not in CLIMATE_SCENARIOS:
        raise ValueError(f"unknown climate scenario: {climate_scenario}")
    expected_strength = CLIMATE_SCENARIOS[climate_scenario]
    if not np.isclose(float(climate_calibration_strength), expected_strength):
        raise ValueError("climate scenario and calibration strength do not match")
    config = load_benchmark_config("smoke")
    env = build_environment(
        config,
        int(season["start_day"]),
        episode_days=int(season["season_days"]),
        growth_year=int(season["growth_year"]),
        dt_seconds=3600,
        calibration_strength=float(climate_calibration_strength),
    )
    params = resolve_classical_params("baseline", _agent_params("baseline"), env)
    controller = build_classical_controller("baseline", params, env)
    rows: list[dict[str, object]] = []
    start = pd.Timestamp(year=int(season["growth_year"]), month=1, day=1) + pd.Timedelta(
        days=int(season["start_day"])
    )
    env.reset(seed=int(seed))
    limit = min(env.N, int(max_steps) if max_steps is not None else env.N)
    try:
        for timestep in range(limit):
            target = np.asarray(controller.predict(_step_context(env)), dtype=np.float32)
            action = full_control_target_to_action(
                env.u,
                target,
                delta=env.delta_u_max[env.action_scheme.controlled_idx],
            )
            _observation, _reward, terminated, truncated, info = env.step(action)
            controls = np.asarray(info["controls"], dtype=float)
            indoor = np.asarray(env.obs["IndoorClimateObservations"], dtype=float)
            row = {
                "season_id": str(season["season_id"]),
                "climate_scenario": str(climate_scenario),
                "climate_calibration_strength": float(climate_calibration_strength),
                "growth_year": int(season["growth_year"]),
                "start_day": int(season["start_day"]),
                "timestep": int(timestep),
                "timestamp": start + pd.Timedelta(seconds=(timestep + 1) * env.dt),
                "air_temperature": float(indoor[1]),
                "relative_humidity": float(indoor[2]),
                "crop_model": str(info["crop_model"]),
                **{
                    name: float(controls[index])
                    for index, name in enumerate(
                        ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")
                    )
                },
                **{
                    key: float(info[key])
                    for key in (
                        "fruit_harvest_rate_mg_m2_s",
                        "fruit_allocation_rate_mg_m2_s",
                        "harvested_dry_matter_mg_m2",
                        "allocated_fruit_dry_matter_mg_m2",
                        "dry_matter_fraction",
                        "c_buffer_mg_m2",
                        "c_leaf_mg_m2",
                        "c_stem_mg_m2",
                        "c_fruit_mg_m2",
                        "c_fruit_previous_mg_m2",
                    )
                },
            }
            if row["uBoil"] != 0.0 or row["uCO2"] != 0.0:
                raise RuntimeError("heating and CO2 controls must remain disabled")
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()
    return pd.DataFrame(rows)


def canonical_chengdu_seasons() -> pd.DataFrame:
    """Return the audited 2023-2025 Chengdu 120-day weather windows."""
    return pd.DataFrame(
        [
            ("2023_spring", 2023, 59),
            ("2023_autumn", 2023, 226),
            ("2024_spring", 2024, 60),
            ("2024_autumn", 2024, 227),
            ("2025_spring", 2025, 59),
            ("2025_autumn", 2025, 226),
        ],
        columns=("season_id", "growth_year", "start_day"),
    ).assign(season_days=120)


def sample_harvest_scenarios(
    season_ids: Iterable[str],
    *,
    draws_per_climate: int,
    seed: int,
    process_priors: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Stratify each weather season across climate sensitivity and cohort priors."""
    seasons = [str(value) for value in season_ids]
    if not seasons or len(set(seasons)) != len(seasons):
        raise ValueError("season_ids must be non-empty and unique")
    if int(draws_per_climate) <= 0:
        raise ValueError("draws_per_climate must be positive")
    base_draws = [
        (season, draw)
        for season in seasons
        for draw in range(int(draws_per_climate))
    ]
    unit = qmc.LatinHypercube(d=5, seed=int(seed)).random(len(base_draws))
    sampled_values = {
        (season, draw): (index, values)
        for index, ((season, draw), values) in enumerate(
            zip(base_draws, unit, strict=True)
        )
    }
    combinations = [
        (season, climate_name, strength, draw)
        for season in seasons
        for climate_name, strength in CLIMATE_SCENARIOS.items()
        for draw in range(int(draws_per_climate))
    ]
    priors = load_transfer_priors() if process_priors is None else process_priors
    global_maturity_bounds = tuple(
        float(value) for value in priors["maturity_thermal_time_deg_day"]
    )
    maturity_by_season = {
        str(season): tuple(float(value) for value in bounds)
        for season, bounds in dict(
            priors.get("maturity_thermal_time_deg_day_by_season", {})
        ).items()
    }
    pick_low, pick_high = (int(value) for value in priors["pick_interval_days"])
    dmf_low, dmf_high = TARGET_DRY_MATTER_FRACTION_BOUNDS
    rows: list[dict[str, object]] = []
    for season, climate_name, strength, draw in combinations:
        lhs_index, values = sampled_values[(season, draw)]
        maturity_low, maturity_high = maturity_by_season.get(
            season, global_maturity_bounds
        )
        if not 0.0 < maturity_low < maturity_high:
            raise ValueError(f"maturity thermal-time interval is invalid for {season}")
        rows.append(
            {
                "scenario_id": f"SIM_{season}_{climate_name}_{draw:03d}",
                "season_id": season,
                "climate_scenario": climate_name,
                "climate_calibration_strength": float(strength),
                "base_temperature_c": 10.0,
                "maturity_thermal_time_deg_day": float(
                    maturity_low + (maturity_high - maturity_low) * values[0]
                ),
                "dry_matter_fraction": float(dmf_low + (dmf_high - dmf_low) * values[1]),
                "minimum_pick_fresh_kg_m2": float(0.15 * values[2]),
                "pick_interval_days": int(
                    pick_low
                    if pick_low == pick_high
                    else pick_low + np.floor((pick_high - pick_low + 1.0) * values[3])
                ),
                "pick_hour": int(7 + np.floor(4.0 * values[4])),
                "initial_fruit_maturity_fraction": 0.0,
                "cohort_draw": int(draw),
                "lhs_index": int(lhs_index),
                "sampling_seed": int(seed),
                "maturity_prior_source": str(priors.get("source_path", "external_observed")),
                "maturity_prior_scope": (
                    "season_specific" if season in maturity_by_season else "global"
                ),
                "maturity_prior_dois": ";".join(priors.get("source_dois", [])),
                "transfer_calibration_status": str(
                    priors["transfer_calibration_status"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _scheduled_pick_mask(
    timestamps: pd.Series,
    *,
    interval_days: int,
    local_hour: int,
) -> pd.Series:
    times = pd.to_datetime(timestamps, errors="coerce")
    if times.isna().any():
        raise ValueError("trajectory timestamps must be valid")
    if interval_days <= 0 or not 0 <= local_hour <= 23:
        raise ValueError("pick schedule is invalid")
    first_day = times.iloc[0].normalize()
    elapsed_days = (times.dt.normalize() - first_day).dt.days
    return elapsed_days.mod(int(interval_days)).eq(0) & times.dt.hour.eq(int(local_hour))


def _simulate_fractional_loss_harvest(
    drivers: pd.DataFrame,
    parameters: HarvestCohortParameters,
) -> pd.DataFrame:
    """Age gross allocation and transfer source losses as inventory fractions."""
    first = drivers.iloc[0]
    model = HarvestCohortModel(
        parameters,
        initial_fruit_dry_matter_kg_m2=float(
            first["initial_fruit_dry_matter_kg_m2"]
        ),
        initial_fruit_maturity_fraction=float(
            first["initial_fruit_maturity_fraction"]
        ),
    )
    rows: list[dict[str, object]] = []
    cumulative_loss = 0.0
    cumulative_stress_loss = 0.0
    for row in drivers.itertuples(index=False):
        gross_allocation = float(row.gross_fruit_allocation_kg_m2)
        retention = float(row.reproductive_heat_retention)
        retained_allocation = gross_allocation * retention
        stress_loss = gross_allocation - retained_allocation
        growth_result = model.step(
            net_fruit_dry_matter_change_kg_m2=retained_allocation,
            air_temperature_c=float(row.air_temperature_c),
            dt_hours=float(row.dt_hours),
            pick=False,
        )
        applied_loss = (
            float(row.source_nonharvest_loss_fraction)
            * growth_result.standing_dry_matter_kg_m2
        )
        final_result = model.step(
            net_fruit_dry_matter_change_kg_m2=-applied_loss,
            air_temperature_c=float(row.air_temperature_c),
            dt_hours=0.0,
            pick=bool(row.pick),
        )
        cumulative_loss += applied_loss
        cumulative_stress_loss += stress_loss
        rows.append(
            {
                "timestamp": pd.Timestamp(row.timestamp),
                "season_id": str(row.season_id),
                "harvested_dry_matter_kg_m2": final_result.harvested_dry_matter_kg_m2,
                "fresh_kg_m2": final_result.harvested_fresh_kg_m2,
                "cumulative_fresh_kg_m2": final_result.cumulative_harvested_fresh_kg_m2,
                "mature_dry_matter_kg_m2": final_result.mature_dry_matter_kg_m2,
                "standing_dry_matter_kg_m2": final_result.standing_dry_matter_kg_m2,
                "applied_nonharvest_loss_dry_matter_kg_m2": applied_loss,
                "cumulative_nonharvest_loss_dry_matter_kg_m2": cumulative_loss,
                "reproductive_heat_stress_loss_dry_matter_kg_m2": stress_loss,
                "cumulative_reproductive_heat_stress_loss_dry_matter_kg_m2": cumulative_stress_loss,
                "retained_fruit_allocation_dry_matter_kg_m2": retained_allocation,
            }
        )
    return pd.DataFrame(rows)


def simulate_cohort_scenario(
    trajectory: pd.DataFrame,
    sample: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Apply one sampled harvest protocol to a GreenLight state trajectory."""
    required = {
        "season_id",
        "timestamp",
        "c_fruit_previous_mg_m2",
        "c_fruit_mg_m2",
        "harvested_dry_matter_mg_m2",
        "allocated_fruit_dry_matter_mg_m2",
        "air_temperature",
        "uBoil",
        "uCO2",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"GreenLight trajectory missing columns: {missing}")
    frame = trajectory.copy()
    if frame.empty:
        raise ValueError("GreenLight trajectory must not be empty")
    if not frame["season_id"].astype(str).eq(str(sample["season_id"])).all():
        raise ValueError("trajectory and sampled season_id must match")
    disabled = frame[["uBoil", "uCO2"]].apply(pd.to_numeric, errors="coerce")
    if disabled.isna().any().any() or not np.allclose(disabled.to_numpy(float), 0.0):
        raise ValueError("heating and CO2 controls must remain disabled")
    scenario_id = str(sample["scenario_id"])
    frame["pick"] = _scheduled_pick_mask(
        frame["timestamp"],
        interval_days=int(sample["pick_interval_days"]),
        local_hour=int(sample["pick_hour"]),
    )
    frame["pick_source"] = "simulated_external_prior_protocol"
    frame["season_complete"] = True
    frame["greenhouse_id"] = -1
    frame["greenhouse_code"] = "SIM_CHENGDU_GREENHOUSE"
    frame["planting_code"] = scenario_id
    times = pd.to_datetime(frame["timestamp"], errors="raise")
    if len(times) > 1:
        intervals = times.diff().dropna().dt.total_seconds()
        if not intervals.eq(intervals.iloc[0]).all():
            raise ValueError("GreenLight trajectory must use a constant timestep")
        dt_seconds = float(intervals.iloc[0])
    else:
        raise ValueError("GreenLight trajectory must contain at least two rows")
    drivers = build_harvest_drivers_from_greenlight(
        frame,
        dt_seconds=dt_seconds,
        driver_model_status="diagnostic_simulated_prior_model",
        initial_fruit_maturity_fraction=float(
            sample["initial_fruit_maturity_fraction"]
        ),
    )
    allocation_mg = pd.to_numeric(
        frame["allocated_fruit_dry_matter_mg_m2"], errors="coerce"
    )
    source_before_loss_mg = (
        pd.to_numeric(frame["c_fruit_previous_mg_m2"], errors="coerce")
        + allocation_mg
    )
    nonharvest_loss_mg = (
        source_before_loss_mg
        - pd.to_numeric(frame["harvested_dry_matter_mg_m2"], errors="coerce")
        - pd.to_numeric(frame["c_fruit_mg_m2"], errors="coerce")
    ).clip(lower=0.0)
    if allocation_mg.isna().any() or (allocation_mg < 0.0).any():
        raise ValueError("GreenLight fruit allocation must be finite and non-negative")
    loss_fraction = np.divide(
        nonharvest_loss_mg.to_numpy(float),
        source_before_loss_mg.to_numpy(float),
        out=np.zeros(len(frame), dtype=float),
        where=source_before_loss_mg.to_numpy(float) > 0.0,
    )
    drivers["gross_fruit_allocation_kg_m2"] = allocation_mg.to_numpy(float) * 1e-6
    drivers["source_nonharvest_loss_fraction"] = np.clip(loss_fraction, 0.0, 1.0)
    drivers["source_nonharvest_loss_kg_m2"] = nonharvest_loss_mg.to_numpy(float) * 1e-6
    if "heat_stress_floor_retention" in sample and pd.notna(
        sample["heat_stress_floor_retention"]
    ):
        stress_parameters = ReproductiveHeatStressParameters(
            optimal_mean_temperature_c=float(
                sample["heat_stress_optimal_mean_temperature_c"]
            ),
            severe_mean_temperature_c=float(
                sample["heat_stress_severe_mean_temperature_c"]
            ),
            floor_retention=float(sample["heat_stress_floor_retention"]),
            trailing_window_hours=int(sample["heat_stress_trailing_window_hours"]),
        )
        drivers["reproductive_heat_retention"] = trailing_heat_retention(
            drivers["air_temperature_c"].to_numpy(dtype=float), stress_parameters
        )
        heat_stress_scenario = str(sample.get("heat_stress_scenario", "unspecified"))
    else:
        drivers["reproductive_heat_retention"] = 1.0
        heat_stress_scenario = "disabled"
    drivers["fruit_change_source"] = (
        "greenlight_allocation_and_fractional_nonharvest_loss"
    )
    parameters = HarvestCohortParameters(
        base_temperature_c=float(sample["base_temperature_c"]),
        maturity_thermal_time_deg_day=float(
            sample["maturity_thermal_time_deg_day"]
        ),
        dry_matter_fraction=float(sample["dry_matter_fraction"]),
        minimum_pick_fresh_kg_m2=float(sample["minimum_pick_fresh_kg_m2"]),
    )
    prediction = _simulate_fractional_loss_harvest(drivers, parameters)
    event_rows = prediction["fresh_kg_m2"].gt(0.0)
    events = prediction.loc[event_rows].copy().reset_index(drop=True)
    events["synthetic_event_id"] = [
        f"{scenario_id}_{index:03d}" for index in range(len(events))
    ]
    events["scenario_id"] = scenario_id
    events["synthetic_greenhouse_code"] = "SIM_CHENGDU_GREENHOUSE"
    events["synthetic_planting_code"] = scenario_id
    events["evidence_class"] = "simulated_prior"
    events["target_eligible"] = False
    events["dry_matter_fraction"] = parameters.dry_matter_fraction

    initial_dm = float(drivers["initial_fruit_dry_matter_kg_m2"].sum())
    gross_allocation_dm = float(drivers["gross_fruit_allocation_kg_m2"].sum())
    applied_loss_dm = float(
        prediction["applied_nonharvest_loss_dry_matter_kg_m2"].sum()
    )
    reproductive_stress_loss_dm = float(
        prediction["reproductive_heat_stress_loss_dry_matter_kg_m2"].sum()
    )
    retained_allocation_dm = float(
        prediction["retained_fruit_allocation_dry_matter_kg_m2"].sum()
    )
    final_standing = float(prediction["standing_dry_matter_kg_m2"].iloc[-1])
    harvested_dm = float(prediction["harvested_dry_matter_kg_m2"].sum())
    balance_error = (
        initial_dm
        + gross_allocation_dm
        - applied_loss_dm
        - reproductive_stress_loss_dm
        - final_standing
        - harvested_dm
    )
    harvested_times = pd.to_datetime(events["timestamp"])
    intervals = harvested_times.diff().dropna().dt.total_seconds() / 86400.0
    summary: dict[str, object] = {
        "scenario_id": scenario_id,
        "season_id": str(sample["season_id"]),
        "climate_scenario": str(sample["climate_scenario"]),
        "event_count": int(len(events)),
        "first_harvest_datetime": (
            harvested_times.iloc[0].isoformat() if len(events) else None
        ),
        "peak_harvest_datetime": (
            pd.Timestamp(events.loc[events["fresh_kg_m2"].idxmax(), "timestamp"]).isoformat()
            if len(events)
            else None
        ),
        "median_inter_pick_days": float(intervals.median()) if len(intervals) else None,
        "partial_yield_kg_m2": float(events["fresh_kg_m2"].sum()),
        "final_standing_dry_matter_kg_m2": final_standing,
        "harvested_dry_matter_kg_m2": harvested_dm,
        "gross_fruit_allocation_dry_matter_kg_m2": gross_allocation_dm,
        "applied_nonharvest_loss_dry_matter_kg_m2": applied_loss_dm,
        "reproductive_heat_stress_loss_dry_matter_kg_m2": reproductive_stress_loss_dm,
        "retained_fruit_allocation_dry_matter_kg_m2": retained_allocation_dm,
        "mean_reproductive_heat_retention": (
            retained_allocation_dm / gross_allocation_dm
            if gross_allocation_dm > 0.0
            else 1.0
        ),
        "heat_stress_scenario": heat_stress_scenario,
        "dry_matter_balance_error_kg_m2": float(balance_error),
        "evidence_class": "simulated_prior",
        "target_eligible": False,
    }
    return drivers, events, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run target-ineligible ChengduPhysics/GreenLight harvest priors."
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "results/chengdu_agri_greenhouse_001/harvest_model/"
            "simulated_prior/greenlight_cohort"
        ),
    )
    parser.add_argument("--draws-per-climate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument(
        "--process-prior",
        default=str(DEFAULT_PROCESS_PRIOR_PATH),
        help="WUR external process-prior JSON used for maturity and pick cadence.",
    )
    parser.add_argument(
        "--reuse-existing-trajectories",
        action="store_true",
        help="Reuse greenlight_state_trajectories.csv and recompute cohort outputs only.",
    )
    args = parser.parse_args()

    seasons = canonical_chengdu_seasons()
    process_priors = load_transfer_priors(args.process_prior)
    samples = sample_harvest_scenarios(
        seasons["season_id"],
        draws_per_climate=int(args.draws_per_climate),
        seed=int(args.seed),
        process_priors=process_priors,
    )
    trajectory_path = Path(args.output_dir) / "greenlight_state_trajectories.csv"
    if args.reuse_existing_trajectories:
        precomputed = load_greenlight_trajectory_cache(trajectory_path)
    else:
        precomputed = collect_greenlight_trajectories(
            seasons,
            seed=int(args.seed),
            max_workers=int(args.max_workers),
        )

    def cached_runner(
        season: pd.Series,
        *,
        climate_scenario: str,
        climate_calibration_strength: float,
        seed: int,
    ) -> pd.DataFrame:
        del climate_calibration_strength, seed
        return precomputed[(str(season["season_id"]), climate_scenario)].copy()

    trajectories, drivers, events, summaries, audit = assemble_high_fidelity_ensemble(
        seasons,
        samples,
        trajectory_runner=cached_runner,
        seed=int(args.seed),
    )
    write_ensemble_artifacts(
        args.output_dir,
        frames={
            "greenlight_state_trajectories.csv": trajectories,
            "harvest_drivers.csv": drivers,
            "sampled_parameters.csv": samples,
            "synthetic_harvest_events.csv": events,
            "synthetic_season_summaries.csv": summaries,
        },
        audit=audit,
    )


if __name__ == "__main__":
    main()
