from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.controllers.benchmark_protocol import PROJECT_ROOT
from experiments.controllers.evaluate_six_season_controllers import (
    canonical_evaluation_seasons,
    trajectory_filename,
)
from experiments.crop.simulate_chengdu_greenlight_harvest_priors import (
    load_transfer_priors,
    sample_harvest_scenarios,
    simulate_cohort_scenario,
)


def load_regional_projection_priors(path: str | Path) -> dict[str, object]:
    prior_path = Path(path)
    if not prior_path.is_absolute():
        prior_path = PROJECT_ROOT / prior_path
    payload = json.loads(prior_path.read_text(encoding="utf-8"))
    if payload.get("evidence_class") != "external_transfer_prior":
        raise ValueError("regional projection prior must be external_transfer_prior")
    if bool(payload.get("target_eligible", True)):
        raise ValueError("regional projection prior must remain target-ineligible")
    maturity = payload.get("maturity_thermal_time", {})
    bounds = (float(maturity["low_deg_day"]), float(maturity["high_deg_day"]))
    if not 0.0 < bounds[0] < bounds[1]:
        raise ValueError("regional maturity thermal-time interval is invalid")
    priors = load_transfer_priors()
    season_intervals = maturity.get("season_intervals_deg_day", {})
    if season_intervals:
        parsed_intervals: dict[str, tuple[float, float]] = {}
        for season_id, interval in sorted(season_intervals.items()):
            low = float(interval["low_deg_day"])
            central = float(interval["central_deg_day"])
            high = float(interval["high_deg_day"])
            if not 0.0 < low < central < high:
                raise ValueError(
                    f"regional maturity interval is invalid for season {season_id}"
                )
            parsed_intervals[str(season_id)] = (low, high)
        priors["maturity_thermal_time_deg_day_by_season"] = parsed_intervals
    priors.update(
        {
            "maturity_thermal_time_deg_day": bounds,
            "source_path": prior_path.as_posix(),
            "transfer_calibration_status": str(payload["calibration_status"]),
        }
    )
    return priors


def prepare_hourly_harvest_trajectory(
    trajectory: pd.DataFrame,
    *,
    season_id: str,
    growth_year: int,
    start_day: int,
) -> pd.DataFrame:
    required = {
        "timestep",
        "air_temperature",
        "c_fruit_previous_mg_m2",
        "c_fruit_mg_m2",
        "harvested_dry_matter_mg_m2",
        "allocated_fruit_dry_matter_mg_m2",
        "uBoil",
        "uCO2",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"controller trajectory missing columns: {', '.join(missing)}")
    if trajectory.empty or len(trajectory) % 4:
        raise ValueError("controller trajectory must contain complete hours")
    frame = trajectory.sort_values("timestep", kind="stable").reset_index(drop=True)
    if not np.array_equal(frame["timestep"].to_numpy(dtype=int), np.arange(len(frame))):
        raise ValueError("controller trajectory timesteps must be contiguous from zero")
    frame["hour_index"] = frame.index // 4
    hourly = frame.groupby("hour_index", sort=True).agg(
        air_temperature=("air_temperature", "mean"),
        c_fruit_previous_mg_m2=("c_fruit_previous_mg_m2", "first"),
        c_fruit_mg_m2=("c_fruit_mg_m2", "last"),
        harvested_dry_matter_mg_m2=("harvested_dry_matter_mg_m2", "sum"),
        allocated_fruit_dry_matter_mg_m2=("allocated_fruit_dry_matter_mg_m2", "sum"),
        uBoil=("uBoil", "max"),
        uCO2=("uCO2", "max"),
    ).reset_index()
    start = pd.Timestamp(year=int(growth_year), month=1, day=1) + pd.Timedelta(
        days=int(start_day)
    )
    hourly["timestamp"] = start + pd.to_timedelta(hourly["hour_index"] + 1, unit="h")
    hourly["season_id"] = str(season_id)
    return hourly.drop(columns="hour_index")


def _project_episode(
    trajectory_path: Path,
    *,
    algorithm: str,
    seed: int,
    season_data: dict[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    season_id = str(season_data["season_id"])
    hourly = prepare_hourly_harvest_trajectory(
        pd.read_csv(trajectory_path, low_memory=False),
        season_id=season_id,
        growth_year=int(season_data["growth_year"]),
        start_day=int(season_data["start_day"]),
    )
    start = pd.Timestamp(year=int(season_data["growth_year"]), month=1, day=1) + pd.Timedelta(
        days=int(season_data["start_day"])
    )
    records: list[dict[str, Any]] = []
    for sample_data in samples:
        sample = pd.Series(sample_data).copy()
        source_scenario_id = str(sample["scenario_id"])
        sample["scenario_id"] = (
            f"CTRL_{algorithm}_seed_{int(seed)}__{source_scenario_id}"
        )
        _drivers, _events, summary = simulate_cohort_scenario(hourly, sample)
        first = summary["first_harvest_datetime"]
        records.append(
            {
                "algorithm": str(algorithm),
                "seed": int(seed),
                "season_id": season_id,
                "growth_year": int(season_data["growth_year"]),
                "start_day": int(season_data["start_day"]),
                "cohort_draw": int(sample["cohort_draw"]),
                "scenario_id": str(sample["scenario_id"]),
                "maturity_thermal_time_deg_day": float(
                    sample["maturity_thermal_time_deg_day"]
                ),
                "dry_matter_fraction": float(sample["dry_matter_fraction"]),
                "minimum_pick_fresh_kg_m2": float(
                    sample["minimum_pick_fresh_kg_m2"]
                ),
                "pick_interval_days": int(sample["pick_interval_days"]),
                "pick_hour": int(sample["pick_hour"]),
                "first_harvest_day": (
                    (pd.Timestamp(first) - start).total_seconds() / 86400.0
                    if first is not None
                    else np.nan
                ),
                "transfer_calibration_status": str(
                    sample["transfer_calibration_status"]
                ),
                **summary,
            }
        )
    return records


def aggregate_projected_yield(
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = scenarios.groupby(["algorithm", "seed", "season_id"], sort=True)
    episodes = grouped.agg(
        projected_harvest_median_kg_m2=("partial_yield_kg_m2", "median"),
        projected_harvest_mean_kg_m2=("partial_yield_kg_m2", "mean"),
        projected_harvest_ci95_low_kg_m2=(
            "partial_yield_kg_m2", lambda values: values.quantile(0.025)
        ),
        projected_harvest_ci95_high_kg_m2=(
            "partial_yield_kg_m2", lambda values: values.quantile(0.975)
        ),
        projected_first_harvest_day_median=("first_harvest_day", "median"),
        projected_event_count_median=("event_count", "median"),
        max_abs_mass_balance_error_kg_m2=(
            "dry_matter_balance_error_kg_m2",
            lambda values: float(np.max(np.abs(values.to_numpy(dtype=float)))),
        ),
    ).reset_index()
    seasons = episodes.groupby(["algorithm", "season_id"], as_index=False).agg(
        projected_harvest_kg_m2=("projected_harvest_median_kg_m2", "mean"),
        projected_first_harvest_day=("projected_first_harvest_day_median", "mean"),
        projected_event_count=("projected_event_count_median", "mean"),
    )
    records: list[dict[str, Any]] = []
    statuses = scenarios["transfer_calibration_status"].dropna().astype(str).unique()
    if len(statuses) != 1:
        raise ValueError("projected scenarios must use one transfer calibration status")
    transfer_status = str(statuses[0])
    for algorithm, group in seasons.groupby("algorithm", sort=True):
        values = group["projected_harvest_kg_m2"].to_numpy(dtype=float)
        first_days = group["projected_first_harvest_day"].to_numpy(dtype=float)
        records.append(
            {
                "algorithm": str(algorithm),
                "season_count": int(len(group)),
                "projected_harvest_mean_kg_m2": float(values.mean()),
                "projected_harvest_std_kg_m2": float(values.std(ddof=1)),
                "projected_harvest_median_kg_m2": float(np.median(values)),
                "projected_harvest_min_kg_m2": float(values.min()),
                "projected_harvest_max_kg_m2": float(values.max()),
                "projected_first_harvest_day_mean": float(np.nanmean(first_days)),
                "evidence_class": "simulated_prior",
                "target_eligible": False,
                "transfer_calibration_status": transfer_status,
            }
        )
    return episodes, seasons, pd.DataFrame(records)


def run_projection(
    input_dir: str | Path,
    *,
    draws_per_season: int = 16,
    sampling_seed: int = 20260818,
    max_workers: int = 8,
    transfer_prior_path: str | Path | None = None,
    artifact_prefix: str = "",
) -> Path:
    root = Path(input_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    metrics = pd.read_csv(root / "episode_metrics.csv")
    season_lookup = {
        season.season_id: season for season in canonical_evaluation_seasons()
    }
    process_priors = (
        None
        if transfer_prior_path is None
        else load_regional_projection_priors(transfer_prior_path)
    )
    samples = sample_harvest_scenarios(
        season_lookup,
        draws_per_climate=int(draws_per_season),
        seed=int(sampling_seed),
        process_priors=process_priors,
    )
    samples = samples.loc[samples["climate_scenario"].eq("calibrated")]
    sample_lookup = {
        season_id: group.to_dict(orient="records")
        for season_id, group in samples.groupby("season_id", sort=False)
    }
    tasks = []
    for row in metrics.itertuples(index=False):
        season = season_lookup[str(row.season_id)]
        path = root / "trajectories" / trajectory_filename(
            str(row.algorithm), int(row.seed), season
        )
        tasks.append((path, str(row.algorithm), int(row.seed), season))
    records: list[dict[str, Any]] = []
    workers = max(1, min(int(max_workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _project_episode,
                path,
                algorithm=algorithm,
                seed=seed,
                season_data=season.__dict__,
                samples=sample_lookup[season.season_id],
            ): f"{algorithm}/seed_{seed}/{season.season_id}"
            for path, algorithm, seed, season in tasks
        }
        for future in as_completed(futures):
            unit_id = futures[future]
            records.extend(future.result())
            print(f"yield projected: {unit_id}", flush=True)
    scenarios = pd.DataFrame(records).sort_values(
        ["algorithm", "seed", "season_id", "cohort_draw"]
    ).reset_index(drop=True)
    episodes, seasons, algorithms = aggregate_projected_yield(scenarios)
    for filename, frame in (
        ("cohort_yield_scenarios.csv", scenarios),
        ("cohort_yield_by_episode.csv", episodes),
        ("cohort_yield_by_season.csv", seasons),
        ("cohort_yield_by_algorithm.csv", algorithms),
    ):
        path = root / f"{artifact_prefix}{filename}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    audit = {
        "controller_episode_count": int(len(tasks)),
        "draws_per_episode": int(draws_per_season),
        "scenario_count": int(len(scenarios)),
        "sampling_seed": int(sampling_seed),
        "hourly_driver_rows_per_episode": 120 * 24,
        "max_abs_mass_balance_error_kg_m2": float(
            scenarios["dry_matter_balance_error_kg_m2"].abs().max()
        ),
        "evidence_class": "simulated_prior",
        "target_eligible": False,
        "transfer_calibration_status": str(
            scenarios["transfer_calibration_status"].iloc[0]
        ),
        "transfer_prior_path": (
            None if transfer_prior_path is None else str(Path(transfer_prior_path).as_posix())
        ),
    }
    (root / f"{artifact_prefix}cohort_yield_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=(
            "results/chengdu_agri_greenhouse_001/controller_benchmark/"
            "six_season_120d_guarded_v2"
        ),
    )
    parser.add_argument("--draws-per-season", type=int, default=16)
    parser.add_argument("--sampling-seed", type=int, default=20260818)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--transfer-prior-path")
    parser.add_argument("--artifact-prefix", default="")
    args = parser.parse_args()
    print(
        run_projection(
            args.input_dir,
            draws_per_season=args.draws_per_season,
            sampling_seed=args.sampling_seed,
            max_workers=args.max_workers,
            transfer_prior_path=args.transfer_prior_path,
            artifact_prefix=args.artifact_prefix,
        )
    )


if __name__ == "__main__":
    main()
