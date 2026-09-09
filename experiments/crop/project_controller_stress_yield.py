from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from experiments.controllers.benchmark_protocol import PROJECT_ROOT
from experiments.controllers.evaluate_six_season_controllers import (
    canonical_evaluation_seasons,
    trajectory_filename,
)
from experiments.crop.project_controller_cohort_yield import (
    load_regional_projection_priors,
    prepare_hourly_harvest_trajectory,
)
from experiments.crop.simulate_chengdu_greenlight_harvest_priors import (
    sample_harvest_scenarios,
    simulate_cohort_scenario,
)


def expand_heat_stress_samples(
    samples: pd.DataFrame,
    *,
    stress_scenarios: Mapping[str, float],
    optimal_mean_temperature_c: float,
    severe_mean_temperature_c: float,
    trailing_window_hours: int,
) -> pd.DataFrame:
    if samples.empty or not stress_scenarios:
        raise ValueError("samples and heat-stress scenarios must be non-empty")
    frames: list[pd.DataFrame] = []
    for name, floor in stress_scenarios.items():
        if not 0.0 <= float(floor) <= 1.0:
            raise ValueError(f"invalid heat-stress floor for {name}")
        frame = samples.copy()
        frame["scenario_id"] = (
            frame["scenario_id"].astype(str) + f"__heat_{str(name)}"
        )
        frame["heat_stress_scenario"] = str(name)
        frame["heat_stress_floor_retention"] = float(floor)
        frame["heat_stress_optimal_mean_temperature_c"] = float(
            optimal_mean_temperature_c
        )
        frame["heat_stress_severe_mean_temperature_c"] = float(
            severe_mean_temperature_c
        )
        frame["heat_stress_trailing_window_hours"] = int(trailing_window_hours)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["season_id", "cohort_draw", "heat_stress_scenario"], kind="stable"
    ).reset_index(drop=True)


def aggregate_stress_projection(
    scenarios: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = ["algorithm", "seed", "season_id", "heat_stress_scenario"]
    episodes = scenarios.groupby(keys, as_index=False, sort=True).agg(
        projected_harvest_median_kg_m2=("partial_yield_kg_m2", "median"),
        projected_first_harvest_day_median=("first_harvest_day", "median"),
        reproductive_heat_stress_loss_mean_kg_m2=(
            "reproductive_heat_stress_loss_dry_matter_kg_m2", "mean"
        ),
        mean_reproductive_heat_retention=("mean_reproductive_heat_retention", "mean"),
        max_abs_mass_balance_error_kg_m2=(
            "dry_matter_balance_error_kg_m2",
            lambda values: float(np.max(np.abs(values.to_numpy(dtype=float)))),
        ),
    )
    seasons = episodes.groupby(
        ["algorithm", "season_id", "heat_stress_scenario"], as_index=False, sort=True
    ).agg(
        projected_harvest_kg_m2=("projected_harvest_median_kg_m2", "mean"),
        projected_first_harvest_day=("projected_first_harvest_day_median", "mean"),
        reproductive_heat_stress_loss_mean_kg_m2=(
            "reproductive_heat_stress_loss_mean_kg_m2", "mean"
        ),
        mean_reproductive_heat_retention=("mean_reproductive_heat_retention", "mean"),
    )
    algorithms = seasons.groupby(
        ["algorithm", "heat_stress_scenario"], as_index=False, sort=True
    ).agg(
        projected_harvest_mean_kg_m2=("projected_harvest_kg_m2", "mean"),
        projected_harvest_std_kg_m2=("projected_harvest_kg_m2", "std"),
        first_harvest_day_mean=("projected_first_harvest_day", "mean"),
        reproductive_heat_stress_loss_mean_kg_m2=(
            "reproductive_heat_stress_loss_mean_kg_m2", "mean"
        ),
        mean_reproductive_heat_retention=("mean_reproductive_heat_retention", "mean"),
    )
    return episodes, seasons, algorithms


def _climate_exposure(frame: pd.DataFrame) -> dict[str, float]:
    temperature = frame["air_temperature"].to_numpy(dtype=float)
    result = {
        "hours_air_temperature_above_32c": float(np.count_nonzero(temperature > 32.0) / 4.0),
        "hours_air_temperature_above_35c": float(np.count_nonzero(temperature > 35.0) / 4.0),
    }
    if "relative_humidity" in frame:
        humidity = np.clip(frame["relative_humidity"].to_numpy(dtype=float), 0.0, 100.0)
        saturation = 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))
        vpd = saturation * (1.0 - humidity / 100.0)
        result.update(
            {
                "hours_relative_humidity_above_90pct": float(np.count_nonzero(humidity > 90.0) / 4.0),
                "hours_vpd_below_0_2_kpa": float(np.count_nonzero(vpd < 0.2) / 4.0),
                "hours_vpd_above_2_kpa": float(np.count_nonzero(vpd > 2.0) / 4.0),
            }
        )
    return result


def _project_stress_episode(
    trajectory_path: Path,
    *,
    algorithm: str,
    seed: int,
    season_data: dict[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = pd.read_csv(trajectory_path, low_memory=False)
    exposure = _climate_exposure(raw)
    season_id = str(season_data["season_id"])
    hourly = prepare_hourly_harvest_trajectory(
        raw,
        season_id=season_id,
        growth_year=int(season_data["growth_year"]),
        start_day=int(season_data["start_day"]),
    )
    start = pd.Timestamp(year=int(season_data["growth_year"]), month=1, day=1) + pd.Timedelta(
        days=int(season_data["start_day"])
    )
    records: list[dict[str, Any]] = []
    for sample_data in samples:
        sample = pd.Series(sample_data)
        source_scenario_id = str(sample["scenario_id"])
        sample["scenario_id"] = f"CTRL_{algorithm}_seed_{int(seed)}__{source_scenario_id}"
        _drivers, _events, summary = simulate_cohort_scenario(hourly, sample)
        first = summary["first_harvest_datetime"]
        records.append(
            {
                "algorithm": str(algorithm),
                "seed": int(seed),
                "season_id": season_id,
                "cohort_draw": int(sample["cohort_draw"]),
                "heat_stress_scenario": str(sample["heat_stress_scenario"]),
                "scenario_id": str(sample["scenario_id"]),
                "maturity_thermal_time_deg_day": float(sample["maturity_thermal_time_deg_day"]),
                "dry_matter_fraction": float(sample["dry_matter_fraction"]),
                "first_harvest_day": (
                    (pd.Timestamp(first) - start).total_seconds() / 86400.0
                    if first is not None
                    else np.nan
                ),
                **exposure,
                **summary,
            }
        )
    return records


def run_projection(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / config["controller_benchmark_root"]
    metrics = pd.read_csv(root / "episode_metrics.csv")
    seasons = {item.season_id: item for item in canonical_evaluation_seasons()}
    priors = load_regional_projection_priors(config["regional_prior_path"])
    samples = sample_harvest_scenarios(
        seasons,
        draws_per_climate=int(config["draws_per_episode"]),
        seed=int(config["sampling_seed"]),
        process_priors=priors,
    )
    samples = samples.loc[samples["climate_scenario"].eq("calibrated")]
    stress = config["reproductive_heat_stress"]
    samples = expand_heat_stress_samples(
        samples,
        stress_scenarios=stress["floor_retention_by_scenario"],
        optimal_mean_temperature_c=float(stress["optimal_mean_temperature_c"]),
        severe_mean_temperature_c=float(stress["severe_mean_temperature_c"]),
        trailing_window_hours=int(stress["trailing_window_hours"]),
    )
    lookup = {
        season_id: group.to_dict(orient="records")
        for season_id, group in samples.groupby("season_id", sort=False)
    }
    workers = max(1, min(int(config.get("max_workers", 8)), len(metrics)))
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for row in metrics.itertuples(index=False):
            season = seasons[str(row.season_id)]
            trajectory = root / "trajectories" / trajectory_filename(
                str(row.algorithm), int(row.seed), season
            )
            future = executor.submit(
                _project_stress_episode,
                trajectory,
                algorithm=str(row.algorithm),
                seed=int(row.seed),
                season_data=season.__dict__,
                samples=lookup[season.season_id],
            )
            futures[future] = f"{row.algorithm}/seed_{row.seed}/{row.season_id}"
        for future in as_completed(futures):
            records.extend(future.result())
            print(f"stress yield projected: {futures[future]}", flush=True)
    scenarios = pd.DataFrame(records).sort_values(
        ["algorithm", "seed", "season_id", "cohort_draw", "heat_stress_scenario"]
    ).reset_index(drop=True)
    episodes, season_frame, algorithms = aggregate_stress_projection(scenarios)
    prefix = str(config.get("artifact_prefix", "regional_v3_"))
    for name, frame in (
        ("stress_yield_scenarios.csv", scenarios),
        ("stress_yield_by_episode.csv", episodes),
        ("stress_yield_by_season.csv", season_frame),
        ("stress_yield_by_algorithm.csv", algorithms),
    ):
        target = root / f"{prefix}{name}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(target)
    audit = {
        "method": "chengdu_reproductive_heat_stress_v3",
        "controller_episode_count": int(len(metrics)),
        "stress_scenario_count": int(len(stress["floor_retention_by_scenario"])),
        "draws_per_episode": int(config["draws_per_episode"]),
        "scenario_count": int(len(scenarios)),
        "maximum_abs_dry_matter_balance_error_kg_m2": float(
            scenarios["dry_matter_balance_error_kg_m2"].abs().max()
        ),
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
        "target_harvest_validated": False,
    }
    (root / f"{prefix}stress_yield_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/crops/chengdu_reproductive_heat_stress.yml"
    )
    args = parser.parse_args()
    print(run_projection(args.config))


if __name__ == "__main__":
    main()

