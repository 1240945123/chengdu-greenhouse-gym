from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from experiments.controllers.benchmark_protocol import PROJECT_ROOT
from experiments.controllers.evaluate_six_season_controllers import (
    canonical_evaluation_seasons,
    trajectory_filename,
)


def validate_regional_source_registry(registry: pd.DataFrame) -> None:
    required = {"source_id", "evidence_class", "target_eligible", "transfer_role"}
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"regional source registry missing columns: {', '.join(missing)}")
    eligible = registry["target_eligible"].astype(str).str.lower().isin({"true", "1"})
    if eligible.any():
        raise ValueError("regional evidence must have target_eligible=false")
    if not registry["evidence_class"].eq("external_observed").all():
        raise ValueError("regional registry accepts external_observed evidence only")


def _degree_days_at(
    hourly: pd.DataFrame,
    *,
    day: int,
    base_temperature_c: float,
    upper_temperature_c: float,
) -> float:
    required = {"elapsed_hours", "air_temperature_c"}
    missing = sorted(required - set(hourly.columns))
    if missing:
        raise ValueError(f"hourly trajectory missing columns: {', '.join(missing)}")
    frame = hourly.sort_values("elapsed_hours", kind="stable")
    expected = np.arange(1, int(day) * 24 + 1)
    selected = frame.loc[frame["elapsed_hours"].le(int(day) * 24)]
    if len(selected) != len(expected) or not np.array_equal(
        selected["elapsed_hours"].to_numpy(dtype=int), expected
    ):
        raise ValueError(f"trajectory is incomplete through crop day {int(day)}")
    excess = np.maximum(
        np.minimum(
            selected["air_temperature_c"].to_numpy(dtype=float),
            float(upper_temperature_c),
        )
        - float(base_temperature_c),
        0.0,
    )
    return float(excess.sum() / 24.0)


def derive_maturity_thermal_time_prior(
    hourly_by_season: Mapping[str, pd.DataFrame],
    *,
    base_temperature_c: float,
    onset_days: Sequence[int],
    upper_temperature_c: float = 30.0,
) -> dict[str, Any]:
    days = tuple(sorted({int(day) for day in onset_days}))
    if len(days) != 3 or days[0] <= 0:
        raise ValueError("onset_days must contain three positive day values")
    if not hourly_by_season:
        raise ValueError("at least one baseline season is required")
    if float(upper_temperature_c) <= float(base_temperature_c):
        raise ValueError("upper_temperature_c must exceed base_temperature_c")
    all_thresholds: list[float] = []
    central_thresholds: list[float] = []
    season_values: dict[str, dict[str, float]] = {}
    season_intervals: dict[str, dict[str, float]] = {}
    for season_id, hourly in sorted(hourly_by_season.items()):
        values = {
            str(day): _degree_days_at(
                hourly,
                day=day,
                base_temperature_c=base_temperature_c,
                upper_temperature_c=upper_temperature_c,
            )
            for day in days
        }
        season_values[str(season_id)] = values
        season_intervals[str(season_id)] = {
            "low_deg_day": float(values[str(days[0])]),
            "central_deg_day": float(values[str(days[1])]),
            "high_deg_day": float(values[str(days[2])]),
        }
        all_thresholds.extend(values.values())
        central_thresholds.append(values[str(days[1])])
    return {
        "base_temperature_c": float(base_temperature_c),
        "upper_temperature_c": float(upper_temperature_c),
        "onset_days": list(days),
        "target_first_harvest_day": int(days[1]),
        "low_deg_day": float(min(all_thresholds)),
        "central_deg_day": float(np.median(central_thresholds)),
        "high_deg_day": float(max(all_thresholds)),
        "season_count": int(len(hourly_by_season)),
        "season_degree_days": season_values,
        "season_intervals_deg_day": season_intervals,
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
    }


def build_regional_partial_yield_envelope(
    observations: pd.DataFrame,
    *,
    regional_total_yields_kg_m2: Sequence[float],
    wur_crop_start: str,
    wur_crop_duration_days: int,
    regional_crop_duration_days: int,
    evaluation_day: int,
) -> dict[str, Any]:
    required = {"team", "harvest_date", "batch_fresh_kg_m2", "target_eligible"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"WUR harvest observations missing columns: {', '.join(missing)}")
    eligible = observations["target_eligible"].astype(str).str.lower().isin({"true", "1"})
    if eligible.any():
        raise ValueError("WUR transfer observations must be target-ineligible")
    totals = np.asarray(regional_total_yields_kg_m2, dtype=float)
    if totals.size == 0 or not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("regional total yields must be positive finite values")
    if min(wur_crop_duration_days, regional_crop_duration_days, evaluation_day) <= 0:
        raise ValueError("crop durations and evaluation day must be positive")
    mapped_day = int(round(evaluation_day / regional_crop_duration_days * wur_crop_duration_days))
    frame = observations.copy()
    frame["harvest_date"] = pd.to_datetime(frame["harvest_date"], errors="raise")
    frame["crop_day"] = (
        frame["harvest_date"] - pd.Timestamp(wur_crop_start)
    ).dt.days
    fractions: dict[str, float] = {}
    for team, group in frame.groupby("team", sort=True):
        if len(group) < 2 or int(group["crop_day"].max()) <= mapped_day:
            raise ValueError(f"team {team} lacks a complete harvest curve")
        total = float(group["batch_fresh_kg_m2"].sum())
        if total <= 0:
            raise ValueError(f"team {team} has non-positive total harvest")
        partial = float(
            group.loc[group["crop_day"].le(mapped_day), "batch_fresh_kg_m2"].sum()
        )
        fractions[str(team)] = partial / total
    scenarios = np.asarray(
        [value * fraction for value in totals for fraction in fractions.values()],
        dtype=float,
    )
    return {
        "evaluation_day": int(evaluation_day),
        "regional_crop_duration_days": int(regional_crop_duration_days),
        "wur_crop_duration_days": int(wur_crop_duration_days),
        "mapped_wur_crop_day": int(mapped_day),
        "wur_cumulative_fractions": fractions,
        "scenario_count": int(scenarios.size),
        "low_kg_m2": float(scenarios.min()),
        "median_kg_m2": float(np.median(scenarios)),
        "high_kg_m2": float(scenarios.max()),
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
        "applicable_seasons": ["spring"],
    }


def load_baseline_hourly_trajectories(controller_root: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(controller_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    metrics = pd.read_csv(root / "episode_metrics.csv")
    baseline = metrics.loc[metrics["algorithm"].eq("baseline")]
    seasons = {item.season_id: item for item in canonical_evaluation_seasons()}
    if len(baseline) != len(seasons) or baseline["season_id"].nunique() != len(seasons):
        raise ValueError("one complete baseline episode is required for every season")
    result: dict[str, pd.DataFrame] = {}
    for row in baseline.itertuples(index=False):
        season = seasons[str(row.season_id)]
        path = root / "trajectories" / trajectory_filename("baseline", int(row.seed), season)
        trajectory = pd.read_csv(path, usecols=["timestep", "air_temperature"])
        if len(trajectory) != 120 * 24 * 4:
            raise ValueError(f"incomplete baseline trajectory: {path}")
        trajectory["hour_index"] = trajectory["timestep"].astype(int) // 4
        hourly = trajectory.groupby("hour_index", sort=True).agg(
            air_temperature_c=("air_temperature", "mean")
        ).reset_index(drop=True)
        hourly["elapsed_hours"] = np.arange(1, len(hourly) + 1)
        result[str(row.season_id)] = hourly[["elapsed_hours", "air_temperature_c"]]
    return result


def run_calibration(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    registry_path = PROJECT_ROOT / config["sources"]["registry"]
    registry = pd.read_csv(registry_path)
    validate_regional_source_registry(registry)
    timing = config["timing"]
    maturity = derive_maturity_thermal_time_prior(
        load_baseline_hourly_trajectories(config["controller_benchmark_root"]),
        base_temperature_c=float(timing["base_temperature_c"]),
        onset_days=timing["onset_days"],
        upper_temperature_c=float(timing["upper_temperature_c"]),
    )
    yield_rows = registry.loc[
        registry["transfer_role"].eq("regional_total_yield_constraint")
    ]
    regional_yields = yield_rows["observed_yield_kg_m2"].dropna().to_numpy(dtype=float)
    shape = config["partial_yield_shape"]
    envelope = build_regional_partial_yield_envelope(
        pd.read_csv(PROJECT_ROOT / shape["wur_observations"]),
        regional_total_yields_kg_m2=regional_yields,
        wur_crop_start=str(shape["wur_crop_start"]),
        wur_crop_duration_days=int(shape["wur_crop_duration_days"]),
        regional_crop_duration_days=int(shape["regional_crop_duration_days"]),
        evaluation_day=int(shape["evaluation_day"]),
    )
    result = {
        "method": str(config.get("method", "chengdu_regional_external_transfer_v2_season_adaptive")),
        "maturity_thermal_time": maturity,
        "spring_partial_yield_envelope": envelope,
        "source_ids": registry["source_id"].astype(str).tolist(),
        "regional_yield_source_ids": yield_rows["source_id"].astype(str).tolist(),
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
        "target_validation_permitted": False,
        "calibration_status": "regional_external_transfer_not_target_validated",
        "limitations": [
            "no_dated_target_greenhouse_harvest_events",
            "regional_facility_and_cultivar_mismatch",
            "spring_yield_envelope_not_applicable_to_autumn",
            "time_warped_wur_curve_is_a_transfer_assumption",
            "season_specific_thresholds_are_derived_from_baseline_controller_climate",
        ],
    }
    output = PROJECT_ROOT / config["output_json"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/crops/chengdu_regional_yield_transfer.yml"
    )
    args = parser.parse_args()
    print(run_calibration(args.config))


if __name__ == "__main__":
    main()
