from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def build_paired_climate_sensitivity(
    summaries: pd.DataFrame,
    sampled_parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Compare climate levels within identical season/cohort parameter draws."""
    required = {"scenario_id", "season_id", "cohort_draw", "climate_scenario"}
    missing = sorted(required - set(sampled_parameters.columns))
    if missing:
        raise ValueError(f"sampled parameters missing paired columns: {missing}")
    metrics = summaries[
        ["scenario_id", "partial_yield_kg_m2", "event_count", "first_harvest_datetime"]
    ].copy()
    metrics["first_harvest_datetime"] = pd.to_datetime(
        metrics["first_harvest_datetime"], errors="coerce"
    )
    merged = sampled_parameters[list(required)].merge(
        metrics, on="scenario_id", how="left", validate="one_to_one"
    )
    expected = {"calibrated_low", "calibrated", "calibrated_high"}
    rows: list[dict[str, Any]] = []
    for (season_id, cohort_draw), group in merged.groupby(
        ["season_id", "cohort_draw"], sort=True
    ):
        if set(group["climate_scenario"].astype(str)) != expected:
            raise ValueError("each paired draw must contain all three climate scenarios")
        indexed = group.set_index("climate_scenario")
        high = indexed.loc["calibrated_high"]
        low = indexed.loc["calibrated_low"]
        middle = indexed.loc["calibrated"]
        rows.append(
            {
                "season_id": str(season_id),
                "cohort_draw": int(cohort_draw),
                "yield_low_kg_m2": float(low["partial_yield_kg_m2"]),
                "yield_calibrated_kg_m2": float(middle["partial_yield_kg_m2"]),
                "yield_high_kg_m2": float(high["partial_yield_kg_m2"]),
                "yield_high_minus_low_kg_m2": float(
                    high["partial_yield_kg_m2"] - low["partial_yield_kg_m2"]
                ),
                "event_count_high_minus_low": int(
                    high["event_count"] - low["event_count"]
                ),
                "first_harvest_high_minus_low_days": float(
                    (
                        high["first_harvest_datetime"]
                        - low["first_harvest_datetime"]
                    ).total_seconds()
                    / 86400.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _quantiles(
    values: pd.Series,
    prefix: str,
    unit: str | None = None,
) -> dict[str, float]:
    def key(name: str) -> str:
        return f"{prefix}_{name}_{unit}" if unit else f"{prefix}_{name}"

    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {key(name): float("nan") for name in (
            "ci95_low", "ci80_low", "median", "ci80_high", "ci95_high",
            "ci80_width", "ci95_width",
        )}
    q025, q10, q50, q90, q975 = numeric.quantile([0.025, 0.10, 0.50, 0.90, 0.975])
    return {
        key("ci95_low"): float(q025),
        key("ci80_low"): float(q10),
        key("median"): float(q50),
        key("ci80_high"): float(q90),
        key("ci95_high"): float(q975),
        key("ci80_width"): float(q90 - q10),
        key("ci95_width"): float(q975 - q025),
    }


def build_plausibility_report(
    summaries: pd.DataFrame,
    events: pd.DataFrame,
    trajectories: pd.DataFrame,
    external_priors: dict[str, Any],
    standing_crop: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Summarize scenario-level uncertainty without claiming target accuracy."""
    if summaries.empty or trajectories.empty:
        raise ValueError("summaries and trajectories must not be empty")
    if not summaries["evidence_class"].astype(str).eq("simulated_prior").all():
        raise ValueError("all summaries must be simulated_prior")
    if summaries["target_eligible"].astype(bool).any():
        raise ValueError("simulated summaries must be target ineligible")
    if not events.empty and (
        not events["evidence_class"].astype(str).eq("simulated_prior").all()
        or events["target_eligible"].astype(bool).any()
    ):
        raise ValueError("all events must be target-ineligible simulated priors")

    scenario = summaries.copy()
    scenario["first_harvest_datetime"] = pd.to_datetime(
        scenario["first_harvest_datetime"], errors="coerce"
    )
    scenario["peak_harvest_datetime"] = pd.to_datetime(
        scenario["peak_harvest_datetime"], errors="coerce"
    )
    trajectory_times = trajectories.copy()
    trajectory_times["timestamp"] = pd.to_datetime(
        trajectory_times["timestamp"], errors="coerce"
    )
    season_starts = trajectory_times.groupby("season_id")["timestamp"].min().dt.normalize()
    scenario["season_start"] = scenario["season_id"].map(season_starts)
    scenario["first_harvest_crop_day"] = (
        scenario["first_harvest_datetime"] - scenario["season_start"]
    ).dt.total_seconds() / 86400.0
    scenario["peak_harvest_crop_day"] = (
        scenario["peak_harvest_datetime"] - scenario["season_start"]
    ).dt.total_seconds() / 86400.0
    if events.empty:
        scenario["mean_batch_kg_m2"] = np.nan
    else:
        batch = events.groupby("scenario_id")["fresh_kg_m2"].mean()
        scenario["mean_batch_kg_m2"] = scenario["scenario_id"].map(batch)

    season_rows: list[dict[str, Any]] = []
    for season_id, group in scenario.groupby("season_id", sort=True):
        row: dict[str, Any] = {
            "season_id": str(season_id),
            "scenario_count": int(len(group)),
            "zero_harvest_scenario_count": int(group["event_count"].eq(0).sum()),
            "event_count_median": float(group["event_count"].median()),
            "mass_balance_max_abs_kg_m2": float(
                group["dry_matter_balance_error_kg_m2"].abs().max()
            ),
        }
        row.update(_quantiles(group["partial_yield_kg_m2"], "partial_yield", "kg_m2"))
        row.update(_quantiles(group["mean_batch_kg_m2"], "mean_batch", "kg_m2"))
        row.update(_quantiles(group["first_harvest_crop_day"], "first_harvest_crop_day"))
        row.update(_quantiles(group["median_inter_pick_days"], "median_inter_pick_days"))
        row.update(_quantiles(group["peak_harvest_crop_day"], "peak_harvest_crop_day"))
        season_rows.append(row)
    season_table = pd.DataFrame(season_rows)

    climate_rows: list[dict[str, Any]] = []
    for (season_id, climate), group in trajectory_times.groupby(
        ["season_id", "climate_scenario"], sort=True
    ):
        temperature = pd.to_numeric(group["air_temperature"], errors="coerce")
        humidity = pd.to_numeric(group["relative_humidity"], errors="coerce")
        climate_rows.append(
            {
                "season_id": str(season_id),
                "climate_scenario": str(climate),
                "hour_count": int(len(group)),
                "air_temperature_mean_c": float(temperature.mean()),
                "air_temperature_mae_outside_16_28_c": float(
                    (16.0 - temperature).clip(lower=0.0).add(
                        (temperature - 28.0).clip(lower=0.0)
                    ).mean()
                ),
                "relative_humidity_mean_pct": float(humidity.mean()),
                "relative_humidity_mae_outside_60_85_pct": float(
                    (60.0 - humidity).clip(lower=0.0).add(
                        (humidity - 85.0).clip(lower=0.0)
                    ).mean()
                ),
                "ventilation_mean": float(group["uVent"].mean()),
                "lamp_mean": float(group["uLamp"].mean()),
                "heating_max": float(group["uBoil"].max()),
                "co2_max": float(group["uCO2"].max()),
            }
        )
    climate_table = pd.DataFrame(climate_rows)

    external_metrics = external_priors["metrics"]
    valid_standing = pd.to_numeric(
        standing_crop.get("ripe_fruit_fresh_kg_m2", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    report: dict[str, Any] = {
        "method": "greenlight_cohort_plausibility_report",
        "comparison_scope": "plausibility_only_not_target_accuracy",
        "target_validated": False,
        "scenario_count": int(len(scenario)),
        "event_count": int(len(events)),
        "season_count": int(scenario["season_id"].nunique()),
        "overall": {
            **_quantiles(scenario["partial_yield_kg_m2"], "partial_yield", "kg_m2"),
            **_quantiles(scenario["mean_batch_kg_m2"], "mean_batch", "kg_m2"),
            **_quantiles(scenario["first_harvest_crop_day"], "first_harvest_crop_day"),
            **_quantiles(scenario["median_inter_pick_days"], "median_inter_pick_days"),
            "zero_harvest_scenario_count": int(scenario["event_count"].eq(0).sum()),
        },
        "wur_comparison": {
            "source_role": "external_plausibility_prior_only",
            "first_harvest_day_estimate": float(
                external_metrics["first_harvest_days_from_crop_start"]["estimate"]
            ),
            "mean_batch_kg_m2_estimate": float(
                external_metrics["mean_batch_kg_m2"]["estimate"]
            ),
            "median_interpick_days_estimate": float(
                external_metrics["median_interpick_days"]["estimate"]
            ),
            "full_season_yield_kg_m2_estimate": float(
                external_metrics["seasonal_yield_kg_m2"]["estimate"]
            ),
            "yield_comparison": "partial_120_day_vs_full_external_season_not_directly_comparable",
        },
        "pidu_standing_crop_comparison": {
            "role": "standing_crop_range_plausibility_not_harvest_accuracy",
            "observation_count": int(len(valid_standing)),
            "ripe_fruit_fresh_kg_m2_min": (
                float(valid_standing.min()) if len(valid_standing) else None
            ),
            "ripe_fruit_fresh_kg_m2_max": (
                float(valid_standing.max()) if len(valid_standing) else None
            ),
        },
        "quality_gates": {
            "all_target_ineligible": True,
            "maximum_abs_mass_balance_error_kg_m2": float(
                scenario["dry_matter_balance_error_kg_m2"].abs().max()
            ),
            "heating_and_co2_disabled": bool(
                climate_table[["heating_max", "co2_max"]].eq(0.0).all().all()
            ),
        },
        "limitations": [
            "no_target_harvest_events_for_calibration_or_accuracy_metrics",
            "six_120_day_windows_are_partial_seasons",
            "wur_full_season_yield_is_not_a_direct_target_benchmark",
            "pidu_standing_crop_samples_are_not_harvest_batches",
            "climate_sensitivity_levels_are_not_confidence_intervals",
        ],
    }
    return report, season_table, climate_table


def _markdown_report(report: dict[str, Any], season_table: pd.DataFrame) -> str:
    lines = [
        "# Chengdu GreenLight/cohort plausibility report",
        "",
        "Status: diagnostic simulated prior; not target harvest validation.",
        "",
        "| Season | Scenarios | Median partial yield (kg/m2) | 80% interval | Median first harvest day | Median batches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in season_table.itertuples(index=False):
        lines.append(
            f"| {row.season_id} | {row.scenario_count} | "
            f"{row.partial_yield_median_kg_m2:.3f} | "
            f"{row.partial_yield_ci80_low_kg_m2:.3f}-{row.partial_yield_ci80_high_kg_m2:.3f} | "
            f"{row.first_harvest_crop_day_median:.1f} | {row.event_count_median:.1f} |"
        )
    lines.extend([
        "",
        f"Maximum dry-matter balance error: {report['quality_gates']['maximum_abs_mass_balance_error_kg_m2']:.3e} kg/m2.",
        "",
        "The WUR comparison is an external plausibility prior. Its full-season yield is not directly comparable with these 120-day partial-season outputs.",
        "Target accuracy remains unavailable until measured Pidu harvest dates and batch weights are collected.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report GreenLight harvest prior plausibility.")
    parser.add_argument(
        "--ensemble-dir",
        default="results/chengdu_agri_greenhouse_001/harvest_model/simulated_prior/greenlight_cohort",
    )
    parser.add_argument(
        "--external-prior",
        default="data/processed/external/wur_agc2/external_harvest_priors.json",
    )
    parser.add_argument(
        "--standing-crop",
        default="results/chengdu_agri_greenhouse_001/harvest_model/processed/standing_crop_samples.csv",
    )
    args = parser.parse_args()
    root = Path(args.ensemble_dir)
    report, seasons, climate = build_plausibility_report(
        pd.read_csv(root / "synthetic_season_summaries.csv"),
        pd.read_csv(root / "synthetic_harvest_events.csv"),
        pd.read_csv(root / "greenlight_state_trajectories.csv"),
        json.loads(Path(args.external_prior).read_text(encoding="utf-8")),
        pd.read_csv(args.standing_crop),
    )
    paired = build_paired_climate_sensitivity(
        pd.read_csv(root / "synthetic_season_summaries.csv"),
        pd.read_csv(root / "sampled_parameters.csv"),
    )
    report["paired_climate_sensitivity"] = {
        "pair_count": int(len(paired)),
        "yield_high_minus_low_median_kg_m2": float(
            paired["yield_high_minus_low_kg_m2"].median()
        ),
        "yield_high_minus_low_min_kg_m2": float(
            paired["yield_high_minus_low_kg_m2"].min()
        ),
        "yield_high_minus_low_max_kg_m2": float(
            paired["yield_high_minus_low_kg_m2"].max()
        ),
        "interpretation": "paired_sensitivity_not_confidence_interval",
    }
    outputs = {
        root / "plausibility_by_season.csv": seasons,
        root / "climate_diagnostics.csv": climate,
        root / "paired_climate_sensitivity.csv": paired,
    }
    for path, frame in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    report_path = root / "plausibility_report.json"
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    markdown_path = root / "plausibility_report.md"
    temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary_markdown.write_text(_markdown_report(report, seasons), encoding="utf-8")
    temporary_markdown.replace(markdown_path)


if __name__ == "__main__":
    main()
