from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


TRAINING_SCENARIOS = ((2023, 59), (2023, 226), (2024, 60), (2024, 227))
VALIDATION_SCENARIOS = ((2025, 59), (2025, 74), (2025, 89))
CONFIRMATION_SCENARIO = (2025, 104)
HISTORICAL_SMOKE_SCENARIO = (2025, 226)

_REQUIRED_HEALTH_COLUMNS = (
    "completed_episode",
    "numerical_failure",
    "all_values_finite",
    "residual_fallback_count",
    "crop_carbon_nonnegative",
)


def _healthy_rows(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["completed_episode"].astype(bool)
        & ~frame["numerical_failure"].astype(bool)
        & frame["all_values_finite"].astype(bool)
        & frame["residual_fallback_count"].astype(int).eq(0)
        & frame["crop_carbon_nonnegative"].astype(bool)
    )


def aggregate_validation_candidates(
    rows: pd.DataFrame,
    *,
    expected_scenarios: Sequence[tuple[int, int]] = VALIDATION_SCENARIOS,
) -> pd.DataFrame:
    required = {
        "candidate_id",
        "growth_year",
        "start_day",
        "cumulative_reward",
        "temperature_band_mae_c",
        "relative_humidity_band_mae_percent",
        "total_executed_action_variation",
        "mean_controller_inference_ms",
        *_REQUIRED_HEALTH_COLUMNS,
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Validation rows missing columns: {sorted(missing)}")
    expected = {(int(year), int(day)) for year, day in expected_scenarios}
    records = []
    for candidate_id, group in rows.groupby("candidate_id", sort=True):
        observed = set(zip(group["growth_year"].astype(int), group["start_day"].astype(int)))
        complete = observed == expected and len(group) == len(expected)
        healthy = bool(complete and _healthy_rows(group).all())
        records.append(
            {
                "candidate_id": str(candidate_id),
                "scenario_count": int(len(group)),
                "validation_complete": bool(complete),
                "feasible": healthy,
                "mean_cumulative_reward": float(group["cumulative_reward"].mean()),
                "mean_temperature_band_mae_c": float(group["temperature_band_mae_c"].mean()),
                "mean_relative_humidity_band_mae_percent": float(
                    group["relative_humidity_band_mae_percent"].mean()
                ),
                "mean_total_executed_action_variation": float(
                    group["total_executed_action_variation"].mean()
                ),
                "mean_controller_inference_ms": float(
                    group["mean_controller_inference_ms"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def select_best_candidate(summary: pd.DataFrame) -> dict:
    feasible = summary.loc[summary["feasible"].astype(bool)].copy()
    if feasible.empty:
        raise ValueError("No feasible validation candidate")
    ranked = feasible.sort_values(
        by=[
            "mean_cumulative_reward",
            "mean_temperature_band_mae_c",
            "mean_relative_humidity_band_mae_percent",
            "mean_total_executed_action_variation",
            "mean_controller_inference_ms",
            "candidate_id",
        ],
        ascending=[False, True, True, True, True, True],
        kind="stable",
    )
    return ranked.iloc[0].to_dict()


def assess_confirmation(
    candidate: Mapping[str, object],
    pid: Mapping[str, object],
) -> dict:
    candidate_health = bool(
        candidate["completed_episode"]
        and not candidate["numerical_failure"]
        and candidate["all_values_finite"]
        and int(candidate["residual_fallback_count"]) == 0
        and candidate["crop_carbon_nonnegative"]
    )
    safety_metric = (
        "active_safety_intervention_fraction"
        if "active_safety_intervention_fraction" in candidate
        and "active_safety_intervention_fraction" in pid
        else "safety_intervention_fraction"
    )
    gates = {
        "health": candidate_health,
        "reward": float(candidate["cumulative_reward"]) > float(pid["cumulative_reward"]),
        "temperature_band_mae": float(candidate["temperature_band_mae_c"])
        <= float(pid["temperature_band_mae_c"]) + 0.25,
        "relative_humidity_band_mae": float(
            candidate["relative_humidity_band_mae_percent"]
        )
        <= float(pid["relative_humidity_band_mae_percent"]) + 1.0,
        "safety_intervention": float(candidate[safety_metric])
        <= float(pid[safety_metric]) + 0.02,
    }
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "pid_candidate_id": str(pid["candidate_id"]),
        "reward_improvement_over_pid": float(candidate["cumulative_reward"])
        - float(pid["cumulative_reward"]),
        "safety_metric": safety_metric,
        "gates": gates,
        "promoted": bool(all(gates.values())),
    }
