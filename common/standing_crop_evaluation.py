from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def evaluate_standing_crop_predictions(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    timezone: str = "Asia/Shanghai",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Score a continuous standing-fruit trajectory at target sample dates."""
    observation_required = {
        "observation_type",
        "observation_date",
        "greenhouse_id",
        "greenhouse_code",
        "sample_plant_id",
        "target_eligible",
        "ripe_fruit_fresh_kg_m2",
    }
    prediction_required = {
        "timestamp",
        "greenhouse_id",
        "greenhouse_code",
        "standing_fresh_kg_m2",
    }
    _require_columns(observations, observation_required, "crop observations")
    _require_columns(predictions, prediction_required, "standing-crop predictions")

    observed = observations.copy()
    observed_ids = pd.to_numeric(observed["greenhouse_id"], errors="coerce")
    observed = observed.loc[
        observed_ids.eq(int(target_greenhouse_id))
        & observed["greenhouse_code"].astype(str).eq(str(target_greenhouse_code))
        & observed["target_eligible"].eq(True)
    ].copy()
    if observed.empty or not observed["observation_type"].eq(
        "standing_crop_sample"
    ).all():
        raise ValueError("target observations must contain only standing_crop_sample rows")
    observed["observation_date"] = observed["observation_date"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    observed["ripe_fruit_fresh_kg_m2"] = pd.to_numeric(
        observed["ripe_fruit_fresh_kg_m2"], errors="coerce"
    )
    observed_values = observed["ripe_fruit_fresh_kg_m2"].to_numpy(dtype=float)
    if (
        observed["observation_date"].isna().any()
        or not np.isfinite(observed_values).all()
        or (observed_values < 0.0).any()
    ):
        raise ValueError("target standing-crop observations must have valid dates and mass")
    if observed.duplicated(["observation_date", "sample_plant_id"]).any():
        raise ValueError("standing-crop sample IDs must be unique within each date")

    predicted = predictions.copy()
    predicted_ids = pd.to_numeric(predicted["greenhouse_id"], errors="coerce")
    identity_matches = predicted_ids.eq(int(target_greenhouse_id)) & predicted[
        "greenhouse_code"
    ].astype(str).eq(str(target_greenhouse_code))
    if predicted.empty or predicted_ids.isna().any() or not identity_matches.all():
        raise ValueError("prediction greenhouse identity does not match target")
    predicted["timestamp"] = predicted["timestamp"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    predicted["standing_fresh_kg_m2"] = pd.to_numeric(
        predicted["standing_fresh_kg_m2"], errors="coerce"
    )
    predicted_values = predicted["standing_fresh_kg_m2"].to_numpy(dtype=float)
    if (
        predicted["timestamp"].isna().any()
        or not np.isfinite(predicted_values).all()
        or (predicted_values < 0.0).any()
    ):
        raise ValueError("standing-crop predictions must have valid timestamps and mass")
    predicted = predicted.sort_values("timestamp", kind="stable")
    if predicted["timestamp"].duplicated().any():
        raise ValueError("standing-crop prediction timestamps must be unique")

    grouped_rows = []
    for date, part in observed.groupby("observation_date", sort=True):
        values = part["ripe_fruit_fresh_kg_m2"].to_numpy(dtype=float)
        count = len(values)
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1)) if count > 1 else None
        standard_error = (
            standard_deviation / float(np.sqrt(count)) if count > 1 else None
        )
        half_width = (
            float(student_t.ppf(0.975, count - 1) * standard_error)
            if count > 1
            else None
        )
        grouped_rows.append(
            {
                "observation_date": date,
                "observed_mean_fresh_kg_m2": mean,
                "observed_sample_count": count,
                "observed_standard_deviation_kg_m2": standard_deviation,
                "observed_standard_error_kg_m2": standard_error,
                "observed_ci95_low_kg_m2": max(0.0, mean - half_width)
                if half_width is not None
                else None,
                "observed_ci95_high_kg_m2": mean + half_width
                if half_width is not None
                else None,
            }
        )
    aligned = pd.DataFrame(grouped_rows)
    if (
        aligned["observation_date"].min() < predicted["timestamp"].min()
        or aligned["observation_date"].max() > predicted["timestamp"].max()
    ):
        raise ValueError("prediction trajectory must cover every observation date")

    origin = predicted["timestamp"].iloc[0]
    prediction_x = (
        predicted["timestamp"] - origin
    ).dt.total_seconds().to_numpy(dtype=float)
    observation_x = (
        aligned["observation_date"] - origin
    ).dt.total_seconds().to_numpy(dtype=float)
    aligned["predicted_standing_fresh_kg_m2"] = np.interp(
        observation_x, prediction_x, predicted_values
    )

    interval_coverages: dict[str, float] = {}
    for level in (80, 95):
        low_column = f"standing_fresh_pi{level}_low_kg_m2"
        high_column = f"standing_fresh_pi{level}_high_kg_m2"
        if low_column not in predicted.columns and high_column not in predicted.columns:
            continue
        if low_column not in predicted.columns or high_column not in predicted.columns:
            raise ValueError(f"both prediction interval {level} columns are required")
        low = pd.to_numeric(predicted[low_column], errors="coerce").to_numpy(dtype=float)
        high = pd.to_numeric(predicted[high_column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(low).all() or not np.isfinite(high).all() or (low > high).any():
            raise ValueError(f"prediction interval {level} bounds must be finite and ordered")
        aligned[f"predicted_pi{level}_low_kg_m2"] = np.interp(
            observation_x, prediction_x, low
        )
        aligned[f"predicted_pi{level}_high_kg_m2"] = np.interp(
            observation_x, prediction_x, high
        )
        covered = (
            aligned["observed_mean_fresh_kg_m2"]
            >= aligned[f"predicted_pi{level}_low_kg_m2"]
        ) & (
            aligned["observed_mean_fresh_kg_m2"]
            <= aligned[f"predicted_pi{level}_high_kg_m2"]
        )
        interval_coverages[f"prediction_interval_{level}_coverage"] = float(
            covered.mean()
        )

    observed_mean = aligned["observed_mean_fresh_kg_m2"].to_numpy(dtype=float)
    predicted_mean = aligned["predicted_standing_fresh_kg_m2"].to_numpy(dtype=float)
    errors = predicted_mean - observed_mean
    denominator = float(np.sum(np.abs(observed_mean)))
    centered_denominator = float(np.sum(np.square(observed_mean - observed_mean.mean())))
    aligned["residual_kg_m2"] = errors
    metrics: dict[str, Any] = {
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "observation_date_count": int(len(aligned)),
        "replicate_count": int(len(observed)),
        "mae_kg_m2": float(np.mean(np.abs(errors))),
        "rmse_kg_m2": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_bias_kg_m2": float(np.mean(errors)),
        "wmape": float(np.sum(np.abs(errors)) / denominator)
        if denominator > 0.0
        else None,
        "r2": float(1.0 - np.sum(np.square(errors)) / centered_denominator)
        if centered_denominator > 0.0
        else None,
        "metrics_scope": "target_standing_fruit_not_harvest_yield",
        **interval_coverages,
    }
    return metrics, aligned


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def _local_timestamp(value: object, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)

