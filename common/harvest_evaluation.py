from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


_CUMULATIVE_INTERVAL_COLUMNS = (
    "cumulative_p025_kg_m2",
    "cumulative_p10_kg_m2",
    "cumulative_p50_kg_m2",
    "cumulative_p90_kg_m2",
    "cumulative_p975_kg_m2",
)
_BATCH_INTERVAL_COLUMNS = (
    "batch_p025_kg_m2",
    "batch_p10_kg_m2",
    "batch_p50_kg_m2",
    "batch_p90_kg_m2",
    "batch_p975_kg_m2",
)
_INTERVAL_COLUMNS = _BATCH_INTERVAL_COLUMNS + _CUMULATIVE_INTERVAL_COLUMNS


def _daily_harvest(frame: pd.DataFrame, label: str) -> pd.Series:
    required = {"timestamp", "fresh_kg_m2"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} data missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{label} harvest data must not be empty")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    masses = pd.to_numeric(frame["fresh_kg_m2"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"{label} timestamps must be valid")
    if not np.isfinite(masses.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} harvest masses must be finite")
    if (masses < 0.0).any():
        raise ValueError(f"{label} harvest masses must be non-negative")
    return masses.groupby(timestamps.dt.normalize()).sum().sort_index()


def _r2_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum(np.square(observed - np.mean(observed))))
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - np.sum(np.square(observed - predicted)) / denominator)


def _harvest_timing_metrics(
    dates: pd.DatetimeIndex,
    observed: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    observed_total = float(np.sum(observed))
    predicted_total = float(np.sum(predicted))
    if observed_total <= 0.0 or predicted_total <= 0.0:
        return {
            "harvest_timing_wasserstein_days": float("nan"),
            "median_harvest_date_error_days": float("nan"),
            "p90_harvest_date_error_days": float("nan"),
        }
    observed_cdf = np.cumsum(observed) / observed_total
    predicted_cdf = np.cumsum(predicted) / predicted_total

    def quantile_date(values: np.ndarray, quantile: float) -> pd.Timestamp:
        threshold = float(quantile) * float(np.sum(values))
        position = int(np.searchsorted(np.cumsum(values), threshold, side="left"))
        return pd.Timestamp(dates[min(position, len(dates) - 1)])

    return {
        "harvest_timing_wasserstein_days": float(
            np.sum(np.abs(observed_cdf - predicted_cdf))
        ),
        "median_harvest_date_error_days": float(
            (
                quantile_date(predicted, 0.50)
                - quantile_date(observed, 0.50)
            ).days
        ),
        "p90_harvest_date_error_days": float(
            (
                quantile_date(predicted, 0.90)
                - quantile_date(observed, 0.90)
            ).days
        ),
    }


def evaluate_harvest_predictions(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate daily harvest batches and their cumulative yield trajectory."""
    observed_has_season = "season_id" in observed.columns
    predicted_has_season = "season_id" in predicted.columns
    if observed_has_season != predicted_has_season:
        raise ValueError("observed and predicted data must both include season_id")
    if observed_has_season:
        return _evaluate_multiseason(observed, predicted)
    observed_daily = _daily_harvest(observed, "observed")
    predicted_daily = _daily_harvest(predicted, "predicted")
    start = min(observed_daily.index.min(), predicted_daily.index.min())
    end = max(observed_daily.index.max(), predicted_daily.index.max())
    dates = pd.date_range(start, end, freq="D")
    observed_values = observed_daily.reindex(dates, fill_value=0.0).to_numpy(dtype=float)
    predicted_values = predicted_daily.reindex(dates, fill_value=0.0).to_numpy(dtype=float)
    observed_cumulative = np.cumsum(observed_values)
    predicted_cumulative = np.cumsum(predicted_values)

    observed_total = float(observed_cumulative[-1])
    predicted_total = float(predicted_cumulative[-1])
    if observed_total <= 0.0:
        raise ValueError("observed harvest must contain positive fresh mass")
    batch_errors = predicted_values - observed_values
    cumulative_errors = predicted_cumulative - observed_cumulative
    first_observed = observed_daily[observed_daily > 0.0].index.min()
    positive_prediction = predicted_daily[predicted_daily > 0.0]
    first_predicted = positive_prediction.index.min() if not positive_prediction.empty else None

    aligned = pd.DataFrame(
        {
            "date": dates,
            "observed_fresh_kg_m2": observed_values,
            "predicted_fresh_kg_m2": predicted_values,
            "observed_cumulative_kg_m2": observed_cumulative,
            "predicted_cumulative_kg_m2": predicted_cumulative,
        }
    )
    metrics = {
        "first_harvest_date_error_days": (
            float((first_predicted - first_observed).days)
            if first_predicted is not None else float("nan")
        ),
        "first_harvest_date_max_abs_error_days": (
            abs(float((first_predicted - first_observed).days))
            if first_predicted is not None else float("nan")
        ),
        "batch_mae_kg_m2": float(np.mean(np.abs(batch_errors))),
        "batch_rmse_kg_m2": float(np.sqrt(np.mean(np.square(batch_errors)))),
        "batch_wmape": float(np.sum(np.abs(batch_errors)) / observed_total),
        "batch_r2": _r2_score(observed_values, predicted_values),
        "cumulative_mae_kg_m2": float(np.mean(np.abs(cumulative_errors))),
        "cumulative_rmse_kg_m2": float(np.sqrt(np.mean(np.square(cumulative_errors)))),
        "cumulative_wmape": float(np.sum(np.abs(cumulative_errors)) / np.sum(observed_cumulative)),
        "cumulative_r2": _r2_score(observed_cumulative, predicted_cumulative),
        "final_observed_yield_kg_m2": observed_total,
        "final_predicted_yield_kg_m2": predicted_total,
        "final_yield_bias_kg_m2": predicted_total - observed_total,
        "final_yield_bias_percent": 100.0 * (predicted_total - observed_total) / observed_total,
        **_harvest_timing_metrics(dates, observed_values, predicted_values),
    }

    interval_dates = pd.to_datetime(predicted["timestamp"], errors="coerce").dt.normalize()
    for column in _INTERVAL_COLUMNS:
        if column not in predicted.columns:
            continue
        values = pd.to_numeric(predicted[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")
        daily = values.groupby(interval_dates).last().sort_index()
        reindexed = daily.reindex(dates)
        if column in _CUMULATIVE_INTERVAL_COLUMNS:
            reindexed = reindexed.ffill()
        aligned[column] = reindexed.fillna(0.0).to_numpy(dtype=float)

    if {"batch_p10_kg_m2", "batch_p90_kg_m2"}.issubset(aligned.columns):
        _add_interval_metrics(
            metrics,
            observed_values,
            aligned["batch_p10_kg_m2"].to_numpy(dtype=float),
            aligned["batch_p90_kg_m2"].to_numpy(dtype=float),
            level=0.80,
            metric_prefix="batch",
            normalization_denominator=observed_total / len(observed_values),
        )
    if {"batch_p025_kg_m2", "batch_p975_kg_m2"}.issubset(aligned.columns):
        _add_interval_metrics(
            metrics,
            observed_values,
            aligned["batch_p025_kg_m2"].to_numpy(dtype=float),
            aligned["batch_p975_kg_m2"].to_numpy(dtype=float),
            level=0.95,
            metric_prefix="batch",
            normalization_denominator=observed_total / len(observed_values),
        )

    if {"cumulative_p10_kg_m2", "cumulative_p90_kg_m2"}.issubset(aligned.columns):
        _add_interval_metrics(
            metrics,
            observed_cumulative,
            aligned["cumulative_p10_kg_m2"].to_numpy(dtype=float),
            aligned["cumulative_p90_kg_m2"].to_numpy(dtype=float),
            level=0.80,
            metric_prefix="cumulative",
            normalization_denominator=observed_total,
        )
    if {"cumulative_p025_kg_m2", "cumulative_p975_kg_m2"}.issubset(aligned.columns):
        _add_interval_metrics(
            metrics,
            observed_cumulative,
            aligned["cumulative_p025_kg_m2"].to_numpy(dtype=float),
            aligned["cumulative_p975_kg_m2"].to_numpy(dtype=float),
            level=0.95,
            metric_prefix="cumulative",
            normalization_denominator=observed_total,
        )
    return metrics, aligned


def _evaluate_multiseason(
    observed: pd.DataFrame,
    predicted: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    observed_seasons = set(observed["season_id"].astype(str))
    predicted_seasons = set(predicted["season_id"].astype(str))
    if observed_seasons != predicted_seasons:
        missing_predictions = sorted(observed_seasons - predicted_seasons)
        missing_observations = sorted(predicted_seasons - observed_seasons)
        raise ValueError(
            "season_id mismatch; "
            f"missing predictions={missing_predictions}, "
            f"missing observations={missing_observations}"
        )
    aligned_parts: list[pd.DataFrame] = []
    onset_errors: list[float] = []
    season_timing_metrics: list[dict[str, float]] = []
    for season_id in sorted(observed_seasons):
        observed_part = observed[
            observed["season_id"].astype(str) == season_id
        ].drop(columns="season_id")
        predicted_part = predicted[
            predicted["season_id"].astype(str) == season_id
        ].drop(columns="season_id")
        season_metrics, aligned = evaluate_harvest_predictions(
            observed_part,
            predicted_part,
        )
        aligned.insert(0, "season_id", season_id)
        aligned_parts.append(aligned)
        onset_errors.append(float(season_metrics["first_harvest_date_error_days"]))
        season_timing_metrics.append(season_metrics)
    combined = pd.concat(aligned_parts, ignore_index=True)
    observed_batch = combined["observed_fresh_kg_m2"].to_numpy(dtype=float)
    predicted_batch = combined["predicted_fresh_kg_m2"].to_numpy(dtype=float)
    observed_cumulative = combined["observed_cumulative_kg_m2"].to_numpy(dtype=float)
    predicted_cumulative = combined["predicted_cumulative_kg_m2"].to_numpy(dtype=float)
    batch_errors = predicted_batch - observed_batch
    cumulative_errors = predicted_cumulative - observed_cumulative
    observed_total = float(
        combined.groupby("season_id", sort=False)["observed_cumulative_kg_m2"].last().sum()
    )
    predicted_total = float(
        combined.groupby("season_id", sort=False)["predicted_cumulative_kg_m2"].last().sum()
    )
    finite_onsets = np.asarray(onset_errors, dtype=float)
    finite_onsets = finite_onsets[np.isfinite(finite_onsets)]

    def timing_summary(name: str, *, absolute: bool = False) -> tuple[float, float]:
        values = np.asarray(
            [float(item[name]) for item in season_timing_metrics], dtype=float
        )
        if not np.isfinite(values).all():
            return float("nan"), float("nan")
        if absolute:
            values = np.abs(values)
        return float(np.mean(values)), float(np.max(np.abs(values)))

    wasserstein_mean, wasserstein_max = timing_summary(
        "harvest_timing_wasserstein_days"
    )
    median_mean_abs, median_max_abs = timing_summary(
        "median_harvest_date_error_days", absolute=True
    )
    p90_mean_abs, p90_max_abs = timing_summary(
        "p90_harvest_date_error_days", absolute=True
    )
    metrics = {
        "first_harvest_date_error_days": (
            float(np.mean(np.abs(finite_onsets))) if len(finite_onsets) else float("nan")
        ),
        "first_harvest_date_max_abs_error_days": (
            float(np.max(np.abs(finite_onsets))) if len(finite_onsets) else float("nan")
        ),
        "harvest_timing_wasserstein_days": wasserstein_mean,
        "harvest_timing_wasserstein_max_days": wasserstein_max,
        "median_harvest_date_error_days": median_mean_abs,
        "median_harvest_date_max_abs_error_days": median_max_abs,
        "p90_harvest_date_error_days": p90_mean_abs,
        "p90_harvest_date_max_abs_error_days": p90_max_abs,
        "batch_mae_kg_m2": float(np.mean(np.abs(batch_errors))),
        "batch_rmse_kg_m2": float(np.sqrt(np.mean(np.square(batch_errors)))),
        "batch_wmape": float(np.sum(np.abs(batch_errors)) / observed_total),
        "batch_r2": _r2_score(observed_batch, predicted_batch),
        "cumulative_mae_kg_m2": float(np.mean(np.abs(cumulative_errors))),
        "cumulative_rmse_kg_m2": float(
            np.sqrt(np.mean(np.square(cumulative_errors)))
        ),
        "cumulative_wmape": float(
            np.sum(np.abs(cumulative_errors)) / np.sum(observed_cumulative)
        ),
        "cumulative_r2": _r2_score(observed_cumulative, predicted_cumulative),
        "final_observed_yield_kg_m2": observed_total,
        "final_predicted_yield_kg_m2": predicted_total,
        "final_yield_bias_kg_m2": predicted_total - observed_total,
        "final_yield_bias_percent": (
            100.0 * (predicted_total - observed_total) / observed_total
        ),
    }
    if {"batch_p10_kg_m2", "batch_p90_kg_m2"}.issubset(combined.columns):
        _add_interval_metrics(
            metrics,
            observed_batch,
            combined["batch_p10_kg_m2"].to_numpy(dtype=float),
            combined["batch_p90_kg_m2"].to_numpy(dtype=float),
            level=0.80,
            metric_prefix="batch",
            normalization_denominator=observed_total / len(observed_batch),
        )
    if {"batch_p025_kg_m2", "batch_p975_kg_m2"}.issubset(combined.columns):
        _add_interval_metrics(
            metrics,
            observed_batch,
            combined["batch_p025_kg_m2"].to_numpy(dtype=float),
            combined["batch_p975_kg_m2"].to_numpy(dtype=float),
            level=0.95,
            metric_prefix="batch",
            normalization_denominator=observed_total / len(observed_batch),
        )
    if {"cumulative_p10_kg_m2", "cumulative_p90_kg_m2"}.issubset(combined.columns):
        _add_interval_metrics(
            metrics,
            observed_cumulative,
            combined["cumulative_p10_kg_m2"].to_numpy(dtype=float),
            combined["cumulative_p90_kg_m2"].to_numpy(dtype=float),
            level=0.80,
            metric_prefix="cumulative",
            normalization_denominator=observed_total,
        )
    if {"cumulative_p025_kg_m2", "cumulative_p975_kg_m2"}.issubset(combined.columns):
        _add_interval_metrics(
            metrics,
            observed_cumulative,
            combined["cumulative_p025_kg_m2"].to_numpy(dtype=float),
            combined["cumulative_p975_kg_m2"].to_numpy(dtype=float),
            level=0.95,
            metric_prefix="cumulative",
            normalization_denominator=observed_total,
        )
    return metrics, combined


def _add_interval_metrics(
    metrics: dict[str, float],
    observed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    level: float,
    metric_prefix: str,
    normalization_denominator: float,
) -> None:
    if np.any(lower > upper):
        raise ValueError("prediction interval lower bound exceeds upper bound")
    label = int(round(level * 100.0))
    width = upper - lower
    alpha = 1.0 - level
    winkler = width.copy()
    below = observed < lower
    above = observed > upper
    winkler[below] += (2.0 / alpha) * (lower[below] - observed[below])
    winkler[above] += (2.0 / alpha) * (observed[above] - upper[above])
    if not np.isfinite(normalization_denominator) or normalization_denominator <= 0.0:
        raise ValueError("interval normalization denominator must be positive")
    metrics[f"{metric_prefix}_interval_{label}_coverage"] = float(
        np.mean((observed >= lower) & (observed <= upper))
    )
    metrics[f"{metric_prefix}_interval_{label}_mean_width_kg_m2"] = float(
        np.mean(width)
    )
    metrics[f"{metric_prefix}_interval_{label}_normalized_mean_width"] = float(
        np.mean(width) / normalization_denominator
    )
    metrics[f"{metric_prefix}_interval_{label}_winkler_score_kg_m2"] = float(
        np.mean(winkler)
    )


def batch_prediction_intervals(
    batch_ensemble_kg_m2: np.ndarray,
) -> Mapping[str, np.ndarray]:
    ensemble = np.asarray(batch_ensemble_kg_m2, dtype=float)
    if ensemble.ndim != 2 or ensemble.shape[0] < 2 or ensemble.shape[1] == 0:
        raise ValueError("batch ensemble must have shape (samples >= 2, steps >= 1)")
    if not np.isfinite(ensemble).all() or np.any(ensemble < 0.0):
        raise ValueError("batch ensemble must be finite and non-negative")
    quantiles = np.quantile(ensemble, [0.025, 0.10, 0.50, 0.90, 0.975], axis=0)
    return {
        name: quantiles[index]
        for index, name in enumerate(_BATCH_INTERVAL_COLUMNS)
    }


def percentile_prediction_intervals(
    cumulative_ensemble_kg_m2: np.ndarray,
) -> Mapping[str, np.ndarray]:
    ensemble = np.asarray(cumulative_ensemble_kg_m2, dtype=float)
    if ensemble.ndim != 2 or ensemble.shape[0] < 2 or ensemble.shape[1] == 0:
        raise ValueError("cumulative ensemble must have shape (samples >= 2, steps >= 1)")
    if not np.isfinite(ensemble).all() or np.any(ensemble < 0.0):
        raise ValueError("cumulative ensemble must be finite and non-negative")
    if np.any(np.diff(ensemble, axis=1) < -1e-12):
        raise ValueError("each cumulative ensemble member must be non-decreasing")
    quantiles = np.quantile(ensemble, [0.025, 0.10, 0.50, 0.90, 0.975], axis=0)
    return {
        name: quantiles[index]
        for index, name in enumerate(_CUMULATIVE_INTERVAL_COLUMNS)
    }
