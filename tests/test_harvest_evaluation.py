from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common.harvest_evaluation import (
    batch_prediction_intervals,
    evaluate_harvest_predictions,
    percentile_prediction_intervals,
)


def test_evaluation_aligns_batches_on_calendar_days_and_reports_timing():
    observed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01 08:00", "2026-05-04 08:00"]),
            "fresh_kg_m2": [1.0, 2.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-02 09:00", "2026-05-04 08:00"]),
            "fresh_kg_m2": [1.0, 1.5],
        }
    )

    metrics, aligned = evaluate_harvest_predictions(observed, predicted)

    assert metrics["first_harvest_date_error_days"] == 1.0
    assert math.isclose(metrics["final_observed_yield_kg_m2"], 3.0)
    assert math.isclose(metrics["final_predicted_yield_kg_m2"], 2.5)
    assert math.isclose(metrics["final_yield_bias_kg_m2"], -0.5)
    assert math.isclose(metrics["final_yield_bias_percent"], -100.0 / 6.0)
    assert math.isclose(metrics["batch_mae_kg_m2"], 0.625)
    assert math.isclose(metrics["batch_wmape"], 2.5 / 3.0)
    assert aligned["date"].tolist() == list(pd.date_range("2026-05-01", "2026-05-04"))
    assert np.allclose(aligned["observed_cumulative_kg_m2"], [1.0, 1.0, 1.0, 3.0])


def test_duplicate_same_day_batches_are_aggregated():
    observed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01 08:00", "2026-05-01 16:00"]),
            "fresh_kg_m2": [0.4, 0.6],
        }
    )
    predicted = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01 12:00"]),
            "fresh_kg_m2": [1.0],
        }
    )

    metrics, aligned = evaluate_harvest_predictions(observed, predicted)

    assert len(aligned) == 1
    assert metrics["batch_mae_kg_m2"] == 0.0
    assert metrics["first_harvest_date_error_days"] == 0.0


def test_full_season_timing_metrics_measure_one_day_mass_shift():
    observed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-01", "2026-05-03"]),
            "fresh_kg_m2": [1.0, 1.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-05-02", "2026-05-04"]),
            "fresh_kg_m2": [1.0, 1.0],
        }
    )

    metrics, _ = evaluate_harvest_predictions(observed, predicted)

    assert metrics["harvest_timing_wasserstein_days"] == 1.0
    assert metrics["median_harvest_date_error_days"] == 1.0
    assert metrics["p90_harvest_date_error_days"] == 1.0


def test_zero_predicted_yield_has_nonfinite_full_season_timing_metrics():
    observed = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-05-01")], "fresh_kg_m2": [1.0]}
    )
    predicted = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-05-01")], "fresh_kg_m2": [0.0]}
    )

    metrics, _ = evaluate_harvest_predictions(observed, predicted)

    assert math.isnan(metrics["harvest_timing_wasserstein_days"])
    assert math.isnan(metrics["median_harvest_date_error_days"])
    assert math.isnan(metrics["p90_harvest_date_error_days"])


def test_prediction_interval_coverage_is_reported_for_cumulative_yield():
    observed = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=3),
            "fresh_kg_m2": [1.0, 0.5, 1.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=3),
            "fresh_kg_m2": [0.9, 0.6, 0.9],
            "batch_p10_kg_m2": [0.8, 0.4, 0.8],
            "batch_p90_kg_m2": [1.2, 0.6, 1.2],
            "batch_p025_kg_m2": [0.7, 0.3, 0.7],
            "batch_p975_kg_m2": [1.3, 0.7, 1.3],
            "cumulative_p10_kg_m2": [0.8, 1.3, 2.2],
            "cumulative_p90_kg_m2": [1.1, 1.7, 2.7],
            "cumulative_p025_kg_m2": [0.7, 1.2, 2.0],
            "cumulative_p975_kg_m2": [1.2, 1.8, 2.9],
        }
    )

    metrics, _ = evaluate_harvest_predictions(observed, predicted)

    assert metrics["cumulative_interval_80_coverage"] == 1.0
    assert metrics["cumulative_interval_95_coverage"] == 1.0
    assert math.isclose(metrics["cumulative_interval_80_mean_width_kg_m2"], 0.4)
    assert math.isclose(
        metrics["cumulative_interval_80_normalized_mean_width"],
        0.16,
    )
    assert math.isclose(metrics["cumulative_interval_80_winkler_score_kg_m2"], 0.4)
    assert metrics["batch_interval_80_coverage"] == 1.0
    assert metrics["batch_interval_95_coverage"] == 1.0
    assert math.isclose(metrics["batch_interval_80_mean_width_kg_m2"], 1.0 / 3.0)
    assert math.isclose(metrics["batch_interval_80_normalized_mean_width"], 0.4)
    assert math.isclose(metrics["batch_interval_80_winkler_score_kg_m2"], 1.0 / 3.0)


def test_percentile_intervals_are_reproducible_and_monotonic():
    ensemble = np.array(
        [
            [0.8, 1.2, 1.9],
            [1.0, 1.5, 2.4],
            [1.2, 1.7, 2.8],
            [0.9, 1.4, 2.2],
        ]
    )

    intervals = percentile_prediction_intervals(ensemble)

    assert list(intervals) == [
        "cumulative_p025_kg_m2",
        "cumulative_p10_kg_m2",
        "cumulative_p50_kg_m2",
        "cumulative_p90_kg_m2",
        "cumulative_p975_kg_m2",
    ]
    assert np.all(intervals["cumulative_p025_kg_m2"] <= intervals["cumulative_p10_kg_m2"])
    assert np.all(intervals["cumulative_p10_kg_m2"] <= intervals["cumulative_p90_kg_m2"])
    assert np.all(np.diff(intervals["cumulative_p50_kg_m2"]) >= 0.0)


def test_batch_percentile_intervals_allow_nonmonotonic_daily_harvest():
    ensemble = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.2, 0.8, 0.1],
            [0.1, 1.2, 0.0],
            [0.0, 0.9, 0.2],
        ]
    )

    intervals = batch_prediction_intervals(ensemble)

    assert list(intervals) == [
        "batch_p025_kg_m2",
        "batch_p10_kg_m2",
        "batch_p50_kg_m2",
        "batch_p90_kg_m2",
        "batch_p975_kg_m2",
    ]
    assert np.all(intervals["batch_p025_kg_m2"] <= intervals["batch_p10_kg_m2"])
    assert np.all(intervals["batch_p10_kg_m2"] <= intervals["batch_p90_kg_m2"])
    assert intervals["batch_p50_kg_m2"][1] > intervals["batch_p50_kg_m2"][2]


def test_invalid_or_empty_measurements_are_rejected():
    empty = pd.DataFrame(columns=["timestamp", "fresh_kg_m2"])
    prediction = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2026-05-01")], "fresh_kg_m2": [1.0]}
    )
    try:
        evaluate_harvest_predictions(empty, prediction)
    except ValueError as exc:
        assert "observed" in str(exc)
    else:
        raise AssertionError("empty observations must be rejected")

    invalid = prediction.copy()
    invalid.loc[0, "fresh_kg_m2"] = -1.0
    try:
        evaluate_harvest_predictions(invalid, prediction)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative harvest mass must be rejected")


def test_multiseason_evaluation_excludes_calendar_gaps_and_scores_each_onset():
    observed = pd.DataFrame(
        {
            "season_id": ["s1", "s1", "s2", "s2"],
            "timestamp": pd.to_datetime(
                ["2024-05-01", "2024-05-03", "2026-05-01", "2026-05-03"]
            ),
            "fresh_kg_m2": [1.0, 1.0, 1.0, 1.0],
        }
    )
    predicted = pd.DataFrame(
        {
            "season_id": ["s1", "s1", "s2", "s2"],
            "timestamp": pd.to_datetime(
                ["2024-05-02", "2024-05-03", "2026-05-02", "2026-05-03"]
            ),
            "fresh_kg_m2": [1.0, 1.0, 1.0, 1.0],
        }
    )

    metrics, aligned = evaluate_harvest_predictions(observed, predicted)

    assert len(aligned) == 6
    assert aligned["season_id"].nunique() == 2
    assert metrics["first_harvest_date_error_days"] == 1.0
    assert metrics["first_harvest_date_max_abs_error_days"] == 1.0
    assert metrics["harvest_timing_wasserstein_days"] == 0.5
    assert metrics["harvest_timing_wasserstein_max_days"] == 0.5
    assert metrics["median_harvest_date_error_days"] == 1.0
    assert metrics["median_harvest_date_max_abs_error_days"] == 1.0
    assert metrics["p90_harvest_date_error_days"] == 0.0
    assert metrics["p90_harvest_date_max_abs_error_days"] == 0.0
    assert metrics["batch_wmape"] == 1.0
