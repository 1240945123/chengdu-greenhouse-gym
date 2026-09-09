from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.standing_crop_evaluation import evaluate_standing_crop_predictions


def _observations(values=(0.0, 1.0, 2.0)) -> pd.DataFrame:
    rows = []
    for date, mean in zip(pd.date_range("2026-04-01", periods=3, tz="Asia/Shanghai"), values):
        for sample, offset in (("1", -0.1), ("2", 0.1)):
            value = 0.0 if mean == 0.0 else mean + offset
            rows.append(
                {
                    "observation_type": "standing_crop_sample",
                    "observation_date": date,
                    "greenhouse_id": 63,
                    "greenhouse_code": "GH-PIDU",
                    "sample_plant_id": sample,
                    "target_eligible": True,
                    "ripe_fruit_fresh_kg_m2": value,
                }
            )
    return pd.DataFrame(rows)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=3, tz="Asia/Shanghai"),
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "standing_fresh_kg_m2": [0.0, 1.2, 1.8],
            "standing_fresh_pi95_low_kg_m2": [0.0, 0.8, 1.5],
            "standing_fresh_pi95_high_kg_m2": [0.2, 1.4, 2.1],
        }
    )


def test_evaluate_standing_crop_predictions_aggregates_replicates_and_scores_dates():
    metrics, aligned = evaluate_standing_crop_predictions(
        _observations(),
        _predictions(),
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert aligned["observed_mean_fresh_kg_m2"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    expected_errors = np.array([0.0, 0.2, -0.2])
    assert metrics["mae_kg_m2"] == pytest.approx(np.mean(np.abs(expected_errors)))
    assert metrics["rmse_kg_m2"] == pytest.approx(np.sqrt(np.mean(expected_errors**2)))
    assert metrics["mean_bias_kg_m2"] == pytest.approx(expected_errors.mean())
    assert metrics["wmape"] == pytest.approx(np.abs(expected_errors).sum() / 3.0)
    assert metrics["observation_date_count"] == 3
    assert metrics["replicate_count"] == 6
    assert metrics["prediction_interval_95_coverage"] == pytest.approx(1.0)
    assert aligned["observed_sample_count"].tolist() == [2, 2, 2]


def test_evaluate_standing_crop_predictions_marks_scale_metrics_undefined_for_all_zero():
    metrics, _ = evaluate_standing_crop_predictions(
        _observations(values=(0.0, 0.0, 0.0)),
        _predictions().assign(standing_fresh_kg_m2=0.0),
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert metrics["wmape"] is None
    assert metrics["r2"] is None


def test_evaluate_standing_crop_predictions_rejects_harvest_rows_and_identity_mismatch():
    harvest_rows = _observations()
    harvest_rows["observation_type"] = "harvest_event"
    with pytest.raises(ValueError, match="standing_crop_sample"):
        evaluate_standing_crop_predictions(
            harvest_rows,
            _predictions(),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
        )

    with pytest.raises(ValueError, match="prediction greenhouse identity"):
        evaluate_standing_crop_predictions(
            _observations(),
            _predictions().assign(greenhouse_code="WRONG"),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
        )
