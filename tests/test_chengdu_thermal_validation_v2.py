from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.reports.evaluate_chengdu_multistep import evaluate_multistep_rollouts


class _StatefulPredictor:
    def initialize_state(self, row):
        return np.array([float(row["x_air_temperature"]), float(row["x_relative_humidity"]), 5.0])

    def predict_state(self, state, row):
        next_state = state + np.array([1.0, -1.0, 2.0])
        return {
            "pred_air_temperature": float(next_state[0]),
            "pred_relative_humidity": float(next_state[1]),
        }, next_state


def _trajectory() -> pd.DataFrame:
    timestamps = pd.date_range("2026-04-01", periods=5, freq="h")
    return pd.DataFrame(
        {
            "timestamp": timestamps[:4],
            "next_timestamp": timestamps[1:],
            "x_air_temperature": [20.0, 21.0, 22.0, 23.0],
            "x_relative_humidity": [70.0, 69.0, 68.0, 67.0],
            "d_air_temperature": [12.0, 13.0, 14.0, 15.0],
            "next_x_air_temperature": [21.0, 22.0, 23.0, 24.0],
            "next_x_relative_humidity": [69.0, 68.0, 67.0, 66.0],
        }
    )


def test_continuous_rollout_propagates_full_predictor_state_and_reports_physical_metrics():
    metrics, rollouts = evaluate_multistep_rollouts(
        _trajectory(),
        predictor=_StatefulPredictor(),
        horizons=[1, 3],
        start_indices=[0],
    )

    assert rollouts.loc[rollouts["horizon"] == 3, "pred_air_temperature"].item() == 23.0
    assert metrics["horizon_3_air_temperature_rmse"] == 0.0
    assert metrics["horizon_3_air_temperature_bias"] == 0.0
    assert metrics["horizon_3_air_temperature_pred_min"] == 23.0
    assert metrics["horizon_3_air_temperature_true_max"] == 23.0
    assert metrics["all_predictions_finite"] is True
    assert metrics["physical_envelope_pass"] is True


def test_rollout_does_not_cross_a_timestamp_gap():
    trajectory = _trajectory()
    trajectory.loc[2:, ["timestamp", "next_timestamp"]] += pd.Timedelta(hours=3)

    metrics, rollouts = evaluate_multistep_rollouts(
        trajectory,
        predictor=_StatefulPredictor(),
        horizons=[1, 3],
        start_indices=[0],
    )

    assert len(rollouts) == 2
    assert "horizon_3_air_temperature_mae" not in metrics


def test_reference_predictors_define_persistence_and_outdoor_following():
    from experiments.reports.evaluate_chengdu_multistep import outdoor_following_predictor, persistence_predictor

    row = _trajectory().iloc[0]
    assert persistence_predictor(row) == {
        "pred_air_temperature": 20.0,
        "pred_relative_humidity": 70.0,
    }
    assert outdoor_following_predictor(row) == {
        "pred_air_temperature": 12.0,
        "pred_relative_humidity": 70.0,
    }


def test_physics_predictor_accepts_an_explicit_versioned_backend():
    from experiments.reports.evaluate_chengdu_multistep import build_physics_predictor
    from experiments.reports.evaluate_chengdu_physics import default_parameter_vector

    predictor = build_physics_predictor(default_parameter_vector(), model_backend="ChengduPhysicsV2")

    assert predictor.model_backend == "ChengduPhysicsV2"
