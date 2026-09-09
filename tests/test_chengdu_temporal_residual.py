from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.reports.chengdu_residual_correction import FEATURE_SETS
from experiments.predictors.train_chengdu_temporal_residual import (
    select_architecture_by_median_score,
    train_single_temporal_model,
)
from experiments.reports.evaluate_chengdu_temporal_residual import (
    causal_warmup_indices,
    write_temperature_humidity_to_physical_state,
)
from glassgym.environments.utils import vaporPres2rh
from glassgym.models.residual.temporal import (
    ArrayStandardizer,
    LSTMResidualModel,
    TCNResidualModel,
    build_temporal_residual_sequences,
)


def _prediction_frame(rows: int = 12) -> pd.DataFrame:
    timestamp = pd.date_range("2026-04-01", periods=rows, freq="h")
    hour = np.arange(rows) % 24
    pred_t = 20.0 + np.sin(2.0 * np.pi * hour / 24.0)
    pred_rh = 70.0 - 2.0 * np.sin(2.0 * np.pi * hour / 24.0)
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "pred_air_temperature": pred_t,
            "pred_relative_humidity": pred_rh,
            "next_x_air_temperature": pred_t + 0.5,
            "next_x_relative_humidity": pred_rh - 3.0,
            "d_air_temperature": pred_t - 2.0,
            "d_relative_humidity": pred_rh + 5.0,
            "d_global_radiation": np.maximum(0.0, 500.0 * np.sin(np.pi * (hour - 6) / 12.0)),
            "d_wind_speed": np.ones(rows),
            "uRoofVent": np.zeros(rows),
            "uFan": np.zeros(rows),
            "uBlScr": np.zeros(rows),
            "uThScr": np.zeros(rows),
        }
    )


def test_temporal_sequences_use_residual_targets_and_expected_windows():
    frame = _prediction_frame(8)

    sequences = build_temporal_residual_sequences(
        frame,
        feature_columns=FEATURE_SETS["periodic_interactions"],
        lookback=3,
    )

    assert sequences.inputs.shape == (6, 3, 16)
    assert sequences.targets.shape == (6, 2)
    np.testing.assert_allclose(sequences.targets[:, 0], 0.5)
    np.testing.assert_allclose(sequences.targets[:, 1], -3.0)
    np.testing.assert_array_equal(sequences.target_row_indices, np.arange(2, 8))


def test_temporal_sequences_never_cross_timestamp_gap():
    frame = _prediction_frame(8)
    frame.loc[4:, "timestamp"] += pd.Timedelta(hours=2)

    sequences = build_temporal_residual_sequences(
        frame,
        feature_columns=FEATURE_SETS["periodic"],
        lookback=3,
    )

    np.testing.assert_array_equal(sequences.target_row_indices, [2, 3, 6, 7])


def test_temporal_sequences_reject_forbidden_indoor_feature():
    with pytest.raises(ValueError, match="leakage"):
        build_temporal_residual_sequences(
            _prediction_frame(),
            feature_columns=["x_air_temperature"],
            lookback=3,
        )


def test_array_standardizer_round_trip_and_constant_column():
    values = np.asarray([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    scaler = ArrayStandardizer.fit(values)

    transformed = scaler.transform(values)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(scaler.inverse_transform(transformed), values, atol=1e-6)
    assert scaler.scale[1] == 1.0


@pytest.mark.parametrize("model_class", [LSTMResidualModel, TCNResidualModel])
def test_temporal_models_return_two_residuals(model_class):
    torch.manual_seed(0)
    model = model_class(input_dim=16, hidden_dim=8)
    inputs = torch.randn(4, 24, 16)

    output = model(inputs)

    assert output.shape == (4, 2)
    assert torch.isfinite(output).all()
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000


def test_temporal_models_are_deterministic_for_fixed_seed():
    inputs = torch.randn(2, 24, 16)
    for model_class in (LSTMResidualModel, TCNResidualModel):
        torch.manual_seed(17)
        first = model_class(input_dim=16, hidden_dim=8)
        torch.manual_seed(17)
        second = model_class(input_dim=16, hidden_dim=8)
        torch.testing.assert_close(first(inputs), second(inputs))


def test_architecture_selection_uses_median_across_every_seed():
    records = [
        {"architecture": "lstm", "seed": 0, "calibration_score": 0.01},
        {"architecture": "lstm", "seed": 1, "calibration_score": 0.90},
        {"architecture": "lstm", "seed": 2, "calibration_score": 1.00},
        {"architecture": "tcn", "seed": 0, "calibration_score": 0.50},
        {"architecture": "tcn", "seed": 1, "calibration_score": 0.51},
        {"architecture": "tcn", "seed": 2, "calibration_score": 0.52},
    ]

    selected = select_architecture_by_median_score(records, required_seeds=[0, 1, 2])

    assert selected["architecture"] == "tcn"
    assert selected["median_calibration_score"] == pytest.approx(0.51)


def test_training_reduces_calibration_loss_on_learnable_residuals():
    rng = np.random.default_rng(4)
    fit_inputs = rng.normal(size=(96, 6, 3)).astype(np.float32)
    calibration_inputs = rng.normal(size=(32, 6, 3)).astype(np.float32)
    fit_targets = np.column_stack(
        [0.4 * fit_inputs[:, -1, 0], -0.3 * fit_inputs[:, -1, 1]]
    ).astype(np.float32)
    calibration_targets = np.column_stack(
        [0.4 * calibration_inputs[:, -1, 0], -0.3 * calibration_inputs[:, -1, 1]]
    ).astype(np.float32)

    result = train_single_temporal_model(
        architecture="lstm",
        fit_inputs=fit_inputs,
        fit_targets=fit_targets,
        calibration_inputs=calibration_inputs,
        calibration_targets=calibration_targets,
        hidden_dim=8,
        seed=3,
        learning_rate=0.02,
        batch_size=32,
        max_epochs=80,
        patience=15,
    )

    assert result.best_epoch >= 1
    assert result.best_calibration_mse < result.initial_calibration_mse * 0.4
    assert np.isfinite(result.calibration_predictions).all()


def test_causal_warmup_excludes_origin_and_stops_at_gap():
    timestamps = pd.Series(pd.date_range("2026-04-01", periods=8, freq="h"))

    assert causal_warmup_indices(timestamps, origin=6, lookback=4) == [3, 4, 5]

    timestamps.iloc[4:] += pd.Timedelta(hours=2)
    assert causal_warmup_indices(timestamps, origin=6, lookback=4) == [4, 5]
    assert 6 not in causal_warmup_indices(timestamps, origin=6, lookback=4)


def test_temporal_correction_is_written_to_temperature_and_vapor_state():
    state = np.zeros(28, dtype=float)
    corrected = write_temperature_humidity_to_physical_state(
        state,
        air_temperature=23.5,
        relative_humidity=81.0,
    )

    assert corrected is state
    assert corrected[2] == pytest.approx(23.5)
    assert vaporPres2rh(corrected[2], corrected[15]) == pytest.approx(81.0)
