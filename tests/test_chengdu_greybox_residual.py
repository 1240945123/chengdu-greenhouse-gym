from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.reports.chengdu_residual_correction import (
    FEATURE_SETS,
    apply_ridge_residual_corrector,
    build_residual_features,
    calibrate_residual_uncertainty,
    fit_ridge_residual_corrector,
    validate_residual_features,
)
from experiments.reports.evaluate_chengdu_multistep import (
    _apply_residual_model,
    build_physics_predictor,
)
from glassgym.configs.default_params import init_default_params
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres
from experiments.reports.evaluate_chengdu_hybrid_uncertainty import evaluate_interval_metrics


def _frame(rows: int = 48) -> pd.DataFrame:
    time = pd.date_range("2026-04-01", periods=rows, freq="h")
    hour = np.arange(rows) % 24
    radiation = np.maximum(0.0, 500.0 * np.sin(np.pi * (hour - 6) / 12.0))
    pred_t = 20.0 + 3.0 * np.sin(2.0 * np.pi * hour / 24.0)
    pred_rh = 75.0 - 8.0 * np.sin(2.0 * np.pi * hour / 24.0)
    t_residual = 1.2 * np.cos(2.0 * np.pi * hour / 24.0)
    rh_residual = -4.0 * np.cos(2.0 * np.pi * hour / 24.0)
    return pd.DataFrame(
        {
            "timestamp": time,
            "pred_air_temperature": pred_t,
            "pred_relative_humidity": pred_rh,
            "next_x_air_temperature": pred_t + t_residual,
            "next_x_relative_humidity": pred_rh + rh_residual,
            "d_air_temperature": pred_t - 2.0,
            "d_relative_humidity": pred_rh + 5.0,
            "d_global_radiation": radiation,
            "d_wind_speed": np.full(rows, 1.0),
            "uRoofVent": (radiation > 100.0).astype(float),
            "uFan": (radiation > 300.0).astype(float),
            "uBlScr": np.zeros(rows),
            "uThScr": np.zeros(rows),
        }
    )


def test_periodic_interaction_features_are_deterministic_and_finite():
    frame = _frame()

    first = build_residual_features(frame, FEATURE_SETS["periodic_interactions"])
    second = build_residual_features(frame, FEATURE_SETS["periodic_interactions"])

    pd.testing.assert_frame_equal(first, second)
    assert {"hour_sin", "hour_cos", "temperature_outdoor_gap", "radiation_hour_sin"}.issubset(first)
    assert np.isfinite(first.to_numpy()).all()


def test_thermal_dynamics_features_use_available_drivers_and_clip_actions():
    frame = _frame(1)
    frame.loc[0, "pred_air_temperature"] = 35.0
    frame.loc[0, "d_air_temperature"] = 30.0
    frame.loc[0, "d_relative_humidity"] = 40.0
    frame.loc[0, "d_global_radiation"] = 800.0
    frame.loc[0, "d_wind_speed"] = 2.0
    frame.loc[0, "uRoofVent"] = 1.2
    frame.loc[0, "uFan"] = -0.2
    frame.loc[0, "uPad"] = 0.5

    features = build_residual_features(frame, FEATURE_SETS["thermal_dynamics"])

    assert features.loc[0, "solar_trapping"] == pytest.approx(0.0)
    assert features.loc[0, "ventilation_heat_exchange"] == pytest.approx(10.0)
    assert features.loc[0, "fan_cooling_demand"] == pytest.approx(0.0)
    assert features.loc[0, "pad_evaporative_potential"] == pytest.approx(1.5)
    assert features.loc[0, "hot_solar_load"] == pytest.approx(4000.0)
    assert np.isfinite(features.to_numpy(dtype=float)).all()


@pytest.mark.parametrize(
    "feature",
    ["x_air_temperature", "next_x_relative_humidity", "true_air_temperature", "corrected_air_temperature"],
)
def test_observed_or_target_feature_names_are_rejected_as_leakage(feature):
    with pytest.raises(ValueError, match="leakage"):
        validate_residual_features([feature])


def test_ridge_model_learns_residual_and_handles_constant_columns():
    frame = _frame()
    features = FEATURE_SETS["periodic"]

    model = fit_ridge_residual_corrector(frame, features, alpha=0.1)
    corrected = apply_ridge_residual_corrector(frame, model, gain=1.0)

    raw_t_mae = np.mean(np.abs(frame["pred_air_temperature"] - frame["next_x_air_temperature"]))
    corrected_t_mae = np.mean(
        np.abs(corrected["pred_air_temperature"] - frame["next_x_air_temperature"])
    )
    assert model["schema_version"] == "ridge_residual_v1"
    assert np.isfinite(np.asarray(model["feature_scales"], dtype=float)).all()
    assert corrected_t_mae < raw_t_mae * 0.1


def test_weighted_ridge_zero_weight_outlier_matches_outlier_removal():
    clean = _frame(24)
    outlier = clean.iloc[[-1]].copy()
    outlier["pred_air_temperature"] = 55.0
    outlier["next_x_air_temperature"] = -50.0
    outlier["pred_relative_humidity"] = 5.0
    outlier["next_x_relative_humidity"] = 150.0
    augmented = pd.concat([clean, outlier], ignore_index=True)

    reference = fit_ridge_residual_corrector(
        clean, FEATURE_SETS["periodic"], alpha=1.0
    )
    weighted = fit_ridge_residual_corrector(
        augmented,
        FEATURE_SETS["periodic"],
        alpha=1.0,
        sample_weight=np.r_[np.ones(len(clean)), 0.0],
    )

    np.testing.assert_allclose(weighted["feature_means"], reference["feature_means"])
    np.testing.assert_allclose(weighted["feature_scales"], reference["feature_scales"])
    for target in reference["target_names"]:
        np.testing.assert_allclose(weighted["coefficients"][target], reference["coefficients"][target])
    assert weighted["weighted_fit"]
    assert weighted["weight_sum"] == pytest.approx(float(len(clean)))
    assert weighted["positive_weight_rows"] == len(clean)


@pytest.mark.parametrize("weights", [[1.0, -1.0], [0.0, 0.0]])
def test_weighted_ridge_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="sample weights"):
        fit_ridge_residual_corrector(
            _frame(2),
            FEATURE_SETS["periodic"],
            alpha=1.0,
            sample_weight=np.asarray(weights),
        )


def test_calibration_quantiles_are_monotonic_and_target_specific():
    fit = _frame(36)
    calibration = _frame(12).copy()
    calibration["next_x_air_temperature"] += np.linspace(-0.5, 0.5, len(calibration))
    model = fit_ridge_residual_corrector(fit, FEATURE_SETS["periodic"], alpha=1.0)

    calibrated = calibrate_residual_uncertainty(calibration, model, gain=0.5)

    for target in ("air_temperature", "relative_humidity"):
        values = calibrated["uncertainty"][target]
        assert 0.0 <= values["q80"] <= values["q90"] <= values["q95"]
    assert calibrated["gain"] == 0.5


def test_ridge_target_gains_can_disable_humidity_without_disabling_temperature():
    frame = _frame(24)
    model = fit_ridge_residual_corrector(
        frame, FEATURE_SETS["periodic"], alpha=1.0
    )
    model["target_gains"] = {"air_temperature": 1.0, "relative_humidity": 0.0}

    corrected = apply_ridge_residual_corrector(frame, model, gain=1.0)

    assert not np.allclose(corrected["pred_air_temperature"], frame["pred_air_temperature"])
    np.testing.assert_allclose(
        corrected["pred_relative_humidity"], frame["pred_relative_humidity"]
    )


def test_ridge_target_gains_must_be_inside_unit_interval():
    frame = _frame(4)
    model = fit_ridge_residual_corrector(
        frame, FEATURE_SETS["periodic"], alpha=1.0
    )
    model["target_gains"] = {"relative_humidity": 1.2}

    with pytest.raises(ValueError, match="target gain"):
        apply_ridge_residual_corrector(frame, model)


def _v4_params() -> np.ndarray:
    params = np.asarray(init_default_params(225), dtype=float)
    params[208:225] = [
        8.0, 0.6, 2.5, 8.0, 0.6, 1.0, 0.6, 3.0, 0.2,
        8.0, 0.0, 0.0, 1.0, 2.0, 0.0001388889, 4.0, 0.5,
    ]
    return params


def test_ridge_application_does_not_depend_on_observed_indoor_row_values():
    model = calibrate_residual_uncertainty(
        _frame(12),
        fit_ridge_residual_corrector(_frame(), FEATURE_SETS["periodic"], alpha=1.0),
        gain=0.5,
    )
    row = _frame(1).iloc[0].copy()
    prediction = {"pred_air_temperature": 20.0, "pred_relative_humidity": 75.0}
    first = _apply_residual_model(row, prediction, model)
    row["x_air_temperature"] = 55.0
    row["x_relative_humidity"] = 1.0
    second = _apply_residual_model(row, prediction, model)

    assert first == second
    assert first["pred_air_temperature"] != prediction["pred_air_temperature"]


def test_hourly_residual_gain_scales_linearly_for_quarter_hour_step():
    model = calibrate_residual_uncertainty(
        _frame(12),
        fit_ridge_residual_corrector(_frame(), FEATURE_SETS["periodic"], alpha=1.0),
        gain=1.0,
    )
    row = _frame(1).iloc[0]
    prediction = {"pred_air_temperature": 20.0, "pred_relative_humidity": 75.0}

    full = _apply_residual_model(row, prediction, model, gain_scale=1.0)
    quarter = _apply_residual_model(row, prediction, model, gain_scale=0.25)

    full_delta = full["pred_air_temperature"] - prediction["pred_air_temperature"]
    quarter_delta = quarter["pred_air_temperature"] - prediction["pred_air_temperature"]
    assert quarter_delta == pytest.approx(0.25 * full_delta)


def test_stateful_ridge_predictor_writes_correction_back_and_counts_fallback():
    fit = _frame()
    model = calibrate_residual_uncertainty(
        fit.iloc[-12:],
        fit_ridge_residual_corrector(fit.iloc[:-12], FEATURE_SETS["periodic"], alpha=1.0),
        gain=0.5,
    )
    row = fit.iloc[0].copy()
    row["x_air_temperature"] = 20.0
    row["x_relative_humidity"] = 75.0
    row["x_co2_concentration"] = 650.0
    row["x_soil_temperature"] = 18.0
    predictor = build_physics_predictor(
        _v4_params(), residual_model=model, model_backend="ChengduPhysicsV4"
    )
    prediction, state = predictor.predict_state(predictor.initialize_state(row), row)

    assert state[2] == pytest.approx(prediction["pred_air_temperature"])
    expected_vapor_pressure = vaporDens2pres(
        state[2], rh2vaporDens(state[2], prediction["pred_relative_humidity"])
    )
    assert state[15] == pytest.approx(expected_vapor_pressure)
    assert predictor.residual_fallback_count == 0

    bad_row = row.copy()
    bad_row["timestamp"] = "not-a-timestamp"
    fallback_prediction, _state = predictor.predict_state(state, bad_row)
    assert np.isfinite(list(fallback_prediction.values())).all()
    assert predictor.residual_fallback_count == 1


def test_interval_metrics_report_coverage_and_fixed_width_by_horizon():
    rollouts = pd.DataFrame(
        {
            "horizon": [1, 1, 24, 24],
            "pred_air_temperature": [20.0, 20.0, 20.0, 20.0],
            "true_air_temperature": [20.5, 22.0, 19.0, 23.0],
            "pred_relative_humidity": [70.0, 70.0, 70.0, 70.0],
            "true_relative_humidity": [72.0, 80.0, 65.0, 80.0],
        }
    )
    model = {
        "uncertainty": {
            "air_temperature": {"q90": 1.5},
            "relative_humidity": {"q90": 10.0},
        }
    }

    metrics = evaluate_interval_metrics(rollouts, model, horizons=[1, 24], quantile_key="q90")

    assert metrics["horizon_1_air_temperature_coverage"] == 0.5
    assert metrics["horizon_24_relative_humidity_coverage"] == 1.0
    assert metrics["horizon_24_air_temperature_mean_width"] == 3.0
    assert metrics["horizon_1_relative_humidity_mean_width"] == 20.0
