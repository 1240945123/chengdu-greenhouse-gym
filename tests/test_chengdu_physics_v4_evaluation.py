import numpy as np
import pandas as pd

from experiments.reports.evaluate_chengdu_multistep import build_physics_predictor, evaluate_multistep_rollouts
from experiments.reports.evaluate_chengdu_physics import (
    CONTROL_SCHEMAS,
    estimate_clear_sky_temperature,
    row_to_model_inputs,
)
from glassgym.configs.default_params import init_default_params


def _sample_row() -> pd.Series:
    return pd.Series(
        {
            "x_air_temperature": 22.0,
            "x_relative_humidity": 75.0,
            "x_co2_concentration": 650.0,
            "x_soil_temperature": 18.0,
            "d_air_temperature": 16.0,
            "d_relative_humidity": 85.0,
            "d_global_radiation": 0.0,
            "d_wind_speed": 1.0,
            "uRoofVent": 0.0,
            "uFan": 0.0,
            "uPad": 0.0,
        }
    )


def test_clear_sky_temperature_is_finite_and_below_outdoor_air():
    sky_temperature = estimate_clear_sky_temperature(20.0, 80.0)

    assert np.isfinite(sky_temperature)
    assert -80.0 < sky_temperature < 20.0


def test_v4_has_explicit_eight_control_schema_and_uses_sky_estimate():
    assert CONTROL_SCHEMAS["ChengduPhysicsV4"] == [
        "uBoil",
        "uCO2",
        "uThScr",
        "uRoofVent",
        "uLamp",
        "uBlScr",
        "uFan",
        "uPad",
    ]

    _x, controls, disturbances = row_to_model_inputs(_sample_row(), model_backend="ChengduPhysicsV4")
    assert controls.shape == (8,)
    assert disturbances[5] == estimate_clear_sky_temperature(16.0, 85.0)


def test_legacy_backend_retains_existing_sky_temperature_default():
    _x, _controls, disturbances = row_to_model_inputs(_sample_row(), model_backend="ChengduPhysicsV3")

    assert disturbances[5] == 10.0


def test_v4_predictor_completes_two_step_stateful_rollout():
    rows = []
    for hour in range(2):
        row = _sample_row().to_dict()
        row.update(
            {
                "timestamp": f"2026-04-01 {hour:02d}:00:00",
                "next_timestamp": f"2026-04-01 {hour + 1:02d}:00:00",
                "next_x_air_temperature": 21.5,
                "next_x_relative_humidity": 76.0,
            }
        )
        rows.append(row)
    params = np.asarray(init_default_params(225), dtype=float)
    params[208:225] = [
        8.0, 0.6, 2.5, 8.0, 0.6, 1.0, 0.6, 3.0, 0.2,
        8.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    ]
    predictor = build_physics_predictor(params, model_backend="ChengduPhysicsV4")

    metrics, rollouts = evaluate_multistep_rollouts(
        pd.DataFrame(rows), predictor=predictor, horizons=[1, 2], start_indices=[0]
    )

    assert metrics["all_predictions_finite"]
    assert metrics["physical_envelope_pass"]
    assert len(rollouts) == 2
