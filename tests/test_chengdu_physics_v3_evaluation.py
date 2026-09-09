from __future__ import annotations

import pandas as pd


def _row() -> pd.Series:
    return pd.Series(
        {
            "x_air_temperature": 25.0,
            "x_relative_humidity": 70.0,
            "x_co2_concentration": 600.0,
            "d_air_temperature": 20.0,
            "d_relative_humidity": 80.0,
            "d_global_radiation": 300.0,
            "d_wind_speed": 1.0,
            "uBoil": 0.0,
            "uCO2": 0.0,
            "uThScr": 0.2,
            "uRoofVent": 0.3,
            "uVent": 0.9,
            "uLamp": 0.0,
            "uBlScr": 0.4,
            "uFan": 0.7,
            "uPad": 0.1,
        }
    )


def test_v3_row_conversion_uses_explicit_eight_input_schema():
    from experiments.reports.evaluate_chengdu_physics import row_to_model_inputs

    _state, control, _disturbance = row_to_model_inputs(_row(), model_backend="ChengduPhysicsV3")

    assert control.tolist() == [0.0, 0.0, 0.2, 0.3, 0.0, 0.4, 0.7, 0.1]


def test_legacy_row_conversion_keeps_the_six_input_schema():
    from experiments.reports.evaluate_chengdu_physics import row_to_model_inputs

    _state, control, _disturbance = row_to_model_inputs(_row(), model_backend="ChengduPhysics")

    assert control.tolist() == [0.0, 0.0, 0.2, 0.9, 0.0, 0.4]


def test_v3_parameter_artifact_expands_vector_to_declared_size(tmp_path):
    import json
    from experiments.reports.evaluate_chengdu_physics import load_parameter_vector

    artifact = tmp_path / "params.json"
    artifact.write_text(
        json.dumps({"num_params": 220, "multipliers": {"216": 0.2, "217": 8.0, "218": 0.0, "219": 0.0}}),
        encoding="utf-8",
    )

    params = load_parameter_vector(artifact)

    assert len(params) == 220
    assert params[217] == 8.0
