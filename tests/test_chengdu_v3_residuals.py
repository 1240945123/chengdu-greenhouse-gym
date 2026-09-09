from __future__ import annotations

import numpy as np
import pandas as pd
import inspect


def _residual_frame() -> pd.DataFrame:
    n = 48
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=n, freq="h"),
            "d_global_radiation": [300.0 if 6 <= hour % 24 < 18 else 0.0 for hour in range(n)],
            "uRoofVent": [1.0 if hour % 4 in (0, 1) else 0.0 for hour in range(n)],
            "uFan": [1.0 if hour % 4 in (1, 2) else 0.0 for hour in range(n)],
            "temperature_error": np.sin(np.arange(n) / 3.0),
            "relative_humidity_error": np.cos(np.arange(n) / 4.0),
        }
    )


def test_residual_diagnostics_reports_requested_autocorrelation_lags():
    from experiments.reports.analyze_chengdu_v3_residuals import residual_diagnostics

    result = residual_diagnostics(_residual_frame(), minimum_group_size=5)

    assert set(result["autocorrelation"]["temperature_error"]) == {"lag_1", "lag_6", "lag_24"}
    assert result["autocorrelation"]["temperature_error"]["lag_1"] is not None


def test_residual_diagnostics_reports_day_night_and_four_control_regimes():
    from experiments.reports.analyze_chengdu_v3_residuals import residual_diagnostics

    result = residual_diagnostics(_residual_frame(), minimum_group_size=5)

    assert set(result["day_night"]) == {"day", "night"}
    assert set(result["control_regimes"]) == {"both_off", "roof_only", "fan_only", "both_on"}
    assert all(group["status"] == "ok" for group in result["control_regimes"].values())


def test_sparse_residual_group_is_reported_not_silently_dropped():
    from experiments.reports.analyze_chengdu_v3_residuals import residual_diagnostics

    frame = _residual_frame()
    frame["uFan"] = 0.0
    result = residual_diagnostics(frame, minimum_group_size=5)

    assert result["control_regimes"]["fan_only"] == {"status": "insufficient", "rows": 0}


def test_residual_prediction_builder_requires_explicit_backend_argument():
    from experiments.reports.analyze_chengdu_v3_residuals import build_residual_predictions

    signature = inspect.signature(build_residual_predictions)
    assert "model_backend" in signature.parameters
    assert signature.parameters["model_backend"].default is inspect.Parameter.empty
