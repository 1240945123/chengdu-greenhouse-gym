import pandas as pd
import pytest

from experiments.reports.adapt_chengdu_summer_residual import (
    ADAPTATION_CONTRACT,
    build_adaptation_weights,
    build_online_roles,
    build_role_audit,
    select_calibration_target_gain,
)


def _frame(start: str, periods: int) -> pd.DataFrame:
    target = pd.date_range(start, periods=periods, freq="h")
    return pd.DataFrame(
        {
            "timestamp": target - pd.Timedelta(hours=1),
            "next_timestamp": target,
            "target_heat_regime": ["normal"] * periods,
        }
    )


def test_online_roles_use_target_time_and_never_overlap():
    train = _frame("2026-07-10 00:00", 24)
    validation = _frame("2026-07-11 00:00", 24)
    replay = _frame("2026-07-12 00:00", 72)

    roles = build_online_roles(
        train,
        validation,
        replay,
        fit_end="2026-07-13 00:00",
        evaluation_start="2026-07-14 00:00",
    )

    assert pd.to_datetime(roles["fit"]["next_timestamp"]).max() < pd.Timestamp("2026-07-13")
    assert pd.to_datetime(roles["calibration"]["next_timestamp"]).min() == pd.Timestamp("2026-07-13")
    assert pd.to_datetime(roles["calibration"]["next_timestamp"]).max() < pd.Timestamp("2026-07-14")
    assert pd.to_datetime(roles["evaluation"]["next_timestamp"]).min() == pd.Timestamp("2026-07-14")
    role_times = [set(pd.to_datetime(frame["next_timestamp"])) for frame in roles.values()]
    assert role_times[0].isdisjoint(role_times[1])
    assert role_times[0].isdisjoint(role_times[2])
    assert role_times[1].isdisjoint(role_times[2])


def test_online_roles_reject_base_data_that_crosses_fit_boundary():
    train = _frame("2026-07-12 12:00", 24)

    with pytest.raises(ValueError, match="base history crosses"):
        build_online_roles(
            train,
            _frame("2026-07-10", 1),
            _frame("2026-07-13", 48),
            fit_end="2026-07-13 00:00",
            evaluation_start="2026-07-14 00:00",
        )


def test_online_adaptation_uses_frozen_hyperparameters_and_capped_heat_weights():
    frame = pd.DataFrame(
        {"target_heat_regime": ["normal", "high", "extreme", "normal"]}
    )

    weights = build_adaptation_weights(frame)

    assert ADAPTATION_CONTRACT == {
        "feature_set": "thermal_dynamics",
        "alpha": 0.1,
        "gain": 1.0,
        "heat_regime_weights": {"normal": 1.0, "high": 5.0, "extreme": 5.0},
    }
    assert weights.tolist() == [1.0, 5.0, 5.0, 1.0]


def test_online_role_audit_records_target_ranges_and_coefficient_scope():
    roles = build_online_roles(
        _frame("2026-07-10", 24),
        _frame("2026-07-11", 24),
        _frame("2026-07-12", 72),
        fit_end="2026-07-13",
        evaluation_start="2026-07-14",
    )

    audit = build_role_audit(roles)

    assert audit["fit"]["rows"] == 72
    assert audit["calibration"]["rows"] == 24
    assert audit["evaluation"]["rows"] == 24
    assert audit["coefficient_fit_role"] == "fit_only"
    assert audit["calibration_updates_coefficients"] is False


def test_calibration_target_gain_can_disable_a_harmful_humidity_correction():
    calibration = pd.DataFrame(
        {
            "pred_air_temperature": [0.0, 1.0],
            "pred_relative_humidity": [50.0, 50.0],
            "next_x_relative_humidity": [50.0, 50.0],
        }
    )
    model = {
        "feature_columns": ["pred_air_temperature"],
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "target_names": ["relative_humidity"],
        "coefficients": {"relative_humidity": [10.0, 0.0]},
    }

    selection = select_calibration_target_gain(
        calibration,
        model,
        target="relative_humidity",
        candidates=(0.0, 0.5, 1.0),
    )

    assert selection["selected_gain"] == 0.0
    assert selection["candidates"][0]["mae"] == 0.0
