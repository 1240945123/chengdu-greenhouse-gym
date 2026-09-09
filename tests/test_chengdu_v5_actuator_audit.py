import pandas as pd
import pytest

from experiments.reports.audit_chengdu_v5_actuators import (
    build_actuator_audit_bundle,
    probe_actuator_response,
    summarize_actuator_support,
    validate_cultivation_window,
)


def test_support_requires_active_rows_switches_and_multiple_levels():
    rows = 60
    frame = pd.DataFrame(
        {
            "supported": [0.0, 0.5, 1.0] * 20,
            "sparse": [0.0] * 50 + [1.0] * 10,
            "constant": [0.5] * rows,
        }
    )

    support = summarize_actuator_support(
        frame,
        ["supported", "sparse", "constant"],
        min_active_rows=30,
        min_switches=20,
        min_unique_levels=3,
    ).set_index("actuator")

    assert bool(support.loc["supported", "empirical_support_pass"])
    assert not bool(support.loc["sparse", "empirical_support_pass"])
    assert not bool(support.loc["constant", "empirical_support_pass"])
    assert support.loc["supported", "active_rows"] == 40
    assert support.loc["supported", "switch_count"] == 59
    assert support.loc["supported", "rounded_unique_levels"] == 3


def test_cultivation_window_rejects_target_at_cutoff():
    valid = pd.DataFrame({"next_timestamp": ["2026-07-11 23:00"]})
    validate_cultivation_window(valid, "2026-07-12 00:00")

    invalid = pd.DataFrame({"next_timestamp": ["2026-07-12 00:00"]})
    with pytest.raises(ValueError, match="outside the tomato cultivation window"):
        validate_cultivation_window(invalid, "2026-07-12 00:00")


def test_counterfactual_probe_changes_only_selected_actuator():
    frame = pd.DataFrame(
        {
            "uFan": [0.25, 0.75],
            "uRoofVent": [0.4, 0.6],
            "x_air_temperature": [30.0, 32.0],
        }
    )
    seen = []

    def predictor(row):
        seen.append(row.copy())
        return {
            "pred_air_temperature": row["x_air_temperature"] - 2.0 * row["uFan"],
            "pred_relative_humidity": 60.0 + row["uFan"],
        }

    result = probe_actuator_response(
        frame,
        actuator="uFan",
        low_level=0.0,
        high_level=1.0,
        predictor=predictor,
    )

    assert result["sample_rows"] == 2
    assert result["air_temperature_delta_median"] == pytest.approx(-2.0)
    assert result["relative_humidity_delta_median"] == pytest.approx(1.0)
    assert result["cooling_response_fraction"] == pytest.approx(1.0)
    assert [row["uRoofVent"] for row in seen] == [0.4, 0.4, 0.6, 0.6]
    assert frame["uFan"].tolist() == [0.25, 0.75]


def test_audit_bundle_is_deterministic_and_promotes_only_passing_cooling_actuator(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=60, freq="h"),
            "next_timestamp": pd.date_range("2026-06-01 01:00", periods=60, freq="h"),
            "x_air_temperature": [30.0] * 60,
            "d_air_temperature": [25.0] * 60,
            "uFan": [0.0, 0.5, 1.0] * 20,
        }
    )
    trajectory = tmp_path / "train.csv"
    frame.to_csv(trajectory, index=False)

    def predictor(row):
        return {
            "pred_air_temperature": 30.0 - 2.0 * row["uFan"],
            "pred_relative_humidity": 70.0 + row["uFan"],
        }

    first = build_actuator_audit_bundle(
        trajectory,
        tmp_path / "first",
        predictor=predictor,
        actuators=["uFan"],
        cooling_actuators=["uFan"],
    )
    second = build_actuator_audit_bundle(
        trajectory,
        tmp_path / "second",
        predictor=predictor,
        actuators=["uFan"],
        cooling_actuators=["uFan"],
    )

    assert first["recommended_action_space"] == ["uFan"]
    assert first == second
    for filename in ("actuator_support.csv", "counterfactual_responses.csv", "audit.json", "audit.md"):
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
