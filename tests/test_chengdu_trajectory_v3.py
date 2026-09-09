from __future__ import annotations

import hashlib

import pandas as pd


def _source_frame(periods: int = 24) -> pd.DataFrame:
    index = list(range(periods))
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=periods, freq="h"),
            "air_temperature": [20.0 + value * 0.1 for value in index],
            "relative_humidity": [75.0 - value * 0.1 for value in index],
            "outdoor_air_temperature": [12.0 + value * 0.1 for value in index],
            "outdoor_relative_humidity": [85.0 - value * 0.1 for value in index],
            "global_radiation": [max(0.0, (value % 12) * 60.0) for value in index],
            "wind_speed": [0.5] * periods,
            "co2_concentration": [600.0] * periods,
            "illumination": [0.0] * periods,
            "soil_temperature": [18.0] * periods,
            "soil_humidity": [60.0] * periods,
            "uBoil": [0.0] * periods,
            "uCO2": [0.0] * periods,
            "uThScr": [0.0] * periods,
            "uVent": [0.5] * periods,
            "uLamp": [0.0] * periods,
            "uBlScr": [0.0] * periods,
            "uPad": [0.0] * periods,
            "roof_window": [0.5] * periods,
            "fan": [0.0, 1.0] * (periods // 2),
            "wet_pad_pump": [0.0] * periods,
            "wet_pad_roll_film": [0.0] * periods,
            "roof_thermal_screen": [0.0] * periods,
            "side_thermal_screen": [0.0] * periods,
            "external_shade": [0.0] * periods,
            "air_temperature__e3038__s1545": [20.0 + value * 0.1 for value in index],
            "air_temperature__e3044__s1561": [20.2 + value * 0.1 for value in index],
            "air_temperature__e3036__s1536": [12.0 + value * 0.1 for value in index],
            "roof_window_known_fraction": [1.0] * periods,
            "roof_window_uncertain_fraction": [0.0] * periods,
            "fan_known_fraction": [1.0] * periods,
            "fan_uncertain_fraction": [0.0] * periods,
        }
    )
    frame.loc[4, "air_temperature__e3044__s1561"] = 25.0
    return frame


def test_v3_bundle_preserves_physical_devices_and_sensor_quality(tmp_path):
    from processing.build_chengdu_trajectory_v3 import build_trajectory_v3_bundle

    source = tmp_path / "aligned.csv"
    output = tmp_path / "v3"
    _source_frame().to_csv(source, index=False)

    manifest = build_trajectory_v3_bundle(source, output)
    train = pd.read_csv(output / "train.csv")

    assert manifest["schema_version"] == "chengdu_trajectory_v3"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    for column in [
        "uRoofVent",
        "uFan",
        "uPad",
        "roof_window_known_fraction",
        "fan_known_fraction",
        "indoor_temperature_sensor_count",
        "indoor_temperature_sensor_disagreement_c",
        "indoor_temperature_single_sensor",
        "identification_eligible",
    ]:
        assert column in train.columns
    disagreement_row = train.loc[pd.to_datetime(train["timestamp"]).eq(pd.Timestamp("2026-04-01 04:00:00"))].iloc[0]
    assert disagreement_row["indoor_temperature_sensor_disagreement_c"] > 3.0
    assert not bool(disagreement_row["identification_eligible"])
    assert manifest["identification_policy"]["max_sensor_disagreement_c"] == 3.0
    assert manifest["device_support"]["fan"]["active_rows"] == 12
    assert "air_temperature__e3036__s1536" not in manifest["temperature_sensor_channels"]


def test_v3_bundle_rejects_missing_physical_roof_or_fan_columns(tmp_path):
    from processing.build_chengdu_trajectory_v3 import build_trajectory_v3_bundle

    source = tmp_path / "aligned.csv"
    frame = _source_frame().drop(columns=["fan"])
    frame.to_csv(source, index=False)

    try:
        build_trajectory_v3_bundle(source, tmp_path / "v3")
    except ValueError as exc:
        assert "fan" in str(exc)
    else:
        raise AssertionError("V3 must not reconstruct fan state from aggregate uVent")
