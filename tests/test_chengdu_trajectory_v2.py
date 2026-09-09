from __future__ import annotations

import hashlib
import json

import pandas as pd


def _source_frame(periods: int = 20) -> pd.DataFrame:
    values = list(range(periods))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=periods, freq="h"),
            "air_temperature": [20.0 + 0.1 * value for value in values],
            "relative_humidity": [70.0 - 0.2 * value for value in values],
            "outdoor_air_temperature": [12.0 + 0.2 * value for value in values],
            "outdoor_relative_humidity": [85.0 - 0.1 * value for value in values],
            "global_radiation": [max(0.0, (value % 12) * 50.0) for value in values],
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
            "roof_window_known_fraction": [1.0] * periods,
            "roof_window_uncertain_fraction": [0.0] * periods,
        }
    )


def test_v2_bundle_is_chronological_audited_and_fingerprinted(tmp_path):
    from processing.build_chengdu_trajectory_v2 import build_trajectory_v2_bundle

    source = tmp_path / "aligned.csv"
    output = tmp_path / "v2"
    frame = _source_frame()
    frame.loc[5, "air_temperature"] = float("nan")
    frame.to_csv(source, index=False)

    manifest = build_trajectory_v2_bundle(source, output)

    assert manifest["schema_version"] == "chengdu_trajectory_v2"
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["source_rows"] == 20
    assert manifest["candidate_transitions"] == 19
    assert manifest["excluded_transitions"]["missing_required_measurement"] == 2
    assert manifest["valid_transitions"] == 17
    assert manifest["split_rows"] == {"train": 11, "val": 2, "test": 4}

    combined = pd.concat([pd.read_csv(output / f"{name}.csv") for name in ("train", "val", "test")])
    timestamps = pd.to_datetime(combined["timestamp"])
    next_timestamps = pd.to_datetime(combined["next_timestamp"])
    assert ((next_timestamps - timestamps) == pd.Timedelta(hours=1)).all()
    assert not combined["x_air_temperature"].equals(combined["d_air_temperature"])
    assert "roof_window_known_fraction" in combined.columns
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest
    for name, expected_hash in manifest["split_sha256"].items():
        assert hashlib.sha256((output / f"{name}.csv").read_bytes()).hexdigest() == expected_hash


def test_v2_bundle_rejects_identical_indoor_and_outdoor_series(tmp_path):
    from processing.build_chengdu_trajectory_v2 import build_trajectory_v2_bundle

    source = tmp_path / "aligned.csv"
    frame = _source_frame()
    frame["outdoor_air_temperature"] = frame["air_temperature"]
    frame["outdoor_relative_humidity"] = frame["relative_humidity"]
    frame.to_csv(source, index=False)

    try:
        build_trajectory_v2_bundle(source, tmp_path / "v2")
    except ValueError as exc:
        assert "identical to indoor" in str(exc)
    else:
        raise AssertionError("identical indoor/outdoor disturbances must be rejected")
