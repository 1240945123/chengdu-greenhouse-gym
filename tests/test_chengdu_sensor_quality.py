from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processing.chengdu_sensor_quality import derive_hourly_variable_quality
from processing.build_chengdu_trajectory_v4 import build_trajectory_v4_bundle


def _hourly_channels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-01", periods=4, freq="h"),
            "air_temperature": [20.2, 45.0, 22.0, 25.0],
            "air_temperature__e3038__s1545": [20.0, 45.0, 20.0, 25.0],
            "air_temperature__e3044__s1561": [20.4, np.nan, 25.0, 25.5],
            "air_temperature__e3038__s1545_observed_fraction": [1.0, 1.0, 1.0, 0.2],
            "air_temperature__e3044__s1561_observed_fraction": [1.0, 0.0, 1.0, 0.2],
        }
    )


def _trajectory_source(periods: int = 12) -> pd.DataFrame:
    timestamp = pd.date_range("2026-07-01", periods=periods, freq="h")
    temperature = np.linspace(20.0, 25.5, periods)
    temperature[5] = 42.0
    humidity = np.linspace(75.0, 64.0, periods)
    second_coverage = np.r_[np.ones(6), np.zeros(periods - 6)]
    second_temperature = temperature + 0.3
    second_temperature[6:] = np.nan
    second_humidity = humidity + 1.0
    second_humidity[6:] = np.nan
    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "air_temperature": temperature,
            "relative_humidity": humidity,
            "outdoor_air_temperature": np.full(periods, 32.0),
            "outdoor_relative_humidity": np.full(periods, 60.0),
            "global_radiation": np.full(periods, 800.0),
            "wind_speed": np.ones(periods),
            "co2_concentration": np.full(periods, 650.0),
            "illumination": np.zeros(periods),
            "soil_temperature": np.full(periods, 20.0),
            "soil_humidity": np.full(periods, 60.0),
            "uBoil": np.zeros(periods),
            "uCO2": np.zeros(periods),
            "uThScr": np.zeros(periods),
            "uVent": np.ones(periods),
            "uLamp": np.zeros(periods),
            "uBlScr": np.zeros(periods),
            "uPad": np.zeros(periods),
            "roof_window": np.ones(periods),
            "fan": np.zeros(periods),
            "wet_pad_pump": np.zeros(periods),
            "wet_pad_roll_film": np.zeros(periods),
            "roof_thermal_screen": np.zeros(periods),
            "side_thermal_screen": np.zeros(periods),
            "external_shade": np.zeros(periods),
            "air_temperature__e3038__s1545": temperature,
            "air_temperature__e3044__s1561": second_temperature,
            "relative_humidity__e3038__s1546": humidity,
            "relative_humidity__e3044__s1562": second_humidity,
        }
    )
    for channel in (
        "air_temperature__e3038__s1545",
        "relative_humidity__e3038__s1546",
    ):
        frame[f"{channel}_observed_fraction"] = 1.0
    for channel in (
        "air_temperature__e3044__s1561",
        "relative_humidity__e3044__s1562",
    ):
        frame[f"{channel}_observed_fraction"] = second_coverage
    for device in (
        "roof_window",
        "fan",
        "wet_pad_pump",
        "wet_pad_roll_film",
        "roof_thermal_screen",
        "side_thermal_screen",
        "external_shade",
    ):
        frame[f"{device}_known_fraction"] = 1.0
        frame[f"{device}_uncertain_fraction"] = 0.0
    return frame


def test_agreeing_effective_channels_are_high_quality():
    quality = derive_hourly_variable_quality(
        _hourly_channels(),
        variable="air_temperature",
        indoor_equipment_ids=(3038, 3044),
        coverage_threshold=0.8,
        high_disagreement=1.0,
        medium_disagreement=3.0,
    )

    row = quality.iloc[0]
    assert row["air_temperature_effective_sensor_count"] == 2
    assert row["air_temperature_sensor_disagreement"] == pytest.approx(0.4)
    assert row["air_temperature_quality_tier"] == "high"
    assert row["air_temperature_quality_weight"] == 1.0


def test_well_covered_single_channel_is_medium_not_invalid():
    quality = derive_hourly_variable_quality(
        _hourly_channels(),
        variable="air_temperature",
        indoor_equipment_ids=(3038, 3044),
    )

    row = quality.iloc[1]
    assert row["air_temperature_effective_sensor_count"] == 1
    assert np.isnan(row["air_temperature_sensor_disagreement"])
    assert row["air_temperature_quality_tier"] == "medium"
    assert row["air_temperature_quality_weight"] == 0.6


def test_excessive_channel_disagreement_is_low_quality():
    quality = derive_hourly_variable_quality(
        _hourly_channels(),
        variable="air_temperature",
        indoor_equipment_ids=(3038, 3044),
        high_disagreement=1.0,
        medium_disagreement=3.0,
    )

    row = quality.iloc[2]
    assert row["air_temperature_effective_sensor_count"] == 2
    assert row["air_temperature_sensor_disagreement"] == pytest.approx(5.0)
    assert row["air_temperature_quality_tier"] == "low"
    assert row["air_temperature_quality_weight"] == 0.0


def test_low_observation_coverage_is_not_counted_as_effective():
    quality = derive_hourly_variable_quality(
        _hourly_channels(),
        variable="air_temperature",
        indoor_equipment_ids=(3038, 3044),
        coverage_threshold=0.8,
    )

    row = quality.iloc[3]
    assert row["air_temperature_available_sensor_count"] == 2
    assert row["air_temperature_effective_sensor_count"] == 0
    assert row["air_temperature_quality_tier"] == "low"
    assert row["air_temperature_quality_weight"] == 0.0


def test_quality_derivation_rejects_missing_channel_metadata():
    with pytest.raises(ValueError, match="channel-level"):
        derive_hourly_variable_quality(
            pd.DataFrame({"timestamp": ["2026-07-01"], "air_temperature": [20.0]}),
            variable="air_temperature",
            indoor_equipment_ids=(3038, 3044),
        )


def test_v4_trajectory_preserves_values_and_retains_extreme_heat(tmp_path):
    source = _trajectory_source()
    source_path = tmp_path / "aligned.csv"
    source.to_csv(source_path, index=False)

    manifest = build_trajectory_v4_bundle(source_path, tmp_path / "v4")
    trajectory = pd.concat(
        [pd.read_csv(tmp_path / "v4" / f"{split}.csv") for split in ("train", "val", "test")],
        ignore_index=True,
    )

    assert manifest["valid_transitions"] == len(source) - 1
    np.testing.assert_allclose(trajectory["x_air_temperature"], source["air_temperature"].iloc[:-1])
    np.testing.assert_allclose(trajectory["next_x_air_temperature"], source["air_temperature"].iloc[1:])
    extreme = trajectory[trajectory["target_heat_regime"] == "extreme"]
    assert len(extreme) == 1
    assert extreme.iloc[0]["next_x_air_temperature"] == 42.0
    assert bool(extreme.iloc[0]["extreme_heat_physical_support"])


def test_v4_transition_weight_uses_current_and_target_temperature_and_humidity(tmp_path):
    source_path = tmp_path / "aligned.csv"
    _trajectory_source().to_csv(source_path, index=False)
    build_trajectory_v4_bundle(source_path, tmp_path / "v4")
    trajectory = pd.concat(
        [pd.read_csv(tmp_path / "v4" / f"{split}.csv") for split in ("train", "val", "test")],
        ignore_index=True,
    )

    assert trajectory.iloc[0]["air_temperature_quality_tier"] == "high"
    assert trajectory.iloc[0]["next_air_temperature_quality_tier"] == "high"
    assert trajectory.iloc[0]["transition_quality_weight"] == 1.0
    boundary = trajectory[pd.to_datetime(trajectory["timestamp"]) == pd.Timestamp("2026-07-01 05:00")].iloc[0]
    assert boundary["air_temperature_quality_tier"] == "high"
    assert boundary["next_air_temperature_quality_tier"] == "medium"
    assert boundary["transition_quality_weight"] == 0.6


def test_v4_manifest_records_quality_distribution_without_overwriting_source(tmp_path):
    source = _trajectory_source()
    source_path = tmp_path / "aligned.csv"
    source.to_csv(source_path, index=False)

    manifest = build_trajectory_v4_bundle(source_path, tmp_path / "v4")

    assert manifest["schema_version"] == "chengdu_trajectory_v4_quality_aware"
    assert manifest["quality_policy"]["single_sensor_weight"] == 0.6
    assert manifest["quality_counts"]["all"]["extreme"] == 1
    assert manifest["quality_counts"]["all"]["medium_weight"] >= 1
