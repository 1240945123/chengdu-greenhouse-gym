import numpy as np
import pandas as pd
import pytest

from processing.chengdu_era5 import (
    build_archive_request,
    parse_archive_payload,
    to_greenlight_weather,
)


def archive_payload(start="2023-01-01", periods=48):
    timestamps = pd.date_range(start, periods=periods, freq="h")
    hours = np.arange(periods)
    return {
        "latitude": 30.8,
        "longitude": 103.9,
        "timezone": "Asia/Shanghai",
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "shortwave_radiation": "W/m²",
            "wind_speed_10m": "m/s",
            "precipitation": "mm",
            "soil_temperature_0_to_7cm": "°C",
            "soil_temperature_28_to_100cm": "°C",
        },
        "hourly": {
            "time": timestamps.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": (15 + hours * 0.01).tolist(),
            "relative_humidity_2m": (75 - hours * 0.01).tolist(),
            "shortwave_radiation": np.maximum(0, 500 * np.sin((hours % 24 - 6) * np.pi / 12)).tolist(),
            "wind_speed_10m": np.full(periods, 1.2).tolist(),
            "precipitation": np.zeros(periods).tolist(),
            "soil_temperature_0_to_7cm": np.full(periods, 16.0).tolist(),
            "soil_temperature_28_to_100cm": np.full(periods, 17.0).tolist(),
        },
    }


def test_archive_request_is_explicit_and_reproducible():
    request = build_archive_request(
        latitude=30.7980107,
        longitude=103.8986134,
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    assert request["models"] == "era5"
    assert request["timezone"] == "Asia/Shanghai"
    assert request["wind_speed_unit"] == "ms"
    assert "shortwave_radiation" in request["hourly"]


def test_parse_archive_payload_requires_equal_complete_hourly_arrays():
    frame = parse_archive_payload(archive_payload())
    assert len(frame) == 48
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert frame.notna().all().all()

    broken = archive_payload()
    broken["hourly"]["wind_speed_10m"] = broken["hourly"]["wind_speed_10m"][:-1]
    with pytest.raises(ValueError, match="equal length"):
        parse_archive_payload(broken)


def test_greenlight_conversion_keeps_calendar_position():
    payload = archive_payload(start="2023-03-01", periods=24)
    payload["hourly"]["precipitation"][3] = 1.25
    frame = parse_archive_payload(payload)
    converted = to_greenlight_weather(frame)

    assert converted["time"].iloc[0] == 59 * 86400.0
    assert converted["day number"].iloc[0] == 60
    assert converted["CO2 concentration"].eq(400.0).all()
    assert converted["global radiation"].ge(0).all()
    assert converted["RH"].between(0, 100).all()
    assert "precipitation accumulation" in converted.columns
    assert converted["precipitation accumulation"].iloc[3] == pytest.approx(1.25)
    assert converted["precipitation accumulation"].sum() == pytest.approx(
        frame["precipitation"].sum()
    )
