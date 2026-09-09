import numpy as np
import pandas as pd

from processing.chengdu_era5 import (
    apply_local_bias_correction,
    build_six_seasons,
    fit_local_bias_correction,
)


def full_year(year):
    timestamps = pd.date_range(
        f"{year}-01-01", f"{year}-12-31 23:00", freq="h"
    )
    hour = np.arange(len(timestamps))
    radiation = np.maximum(0, 500 * np.sin((hour % 24 - 6) * np.pi / 12))
    return pd.DataFrame({
        "timestamp": timestamps,
        "air_temperature": 20 + 5 * np.sin(hour * 2 * np.pi / (24 * 365)),
        "relative_humidity": 70 - 10 * np.sin(hour * 2 * np.pi / (24 * 365)),
        "global_radiation": radiation,
        "wind_speed": np.full(len(hour), 1.5),
        "precipitation": np.zeros(len(hour)),
        "soil_temperature_0_to_7cm": np.full(len(hour), 18.0),
        "soil_temperature_28_to_100cm": np.full(len(hour), 17.0),
    })


def test_bias_correction_is_bounded_and_preserves_night_zero():
    era = full_year(2023).iloc[: 24 * 30].copy()
    observed = era.copy()
    observed["air_temperature"] += 2.0
    observed["relative_humidity"] += 4.0
    observed["global_radiation"] *= 1.1
    observed["wind_speed"] *= 0.8

    correction = fit_local_bias_correction(era, observed)
    corrected = apply_local_bias_correction(era, correction)

    assert corrected["relative_humidity"].between(0, 100).all()
    assert corrected["wind_speed"].ge(0).all()
    assert corrected.loc[era["global_radiation"].eq(0), "global_radiation"].eq(0).all()
    assert corrected["air_temperature"].mean() > era["air_temperature"].mean()


def test_builds_exactly_six_120_day_seasons_with_leap_year_boundaries():
    seasons = build_six_seasons({year: full_year(year) for year in (2023, 2024, 2025)})

    assert set(seasons) == {
        "2023_spring", "2023_autumn", "2024_spring",
        "2024_autumn", "2025_spring", "2025_autumn",
    }
    assert all(len(frame) == 2880 for frame in seasons.values())
    assert seasons["2023_spring"]["timestamp"].iloc[0] == pd.Timestamp("2023-03-01")
    assert seasons["2023_spring"]["timestamp"].iloc[-1] == pd.Timestamp("2023-06-28 23:00")
    assert seasons["2024_spring"]["timestamp"].iloc[0] == pd.Timestamp("2024-03-01")
    assert seasons["2025_autumn"]["timestamp"].iloc[-1] == pd.Timestamp("2025-12-12 23:00")
