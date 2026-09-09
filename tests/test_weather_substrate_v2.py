from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from glassgym.components.weather import WeatherRepository
from glassgym.core import types as core_types
from glassgym.environments import utils as weather_utils


load_weather_data_v2 = getattr(
    weather_utils,
    "load_weather_data_v2",
    weather_utils.load_weather_data,
)


SECONDS_PER_DAY = 86_400


def _expected_loader_rows(kwargs: dict[str, object]) -> int:
    duration_days = float(kwargs["n_days"]) + float(kwargs["pred_horizon"])
    return int(np.ceil(duration_days * SECONDS_PER_DAY / float(kwargs["h"])))


def _write_hourly_weather(
    root: Path,
    *,
    location: str,
    year: int,
    days: int,
    temperature: float,
    radiation: np.ndarray | None = None,
    precipitation_accumulation: np.ndarray | None = None,
) -> None:
    sample_count = days * 24
    time = np.arange(sample_count, dtype=np.float64) * 3_600.0
    if radiation is None:
        hour = np.arange(sample_count) % 24
        radiation = np.where((hour >= 6) & (hour < 18), 400.0, 0.0)
    frame = pd.DataFrame(
        {
            "time": time,
            "global radiation": np.asarray(radiation, dtype=np.float64),
            "wind speed": np.full(sample_count, 2.0),
            "air temperature": np.full(sample_count, temperature),
            "sky temperature": np.full(sample_count, temperature - 5.0),
            "RH": np.full(sample_count, 70.0),
        }
    )
    if precipitation_accumulation is not None:
        frame["precipitation accumulation"] = np.asarray(
            precipitation_accumulation,
            dtype=np.float64,
        )
    target = root / location
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / f"{year}.csv", index=False)


def test_hourly_source_uses_exact_15_minute_cross_year_target_grid(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=10.0,
    )
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2024,
        days=366,
        temperature=20.0,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=300,
        n_days=180,
        pred_horizon=0.5,
        h=900,
        nd=10,
    )

    expected_rows = int(np.ceil((180.0 + 0.5) * SECONDS_PER_DAY / 900.0))
    assert weather.shape == (expected_rows, 10)
    assert np.any(np.isclose(weather[:, 1], 10.0))
    assert np.any(np.isclose(weather[:, 1], 20.0))


def test_radiation_resampling_preserves_interval_energy(tmp_path: Path) -> None:
    location = "Synthetic"
    sample_count = 365 * 24
    radiation = np.zeros(sample_count, dtype=np.float64)
    radiation[6:18] = 400.0
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        radiation=radiation,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=900,
        nd=10,
    )

    source_energy_j_m2 = float(np.sum(radiation[:24]) * 3_600.0)
    target_energy_j_m2 = float(np.sum(weather[:, 0]) * 900.0)
    assert weather.shape == (96, 10)
    assert target_energy_j_m2 == source_energy_j_m2
    assert weather[:, 1].min() >= 15.0
    assert weather[:, 1].max() <= 15.0


def test_hourly_grid_handles_leap_year_boundary_and_forecast_padding(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2024,
        days=366,
        temperature=14.0,
    )
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2025,
        days=365,
        temperature=25.0,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2024,
        start_day=365,
        n_days=1,
        pred_horizon=0.25,
        h=3_600,
        nd=10,
    )

    assert weather.shape == (30, 10)
    np.testing.assert_allclose(weather[:24, 1], 14.0)
    np.testing.assert_allclose(weather[24:, 1], 25.0)


def test_cross_year_grid_fails_when_next_year_source_is_missing(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=10.0,
    )

    with pytest.raises(FileNotFoundError, match=r"2024\.csv"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=364,
            n_days=2,
            pred_horizon=0,
            h=3_600,
            nd=10,
        )


def test_weather_cache_identity_includes_all_load_arguments(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_loader(**kwargs: object) -> np.ndarray:
        calls.append(kwargs)
        rows = int(float(kwargs["n_days"]) * SECONDS_PER_DAY / float(kwargs["h"]))
        return np.full((rows, int(kwargs["nd"])), len(calls), dtype=np.float64)

    repository = WeatherRepository(tmp_path, fake_loader)
    common = {
        "location": "Synthetic",
        "growth_year": 2023,
        "start_day": 10,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }

    one_day = repository.load(season_length=1, **common)
    two_days = repository.load(season_length=2, **common)

    assert len(calls) == 2
    assert one_day.shape == (24, 10)
    assert two_days.shape == (48, 10)


def test_weather_cache_returns_isolated_arrays(tmp_path: Path) -> None:
    call_count = 0

    def fake_loader(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.ones((24, int(kwargs["nd"])), dtype=np.float64)

    repository = WeatherRepository(tmp_path, fake_loader)
    arguments = {
        "location": "Synthetic",
        "growth_year": 2023,
        "start_day": 0,
        "season_length": 1,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }

    first = repository.load(**arguments)
    first[0, 0] = 999.0
    second = repository.load(**arguments)

    assert call_count == 1
    assert second[0, 0] == 1.0
    assert not np.shares_memory(first, second)


def test_weather_cache_invalidates_when_source_checksum_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Synthetic" / "2023.csv"
    source.parent.mkdir(parents=True)
    source.write_text("version-one", encoding="utf-8")
    call_count = 0

    def fake_loader(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.full((24, int(kwargs["nd"])), call_count, dtype=np.float64)

    repository = WeatherRepository(tmp_path, fake_loader)
    arguments = {
        "location": "Synthetic",
        "growth_year": 2023,
        "start_day": 0,
        "season_length": 1,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }

    first = repository.load(**arguments)
    source.write_text("version-two", encoding="utf-8")
    second = repository.load(**arguments)

    assert call_count == 2
    assert first[0, 0] == 1.0
    assert second[0, 0] == 2.0


@pytest.mark.parametrize(
    ("argument", "changed_value"),
    [
        ("location", "Other"),
        ("growth_year", 2024),
        ("start_day", 1),
        ("pred_horizon", 0.5),
        ("dt", 900),
        ("nd", 11),
    ],
)
def test_weather_cache_identity_tracks_each_load_argument(
    tmp_path: Path,
    argument: str,
    changed_value: object,
) -> None:
    call_count = 0

    def fake_loader(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.zeros(
            (_expected_loader_rows(kwargs), int(kwargs["nd"])),
            dtype=np.float64,
        )

    repository = WeatherRepository(tmp_path, fake_loader)
    baseline = {
        "location": "Synthetic",
        "growth_year": 2023,
        "start_day": 0,
        "season_length": 1,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }
    changed = {**baseline, argument: changed_value}

    repository.load(**baseline)
    repository.load(**changed)

    assert call_count == 2


def test_cross_year_cache_identity_includes_next_year_checksum(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    first_source = tmp_path / location / "2023.csv"
    second_source = tmp_path / location / "2024.csv"
    first_source.parent.mkdir(parents=True)
    first_source.write_text("first-year", encoding="utf-8")
    second_source.write_text("next-year-v1", encoding="utf-8")
    call_count = 0

    def fake_loader(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.zeros(
            (_expected_loader_rows(kwargs), int(kwargs["nd"])),
            dtype=np.float64,
        )

    repository = WeatherRepository(tmp_path, fake_loader)
    arguments = {
        "location": location,
        "growth_year": 2023,
        "start_day": 300,
        "season_length": 180,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }

    repository.load(**arguments)
    second_source.write_text("next-year-v2", encoding="utf-8")
    repository.load(**arguments)

    assert call_count == 2


def test_legacy_weather_loader_remains_explicitly_available(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )

    legacy = weather_utils.load_weather_data_legacy(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=900,
        nd=10,
    )

    assert legacy.shape == (100, 10)


def test_weather_scenario_v2_manifest_is_complete_and_stable(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    source_2023 = tmp_path / location / "2023.csv"
    source_2024 = tmp_path / location / "2024.csv"
    source_2023.parent.mkdir(parents=True)
    source_2023.write_text("year-2023", encoding="utf-8")
    source_2024.write_text("year-2024", encoding="utf-8")
    repository = WeatherRepository(tmp_path, lambda **_: np.zeros((1, 10)))
    arguments = {
        "location": location,
        "growth_year": 2023,
        "start_day": 300,
        "season_length": 180,
        "pred_horizon": 0.5,
        "dt": 900,
        "nd": 10,
    }

    first = repository.describe(**arguments)
    second = repository.describe(**arguments)

    assert isinstance(first, core_types.WeatherScenarioV2)
    assert first.schema_version == "weather-scenario-v2"
    assert first.expected_rows == 17_328
    assert first.loader_version == "weather-repository-v2"
    assert len(first.source_files) == 2
    assert all(len(source.sha256) == 64 for source in first.source_files)
    assert len(first.fingerprint) == 64
    assert first == second
    assert first.fingerprint == second.fingerprint


def test_original_weather_loader_name_retains_legacy_semantics(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )

    weather = weather_utils.load_weather_data(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=900,
        nd=10,
    )

    assert weather.shape == (100, 10)


def test_manifest_includes_year_crossed_only_by_rounded_target_edge(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    for year in (2023, 2024):
        source = tmp_path / location / f"{year}.csv"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(str(year), encoding="utf-8")
    repository = WeatherRepository(tmp_path, lambda **_: np.zeros((18, 10)))

    scenario = repository.describe(
        location=location,
        growth_year=2023,
        start_day=364,
        season_length=1,
        pred_horizon=0,
        dt=5_000,
        nd=10,
    )

    assert scenario.expected_rows == 18
    assert len(scenario.source_files) == 2
    assert scenario.source_files[-1].path.endswith("2024.csv")


def test_negative_radiation_is_sanitized_before_energy_conservation(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    radiation = np.zeros(365 * 24, dtype=np.float64)
    radiation[0] = -10.0
    radiation[1] = 100.0
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        radiation=radiation,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0.5 / 24,
        n_days=1 / 24,
        pred_horizon=0,
        h=3_600,
        nd=10,
    )

    sanitized_energy = 100.0 * 1_800.0
    assert weather.shape == (1, 10)
    assert float(np.sum(weather[:, 0]) * 3_600.0) == sanitized_energy


def test_checksum_detects_same_size_same_timestamp_source_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Synthetic" / "2023.csv"
    source.parent.mkdir(parents=True)
    source.write_text("version-one", encoding="utf-8")
    original_mtime = source.stat().st_mtime_ns
    call_count = 0

    def fake_loader(**kwargs: object) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return np.full((24, int(kwargs["nd"])), call_count, dtype=np.float64)

    repository = WeatherRepository(tmp_path, fake_loader)
    arguments = {
        "location": "Synthetic",
        "growth_year": 2023,
        "start_day": 0,
        "season_length": 1,
        "pred_horizon": 0,
        "dt": 3_600,
        "nd": 10,
    }
    repository.load(**arguments)
    source.write_text("version-two", encoding="utf-8")
    os.utime(source, ns=(original_mtime, original_mtime))
    repository.load(**arguments)

    assert call_count == 2


@pytest.mark.parametrize(
    "bad_result",
    [
        np.zeros((23, 10), dtype=np.float64),
        np.zeros((24, 9), dtype=np.float64),
        np.full((24, 10), np.nan, dtype=np.float64),
    ],
)
def test_repository_enforces_manifest_shape_and_finite_values(
    tmp_path: Path,
    bad_result: np.ndarray,
) -> None:
    repository = WeatherRepository(tmp_path, lambda **_: bad_result)

    with pytest.raises(ValueError, match="weather manifest"):
        repository.load(
            location="Synthetic",
            growth_year=2023,
            start_day=0,
            season_length=1,
            pred_horizon=0,
            dt=3_600,
            nd=10,
        )


def test_v2_rejects_source_that_does_not_begin_at_year_start(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )
    path = tmp_path / location / "2023.csv"
    frame = pd.read_csv(path)
    frame["time"] += 3_600.0
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="begin at zero"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=0,
            n_days=1,
            pred_horizon=0,
            h=900,
            nd=10,
        )


def test_v2_rejects_interior_source_timestamp_gap(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )
    path = tmp_path / location / "2023.csv"
    frame = pd.read_csv(path).drop(index=10)
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="regular cadence"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=0,
            n_days=1,
            pred_horizon=0,
            h=900,
            nd=10,
        )


def test_soil_temperature_phase_resets_after_leap_year(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2024,
        days=366,
        temperature=14.0,
    )
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2025,
        days=365,
        temperature=14.0,
    )

    first_january = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2024,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=3_600,
        nd=10,
    )
    next_january = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2024,
        start_day=366,
        n_days=1,
        pred_horizon=0,
        h=3_600,
        nd=10,
    )

    assert next_january[0, 6] == pytest.approx(first_january[0, 6])


def test_weather_repository_config_requires_explicit_v2_migration(
    tmp_path: Path,
) -> None:
    legacy = WeatherRepository.from_config({"weather_data_dir": str(tmp_path)})
    migrated = WeatherRepository.from_config(
        {
            "weather_data_dir": str(tmp_path),
            "schema_version": "v2",
        }
    )

    assert legacy.load_weather_data_fn is weather_utils.load_weather_data
    assert legacy.loader_version == "weather-repository-legacy-v1"
    assert migrated.load_weather_data_fn is weather_utils.load_weather_data_v2
    assert migrated.loader_version == "weather-repository-v2"

    with pytest.raises(ValueError, match="weather schema_version"):
        WeatherRepository.from_config(
            {
                "weather_data_dir": str(tmp_path),
                "schema_version": "v3",
            }
        )


@pytest.mark.parametrize("timestep", [1_000, 1_200, 3_601, 5_000])
def test_v2_day_night_derivation_accepts_any_positive_timestep(
    tmp_path: Path,
    timestep: int,
) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=timestep,
        nd=10,
    )

    expected_rows = int(np.ceil(SECONDS_PER_DAY / timestep))
    assert weather.shape == (expected_rows, 10)
    assert np.all(np.isfinite(weather[:, 8:10]))
    assert np.all((weather[:, 8:10] >= 0.0) & (weather[:, 8:10] <= 1.0))


def test_partial_day_request_reports_full_calendar_day_dli(tmp_path: Path) -> None:
    location = "Synthetic"
    radiation = np.full(365 * 24, 100.0, dtype=np.float64)
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        radiation=radiation,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0.5,
        n_days=0.5,
        pred_horizon=0,
        h=3_600,
        nd=10,
    )

    expected_dli = 100.0 * SECONDS_PER_DAY * 1e-6
    np.testing.assert_allclose(weather[:, 7], expected_dli)


def test_disturbance_schema_v2_declares_legacy_and_precipitation_semantics() -> None:
    legacy = core_types.DisturbanceSchemaV2.for_count(10)
    rainfall = core_types.DisturbanceSchemaV2.for_count(11)

    assert legacy.schema_version == "disturbance-schema-v2"
    assert legacy.count == 10
    assert rainfall.count == 11
    assert len(rainfall.fingerprint) == 64
    precipitation = rainfall.fields[10]
    assert precipitation.name == "precipitation_rate"
    assert precipitation.unit == "mm/h"
    assert precipitation.temporal_semantics == "interval_rate"
    assert precipitation.source_column == "precipitation accumulation"


def test_disturbance_schema_v2_canonicalizes_input_and_rejects_custom_fields() -> None:
    canonical = core_types.DisturbanceSchemaV2.for_count(10)
    mutable_fields = list(canonical.fields)
    reconstructed = core_types.DisturbanceSchemaV2(fields=mutable_fields)
    mutable_fields.pop()

    assert isinstance(reconstructed.fields, tuple)
    assert reconstructed == canonical
    assert reconstructed.count == 10

    custom = list(canonical.fields)
    custom[0] = core_types.DisturbanceFieldV2(
        0,
        "custom_radiation",
        "W/m2",
        "interval_mean",
        "global radiation",
    )
    with pytest.raises(ValueError, match="canonical"):
        core_types.DisturbanceSchemaV2(fields=custom)


def test_v2_precipitation_resampling_preserves_accumulation_without_smearing(
    tmp_path: Path,
) -> None:
    location = "Synthetic"
    source_accumulation = np.zeros(365 * 24, dtype=np.float64)
    source_accumulation[6] = 2.0
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        precipitation_accumulation=source_accumulation,
    )

    weather = load_weather_data_v2(
        weather_data_dir=str(tmp_path),
        location=location,
        growth_year=2023,
        start_day=0,
        n_days=1,
        pred_horizon=0,
        h=900,
        nd=11,
    )

    assert weather.shape == (96, 11)
    np.testing.assert_allclose(weather[24:28, 10], 2.0)
    np.testing.assert_allclose(weather[:24, 10], 0.0)
    np.testing.assert_allclose(weather[28:, 10], 0.0)
    assert float(np.sum(weather[:, 10]) * 0.25) == pytest.approx(2.0)


def test_v2_precipitation_rejects_negative_accumulation(tmp_path: Path) -> None:
    location = "Synthetic"
    source_accumulation = np.zeros(365 * 24, dtype=np.float64)
    source_accumulation[6] = -0.1
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        precipitation_accumulation=source_accumulation,
    )

    with pytest.raises(ValueError, match="precipitation accumulation"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=0,
            n_days=1,
            pred_horizon=0,
            h=900,
            nd=11,
        )


def test_v2_precipitation_requires_explicit_source_column(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )

    with pytest.raises(ValueError, match="precipitation accumulation"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=0,
            n_days=1,
            pred_horizon=0,
            h=900,
            nd=11,
        )


def test_v2_rejects_cross_year_source_cadence_change(tmp_path: Path) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
        precipitation_accumulation=np.zeros(365 * 24),
    )
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2024,
        days=366,
        temperature=15.0,
        precipitation_accumulation=np.zeros(366 * 24),
    )
    next_year_path = tmp_path / location / "2024.csv"
    next_year = pd.read_csv(next_year_path)
    half_hour = next_year.loc[next_year.index.repeat(2)].reset_index(drop=True)
    half_hour["time"] = np.arange(len(half_hour), dtype=float) * 1_800.0
    half_hour.to_csv(next_year_path, index=False)

    with pytest.raises(ValueError, match="cadence"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=364,
            n_days=2,
            pred_horizon=0,
            h=900,
            nd=11,
        )


@pytest.mark.parametrize("nd", [9, 12])
def test_v2_rejects_undeclared_disturbance_dimensions(tmp_path: Path, nd: int) -> None:
    location = "Synthetic"
    _write_hourly_weather(
        tmp_path,
        location=location,
        year=2023,
        days=365,
        temperature=15.0,
    )

    with pytest.raises(ValueError, match="DisturbanceSchemaV2"):
        load_weather_data_v2(
            weather_data_dir=str(tmp_path),
            location=location,
            growth_year=2023,
            start_day=0,
            n_days=1,
            pred_horizon=0,
            h=900,
            nd=nd,
        )
