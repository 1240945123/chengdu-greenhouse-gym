from __future__ import annotations

import calendar
import hashlib
import math
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable, Mapping
from glassgym.core.types import WeatherScenario, WeatherScenarioV2, WeatherSourceV2
from glassgym.environments.utils import load_weather_data, load_weather_data_v2

WEATHER_REPOSITORY_SCHEMA_VERSION = "weather-repository-v2"
LEGACY_WEATHER_REPOSITORY_VERSION = "weather-repository-legacy-v1"


class WeatherRepository:
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "WeatherRepository":
        schema = str(config.get("schema_version", "legacy")).strip().lower()
        configured_loader = config.get("load_weather_data_fn")
        if schema in {"legacy", "v1", LEGACY_WEATHER_REPOSITORY_VERSION}:
            loader = load_weather_data
            loader_version = LEGACY_WEATHER_REPOSITORY_VERSION
            allowed_names = {None, "load_weather_data", "load_weather_data_legacy"}
        elif schema in {"v2", WEATHER_REPOSITORY_SCHEMA_VERSION}:
            loader = load_weather_data_v2
            loader_version = WEATHER_REPOSITORY_SCHEMA_VERSION
            allowed_names = {None, "load_weather_data_v2"}
        else:
            raise ValueError(f"Unsupported weather schema_version: {schema}")

        if configured_loader not in allowed_names:
            raise ValueError(
                "weather schema_version conflicts with load_weather_data_fn: "
                f"{configured_loader}"
            )
        return cls(
            weather_data_dir=config["weather_data_dir"],
            load_weather_data_fn=loader,
            loader_version=loader_version,
        )

    def __init__(
        self,
        weather_data_dir: str | Path,
        load_weather_data_fn: Callable[Any, np.ndarray],
        loader_version: str | None = None,
    ) -> None:
        self.weather_data_dir = Path(weather_data_dir)
        self.load_weather_data_fn = load_weather_data_fn
        self.loader_version = loader_version or (
            LEGACY_WEATHER_REPOSITORY_VERSION
            if load_weather_data_fn is load_weather_data
            else WEATHER_REPOSITORY_SCHEMA_VERSION
        )
        self._cache: dict[str, np.ndarray] = {}

    def _source_years(
        self,
        *,
        growth_year: int,
        start_day: int,
        season_length: int,
        pred_horizon: int,
        timestep_seconds: float,
    ) -> tuple[int, ...]:
        years = [growth_year]
        duration = (float(season_length) + float(pred_horizon)) * 86_400.0
        expected_rows = math.ceil(duration / float(timestep_seconds))
        end_seconds = float(start_day) * 86_400.0 + expected_rows * float(timestep_seconds)
        year = growth_year
        year_seconds = (366 if calendar.isleap(year) else 365) * 86_400.0
        while end_seconds > year_seconds:
            end_seconds -= year_seconds
            year += 1
            years.append(year)
            year_seconds = (366 if calendar.isleap(year) else 365) * 86_400.0
        return tuple(years)

    def _file_checksum(self, path: Path) -> str:
        if not path.exists():
            return "missing"

        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _source_identity(
        self,
        *,
        location: str,
        growth_year: int,
        start_day: int,
        season_length: int,
        pred_horizon: int,
        timestep_seconds: float,
    ) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                str(path.resolve()),
                self._file_checksum(path),
            )
            for year in self._source_years(
                growth_year=growth_year,
                start_day=start_day,
                season_length=season_length,
                pred_horizon=pred_horizon,
                timestep_seconds=timestep_seconds,
            )
            for path in [self.weather_data_dir / location / f"{year}.csv"]
        )

    def describe(
        self,
        *,
        location: str,
        growth_year: int,
        start_day: int,
        season_length: int,
        pred_horizon: int,
        dt: int,
        nd: int,
    ) -> WeatherScenarioV2:
        loader_identity = (
            getattr(self.load_weather_data_fn, "__module__", ""),
            getattr(self.load_weather_data_fn, "__qualname__", repr(self.load_weather_data_fn)),
        )
        return WeatherScenarioV2(
            location=location,
            growth_year=growth_year,
            start_day=start_day,
            season_length=season_length,
            pred_horizon=pred_horizon,
            timestep_seconds=dt,
            disturbance_count=nd,
            loader_version=self.loader_version,
            loader_identity=loader_identity,
            source_files=tuple(
                WeatherSourceV2(path=path, sha256=checksum)
                for path, checksum in self._source_identity(
                    location=location,
                    growth_year=growth_year,
                    start_day=start_day,
                    season_length=season_length,
                    pred_horizon=pred_horizon,
                    timestep_seconds=dt,
                )
            ),
        )

    def load(
        self,
        *,
        location: str,
        growth_year: int,
        start_day: int,
        season_length: int,
        pred_horizon: int,
        dt: int,
        nd: int,
    ) -> np.ndarray:
        scenario = self.describe(
            location=location,
            growth_year=growth_year,
            start_day=start_day,
            season_length=season_length,
            pred_horizon=pred_horizon,
            dt=dt,
            nd=nd,
        )
        key = scenario.fingerprint
        if key not in self._cache:
            arr = self.load_weather_data_fn(
                weather_data_dir=self.weather_data_dir,
                location=location,
                growth_year=growth_year,
                start_day=start_day,
                n_days=season_length,
                pred_horizon=pred_horizon,
                h=dt,
                nd=nd,
            )
            cached = np.array(arr, dtype=np.float64, copy=True)
            if self.loader_version == WEATHER_REPOSITORY_SCHEMA_VERSION:
                expected_shape = (scenario.expected_rows, scenario.disturbance_count)
                if cached.shape != expected_shape:
                    raise ValueError(
                        "weather manifest expected shape "
                        f"{expected_shape}, received {cached.shape}"
                    )
                if not np.all(np.isfinite(cached)):
                    raise ValueError("weather manifest requires finite values")
            cached.setflags(write=False)
            self._cache[key] = cached
        return self._cache[key].copy()


class BaseWeatherSampler(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        ...

class FixedWeatherSampler(BaseWeatherSampler):
    def __init__(
        self,
        location: str,
        growth_year: int,
        start_day: int,
    ):
        self.location = location
        self.growth_year = growth_year
        self.start_day = start_day

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        return WeatherScenario(location=self.location, growth_year=self.growth_year, start_day=self.start_day)

class RandomWeatherSampler(BaseWeatherSampler):
    def __init__(
        self,
        locations: List[str],
        growth_years: List[int],
        start_days: List[int] | range,
    ):
        self.locations = locations
        self.growth_years = growth_years
        self.start_days = start_days

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        location = rng.choice(self.locations)
        growth_year = rng.choice(self.growth_years)
        start_day = rng.choice(self.start_days)
        return WeatherScenario(location=location, growth_year=growth_year, start_day=start_day)


class RandomScenarioWeatherSampler(BaseWeatherSampler):
    """Randomly sample complete scenarios without creating invalid cross-products."""

    def __init__(self, scenarios: List[Dict[str, Any]]):
        if not scenarios:
            raise ValueError("At least one weather scenario is required")
        self.scenarios = [
            WeatherScenario(
                location=str(scenario["location"]),
                growth_year=int(scenario["growth_year"]),
                start_day=int(scenario["start_day"]),
            )
            for scenario in scenarios
        ]

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        return self.scenarios[int(rng.integers(0, len(self.scenarios)))]

class CyclingWeatherSampler(BaseWeatherSampler):
    """
    Deterministic cycling through a predefined list of scenarios.
    Useful for repeated comparable evaluation.
    """
    def __init__(self, scenarios: List[Dict[str, Any]]):
        self.scenarios = [
            WeatherScenario(location=scenario["location"], growth_year=scenario["growth_year"], start_day=scenario["start_day"]) for scenario in scenarios
        ]
        self._i = 0

    def sample(self, rng: np.random.Generator, options: Dict[str, Any] | None = None) -> WeatherScenario:
        if options is not None and "scenario_index" in options:
            idx = int(options["scenario_index"]) % len(self.scenarios)
            return self.scenarios[idx]

        scenario = self.scenarios[self._i % len(self.scenarios)]
        self._i += 1
        return scenario

WEATHER_SAMPLERS = {
    "fixed": FixedWeatherSampler,
    "random": RandomWeatherSampler,
    "random_scenarios": RandomScenarioWeatherSampler,
    "cycling": CyclingWeatherSampler,
}
