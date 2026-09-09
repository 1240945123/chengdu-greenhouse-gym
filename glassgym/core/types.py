from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import ClassVar
import numpy as np


@dataclass(frozen=True)
class DisturbanceFieldV2:
    index: int
    name: str
    unit: str
    temporal_semantics: str
    source_column: str | None


_DISTURBANCE_FIELDS_V2 = (
    DisturbanceFieldV2(0, "global_radiation", "W/m2", "interval_mean", "global radiation"),
    DisturbanceFieldV2(1, "outdoor_air_temperature", "degC", "instantaneous_state", "air temperature"),
    DisturbanceFieldV2(2, "outdoor_vapor_pressure", "Pa", "derived_state", "RH"),
    DisturbanceFieldV2(3, "outdoor_co2_density", "mg/m3", "derived_state", None),
    DisturbanceFieldV2(4, "outdoor_wind_speed", "m/s", "instantaneous_state", "wind speed"),
    DisturbanceFieldV2(5, "sky_temperature", "degC", "instantaneous_state", "sky temperature"),
    DisturbanceFieldV2(6, "outdoor_soil_temperature", "degC", "calendar_derived_state", None),
    DisturbanceFieldV2(7, "daily_light_integral", "MJ/m2/day", "full_calendar_day_integral", "global radiation"),
    DisturbanceFieldV2(8, "is_day", "1", "derived_indicator", "global radiation"),
    DisturbanceFieldV2(9, "is_day_smooth", "1", "derived_indicator", "global radiation"),
    DisturbanceFieldV2(10, "precipitation_rate", "mm/h", "interval_rate", "precipitation accumulation"),
)


@dataclass(frozen=True)
class DisturbanceSchemaV2:
    schema_version: ClassVar[str] = "disturbance-schema-v2"

    fields: tuple[DisturbanceFieldV2, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        if len(fields) not in (10, 11) or fields != _DISTURBANCE_FIELDS_V2[:len(fields)]:
            raise ValueError(
                "DisturbanceSchemaV2 must use a canonical 10- or 11-field definition"
            )

    @classmethod
    def for_count(cls, count: int) -> "DisturbanceSchemaV2":
        count = int(count)
        if count not in (10, 11):
            raise ValueError(
                "DisturbanceSchemaV2 supports only the 10-column compatibility "
                "schema or the 11-column precipitation schema"
            )
        return cls(fields=_DISTURBANCE_FIELDS_V2[:count])

    @property
    def count(self) -> int:
        return len(self.fields)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "count": self.count,
            "fields": [asdict(field) for field in self.fields],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ActionFieldV2:
    index: int
    name: str
    unit: str
    low: float
    high: float
    default: float
    primary_enabled: bool


_ACTION_FIELDS_V2 = (
    ActionFieldV2(0, "uBoil", "fraction", 0.0, 1.0, 0.0, False),
    ActionFieldV2(1, "uCO2", "fraction", 0.0, 1.0, 0.0, False),
    ActionFieldV2(2, "uThScr", "fraction", 0.0, 1.0, 0.0, True),
    ActionFieldV2(3, "uVent", "fraction", 0.0, 1.0, 0.0, True),
    ActionFieldV2(4, "uLamp", "fraction", 0.0, 1.0, 0.0, True),
    ActionFieldV2(5, "uBlScr", "fraction", 0.0, 1.0, 0.0, True),
    ActionFieldV2(6, "uPadFan", "fraction", 0.0, 1.0, 0.0, False),
    ActionFieldV2(7, "uPadPump", "fraction", 0.0, 1.0, 0.0, False),
)


@dataclass(frozen=True)
class ActionSchemaV2:
    schema_version: ClassVar[str] = "action-schema-v2"

    fields: tuple[ActionFieldV2, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        object.__setattr__(self, "fields", fields)
        if len(fields) not in (6, 8) or fields != _ACTION_FIELDS_V2[:len(fields)]:
            raise ValueError("ActionSchemaV2 must use a canonical 6- or 8-field definition")

    @classmethod
    def for_count(cls, count: int) -> "ActionSchemaV2":
        count = int(count)
        if count not in (6, 8):
            raise ValueError("ActionSchemaV2 supports only 6 or 8 controls")
        return cls(fields=_ACTION_FIELDS_V2[:count])

    @property
    def count(self) -> int:
        return len(self.fields)

    def to_manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "count": self.count,
            "fields": [asdict(field) for field in self.fields],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

@dataclass(frozen=True)
class WeatherScenario:
    location: str       # location of the recorded weather data
    growth_year: int    # growth year (i.e., year of the simulation)
    start_day: int      # start day of the year (i.e., day of the year of the simulation)


@dataclass(frozen=True)
class WeatherSourceV2:
    path: str
    sha256: str


@dataclass(frozen=True)
class WeatherScenarioV2:
    schema_version: ClassVar[str] = "weather-scenario-v2"

    location: str
    growth_year: int
    start_day: int
    season_length: float
    pred_horizon: float
    timestep_seconds: float
    disturbance_count: int
    loader_version: str
    loader_identity: tuple[str, str]
    source_files: tuple[WeatherSourceV2, ...]
    disturbance_schema_version: str = DisturbanceSchemaV2.schema_version

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", str(self.location))
        object.__setattr__(self, "growth_year", int(self.growth_year))
        object.__setattr__(self, "start_day", int(self.start_day))
        object.__setattr__(self, "season_length", float(self.season_length))
        object.__setattr__(self, "pred_horizon", float(self.pred_horizon))
        object.__setattr__(self, "timestep_seconds", float(self.timestep_seconds))
        object.__setattr__(self, "disturbance_count", int(self.disturbance_count))
        if self.start_day < 0:
            raise ValueError("Weather scenario start_day must be non-negative")
        if self.season_length <= 0 or self.pred_horizon < 0:
            raise ValueError("Weather scenario durations are invalid")
        if self.timestep_seconds <= 0 or self.disturbance_count <= 0:
            raise ValueError("Weather scenario timestep and dimensions must be positive")
        DisturbanceSchemaV2.for_count(self.disturbance_count)
        if self.disturbance_schema_version != DisturbanceSchemaV2.schema_version:
            raise ValueError("Unsupported disturbance schema version")
        if not self.source_files:
            raise ValueError("Weather scenario requires at least one source file")

    @property
    def expected_rows(self) -> int:
        duration_seconds = (self.season_length + self.pred_horizon) * 86_400.0
        return int(math.ceil(duration_seconds / self.timestep_seconds))

    def to_manifest(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = self.schema_version
        payload["expected_rows"] = self.expected_rows
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

@dataclass
class EnvState:
    x: np.ndarray
    u_full: np.ndarray
    p: np.ndarray
    timestep: int
    scenario: WeatherScenario
    weather_data: np.ndarray

@dataclass(frozen=True)
class StepContext:
    t: int # timestep
    dt: int # [s] discretization time (i.e., control interval) for the underlying GreenLight solver
    Np: int # number of future weather predictions
    
    x_prev: np.ndarray          # previous state vector
    x: np.ndarray               # current state vector
    u: np.ndarray               # control input vector from the agent
    p: np.ndarray               # model parameters vector
    d: np.ndarray               # weather disturbances for all timesteps

    hour_of_day: float          # [h] hour of the days at timestep t
    day_of_year: float          # [d] day of the year at timestep t
    forecast: object | None = field(
        default=None,
        kw_only=True,
    )  # issue-time forecast; never realized future truth


@dataclass(frozen=True)
class ControllerStepContextV2:
    t: int
    dt: float
    indoor_temperature: float
    indoor_relative_humidity: float
    u: np.ndarray
    forecast: object
    hour_of_day: float
    day_of_year: float

    def __post_init__(self) -> None:
        controls = np.asarray(self.u, dtype=np.float64).copy()
        if controls.shape != (6,) or not np.all(np.isfinite(controls)):
            raise ValueError("ControllerStepContextV2 requires six finite controls")
        if not np.isfinite(self.indoor_temperature):
            raise ValueError("Controller indoor temperature must be finite")
        if not np.isfinite(self.indoor_relative_humidity):
            raise ValueError("Controller indoor relative humidity must be finite")
        if self.forecast is None or self.forecast.issue_timestep != int(self.t):
            raise ValueError("Controller context requires a matching issue-time forecast")
        controls.setflags(write=False)
        object.__setattr__(self, "u", controls)

    
@dataclass(frozen=True)
class RewardContext(StepContext):
    obs: dict[str, np.ndarray]
    constraints_low: np.ndarray | None = None
    constraints_high: np.ndarray | None = None
    u_prev: np.ndarray | None = None
