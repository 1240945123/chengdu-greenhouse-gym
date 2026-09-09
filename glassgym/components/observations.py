from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gymnasium import spaces

from glassgym.environments.utils import co2dens2ppm, vaporPres2rh
from glassgym.core.types import StepContext

class BaseObservations(ABC):
    def __init__(self, env) -> None:
        self.Np = env.Np
        self.nu = env.nu

    @property
    def key(self) -> str:
        ...

    @property
    def space(self) -> spaces.Box:
        ...

    @abstractmethod
    def compute_obs(self, ctx: StepContext):
        ...

class StateObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "StateObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32)

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        return np.array(ctx.x, dtype=np.float32)

class IndoorClimateObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "IndoorClimateObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["co2_air", "temp_air", "rh_air", "pipe_temp"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        climate_obs = np.array(ctx.x, dtype=np.float32)[[0, 2, 15, 9]]
        climate_obs[0] = co2dens2ppm(climate_obs[1], climate_obs[0]*1e-6)
        climate_obs[2] = vaporPres2rh(climate_obs[1], climate_obs[2])
        return climate_obs

class BasicCropObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "BasicCropObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["tCan24", "cFruit", "tCanSum"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        return np.array(ctx.x, dtype=np.float32)[[21, 25, 26]]

class ControlObservations(BaseObservations):
    """Observer module, which gives control over the observations we want to our RL algorithm to use.
    """
    @property
    def key(self) -> str:
        return "ControlObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=0., high=1., shape=(self.nu,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        names = [
            "uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr",
            "uPadFan", "uPadPump",
        ]
        return names[:self.nu]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        return np.asarray(ctx.u, dtype=np.float32)

class WeatherObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "WeatherObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        weather_obs = np.copy(ctx.d[ctx.t, :5]).astype(np.float32)
        weather_obs[2] = vaporPres2rh(weather_obs[1], weather_obs[2])
        weather_obs[3] = co2dens2ppm(weather_obs[1], weather_obs[3]*1e-6)
        return weather_obs


class WeatherObservationsV2(BaseObservations):
    @property
    def key(self) -> str:
        return "WeatherObservationsV2"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind", "precipitation_rate"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        if ctx.d.shape[1] < 11:
            raise ValueError("WeatherObservationsV2 requires DisturbanceSchemaV2 with precipitation")
        weather_obs = np.copy(ctx.d[ctx.t, [0, 1, 2, 3, 4, 10]]).astype(np.float32)
        weather_obs[2] = vaporPres2rh(weather_obs[1], weather_obs[2])
        weather_obs[3] = co2dens2ppm(weather_obs[1], weather_obs[3] * 1e-6)
        return weather_obs

class TimeObservations(BaseObservations):
    @property
    def key(self) -> str:
        return "TimeObservations"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["day_of_year_sin", "day_of_year_cos", "hour_of_day_sin", "hour_of_day_cos"]

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        day_of_year_sin = np.sin(2 * np.pi * ctx.day_of_year / 365.0)
        day_of_year_cos = np.cos(2 * np.pi * ctx.day_of_year / 365.0)
        hour_of_day_sin = np.sin(2 * np.pi * ctx.hour_of_day / 24.0)
        hour_of_day_cos = np.cos(2 * np.pi * ctx.hour_of_day / 24.0)
        return np.array([day_of_year_sin, day_of_year_cos, hour_of_day_sin, hour_of_day_cos], dtype=np.float32)

class WeatherForecastObservationsLegacy(BaseObservations):
    @property
    def key(self) -> str:
        return "WeatherForecastObservationsLegacy"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(5*self.Np,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind"] * self.Np

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        forecast = []
        for i in range(1, self.Np+1):
            forecast.extend(ctx.d[ctx.t+i][0:5])
        return np.array(forecast, dtype=np.float32)


class WeatherForecastObservationsV2(BaseObservations):
    @property
    def key(self) -> str:
        return "WeatherForecastObservationsV2"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(5*self.Np,), dtype=np.float32)

    @property
    def obs_names(self) -> List[str]:
        return ["radiation", "temperature", "rh", "co2", "wind"] * self.Np

    def compute_obs(self, ctx: StepContext) -> np.ndarray:
        if ctx.forecast is None:
            raise ValueError("WeatherForecastObservationsV2 requires an issue-time forecast")
        if ctx.forecast.issue_timestep != ctx.t:
            raise ValueError("Forecast issue timestep does not match the observation timestep")
        if len(ctx.forecast.values) < self.Np:
            raise ValueError("Forecast horizon is shorter than the observation horizon")
        return np.asarray(ctx.forecast.values[:self.Np, :5], dtype=np.float32).reshape(-1)


OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ControlObservations": ControlObservations,
    "WeatherObservations": WeatherObservations,
    "WeatherObservationsV2": WeatherObservationsV2,
    "WeatherForecastObservationsLegacy": WeatherForecastObservationsLegacy,
    "WeatherForecastObservationsV2": WeatherForecastObservationsV2,
    "TimeObservations": TimeObservations,
}
