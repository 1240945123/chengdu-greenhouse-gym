from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReproductiveHeatStressParameters:
    optimal_mean_temperature_c: float = 25.0
    severe_mean_temperature_c: float = 29.0
    floor_retention: float = 0.5
    trailing_window_hours: int = 240

    def __post_init__(self) -> None:
        values = (
            self.optimal_mean_temperature_c,
            self.severe_mean_temperature_c,
            self.floor_retention,
        )
        if not all(isfinite(float(value)) for value in values):
            raise ValueError("heat-stress parameters must be finite")
        if self.severe_mean_temperature_c <= self.optimal_mean_temperature_c:
            raise ValueError("severe temperature must exceed optimal temperature")
        if not 0.0 <= self.floor_retention <= 1.0:
            raise ValueError("floor retention must be in [0, 1]")
        if isinstance(self.trailing_window_hours, bool) or not isinstance(
            self.trailing_window_hours, int
        ):
            raise ValueError("trailing window hours must be an integer")
        if self.trailing_window_hours <= 0:
            raise ValueError("trailing window hours must be positive")


def trailing_heat_retention(
    air_temperature_c: np.ndarray | pd.Series | list[float],
    parameters: ReproductiveHeatStressParameters,
) -> np.ndarray:
    temperatures = np.asarray(air_temperature_c, dtype=float)
    if temperatures.ndim != 1 or temperatures.size == 0:
        raise ValueError("air temperatures must be a non-empty one-dimensional series")
    if not np.isfinite(temperatures).all():
        raise ValueError("air temperatures must be finite")
    trailing = (
        pd.Series(temperatures)
        .rolling(parameters.trailing_window_hours, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    width = (
        parameters.severe_mean_temperature_c
        - parameters.optimal_mean_temperature_c
    )
    scaled = np.clip(
        (trailing - parameters.optimal_mean_temperature_c) / width,
        0.0,
        1.0,
    )
    smoothstep = scaled * scaled * (3.0 - 2.0 * scaled)
    return 1.0 - (1.0 - parameters.floor_retention) * smoothstep

