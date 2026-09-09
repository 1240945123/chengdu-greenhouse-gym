from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class ClimateStatePostprocessor(Protocol):
    def correct(
        self,
        *,
        previous_state: np.ndarray,
        raw_state: np.ndarray,
        controls: np.ndarray,
        disturbance: np.ndarray,
        dt_seconds: float,
        hour_of_day: float,
        day_of_year: float,
    ) -> tuple[np.ndarray, dict[str, Any]]: ...
