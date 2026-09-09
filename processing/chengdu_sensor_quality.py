from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _channel_pattern(variable: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(variable)}__e(?P<equipment>\d+)__s\d+$")


def discover_sensor_channels(
    frame: pd.DataFrame,
    *,
    variable: str,
    indoor_equipment_ids: tuple[int, ...],
) -> list[str]:
    pattern = _channel_pattern(variable)
    allowed = {int(value) for value in indoor_equipment_ids}
    channels = []
    for column in frame.columns:
        match = pattern.match(column)
        if match and int(match.group("equipment")) in allowed:
            channels.append(column)
    return sorted(channels)


def derive_hourly_variable_quality(
    frame: pd.DataFrame,
    *,
    variable: str,
    indoor_equipment_ids: tuple[int, ...],
    coverage_threshold: float = 0.8,
    high_disagreement: float = 1.0,
    medium_disagreement: float = 3.0,
) -> pd.DataFrame:
    if not 0.0 < float(coverage_threshold) <= 1.0:
        raise ValueError("coverage_threshold must be inside (0, 1]")
    if not 0.0 <= float(high_disagreement) <= float(medium_disagreement):
        raise ValueError("disagreement thresholds must be nonnegative and ordered")
    channels = discover_sensor_channels(
        frame,
        variable=variable,
        indoor_equipment_ids=indoor_equipment_ids,
    )
    if not channels:
        raise ValueError(f"No channel-level {variable} measurements were found")
    coverage_columns = [f"{channel}_observed_fraction" for channel in channels]
    missing_coverage = [column for column in coverage_columns if column not in frame]
    if missing_coverage:
        raise ValueError(f"Missing channel-level observation fractions: {missing_coverage}")

    values = frame[channels].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    coverage = frame[coverage_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    finite = np.isfinite(values)
    available = finite & (coverage > 0.0)
    effective = finite & (coverage >= float(coverage_threshold))
    available_count = available.sum(axis=1).astype(int)
    effective_count = effective.sum(axis=1).astype(int)
    maximum = np.max(np.where(effective, values, -np.inf), axis=1)
    minimum = np.min(np.where(effective, values, np.inf), axis=1)
    disagreement = maximum - minimum
    disagreement[effective_count < 2] = np.nan
    maximum_coverage = np.max(np.where(finite, coverage, 0.0), axis=1)

    high = (effective_count >= 2) & (disagreement <= float(high_disagreement))
    medium = (effective_count >= 1) & (
        (effective_count == 1) | (disagreement <= float(medium_disagreement))
    )
    tier = np.full(len(frame), "low", dtype=object)
    tier[medium] = "medium"
    tier[high] = "high"
    weights = np.select([high, medium], [1.0, 0.6], default=0.0).astype(float)

    result = pd.DataFrame(index=frame.index)
    if "timestamp" in frame:
        result["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    result[f"{variable}_available_sensor_count"] = available_count
    result[f"{variable}_effective_sensor_count"] = effective_count
    result[f"{variable}_maximum_observed_fraction"] = maximum_coverage
    result[f"{variable}_sensor_disagreement"] = disagreement
    result[f"{variable}_quality_tier"] = tier
    result[f"{variable}_quality_weight"] = weights
    return result
