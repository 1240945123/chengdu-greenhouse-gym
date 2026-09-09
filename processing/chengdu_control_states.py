from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeviceSpec:
    column: str
    actions: Mapping[str, float | None]


DEVICE_SPECS: dict[int, DeviceSpec] = {
    43: DeviceSpec("external_shade", {"TURN_ON": 1.0, "TURN_OFF": 0.0, "STOP": None}),
    44: DeviceSpec("roof_thermal_screen", {"TURN_ON": 1.0, "TURN_OFF": 0.0, "STOP": None}),
    45: DeviceSpec("side_thermal_screen", {"TURN_ON": 1.0, "TURN_OFF": 0.0, "STOP": None}),
    46: DeviceSpec(
        "roof_window",
        {"TURN_ON": 1.0, "TURN_ON_2": 0.5, "TURN_OFF": 0.0, "STOP": None},
    ),
    47: DeviceSpec(
        "fan",
        {"TURN_ON": 1.0 / 3.0, "TURN_ON_2": 2.0 / 3.0, "TURN_ON_4": 1.0, "TURN_OFF": 0.0},
    ),
    48: DeviceSpec("supplemental_lamp", {"TURN_ON": 1.0, "TURN_OFF": 0.0}),
    49: DeviceSpec("co2_generator", {"TURN_ON": 1.0, "TURN_OFF": 0.0}),
    50: DeviceSpec("wet_pad_pump", {"TURN_ON": 1.0, "TURN_OFF": 0.0}),
    51: DeviceSpec("wet_pad_roll_film", {"TURN_ON": 1.0, "TURN_OFF": 0.0, "STOP": None}),
}

REQUIRED_EVENT_COLUMNS = {"log_id", "timestamp", "controller_id", "action"}
DEVICE_COLUMNS = tuple(spec.column for spec in DEVICE_SPECS.values())


def reconstruct_minute_controls(
    events: pd.DataFrame,
    minute_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reconstruct commanded targets and their certainty on a minute grid."""
    missing = REQUIRED_EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Missing control event columns: {sorted(missing)}")

    minutes = pd.DatetimeIndex(pd.to_datetime(minute_index)).floor("min")
    if minutes.has_duplicates:
        raise ValueError("minute_index must contain unique minutes")
    if not minutes.is_monotonic_increasing:
        raise ValueError("minute_index must be sorted")

    normalized = events.loc[:, sorted(REQUIRED_EVENT_COLUMNS)].copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    if normalized["timestamp"].isna().any():
        raise ValueError("Control events contain invalid timestamps")
    normalized["log_id"] = pd.to_numeric(normalized["log_id"], errors="raise").astype("int64")
    normalized["controller_id"] = pd.to_numeric(
        normalized["controller_id"], errors="raise"
    ).astype("int64")
    normalized = normalized.drop_duplicates().sort_values(
        ["timestamp", "log_id"], kind="stable"
    )

    unknown_ids = sorted(set(normalized["controller_id"]).difference(DEVICE_SPECS))
    if unknown_ids:
        raise ValueError(f"Unsupported controller IDs: {unknown_ids}")

    output = pd.DataFrame({"timestamp": minutes})
    for controller_id, spec in DEVICE_SPECS.items():
        device_events = normalized.loc[normalized["controller_id"].eq(controller_id)]
        values, known, uncertain = _reconstruct_device(device_events, minutes, controller_id, spec)
        output[spec.column] = values
        output[f"{spec.column}_known"] = known
        output[f"{spec.column}_uncertain"] = uncertain

    return add_aggregate_controls(output)


def _reconstruct_device(
    events: pd.DataFrame,
    minutes: pd.DatetimeIndex,
    controller_id: int,
    spec: DeviceSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros(len(minutes), dtype=float)
    known_values = np.zeros(len(minutes), dtype=bool)
    uncertain_values = np.zeros(len(minutes), dtype=bool)

    records = list(events.itertuples(index=False))
    event_index = 0
    value = 0.0
    known = False
    uncertain = False

    for minute_position, minute in enumerate(minutes):
        while event_index < len(records):
            event = records[event_index]
            if pd.Timestamp(event.timestamp).floor("min") > minute:
                break
            action = str(event.action)
            if action not in spec.actions:
                raise ValueError(f"Unsupported action for controller {controller_id}: {action}")
            target = spec.actions[action]
            if target is None:
                uncertain = True
            else:
                value = float(target)
                known = True
                uncertain = False
            event_index += 1

        values[minute_position] = value
        known_values[minute_position] = known
        uncertain_values[minute_position] = uncertain

    return values, known_values, uncertain_values


def add_aggregate_controls(states: pd.DataFrame) -> pd.DataFrame:
    """Add GreenLight-compatible proxies while retaining physical device columns."""
    missing = set(DEVICE_COLUMNS).difference(states.columns)
    if missing:
        raise ValueError(f"Missing device state columns: {sorted(missing)}")

    result = states.copy()
    result["uBoil"] = 0.0
    result["uCO2"] = 0.0
    result["uVent"] = np.maximum(result["roof_window"], result["fan"])
    result["uPad"] = np.minimum.reduce(
        [
            result["fan"].to_numpy(dtype=float),
            result["wet_pad_pump"].to_numpy(dtype=float),
            result["wet_pad_roll_film"].to_numpy(dtype=float),
        ]
    )
    result["uThScr"] = (
        192.0 * result["roof_thermal_screen"] + 196.0 * result["side_thermal_screen"]
    ) / 388.0
    result["uLamp"] = result["supplemental_lamp"].astype(float)
    result["uBlScr"] = result["external_shade"].astype(float)

    for column in ["uBoil", "uCO2", "uVent", "uPad", "uThScr", "uLamp", "uBlScr"]:
        result[column] = np.clip(pd.to_numeric(result[column], errors="raise"), 0.0, 1.0)
    return result


def derive_hourly_controls(minute_controls: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equal-duration minute command targets and quality coverage by hour."""
    required = {"timestamp", *DEVICE_COLUMNS}
    for device in DEVICE_COLUMNS:
        required.add(f"{device}_known")
        required.add(f"{device}_uncertain")
    missing = required.difference(minute_controls.columns)
    if missing:
        raise ValueError(f"Missing minute control columns: {sorted(missing)}")

    minute = minute_controls.loc[:, sorted(required)].copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], errors="coerce")
    if minute["timestamp"].isna().any():
        raise ValueError("Minute controls contain invalid timestamps")
    if minute["timestamp"].duplicated().any():
        raise ValueError("Minute control timestamps must be unique")
    minute = minute.sort_values("timestamp").reset_index(drop=True)
    minute["hour"] = minute["timestamp"].dt.floor("h")

    aggregations: dict[str, str] = {}
    for device in DEVICE_COLUMNS:
        aggregations[device] = "mean"
        aggregations[f"{device}_known"] = "mean"
        aggregations[f"{device}_uncertain"] = "mean"

    hourly = minute.groupby("hour", sort=True, as_index=False).agg(aggregations)
    hourly = hourly.rename(columns={"hour": "timestamp"})
    for device in DEVICE_COLUMNS:
        hourly = hourly.rename(
            columns={
                f"{device}_known": f"{device}_known_fraction",
                f"{device}_uncertain": f"{device}_uncertain_fraction",
            }
        )
    return add_aggregate_controls(hourly)
