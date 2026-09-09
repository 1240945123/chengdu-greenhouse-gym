from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np
import pandas as pd

from glassgym.models.harvest.cohort_model import scheduled_pick_between


DRIVER_COLUMNS = [
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "timestamp",
    "net_fruit_dry_matter_change_kg_m2",
    "initial_fruit_dry_matter_kg_m2",
    "initial_fruit_maturity_fraction",
    "air_temperature_c",
    "dt_hours",
    "pick",
    "pick_source",
    "season_complete",
    "fruit_change_source",
    "air_temperature_source",
    "driver_model_status",
]


def build_harvest_drivers_from_state_trajectory(
    trajectory: pd.DataFrame,
    *,
    season_id: str,
    greenhouse_id: int,
    greenhouse_code: str,
    planting_code: str,
    initial_fruit_maturity_fraction: float,
    protocol_status: str,
    pick_weekdays: Collection[int] | None,
    pick_local_hour: int | None,
    season_complete: bool,
    driver_model_status: str,
    air_temperature_source: str,
    timezone: str = "Asia/Shanghai",
    maximum_interval_hours: float = 24.0,
    protocol_periods: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "timestamp",
        "standing_dry_kg_m2",
        "native_harvested_dry_kg_m2",
        "air_temperature_c",
    }
    if missing := sorted(required - set(trajectory.columns)):
        raise ValueError(f"crop state trajectory missing columns: {', '.join(missing)}")
    if len(trajectory) < 2:
        raise ValueError("crop state trajectory must contain at least two rows")
    if not str(season_id) or not str(greenhouse_code) or not str(planting_code):
        raise ValueError("season, greenhouse, and planting identities must be non-empty")
    if isinstance(greenhouse_id, bool) or int(greenhouse_id) != greenhouse_id:
        raise ValueError("greenhouse_id must be an integer")
    if not isinstance(season_complete, (bool, np.bool_)):
        raise ValueError("season_complete must be boolean")
    maturity = float(initial_fruit_maturity_fraction)
    if not np.isfinite(maturity) or not 0.0 <= maturity <= 1.0:
        raise ValueError("initial_fruit_maturity_fraction must be in [0, 1]")
    maximum_interval = float(maximum_interval_hours)
    if not np.isfinite(maximum_interval) or maximum_interval <= 0.0:
        raise ValueError("maximum_interval_hours must be finite and positive")
    allowed_temperature_sources = {"observed_indoor", "greenlight_simulated_indoor"}
    if air_temperature_source not in allowed_temperature_sources:
        raise ValueError(
            "air_temperature_source must be observed_indoor or greenlight_simulated_indoor"
        )
    if not str(driver_model_status):
        raise ValueError("driver_model_status must be non-empty")
    if protocol_status not in {"declared", "missing"}:
        raise ValueError("protocol_status must be declared or missing")
    if protocol_periods is not None:
        weekdays = []
        local_hour = None
        pick_source = None
    elif protocol_status == "declared":
        if pick_weekdays is None or not list(pick_weekdays) or pick_local_hour is None:
            raise ValueError("declared protocol requires weekdays and local hour")
        weekdays = list(pick_weekdays)
        local_hour = int(pick_local_hour)
        pick_source = "management_protocol"
    else:
        if pick_weekdays not in (None, []) or pick_local_hour is not None:
            raise ValueError("missing protocol cannot define weekdays or local hour")
        weekdays = []
        local_hour = None
        pick_source = "management_protocol_missing"

    frame = trajectory.copy()
    frame["timestamp"] = frame["timestamp"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    if frame["timestamp"].isna().any():
        raise ValueError("crop state timestamps must be valid")
    if frame["timestamp"].duplicated().any():
        raise ValueError("crop state timestamps must be unique")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("crop state timestamps must be strictly chronological")
    numeric_columns = [
        "standing_dry_kg_m2",
        "native_harvested_dry_kg_m2",
        "air_temperature_c",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("crop state masses and temperatures must be finite")
    if (frame[["standing_dry_kg_m2", "native_harvested_dry_kg_m2"]] < 0.0).any().any():
        raise ValueError("crop state masses must be non-negative")
    if abs(float(frame.loc[0, "native_harvested_dry_kg_m2"])) > 1e-12:
        raise ValueError("first crop state native harvest must be zero")

    intervals = (
        frame["timestamp"].diff().iloc[1:].dt.total_seconds().to_numpy(dtype=float)
        / 3600.0
    )
    if (intervals <= 0.0).any():
        raise ValueError("crop state intervals must be positive")
    if (intervals > maximum_interval + 1e-9).any():
        raise ValueError(
            f"crop state intervals must not exceed {maximum_interval:g} hours"
        )
    standing = frame["standing_dry_kg_m2"].to_numpy(dtype=float)
    native_harvest = frame["native_harvested_dry_kg_m2"].to_numpy(dtype=float)
    net_change = standing[1:] - standing[:-1] + native_harvest[1:]
    if (net_change < -standing[:-1] - 1e-12).any():
        raise ValueError("net fruit loss exceeds the preceding standing fruit mass")

    protocol_audit: dict[str, Any]
    protocol_blockers: list[str]
    if protocol_periods is not None:
        picks, pick_sources, protocol_audit, protocol_blockers = (
            _picks_from_protocol_periods(
                frame,
                protocol_periods,
                season_id=str(season_id),
                greenhouse_id=int(greenhouse_id),
                greenhouse_code=str(greenhouse_code),
                planting_code=str(planting_code),
                timezone=timezone,
            )
        )
        protocol_status_for_audit = "period_records"
    elif protocol_status == "declared":
        picks = [
            scheduled_pick_between(
                frame.loc[index - 1, "timestamp"],
                frame.loc[index, "timestamp"],
                weekdays=weekdays,
                local_hour=local_hour,
                timezone=timezone,
            )
            is not None
            for index in range(1, len(frame))
        ]
        pick_sources = [str(pick_source)] * len(picks)
        protocol_audit = {
            "protocol_coverage_fraction": 1.0,
            "confirmed_protocol_interval_count": int(len(picks)),
            "provisional_protocol_interval_count": 0,
            "uncovered_protocol_interval_count": 0,
            "protocol_ids": [],
            "protocol_confirmation_sources": [],
        }
        protocol_blockers = []
        protocol_status_for_audit = protocol_status
    else:
        picks = [False] * (len(frame) - 1)
        pick_sources = [str(pick_source)] * len(picks)
        protocol_audit = {
            "protocol_coverage_fraction": 0.0,
            "confirmed_protocol_interval_count": 0,
            "provisional_protocol_interval_count": 0,
            "uncovered_protocol_interval_count": int(len(picks)),
            "protocol_ids": [],
            "protocol_confirmation_sources": [],
        }
        protocol_blockers = ["management_protocol_missing"]
        protocol_status_for_audit = protocol_status

    initial_inventory = np.zeros(len(frame) - 1, dtype=float)
    initial_inventory[0] = standing[0]
    initial_maturity = np.zeros(len(frame) - 1, dtype=float)
    initial_maturity[0] = maturity
    drivers = pd.DataFrame(
        {
            "season_id": str(season_id),
            "greenhouse_id": int(greenhouse_id),
            "greenhouse_code": str(greenhouse_code),
            "planting_code": str(planting_code),
            "timestamp": frame["timestamp"].iloc[1:].reset_index(drop=True),
            "net_fruit_dry_matter_change_kg_m2": net_change,
            "initial_fruit_dry_matter_kg_m2": initial_inventory,
            "initial_fruit_maturity_fraction": initial_maturity,
            "air_temperature_c": frame["air_temperature_c"].iloc[1:].to_numpy(dtype=float),
            "dt_hours": intervals,
            "pick": picks,
            "pick_source": pick_sources,
            "season_complete": bool(season_complete),
            "fruit_change_source": "greenlight_fruit_state_balance",
            "air_temperature_source": air_temperature_source,
            "driver_model_status": str(driver_model_status),
        }
    )[DRIVER_COLUMNS]

    blockers: list[str] = []
    blockers.extend(protocol_blockers)
    if not season_complete:
        blockers.append("season_incomplete")
    if driver_model_status != "accepted_target_crop_model":
        blockers.append("crop_model_not_accepted")
    accounting_expected = standing[-1] - standing[0] + native_harvest[1:].sum()
    audit = {
        "season_id": str(season_id),
        "greenhouse_id": int(greenhouse_id),
        "greenhouse_code": str(greenhouse_code),
        "planting_code": str(planting_code),
        "state_row_count": int(len(frame)),
        "driver_row_count": int(len(drivers)),
        "state_start": frame.loc[0, "timestamp"].isoformat(),
        "state_end": frame.loc[len(frame) - 1, "timestamp"].isoformat(),
        "span_days": float(
            (frame.loc[len(frame) - 1, "timestamp"] - frame.loc[0, "timestamp"])
            .total_seconds()
            / 86400.0
        ),
        "maximum_interval_hours": float(intervals.max()),
        "protocol_status": protocol_status_for_audit,
        "pick_weekdays": weekdays,
        "pick_local_hour": local_hour,
        "pick_permission_count": int(sum(picks)),
        **protocol_audit,
        "initial_fruit_dry_matter_kg_m2": float(standing[0]),
        "initial_fruit_maturity_fraction": maturity,
        "native_harvest_added_back_kg_m2": float(native_harvest[1:].sum()),
        "net_fruit_dry_matter_change_kg_m2": float(net_change.sum()),
        "mass_accounting_residual_kg_m2": float(
            net_change.sum() - accounting_expected
        ),
        "season_complete": bool(season_complete),
        "driver_model_status": str(driver_model_status),
        "air_temperature_source": air_temperature_source,
        "calibration_eligible": not blockers,
        "blocking_reasons": blockers,
    }
    return drivers, audit


def _picks_from_protocol_periods(
    trajectory: pd.DataFrame,
    protocol_periods: pd.DataFrame,
    *,
    season_id: str,
    greenhouse_id: int,
    greenhouse_code: str,
    planting_code: str,
    timezone: str,
) -> tuple[list[bool], list[str], dict[str, Any], list[str]]:
    required = {
        "protocol_id",
        "season_id",
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "effective_start",
        "effective_end",
        "protocol_status",
        "pick_weekdays",
        "pick_local_hour",
        "confirmation_source",
    }
    if missing := sorted(required - set(protocol_periods.columns)):
        raise ValueError(f"protocol periods missing columns: {', '.join(missing)}")
    periods = protocol_periods.copy()
    periods = periods[periods["season_id"].astype(str).eq(str(season_id))].copy()
    timestamp_dtype = pd.DatetimeTZDtype(tz=timezone)
    periods["effective_start"] = pd.Series(
        periods["effective_start"].map(
            lambda value: _local_timestamp(value, timezone)
        ).tolist(),
        index=periods.index,
        dtype=timestamp_dtype,
    )
    periods["effective_end"] = pd.Series(
        periods["effective_end"].map(
            lambda value: (
                pd.NaT if pd.isna(value) else _local_timestamp(value, timezone)
            )
        ).tolist(),
        index=periods.index,
        dtype=timestamp_dtype,
    )
    if not periods.empty:
        identity_matches = (
            pd.to_numeric(periods["greenhouse_id"], errors="coerce").eq(
                int(greenhouse_id)
            )
            & periods["greenhouse_code"].astype(str).eq(str(greenhouse_code))
            & periods["planting_code"].astype(str).eq(str(planting_code))
        )
        if not identity_matches.all():
            raise ValueError("protocol period identity does not match driver season")

    picks: list[bool] = []
    sources: list[str] = []
    confirmed_count = 0
    provisional_count = 0
    uncovered_count = 0
    for index in range(1, len(trajectory)):
        interval_start = trajectory.loc[index - 1, "timestamp"]
        interval_end = trajectory.loc[index, "timestamp"]
        covering = periods[
            periods["effective_start"].le(interval_start)
            & (
                periods["effective_end"].isna()
                | periods["effective_end"].ge(interval_end)
            )
        ]
        if len(covering) > 1:
            raise ValueError("multiple protocol periods cover one driver interval")
        if covering.empty:
            picks.append(False)
            sources.append("management_protocol_missing")
            uncovered_count += 1
            continue
        period = covering.iloc[0]
        status = str(period["protocol_status"])
        if status not in {"confirmed", "provisional"}:
            raise ValueError("protocol period status must be confirmed or provisional")
        source = (
            "management_protocol"
            if status == "confirmed"
            else "provisional_management_protocol"
        )
        sources.append(source)
        confirmed_count += int(status == "confirmed")
        provisional_count += int(status == "provisional")
        picks.append(
            scheduled_pick_between(
                interval_start,
                interval_end,
                weekdays=list(period["pick_weekdays"]),
                local_hour=int(period["pick_local_hour"]),
                timezone=timezone,
            )
            is not None
        )
    total = len(picks)
    blockers: list[str] = []
    if uncovered_count:
        blockers.append("management_protocol_missing")
    if provisional_count:
        blockers.append("management_protocol_not_confirmed")
    audit = {
        "protocol_coverage_fraction": (
            float((total - uncovered_count) / total) if total else 0.0
        ),
        "confirmed_protocol_interval_count": int(confirmed_count),
        "provisional_protocol_interval_count": int(provisional_count),
        "uncovered_protocol_interval_count": int(uncovered_count),
        "protocol_ids": periods["protocol_id"].astype(str).tolist(),
        "protocol_confirmation_sources": sorted(
            set(periods["confirmation_source"].astype(str))
        ),
    }
    return picks, sources, audit, blockers


def _local_timestamp(value: object, timezone: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)
