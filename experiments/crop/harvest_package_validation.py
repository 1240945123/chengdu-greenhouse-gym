from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from glassgym.models.harvest import scheduled_pick_between
from processing.chengdu_harvest_protocols import assess_harvest_protocol_coverage


_DRIVER_COLUMNS = {
    "season_id",
    "timestamp",
    "net_fruit_dry_matter_change_kg_m2",
    "air_temperature_c",
    "dt_hours",
    "pick",
    "pick_source",
    "driver_model_status",
    "season_complete",
    "fruit_change_source",
    "air_temperature_source",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "initial_fruit_dry_matter_kg_m2",
    "initial_fruit_maturity_fraction",
}
_EVENT_COLUMNS = {
    "season_id",
    "timestamp",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "target_eligible",
    "fresh_kg_m2",
    "season_complete_evidence",
}


def assess_harvest_calibration_package(
    drivers: pd.DataFrame,
    harvest_events: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    minimum_driver_span_days: float = 90.0,
    minimum_harvest_events_per_season: int = 8,
    minimum_harvest_span_days: float = 30.0,
    maximum_driver_interval_hours: float = 24.0,
    minimum_driver_coverage_fraction: float = 0.95,
    timezone: str = "Asia/Shanghai",
    protocols: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assess complete measured-harvest and driver coverage without fitting."""
    if missing := sorted(_DRIVER_COLUMNS - set(drivers.columns)):
        raise ValueError(f"driver columns missing: {', '.join(missing)}")
    if missing := sorted(_EVENT_COLUMNS - set(harvest_events.columns)):
        raise ValueError(f"harvest event columns missing: {', '.join(missing)}")
    if drivers.empty:
        raise ValueError("drivers must not be empty")
    if harvest_events.empty:
        raise ValueError("harvest events must not be empty")

    driver_frame = drivers.copy()
    event_frame = harvest_events.copy()
    driver_frame["season_id"] = _required_text(driver_frame["season_id"], "driver season_id")
    event_frame["season_id"] = _required_text(event_frame["season_id"], "harvest season_id")
    driver_frame["timestamp"] = _local_timestamps(driver_frame["timestamp"], timezone)
    event_frame["timestamp"] = _local_timestamps(event_frame["timestamp"], timezone)
    if driver_frame["timestamp"].isna().any() or event_frame["timestamp"].isna().any():
        raise ValueError("driver and harvest timestamps must be valid")
    if driver_frame.duplicated(["season_id", "timestamp"]).any():
        raise ValueError("driver timestamps must be unique within each season")

    driver_numeric = [
        "greenhouse_id",
        "net_fruit_dry_matter_change_kg_m2",
        "air_temperature_c",
        "dt_hours",
        "initial_fruit_dry_matter_kg_m2",
        "initial_fruit_maturity_fraction",
    ]
    event_numeric = ["greenhouse_id", "fresh_kg_m2"]
    for frame, columns in ((driver_frame, driver_numeric), (event_frame, event_numeric)):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError("driver and harvest numeric values must be finite")
    if (driver_frame["dt_hours"] <= 0.0).any():
        raise ValueError("driver dt_hours must be positive")
    if (event_frame["fresh_kg_m2"] < 0.0).any():
        raise ValueError("harvest fresh_kg_m2 must be non-negative")

    driver_frame["pick"] = _strict_boolean(driver_frame["pick"], "pick")
    driver_frame["season_complete"] = _strict_boolean(
        driver_frame["season_complete"], "season_complete"
    )
    event_frame["target_eligible"] = _strict_boolean(
        event_frame["target_eligible"], "target_eligible"
    )
    event_frame["season_complete_evidence"] = _strict_boolean(
        event_frame["season_complete_evidence"], "season_complete_evidence"
    )
    _validate_target_identity(
        driver_frame,
        target_greenhouse_id=target_greenhouse_id,
        target_greenhouse_code=target_greenhouse_code,
        kind="driver",
    )
    _validate_target_identity(
        event_frame,
        target_greenhouse_id=target_greenhouse_id,
        target_greenhouse_code=target_greenhouse_code,
        kind="harvest event",
    )
    if not event_frame["target_eligible"].all():
        raise ValueError("all harvest events must be target eligible")

    driver_starts = driver_frame.groupby("season_id")["timestamp"].min()
    event_starts = event_frame.groupby("season_id")["timestamp"].min()
    season_ids = sorted(
        set(driver_starts.index) | set(event_starts.index),
        key=lambda season: min(
            driver_starts.get(season, pd.Timestamp.max),
            event_starts.get(season, pd.Timestamp.max),
        ),
    )
    readiness = [
        _assess_season(
            season_id,
            driver_frame[driver_frame["season_id"].eq(season_id)],
            event_frame[event_frame["season_id"].eq(season_id)],
            minimum_driver_span_days=float(minimum_driver_span_days),
            minimum_harvest_events=int(minimum_harvest_events_per_season),
            minimum_harvest_span_days=float(minimum_harvest_span_days),
            maximum_driver_interval_hours=float(maximum_driver_interval_hours),
            minimum_driver_coverage_fraction=float(minimum_driver_coverage_fraction),
        )
        for season_id in season_ids
    ]
    protocol_coverage: dict[str, object] | None = None
    if protocols is not None:
        horizon_rows = []
        for season_id, season in driver_frame.groupby("season_id", sort=False):
            season = season.sort_values("timestamp", kind="stable")
            first_interval_start = season["timestamp"].iloc[0] - pd.Timedelta(
                hours=float(season["dt_hours"].iloc[0])
            )
            horizon_rows.append(
                {
                    "season_id": str(season_id),
                    "greenhouse_id": int(season["greenhouse_id"].iloc[0]),
                    "greenhouse_code": str(season["greenhouse_code"].iloc[0]),
                    "planting_code": str(season["planting_code"].iloc[0]),
                    "horizon_start": first_interval_start,
                    "horizon_end": season["timestamp"].max(),
                }
            )
        protocol_coverage = assess_harvest_protocol_coverage(
            protocols,
            pd.DataFrame(horizon_rows),
            timezone=timezone,
        )
        protocol_rows = {
            str(row["season_id"]): row
            for row in protocol_coverage["season_readiness"]
        }
        for row in readiness:
            coverage = protocol_rows.get(str(row["season_id"]))
            if coverage is None:
                row["blocking_reasons"].append("management_protocol_missing")
            else:
                row["management_protocol_coverage_fraction"] = coverage[
                    "coverage_fraction"
                ]
                row["blocking_reasons"].extend(coverage["blocking_reasons"])
            season_drivers = driver_frame[
                driver_frame["season_id"].eq(str(row["season_id"]))
            ]
            mismatch_count = _protocol_pick_mismatch_count(
                season_drivers,
                protocols,
                timezone=timezone,
            )
            row["management_protocol_pick_mismatch_count"] = mismatch_count
            if mismatch_count:
                row["blocking_reasons"].append(
                    "management_protocol_pick_mismatch"
                )
            row["blocking_reasons"] = list(dict.fromkeys(row["blocking_reasons"]))
    candidate_rows = [row for row in readiness if not row["blocking_reasons"]]
    for previous, following in zip(candidate_rows, candidate_rows[1:]):
        if pd.Timestamp(previous["driver_end"]) >= pd.Timestamp(
            following["driver_start"]
        ):
            following["blocking_reasons"].append("season_overlap")
    qualifying = [
        str(row["season_id"]) for row in readiness if not row["blocking_reasons"]
    ]
    split = _chronological_split(qualifying)
    blocking_reasons = [
        f"{row['season_id']}:{reason}"
        for row in readiness
        for reason in row["blocking_reasons"]
    ]
    if len(qualifying) < 2:
        blocking_reasons.append("independent_target_season_missing")
    return {
        "scope": "calibration_input_package_coverage_only",
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "driver_row_count": int(len(driver_frame)),
        "harvest_event_count": int(len(event_frame)),
        "driver_season_count": int(driver_frame["season_id"].nunique()),
        "harvest_season_count": int(event_frame["season_id"].nunique()),
        "qualifying_complete_seasons": qualifying,
        "season_readiness": readiness,
        "split": split,
        "calibration_package_ready": len(qualifying) >= 1,
        "independent_validation_package_ready": len(qualifying) >= 2,
        "management_protocol_coverage_ready": (
            protocol_coverage["all_horizons_confirmed_and_covered"]
            if protocol_coverage is not None
            else None
        ),
        "management_protocol_coverage": protocol_coverage,
        "harvest_parameters_fitted": False,
        "target_harvest_validated": False,
        "blocking_reasons": blocking_reasons,
    }


def require_independent_validation_package(assessment: dict[str, Any]) -> None:
    if assessment.get("independent_validation_package_ready") is True:
        return
    reasons = assessment.get("blocking_reasons", [])
    detail = ", ".join(map(str, reasons)) or "independent target season missing"
    raise ValueError(
        "harvest calibration package is not ready for independent validation: "
        + detail
    )


def _assess_season(
    season_id: str,
    drivers: pd.DataFrame,
    events: pd.DataFrame,
    *,
    minimum_driver_span_days: float,
    minimum_harvest_events: int,
    minimum_harvest_span_days: float,
    maximum_driver_interval_hours: float,
    minimum_driver_coverage_fraction: float,
) -> dict[str, Any]:
    blockers: list[str] = []
    if drivers.empty:
        blockers.append("drivers_missing")
    if events.empty:
        blockers.append("harvest_events_missing")
    if drivers.empty or events.empty:
        return {
            "season_id": str(season_id),
            "driver_row_count": int(len(drivers)),
            "harvest_event_count": int(len(events)),
            "blocking_reasons": blockers,
        }

    drivers = drivers.sort_values("timestamp", kind="stable")
    events = events.sort_values("timestamp", kind="stable")
    driver_span = _span_days(drivers["timestamp"])
    harvest_span = _span_days(events["timestamp"])
    nominal_interval = float(drivers["dt_hours"].iloc[0])
    gaps = drivers["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    expected_rows = int(np.floor(driver_span * 24.0 / nominal_interval)) + 1
    coverage_fraction = len(drivers) / max(expected_rows, 1)

    if not drivers["season_complete"].all() or not events[
        "season_complete_evidence"
    ].all():
        blockers.append("season_incomplete")
    if not drivers["driver_model_status"].astype(str).eq(
        "accepted_target_crop_model"
    ).all():
        blockers.append("crop_model_not_accepted")
    if not drivers["pick_source"].astype(str).eq("management_protocol").all():
        blockers.append("management_protocol_missing")
    if not drivers["fruit_change_source"].astype(str).eq(
        "greenlight_fruit_state_balance"
    ).all():
        blockers.append("fruit_change_source_invalid")
    if not set(drivers["air_temperature_source"].astype(str)).issubset(
        {"observed_indoor", "greenlight_simulated_indoor"}
    ):
        blockers.append("air_temperature_source_invalid")
    if driver_span < minimum_driver_span_days:
        blockers.append("driver_span_too_short")
    if len(events) < minimum_harvest_events:
        blockers.append("harvest_event_count_too_low")
    if harvest_span < minimum_harvest_span_days:
        blockers.append("harvest_span_too_short")
    if not np.allclose(drivers["dt_hours"].to_numpy(dtype=float), nominal_interval):
        blockers.append("driver_interval_not_fixed")
    if gaps.empty or (gaps > maximum_driver_interval_hours + 1e-8).any():
        blockers.append("driver_interval_exceeds_limit")
    elif not np.allclose(
        gaps.to_numpy(dtype=float),
        drivers["dt_hours"].iloc[1:].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    ):
        blockers.append("driver_dt_timestamp_mismatch")
    if coverage_fraction < minimum_driver_coverage_fraction:
        blockers.append("driver_coverage_too_sparse")
    if drivers["timestamp"].min() > events["timestamp"].min() or drivers[
        "timestamp"
    ].max() < events["timestamp"].max():
        blockers.append("drivers_do_not_cover_harvest_events")
    if set(drivers["planting_code"].astype(str)) != set(
        events["planting_code"].astype(str)
    ) or drivers["planting_code"].astype(str).nunique() != 1:
        blockers.append("planting_identity_mismatch")
    initial_mass = drivers["initial_fruit_dry_matter_kg_m2"]
    initial_age = drivers["initial_fruit_maturity_fraction"]
    if (initial_mass < 0.0).any() or not initial_mass.iloc[1:].eq(0.0).all():
        blockers.append("initial_fruit_inventory_invalid")
    if not initial_age.between(0.0, 1.0).all() or not initial_age.iloc[1:].eq(0.0).all():
        blockers.append("initial_fruit_maturity_invalid")
    return {
        "season_id": str(season_id),
        "driver_row_count": int(len(drivers)),
        "harvest_event_count": int(len(events)),
        "driver_start": pd.Timestamp(drivers["timestamp"].min()).isoformat(),
        "driver_end": pd.Timestamp(drivers["timestamp"].max()).isoformat(),
        "driver_span_days": float(driver_span),
        "driver_coverage_fraction": float(coverage_fraction),
        "harvest_start": pd.Timestamp(events["timestamp"].min()).isoformat(),
        "harvest_end": pd.Timestamp(events["timestamp"].max()).isoformat(),
        "harvest_span_days": float(harvest_span),
        "season_complete": bool(
            drivers["season_complete"].all()
            and events["season_complete_evidence"].all()
        ),
        "driver_model_status": sorted(
            set(drivers["driver_model_status"].astype(str))
        ),
        "blocking_reasons": list(dict.fromkeys(blockers)),
    }


def _chronological_split(seasons: list[str]) -> dict[str, list[str]]:
    if len(seasons) < 2:
        return {
            "train_seasons": seasons.copy(),
            "validation_seasons": [],
            "test_seasons": [],
        }
    if len(seasons) == 2:
        return {
            "train_seasons": [seasons[0]],
            "validation_seasons": [],
            "test_seasons": [seasons[1]],
        }
    return {
        "train_seasons": seasons[:-2],
        "validation_seasons": [seasons[-2]],
        "test_seasons": [seasons[-1]],
    }


def _protocol_pick_mismatch_count(
    drivers: pd.DataFrame,
    protocols: pd.DataFrame,
    *,
    timezone: str,
) -> int:
    if drivers.empty:
        return 0
    periods = protocols[
        protocols["season_id"].astype(str).eq(str(drivers["season_id"].iloc[0]))
    ].copy()
    if periods.empty:
        return 0
    periods["effective_start"] = _local_timestamps(
        periods["effective_start"], timezone
    )
    periods["effective_end"] = _local_timestamps(
        periods["effective_end"], timezone
    )
    mismatches = 0
    for driver in drivers.itertuples(index=False):
        interval_end = pd.Timestamp(driver.timestamp)
        interval_start = interval_end - pd.Timedelta(hours=float(driver.dt_hours))
        covering = []
        for period in periods.itertuples(index=False):
            if period.effective_start > interval_start:
                continue
            if not pd.isna(period.effective_end) and period.effective_end < interval_end:
                continue
            covering.append(period)
        if len(covering) > 1:
            raise ValueError("multiple protocol periods cover one driver interval")
        if not covering:
            continue
        period = covering[0]
        expected = scheduled_pick_between(
            interval_start,
            interval_end,
            weekdays=list(period.pick_weekdays),
            local_hour=int(period.pick_local_hour),
            timezone=timezone,
        ) is not None
        mismatches += int(bool(driver.pick) != expected)
    return int(mismatches)


def _required_text(values: pd.Series, name: str) -> pd.Series:
    parsed = values.astype("string").str.strip()
    if parsed.isna().any() or parsed.eq("").any():
        raise ValueError(f"{name} values must be present")
    return parsed.astype(str)


def _strict_boolean(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = values.map(
        lambda value: mapping.get(
            value.lower().strip() if isinstance(value, str) else value
        )
    )
    if parsed.isna().any():
        raise ValueError(f"{name} must contain strict boolean values")
    return parsed.astype(bool)


def _validate_target_identity(
    frame: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    kind: str,
) -> None:
    if not frame["greenhouse_id"].eq(int(target_greenhouse_id)).all() or not frame[
        "greenhouse_code"
    ].astype(str).eq(str(target_greenhouse_code)).all():
        raise ValueError(f"{kind} greenhouse identity does not match the target greenhouse")


def _span_days(timestamps: pd.Series) -> float:
    return float(
        (timestamps.max() - timestamps.min()).total_seconds() / 86400.0
    )


def _local_timestamps(values: pd.Series, timezone: str) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(timezone)
        return timestamp.tz_convert(timezone)

    return values.map(parse)
