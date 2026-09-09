from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd


HARVEST_PROTOCOL_COLUMNS = [
    "protocol_id",
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "timezone",
    "effective_start",
    "effective_end",
    "protocol_status",
    "pick_weekdays",
    "pick_local_hour",
    "confirmation_source",
    "source_record_id",
    "notes",
    "target_eligible",
]

_REQUIRED_INPUT_COLUMNS = {
    "protocol_id",
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "timezone",
    "effective_start_datetime",
    "effective_end_datetime",
    "protocol_status",
    "pick_weekdays",
    "pick_local_hour",
    "confirmation_source",
    "source_record_id",
}

_HORIZON_COLUMNS = {
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "horizon_start",
    "horizon_end",
}


def load_harvest_protocols(
    path: str | Path,
    *,
    target_greenhouse_ids: Collection[int],
    target_greenhouse_codes: Collection[str],
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Load traceable season-specific picking schedule periods."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xlsm"}:
        try:
            frame = pd.read_excel(source, sheet_name="HarvestProtocol")
        except ValueError as exc:
            if "Worksheet named" in str(exc):
                raise ValueError(
                    "harvest workbook must contain a HarvestProtocol sheet"
                ) from exc
            raise
    else:
        raise ValueError("harvest protocol input must be CSV, XLSX, or XLSM")
    frame = frame.dropna(how="all").reset_index(drop=True)
    if missing := sorted(_REQUIRED_INPUT_COLUMNS - set(frame.columns)):
        raise ValueError(f"harvest protocol file missing columns: {', '.join(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=HARVEST_PROTOCOL_COLUMNS)
    if "notes" not in frame.columns:
        frame["notes"] = pd.NA

    text_columns = [
        "protocol_id",
        "season_id",
        "greenhouse_code",
        "planting_code",
        "timezone",
        "protocol_status",
        "confirmation_source",
        "source_record_id",
    ]
    for column in text_columns:
        parsed = frame[column].astype("string").str.strip()
        if parsed.isna().any() or parsed.eq("").any():
            raise ValueError(f"{column} values must be present")
        frame[column] = parsed.astype(str)
    if frame["protocol_id"].duplicated().any():
        raise ValueError("duplicate protocol_id values are not allowed")
    status = frame["protocol_status"].str.lower()
    if not status.isin({"confirmed", "provisional"}).all():
        raise ValueError("protocol_status must be confirmed or provisional")
    frame["protocol_status"] = status
    if not frame["timezone"].eq(str(timezone)).all():
        raise ValueError(f"protocol timezone must be {timezone}")

    frame["greenhouse_id"] = pd.to_numeric(frame["greenhouse_id"], errors="coerce")
    if frame["greenhouse_id"].isna().any() or not np.equal(
        frame["greenhouse_id"], np.floor(frame["greenhouse_id"])
    ).all():
        raise ValueError("protocol greenhouse_id must be a finite integer")
    frame["greenhouse_id"] = frame["greenhouse_id"].astype("int64")
    frame["pick_local_hour"] = pd.to_numeric(
        frame["pick_local_hour"], errors="coerce"
    )
    if frame["pick_local_hour"].isna().any() or not np.equal(
        frame["pick_local_hour"], np.floor(frame["pick_local_hour"])
    ).all() or not frame["pick_local_hour"].between(0, 23).all():
        raise ValueError("pick local hour must be an integer in [0, 23]")
    frame["pick_local_hour"] = frame["pick_local_hour"].astype("int64")
    frame["pick_weekdays"] = frame["pick_weekdays"].map(_parse_weekdays)

    timestamp_dtype = pd.DatetimeTZDtype(tz=timezone)
    frame["effective_start"] = pd.Series(
        frame["effective_start_datetime"].map(
            lambda value: _local_timestamp(value, timezone)
        ).tolist(),
        index=frame.index,
        dtype=timestamp_dtype,
    )
    frame["effective_end"] = pd.Series(
        frame["effective_end_datetime"].map(
            lambda value: _optional_local_timestamp(value, timezone)
        ).tolist(),
        index=frame.index,
        dtype=timestamp_dtype,
    )
    if frame["effective_start"].isna().any():
        raise ValueError("protocol effective start timestamps must be valid")
    ended = frame["effective_end"].notna()
    if ended.any() and (
        frame.loc[ended, "effective_end"] <= frame.loc[ended, "effective_start"]
    ).any():
        raise ValueError("protocol effective end must be after effective start")

    target_ids = {int(value) for value in target_greenhouse_ids}
    target_codes = {str(value) for value in target_greenhouse_codes}
    id_match = frame["greenhouse_id"].isin(target_ids)
    code_match = frame["greenhouse_code"].isin(target_codes)
    if (id_match != code_match).any():
        raise ValueError("target greenhouse ID and code must agree")
    frame["target_eligible"] = id_match & code_match
    identity = ["greenhouse_id", "greenhouse_code", "planting_code", "timezone"]
    if (frame.groupby("season_id")[identity].nunique(dropna=False) > 1).any().any():
        raise ValueError("each protocol season must have one greenhouse and planting identity")

    frame = frame.sort_values(
        ["season_id", "effective_start", "protocol_id"], kind="stable"
    ).reset_index(drop=True)
    for season_id, periods in frame.groupby("season_id", sort=False):
        rows = list(periods.itertuples(index=False))
        for previous, current in zip(rows, rows[1:]):
            if pd.isna(previous.effective_end):
                raise ValueError(
                    f"open-ended protocol period overlaps later period in season {season_id}"
                )
            if current.effective_start < previous.effective_end:
                raise ValueError(f"protocol periods overlap in season {season_id}")
    return frame[HARVEST_PROTOCOL_COLUMNS]


def assess_harvest_protocol_coverage(
    protocols: pd.DataFrame,
    horizons: pd.DataFrame,
    *,
    timezone: str = "Asia/Shanghai",
) -> dict[str, object]:
    """Assess continuous, confirmed protocol coverage over season horizons."""
    if missing := sorted(_HORIZON_COLUMNS - set(horizons.columns)):
        raise ValueError(f"protocol horizon columns missing: {', '.join(missing)}")
    frame = horizons.copy()
    timestamp_dtype = pd.DatetimeTZDtype(tz=timezone)
    for column in ("horizon_start", "horizon_end"):
        frame[column] = pd.Series(
            frame[column].map(lambda value: _local_timestamp(value, timezone)).tolist(),
            index=frame.index,
            dtype=timestamp_dtype,
        )
    if frame[["horizon_start", "horizon_end"]].isna().any().any():
        raise ValueError("protocol coverage horizons must have valid timestamps")
    if (frame["horizon_end"] <= frame["horizon_start"]).any():
        raise ValueError("protocol coverage horizon end must be after start")
    if frame["season_id"].astype(str).duplicated().any():
        raise ValueError("protocol coverage horizons require unique season_id values")

    readiness: list[dict[str, object]] = []
    for horizon in frame.itertuples(index=False):
        season_periods = protocols[
            protocols["season_id"].astype(str).eq(str(horizon.season_id))
        ].copy()
        if not season_periods.empty:
            identity = (
                pd.to_numeric(season_periods["greenhouse_id"], errors="coerce").eq(
                    int(horizon.greenhouse_id)
                )
                & season_periods["greenhouse_code"].astype(str).eq(
                    str(horizon.greenhouse_code)
                )
                & season_periods["planting_code"].astype(str).eq(
                    str(horizon.planting_code)
                )
            )
            if not identity.all():
                raise ValueError("protocol identity does not match coverage horizon")
        total_seconds = (horizon.horizon_end - horizon.horizon_start).total_seconds()
        covered_seconds = 0.0
        provisional_seconds = 0.0
        for period in season_periods.itertuples(index=False):
            start = max(period.effective_start, horizon.horizon_start)
            period_end = (
                horizon.horizon_end
                if pd.isna(period.effective_end)
                else min(period.effective_end, horizon.horizon_end)
            )
            if period_end <= start:
                continue
            duration = (period_end - start).total_seconds()
            covered_seconds += duration
            if str(period.protocol_status) != "confirmed":
                provisional_seconds += duration
        coverage_fraction = min(covered_seconds / total_seconds, 1.0)
        blockers: list[str] = []
        if coverage_fraction < 1.0 - 1e-9:
            blockers.append("management_protocol_missing")
        if provisional_seconds > 0.0:
            blockers.append("management_protocol_not_confirmed")
        readiness.append(
            {
                "season_id": str(horizon.season_id),
                "horizon_start": horizon.horizon_start.isoformat(),
                "horizon_end": horizon.horizon_end.isoformat(),
                "protocol_row_count": int(len(season_periods)),
                "coverage_fraction": float(coverage_fraction),
                "confirmed_coverage_fraction": float(
                    max(covered_seconds - provisional_seconds, 0.0) / total_seconds
                ),
                "blocking_reasons": blockers,
            }
        )
    return {
        "all_horizons_confirmed_and_covered": bool(readiness)
        and all(not row["blocking_reasons"] for row in readiness),
        "season_readiness": readiness,
    }


def _parse_weekdays(value: object) -> tuple[int, ...]:
    if pd.isna(value):
        raise ValueError("pick weekdays must be present")
    if isinstance(value, (int, np.integer)):
        parts = [int(value)]
    else:
        text = str(value).strip()
        if not text:
            raise ValueError("pick weekdays must be present")
        try:
            parts = [int(item.strip()) for item in text.split(",")]
        except ValueError as exc:
            raise ValueError("pick weekdays must be comma-separated integers") from exc
    if not parts or len(parts) != len(set(parts)) or any(day < 0 or day > 6 for day in parts):
        raise ValueError("pick weekdays must be unique integers in [0, 6]")
    return tuple(sorted(parts))


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


def _optional_local_timestamp(value: object, timezone: str) -> pd.Timestamp:
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    return _local_timestamp(value, timezone)
