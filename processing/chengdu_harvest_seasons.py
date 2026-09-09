from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HARVEST_SEASON_COLUMNS = [
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "cultivar",
    "season_start",
    "season_end",
    "completion_status",
    "completion_source",
    "source_record_id",
    "target_eligible",
]

_REQUIRED_COLUMNS = {
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "cultivar",
    "season_start_datetime",
    "season_end_datetime",
    "completion_status",
    "completion_source",
    "source_record_id",
}


def load_harvest_seasons(
    path: str | Path,
    *,
    target_greenhouse_ids: Collection[int],
    target_greenhouse_codes: Collection[str],
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xlsm"}:
        try:
            frame = pd.read_excel(source, sheet_name="HarvestSeasons")
        except ValueError as exc:
            if "Worksheet named" in str(exc):
                raise ValueError(
                    "harvest workbook must contain a HarvestSeasons sheet"
                ) from exc
            raise
    else:
        raise ValueError("harvest season manifest must be CSV, XLSX, or XLSM")
    frame = frame.dropna(how="all").reset_index(drop=True)
    if missing := sorted(_REQUIRED_COLUMNS - set(frame.columns)):
        raise ValueError(f"harvest season manifest missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("harvest season manifest must not be empty")
    if frame["season_id"].isna().any() or frame["season_id"].astype(str).duplicated().any():
        raise ValueError("season_id values must be present and unique")

    frame["greenhouse_id"] = pd.to_numeric(frame["greenhouse_id"], errors="coerce")
    if frame["greenhouse_id"].isna().any() or not np.equal(
        frame["greenhouse_id"], np.floor(frame["greenhouse_id"])
    ).all():
        raise ValueError("greenhouse_id must be a finite integer")
    frame["greenhouse_id"] = frame["greenhouse_id"].astype("int64")
    identity_columns = [
        "season_id",
        "greenhouse_code",
        "planting_code",
        "cultivar",
        "source_record_id",
    ]
    for column in identity_columns:
        values = frame[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"{column} source identity must be present")
        frame[column] = values.astype(str)

    frame["season_start"] = frame["season_start_datetime"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    frame["season_end"] = frame["season_end_datetime"].map(
        lambda value: _optional_local_timestamp(value, timezone)
    )
    if frame["season_start"].isna().any():
        raise ValueError("season start timestamps must be valid")
    status = frame["completion_status"].astype("string").str.strip().str.lower()
    if status.isna().any() or not status.isin({"complete", "ongoing"}).all():
        raise ValueError("completion status must be complete or ongoing")
    frame["completion_status"] = status.astype(str)
    complete = frame["completion_status"].eq("complete")
    ongoing = ~complete
    completion_source = frame["completion_source"].astype("string").str.strip()
    if completion_source.loc[complete].isna().any() or completion_source.loc[complete].eq("").any():
        raise ValueError("complete seasons require a completion source")
    if frame.loc[complete, "season_end"].isna().any():
        raise ValueError("complete seasons require a valid season end")
    if frame.loc[ongoing, "season_end"].notna().any():
        raise ValueError("ongoing seasons cannot declare a season end")
    if (
        frame.loc[complete, "season_end"].reset_index(drop=True)
        <= frame.loc[complete, "season_start"].reset_index(drop=True)
    ).any():
        raise ValueError("season end must be after season start")
    frame["completion_source"] = completion_source

    target_ids = {int(value) for value in target_greenhouse_ids}
    target_codes = {str(value) for value in target_greenhouse_codes}
    id_match = frame["greenhouse_id"].isin(target_ids)
    code_match = frame["greenhouse_code"].isin(target_codes)
    if (id_match != code_match).any():
        raise ValueError("target greenhouse ID and code must agree")
    frame["target_eligible"] = id_match & code_match
    return frame[HARVEST_SEASON_COLUMNS].sort_values(
        "season_start", kind="stable"
    ).reset_index(drop=True)


def bind_harvest_events_to_seasons(
    harvest_events: pd.DataFrame,
    seasons: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_required = {
        "season_id",
        "timestamp",
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "cultivar",
    }
    season_required = set(HARVEST_SEASON_COLUMNS)
    if missing := sorted(event_required - set(harvest_events.columns)):
        raise ValueError(f"harvest events missing season-binding columns: {', '.join(missing)}")
    if missing := sorted(season_required - set(seasons.columns)):
        raise ValueError(f"harvest seasons missing columns: {', '.join(missing)}")
    if harvest_events.empty:
        raise ValueError("harvest events must not be empty when binding seasons")
    unknown = sorted(
        set(harvest_events["season_id"].astype(str))
        - set(seasons["season_id"].astype(str))
    )
    if unknown:
        raise ValueError(f"harvest events reference unknown seasons: {', '.join(unknown)}")

    season_details = seasons.rename(
        columns={
            "greenhouse_id": "manifest_greenhouse_id",
            "greenhouse_code": "manifest_greenhouse_code",
            "planting_code": "manifest_planting_code",
            "cultivar": "manifest_cultivar",
            "target_eligible": "manifest_target_eligible",
            "completion_source": "season_completion_source",
            "source_record_id": "season_manifest_source_record_id",
        }
    )
    bound = harvest_events.copy()
    bound["season_id"] = bound["season_id"].astype(str)
    bound["timestamp"] = pd.to_datetime(bound["timestamp"], errors="coerce")
    if bound["timestamp"].isna().any():
        raise ValueError("harvest event timestamps must be valid")
    bound = bound.merge(season_details, on="season_id", how="left", validate="many_to_one")
    identity_matches = (
        pd.to_numeric(bound["greenhouse_id"], errors="coerce").eq(
            bound["manifest_greenhouse_id"]
        )
        & bound["greenhouse_code"].astype(str).eq(bound["manifest_greenhouse_code"])
        & bound["planting_code"].astype(str).eq(bound["manifest_planting_code"])
        & bound["cultivar"].astype(str).eq(bound["manifest_cultivar"])
    )
    if not identity_matches.all():
        raise ValueError("harvest event and season manifest identity mismatch")
    within_start = bound["timestamp"] >= bound["season_start"]
    complete = bound["completion_status"].eq("complete")
    within_end = ~complete | (bound["timestamp"] <= bound["season_end"])
    if not (within_start & within_end).all():
        raise ValueError("harvest event timestamp falls outside season bounds")
    bound["season_complete_evidence"] = complete
    bound = bound.drop(
        columns=[
            "manifest_greenhouse_id",
            "manifest_greenhouse_code",
            "manifest_planting_code",
            "manifest_cultivar",
            "manifest_target_eligible",
        ],
        errors="ignore",
    )

    season_complete = seasons["completion_status"].eq("complete")
    event_complete = bound["season_complete_evidence"]
    audit = {
        "season_count": int(len(seasons)),
        "complete_season_count": int(season_complete.sum()),
        "ongoing_season_count": int((~season_complete).sum()),
        "event_count": int(len(bound)),
        "complete_season_event_count": int(event_complete.sum()),
        "ongoing_season_event_count": int((~event_complete).sum()),
        "complete_seasons": seasons.loc[
            season_complete, "season_id"
        ].astype(str).tolist(),
        "ongoing_seasons": seasons.loc[
            ~season_complete, "season_id"
        ].astype(str).tolist(),
    }
    return bound, audit


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
