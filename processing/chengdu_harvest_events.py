from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np
import pandas as pd


HARVEST_EVENT_COLUMNS = [
    "harvest_event_id",
    "event_type",
    "harvest_datetime",
    "timestamp",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "season_id",
    "cultivar",
    "source_site",
    "evidence_class",
    "target_eligible",
    "harvested_fresh_kg",
    "harvested_area_m2",
    "normalization_area_m2",
    "harvest_coverage_fraction",
    "local_harvested_fresh_kg_m2",
    "harvested_fresh_kg_m2",
    "fresh_kg_m2",
    "marketable_fresh_kg",
    "rejected_fresh_kg",
    "grade",
    "batch_id",
    "source_record_id",
]

_REQUIRED_INPUT_COLUMNS = {
    "harvest_event_id",
    "harvest_datetime",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "season_id",
    "cultivar",
    "harvested_fresh_kg",
    "harvested_area_m2",
}

_EVIDENCE_CLASSES = {"target_observed", "external_observed", "simulated_prior"}


def load_harvest_events(
    path: str | Path,
    *,
    target_greenhouse_ids: Collection[int],
    target_greenhouse_codes: Collection[str],
    normalization_areas_m2: Mapping[str, float],
    source_sites: Mapping[str, str] | None = None,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Load measured picking events without inferring mass or harvested area."""
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(source_path, encoding="utf-8-sig")
    elif suffix in {".xlsx", ".xlsm"}:
        try:
            frame = pd.read_excel(source_path, sheet_name="HarvestEvents")
        except ValueError as exc:
            if "Worksheet named" in str(exc):
                raise ValueError(
                    "harvest workbook must contain a HarvestEvents sheet"
                ) from exc
            raise
    else:
        raise ValueError("harvest event input must be CSV, XLSX, or XLSM")
    frame = frame.dropna(how="all").reset_index(drop=True)
    missing = sorted(_REQUIRED_INPUT_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"harvest event file missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("harvest event file must not be empty")
    if frame["harvest_event_id"].isna().any():
        raise ValueError("harvest_event_id must not be empty")
    if frame["harvest_event_id"].astype(str).duplicated().any():
        raise ValueError("duplicate harvest_event_id values are not allowed")

    timestamps = frame["harvest_datetime"].map(
        lambda value: _local_timestamp(value, timezone)
    )
    if timestamps.isna().any():
        raise ValueError("harvest_datetime must contain valid dates")
    frame["harvest_datetime"] = timestamps
    frame["timestamp"] = timestamps

    for column in (
        "harvested_fresh_kg",
        "harvested_area_m2",
        "marketable_fresh_kg",
        "rejected_fresh_kg",
    ):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["harvested_fresh_kg", "harvested_area_m2"]].isna().any().any():
        raise ValueError("harvested_fresh_kg and harvested_area_m2 must be present and numeric")
    if (frame["harvested_fresh_kg"] < 0.0).any():
        raise ValueError("harvested_fresh_kg must be non-negative")
    if (frame["harvested_area_m2"] <= 0.0).any():
        raise ValueError("harvested_area_m2 must be positive")
    numeric_values = frame[
        [
            "harvested_fresh_kg",
            "harvested_area_m2",
            "marketable_fresh_kg",
            "rejected_fresh_kg",
        ]
    ].to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("harvest masses and area must be finite")
    for column in ("marketable_fresh_kg", "rejected_fresh_kg"):
        if (frame[column].dropna() < 0.0).any():
            raise ValueError(f"{column} must be non-negative")
    components = frame[["marketable_fresh_kg", "rejected_fresh_kg"]].fillna(0.0).sum(axis=1)
    if (components > frame["harvested_fresh_kg"] + 1e-9).any():
        raise ValueError("marketable and rejected fresh mass cannot exceed total harvest")

    frame["greenhouse_id"] = pd.to_numeric(frame["greenhouse_id"], errors="coerce")
    if frame["greenhouse_id"].isna().any() or not np.isfinite(
        frame["greenhouse_id"].to_numpy(dtype=float)
    ).all():
        raise ValueError("greenhouse_id must be a finite integer")
    if not np.equal(frame["greenhouse_id"], np.floor(frame["greenhouse_id"])).all():
        raise ValueError("greenhouse_id must be a finite integer")
    frame["greenhouse_id"] = frame["greenhouse_id"].astype("int64")
    for column in (
        "grade",
        "batch_id",
        "source_record_id",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA
    if frame[["greenhouse_code", "planting_code", "season_id", "cultivar"]].isna().any().any():
        raise ValueError("greenhouse, planting, season, and cultivar identifiers are required")

    area_mapping: dict[str, float] = {}
    for code, value in normalization_areas_m2.items():
        area = float(value)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("normalization areas must be finite and positive")
        area_mapping[str(code)] = area
    greenhouse_codes = frame["greenhouse_code"].astype(str)
    missing_area_codes = sorted(set(greenhouse_codes) - set(area_mapping))
    if missing_area_codes:
        raise ValueError(
            "normalization area missing for greenhouse codes: "
            + ", ".join(missing_area_codes)
        )
    frame["normalization_area_m2"] = greenhouse_codes.map(area_mapping).astype(float)
    if (
        frame["harvested_area_m2"]
        > frame["normalization_area_m2"] + 1e-9
    ).any():
        raise ValueError("harvested_area_m2 cannot exceed normalization area")

    sites = source_sites or {}
    target_ids = {int(identifier) for identifier in target_greenhouse_ids}
    target_codes = {str(code) for code in target_greenhouse_codes}
    id_match = frame["greenhouse_id"].isin(target_ids)
    code_match = frame["greenhouse_code"].astype(str).isin(target_codes)
    if (id_match != code_match).any():
        raise ValueError("target greenhouse ID and code must agree")
    if "evidence_class" not in frame.columns:
        frame["evidence_class"] = "target_observed"
    frame["evidence_class"] = frame["evidence_class"].astype(str).str.strip()
    unknown_evidence = sorted(set(frame["evidence_class"]) - _EVIDENCE_CLASSES)
    if unknown_evidence:
        raise ValueError(
            "unknown harvest evidence_class values: " + ", ".join(unknown_evidence)
        )
    frame["event_type"] = "harvest_event"
    frame["source_site"] = frame["greenhouse_code"].astype(str).map(
        lambda code: sites.get(code, code)
    )
    frame["target_eligible"] = (
        id_match
        & code_match
        & frame["evidence_class"].eq("target_observed")
    )
    frame["harvest_coverage_fraction"] = (
        frame["harvested_area_m2"] / frame["normalization_area_m2"]
    )
    frame["local_harvested_fresh_kg_m2"] = (
        frame["harvested_fresh_kg"] / frame["harvested_area_m2"]
    )
    frame["harvested_fresh_kg_m2"] = (
        frame["harvested_fresh_kg"] / frame["normalization_area_m2"]
    )
    frame["fresh_kg_m2"] = frame["harvested_fresh_kg_m2"]
    duplicate_key = [
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "harvest_datetime",
        "harvested_fresh_kg",
        "harvested_area_m2",
    ]
    if frame.duplicated(duplicate_key, keep=False).any():
        raise ValueError("harvest business duplicate detected despite distinct event IDs")
    identity_columns = ["greenhouse_id", "greenhouse_code", "planting_code", "cultivar"]
    if (frame.groupby("season_id")[identity_columns].nunique(dropna=False) > 1).any().any():
        raise ValueError("each season_id must map to one greenhouse, planting, and cultivar identity")
    return frame[HARVEST_EVENT_COLUMNS].sort_values(
        ["harvest_datetime", "harvest_event_id"], kind="stable"
    ).reset_index(drop=True)


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
