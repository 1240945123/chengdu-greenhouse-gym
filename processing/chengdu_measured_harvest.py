from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "harvest_event_id",
    "season_id",
    "timestamp",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "cultivar",
    "harvested_fresh_kg",
    "fresh_kg_m2",
    "season_complete_evidence",
}


def build_measured_harvest_artifacts(
    harvest_events: pd.DataFrame,
) -> dict[str, Any]:
    """Build physical-event and daily cumulative measured harvest artifacts."""
    if missing := sorted(_REQUIRED_COLUMNS - set(harvest_events.columns)):
        raise ValueError(f"measured harvest events missing columns: {', '.join(missing)}")
    if harvest_events.empty:
        raise ValueError("measured harvest events must not be empty")
    frame = harvest_events.copy()
    frame["harvest_event_id"] = frame["harvest_event_id"].astype("string").str.strip()
    if frame["harvest_event_id"].isna().any() or frame["harvest_event_id"].eq("").any():
        raise ValueError("measured harvest event IDs must be present")
    if frame["harvest_event_id"].duplicated().any():
        raise ValueError("measured harvest event IDs must be unique")
    frame["season_id"] = frame["season_id"].astype("string").str.strip()
    if frame["season_id"].isna().any() or frame["season_id"].eq("").any():
        raise ValueError("measured harvest season identity must be present")
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], errors="coerce", format="mixed"
    )
    if frame["timestamp"].isna().any():
        raise ValueError("measured harvest timestamps must be valid")
    for column in ("harvested_fresh_kg", "fresh_kg_m2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    masses = frame[["harvested_fresh_kg", "fresh_kg_m2"]].to_numpy(dtype=float)
    if not np.isfinite(masses).all():
        raise ValueError("measured harvest masses must be finite")
    if (masses < 0.0).any():
        raise ValueError("measured harvest masses must be non-negative")
    complete = _strict_boolean_series(
        frame["season_complete_evidence"], "season_complete_evidence"
    )
    frame["season_complete_evidence"] = complete
    for column in ("batch_id", "source_record_id"):
        if column not in frame.columns:
            frame[column] = pd.NA
    identity_columns = [
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "cultivar",
        "season_complete_evidence",
    ]
    if (frame.groupby("season_id")[identity_columns].nunique(dropna=False) > 1).any().any():
        raise ValueError("each measured harvest season must have one identity and completion status")

    frame = frame.sort_values(
        ["season_id", "timestamp", "harvest_event_id"], kind="stable"
    ).reset_index(drop=True)
    frame["harvest_date"] = frame["timestamp"].dt.normalize()
    frame["event_sequence"] = frame.groupby("season_id", sort=False).cumcount() + 1
    frame["cumulative_harvested_fresh_kg"] = frame.groupby(
        "season_id", sort=False
    )["harvested_fresh_kg"].cumsum()
    frame["cumulative_fresh_kg_m2"] = frame.groupby("season_id", sort=False)[
        "fresh_kg_m2"
    ].cumsum()

    daily = (
        frame.groupby(["season_id", "harvest_date"], sort=False, as_index=False)
        .agg(
            greenhouse_id=("greenhouse_id", "first"),
            greenhouse_code=("greenhouse_code", "first"),
            planting_code=("planting_code", "first"),
            cultivar=("cultivar", "first"),
            season_complete_evidence=("season_complete_evidence", "first"),
            physical_event_count=("harvest_event_id", "size"),
            batch_fresh_kg=("harvested_fresh_kg", "sum"),
            batch_fresh_kg_m2=("fresh_kg_m2", "sum"),
        )
        .sort_values(["season_id", "harvest_date"], kind="stable")
        .reset_index(drop=True)
    )
    daily["picking_day_sequence"] = daily.groupby("season_id", sort=False).cumcount() + 1
    daily["cumulative_fresh_kg"] = daily.groupby("season_id", sort=False)[
        "batch_fresh_kg"
    ].cumsum()
    daily["cumulative_fresh_kg_m2"] = daily.groupby("season_id", sort=False)[
        "batch_fresh_kg_m2"
    ].cumsum()

    event_totals = frame.groupby("season_id", sort=False)[
        ["harvested_fresh_kg", "fresh_kg_m2"]
    ].sum()
    daily_totals = daily.groupby("season_id", sort=False)[
        ["batch_fresh_kg", "batch_fresh_kg_m2"]
    ].sum()
    if not np.allclose(
        event_totals["harvested_fresh_kg"], daily_totals["batch_fresh_kg"]
    ) or not np.allclose(
        event_totals["fresh_kg_m2"], daily_totals["batch_fresh_kg_m2"]
    ):
        raise ValueError("measured event and daily harvest totals do not reconcile")

    summaries: list[dict[str, object]] = []
    for season_id, season in frame.groupby("season_id", sort=False):
        season_daily = daily[daily["season_id"].astype(str).eq(str(season_id))]
        first = season["timestamp"].min()
        last = season["timestamp"].max()
        summaries.append(
            {
                "season_id": str(season_id),
                "physical_event_count": int(len(season)),
                "picking_day_count": int(len(season_daily)),
                "first_harvest_timestamp": pd.Timestamp(first).isoformat(),
                "last_harvest_timestamp": pd.Timestamp(last).isoformat(),
                "event_span_days": float((last - first).total_seconds() / 86400.0),
                "measured_total_fresh_kg": float(season["harvested_fresh_kg"].sum()),
                "final_measured_yield_kg_m2": float(season["fresh_kg_m2"].sum()),
                "season_complete_evidence": bool(
                    season["season_complete_evidence"].iloc[0]
                ),
                "batch_id_recorded_count": _recorded_count(season["batch_id"]),
                "source_record_id_recorded_count": _recorded_count(
                    season["source_record_id"]
                ),
            }
        )
    return {
        "event_trajectory": frame,
        "daily_trajectory": daily,
        "season_summary": summaries,
        "batch_semantics": "greenhouse_calendar_day_total",
    }


def _recorded_count(values: pd.Series) -> int:
    text = values.astype("string").str.strip()
    return int((text.notna() & text.ne("")).sum())


def _strict_boolean_series(values: pd.Series, name: str) -> pd.Series:
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
