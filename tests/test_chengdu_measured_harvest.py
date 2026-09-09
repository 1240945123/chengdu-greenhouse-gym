from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from processing.chengdu_measured_harvest import build_measured_harvest_artifacts


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "harvest_event_id": ["H1", "H2", "H3", "H4"],
            "season_id": ["s1", "s1", "s1", "s2"],
            "timestamp": pd.to_datetime(
                [
                    "2026-05-01 08:00:00+08:00",
                    "2026-05-01 16:00:00+08:00",
                    "2026-05-03 08:00:00+08:00",
                    "2027-05-01 08:00:00+08:00",
                ]
            ),
            "greenhouse_id": [63] * 4,
            "greenhouse_code": ["GH-PIDU"] * 4,
            "planting_code": ["P1", "P1", "P1", "P2"],
            "cultivar": ["Sainirui"] * 4,
            "harvested_fresh_kg": [48.0, 48.0, 96.0, 192.0],
            "fresh_kg_m2": [0.25, 0.25, 0.50, 1.0],
            "batch_id": ["B1", "B2", "B3", "B4"],
            "source_record_id": ["R1", None, "R3", "R4"],
            "season_complete_evidence": [True, True, True, False],
        }
    )


def test_measured_trajectories_preserve_events_aggregate_days_and_reset_seasons():
    result = build_measured_harvest_artifacts(_events())
    event = result["event_trajectory"]
    daily = result["daily_trajectory"]

    assert event["event_sequence"].tolist() == [1, 2, 3, 1]
    assert event["cumulative_harvested_fresh_kg"].tolist() == [48.0, 96.0, 192.0, 192.0]
    assert event["cumulative_fresh_kg_m2"].tolist() == [0.25, 0.50, 1.0, 1.0]
    assert daily["physical_event_count"].tolist() == [2, 1, 1]
    assert daily["batch_fresh_kg"].tolist() == [96.0, 96.0, 192.0]
    assert daily["batch_fresh_kg_m2"].tolist() == [0.50, 0.50, 1.0]
    assert daily["cumulative_fresh_kg_m2"].tolist() == [0.50, 1.0, 1.0]
    assert result["batch_semantics"] == "greenhouse_calendar_day_total"


def test_measured_season_summary_reports_yield_completion_and_traceability():
    result = build_measured_harvest_artifacts(_events())
    summaries = {item["season_id"]: item for item in result["season_summary"]}

    assert summaries["s1"]["physical_event_count"] == 3
    assert summaries["s1"]["picking_day_count"] == 2
    assert summaries["s1"]["measured_total_fresh_kg"] == 192.0
    assert summaries["s1"]["final_measured_yield_kg_m2"] == 1.0
    assert summaries["s1"]["season_complete_evidence"] is True
    assert summaries["s1"]["batch_id_recorded_count"] == 3
    assert summaries["s1"]["source_record_id_recorded_count"] == 2
    assert summaries["s2"]["season_complete_evidence"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.assign(timestamp="bad"), "timestamp"),
        (lambda frame: frame.assign(harvested_fresh_kg=-1.0), "non-negative"),
        (lambda frame: frame.assign(fresh_kg_m2=np.inf), "finite"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "event ID"),
        (lambda frame: frame.drop(columns="planting_code"), "missing"),
    ],
)
def test_measured_trajectory_rejects_invalid_canonical_events(mutator, message):
    with pytest.raises(ValueError, match=message):
        build_measured_harvest_artifacts(mutator(_events()))
