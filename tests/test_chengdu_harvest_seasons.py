from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from processing.chengdu_harvest_seasons import (
    bind_harvest_events_to_seasons,
    load_harvest_seasons,
)


def _write_manifest(path: Path, **overrides: object) -> Path:
    rows = [
        {
            "season_id": "s_complete",
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "planting_code": "P1",
            "cultivar": "Sainirui",
            "season_start_datetime": "2025-03-01 00:00:00",
            "season_end_datetime": "2025-08-31 23:59:59",
            "completion_status": "complete",
            "completion_source": "crop_clearance_log",
            "source_record_id": "CLEAR-2025",
        },
        {
            "season_id": "s_ongoing",
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "planting_code": "P2",
            "cultivar": "Sainirui",
            "season_start_datetime": "2026-03-01 00:00:00",
            "season_end_datetime": None,
            "completion_status": "ongoing",
            "completion_source": None,
            "source_record_id": "PLANT-2026",
        },
    ]
    for key, value in overrides.items():
        rows[0][key] = value
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "harvest_event_id": ["H1", "H2"],
            "source_record_id": ["EVENT-1", "EVENT-2"],
            "season_id": ["s_complete", "s_ongoing"],
            "timestamp": pd.to_datetime(
                ["2025-05-01 08:00:00+08:00", "2026-05-01 08:00:00+08:00"]
            ),
            "greenhouse_id": [63, 63],
            "greenhouse_code": ["GH-PIDU", "GH-PIDU"],
            "planting_code": ["P1", "P2"],
            "cultivar": ["Sainirui", "Sainirui"],
            "target_eligible": [True, True],
            "fresh_kg_m2": [0.5, 0.6],
        }
    )


def test_manifest_loader_reads_xlsx_and_ignores_fully_blank_rows(tmp_path):
    csv_path = _write_manifest(tmp_path / "seasons.csv")
    frame = pd.read_csv(csv_path)
    frame.loc[len(frame)] = [None] * len(frame.columns)
    path = tmp_path / "seasons.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="HarvestSeasons", index=False)

    seasons = load_harvest_seasons(
        path,
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
    )

    assert seasons["season_id"].tolist() == ["s_complete", "s_ongoing"]


def test_manifest_xlsx_loader_rejects_missing_sheet(tmp_path):
    path = tmp_path / "missing.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["wrong sheet"]}).to_excel(
            writer, sheet_name="Other", index=False
        )

    with pytest.raises(ValueError, match="HarvestSeasons"):
        load_harvest_seasons(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )


def test_manifest_binds_complete_and_ongoing_seasons_without_promoting_ongoing(tmp_path):
    seasons = load_harvest_seasons(
        _write_manifest(tmp_path / "seasons.csv"),
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
        timezone="Asia/Shanghai",
    )

    bound, audit = bind_harvest_events_to_seasons(_events(), seasons)

    assert bound["season_complete_evidence"].tolist() == [True, False]
    assert bound.loc[0, "season_completion_source"] == "crop_clearance_log"
    assert bound["source_record_id"].tolist() == ["EVENT-1", "EVENT-2"]
    assert bound.loc[0, "season_manifest_source_record_id"] == "CLEAR-2025"
    assert pd.isna(bound.loc[1, "season_end"])
    assert audit["season_count"] == 2
    assert audit["complete_season_count"] == 1
    assert audit["ongoing_season_count"] == 1
    assert audit["complete_season_event_count"] == 1
    assert audit["ongoing_season_event_count"] == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"completion_source": None}, "source"),
        ({"source_record_id": None}, "source"),
        ({"season_end_datetime": "2025-02-01"}, "after"),
        ({"completion_status": "finished"}, "status"),
    ],
)
def test_manifest_rejects_invalid_completion_evidence(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_harvest_seasons(
            _write_manifest(tmp_path / "seasons.csv", **overrides),
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )


def test_event_binding_rejects_identity_mismatch_and_out_of_bounds(tmp_path):
    seasons = load_harvest_seasons(
        _write_manifest(tmp_path / "seasons.csv"),
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
    )
    mismatch = _events().assign(cultivar="Other")
    with pytest.raises(ValueError, match="identity"):
        bind_harvest_events_to_seasons(mismatch, seasons)

    late = _events().copy()
    late.loc[0, "timestamp"] = pd.Timestamp("2025-09-01", tz="Asia/Shanghai")
    with pytest.raises(ValueError, match="bounds"):
        bind_harvest_events_to_seasons(late, seasons)
