from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from processing.chengdu_harvest_protocols import (
    assess_harvest_protocol_coverage,
    load_harvest_protocols,
)


TARGET_CODE = "GH202602061448266452514"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "protocol_id": "PROTOCOL-1",
        "season_id": "s1",
        "greenhouse_id": 63,
        "greenhouse_code": TARGET_CODE,
        "planting_code": "P1",
        "timezone": "Asia/Shanghai",
        "effective_start_datetime": "2026-04-01 00:00:00",
        "effective_end_datetime": "2026-06-01 00:00:00",
        "protocol_status": "confirmed",
        "pick_weekdays": "0,2,4",
        "pick_local_hour": 8,
        "confirmation_source": "site_manager_signed_schedule",
        "source_record_id": "SOURCE-1",
        "notes": "",
    }
    row.update(overrides)
    return row


def _write(path: Path, rows: list[dict[str, object]]) -> Path:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=list(_row())).to_excel(
            writer, sheet_name="HarvestProtocol", index=False
        )
    return path


def _load(path: Path) -> pd.DataFrame:
    return load_harvest_protocols(
        path,
        target_greenhouse_ids=[63],
        target_greenhouse_codes=[TARGET_CODE],
    )


def test_loads_confirmed_and_provisional_nonoverlapping_protocol_periods(tmp_path: Path):
    path = _write(
        tmp_path / "protocol.xlsx",
        [
            _row(),
            _row(
                protocol_id="PROTOCOL-2",
                effective_start_datetime="2026-06-01 00:00:00",
                effective_end_datetime=None,
                protocol_status="provisional",
                pick_weekdays="1,3,5",
                pick_local_hour=9,
                confirmation_source="researcher_assumption",
                source_record_id="SOURCE-2",
            ),
        ],
    )

    protocols = _load(path)

    assert protocols["protocol_id"].tolist() == ["PROTOCOL-1", "PROTOCOL-2"]
    assert protocols["pick_weekdays"].tolist() == [(0, 2, 4), (1, 3, 5)]
    assert protocols["pick_local_hour"].tolist() == [8, 9]
    assert protocols["effective_start"].dt.tz is not None
    assert pd.isna(protocols.loc[1, "effective_end"])
    assert protocols["target_eligible"].all()


def test_blank_protocol_sheet_returns_empty_canonical_frame(tmp_path: Path):
    protocols = _load(_write(tmp_path / "blank.xlsx", []))

    assert protocols.empty
    assert "pick_weekdays" in protocols.columns


def test_single_open_ended_protocol_period_loads(tmp_path: Path):
    path = _write(
        tmp_path / "open_ended.xlsx",
        [_row(effective_end_datetime=None)],
    )

    protocols = _load(path)

    assert protocols["protocol_id"].tolist() == ["PROTOCOL-1"]
    assert pd.isna(protocols.loc[0, "effective_end"])
    assert protocols["effective_end"].dt.tz is not None


def test_assesses_confirmed_protocol_coverage_across_period_switch(tmp_path: Path):
    path = _write(
        tmp_path / "protocol.xlsx",
        [
            _row(effective_end_datetime="2026-05-01 00:00:00"),
            _row(
                protocol_id="PROTOCOL-2",
                effective_start_datetime="2026-05-01 00:00:00",
                effective_end_datetime=None,
                source_record_id="SOURCE-2",
            ),
        ],
    )
    horizons = pd.DataFrame(
        [
            {
                "season_id": "s1",
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": "P1",
                "horizon_start": "2026-04-01 00:00:00",
                "horizon_end": "2026-06-01 00:00:00",
            }
        ]
    )

    report = assess_harvest_protocol_coverage(_load(path), horizons)

    assert report["all_horizons_confirmed_and_covered"] is True
    assert report["season_readiness"][0]["coverage_fraction"] == pytest.approx(1.0)
    assert report["season_readiness"][0]["blocking_reasons"] == []


def test_protocol_coverage_reports_gap_and_provisional_period(tmp_path: Path):
    path = _write(
        tmp_path / "protocol.xlsx",
        [
            _row(effective_end_datetime="2026-04-15 00:00:00"),
            _row(
                protocol_id="PROTOCOL-2",
                effective_start_datetime="2026-04-20 00:00:00",
                effective_end_datetime=None,
                protocol_status="provisional",
                confirmation_source="researcher_assumption",
                source_record_id="SOURCE-2",
            ),
        ],
    )
    horizons = pd.DataFrame(
        [
            {
                "season_id": "s1",
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": "P1",
                "horizon_start": "2026-04-01 00:00:00",
                "horizon_end": "2026-06-01 00:00:00",
            }
        ]
    )

    report = assess_harvest_protocol_coverage(_load(path), horizons)

    blockers = report["season_readiness"][0]["blocking_reasons"]
    assert "management_protocol_missing" in blockers
    assert "management_protocol_not_confirmed" in blockers
    assert report["all_horizons_confirmed_and_covered"] is False


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row(), _row()], "duplicate protocol_id"),
        ([_row(pick_weekdays="0,7")], "weekdays"),
        ([_row(pick_local_hour=24)], "hour"),
        (
            [
                _row(),
                _row(
                    protocol_id="PROTOCOL-2",
                    effective_start_datetime="2026-05-01",
                    effective_end_datetime="2026-07-01",
                    source_record_id="SOURCE-2",
                ),
            ],
            "overlap",
        ),
        ([_row(greenhouse_code="WRONG")], "ID and code"),
    ],
)
def test_rejects_invalid_or_ambiguous_protocol_records(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
):
    path = _write(tmp_path / "invalid.xlsx", rows)

    with pytest.raises(ValueError, match=message):
        _load(path)
