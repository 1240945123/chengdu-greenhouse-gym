from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.crop.validate_chengdu_harvest_workbook import run_preflight


TARGET_CODE = "GH202602061448266452514"


def _write_workbook(
    path: Path,
    *,
    second_status: str = "complete",
    missing_source_record: bool = False,
) -> Path:
    event_rows = []
    season_rows = []
    for index, (season_id, planting_code, start) in enumerate(
        (("s1", "P1", "2025-03-01"), ("s2", "P2", "2026-03-01"))
    ):
        dates = pd.date_range(start, periods=8, freq="5D")
        for event_index, timestamp in enumerate(dates, start=1):
            event_rows.append(
                {
                    "harvest_event_id": f"H-{season_id}-{event_index}",
                    "harvest_datetime": timestamp + pd.Timedelta(hours=8),
                    "greenhouse_id": 63,
                    "greenhouse_code": TARGET_CODE,
                    "planting_code": planting_code,
                    "season_id": season_id,
                    "cultivar": "Sainirui",
                    "harvested_fresh_kg": 19.2,
                    "harvested_area_m2": 192.0,
                    "marketable_fresh_kg": 18.0,
                    "rejected_fresh_kg": 1.2,
                    "grade": "mixed",
                    "batch_id": f"B-{season_id}-{event_index}",
                    "source_record_id": (
                        None
                        if missing_source_record and index == 0 and event_index == 1
                        else f"R-{season_id}-{event_index}"
                    ),
                }
            )
        status = "complete" if index == 0 else second_status
        season_rows.append(
            {
                "season_id": season_id,
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": planting_code,
                "cultivar": "Sainirui",
                "season_start_datetime": pd.Timestamp(start),
                "season_end_datetime": (
                    dates[-1] + pd.Timedelta(days=1) if status == "complete" else None
                ),
                "completion_status": status,
                "completion_source": (
                    "crop_clearance_record" if status == "complete" else None
                ),
                "source_record_id": f"SEASON-{season_id}",
            }
        )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(event_rows).to_excel(
            writer, sheet_name="HarvestEvents", index=False
        )
        pd.DataFrame(season_rows).to_excel(
            writer, sheet_name="HarvestSeasons", index=False
        )
    return path


def test_preflight_marks_two_complete_qualifying_seasons_measurement_ready(tmp_path: Path):
    workbook = _write_workbook(tmp_path / "harvest.xlsx")

    report = run_preflight(workbook_path=workbook, output_root=tmp_path / "out")

    assert report["target_harvest_event_count"] == 16
    assert report["qualifying_complete_seasons"] == ["s1", "s2"]
    assert report["measurement_calibration_coverage_ready"] is True
    assert report["measurement_independent_validation_coverage_ready"] is True
    assert report["crop_driver_model_accepted"] is False
    assert report["target_harvest_validated"] is False
    assert report["management_protocol_coverage_ready"] is False
    assert "management_protocol_sheet_missing" in report["warnings"]
    assert (tmp_path / "out" / "harvest_workbook_preflight.json").exists()
    event_path = tmp_path / "out" / "measured_harvest_event_trajectory.csv"
    daily_path = tmp_path / "out" / "measured_daily_harvest_trajectory.csv"
    assert event_path.exists() and daily_path.exists()
    daily = pd.read_csv(daily_path)
    totals = daily.groupby("season_id")["cumulative_fresh_kg_m2"].last().tolist()
    assert totals == pytest.approx([0.8, 0.8])


def test_preflight_excludes_ongoing_season_and_reports_traceability_warning(tmp_path: Path):
    workbook = _write_workbook(
        tmp_path / "harvest.xlsx",
        second_status="ongoing",
        missing_source_record=True,
    )

    report = run_preflight(workbook_path=workbook, output_root=tmp_path / "out")

    assert report["qualifying_complete_seasons"] == ["s1"]
    assert report["measurement_calibration_coverage_ready"] is True
    assert report["measurement_independent_validation_coverage_ready"] is False
    assert "ongoing_season_excluded:s2" in report["warnings"]
    assert "source_record_id_missing:s1:1" in report["warnings"]
    assert report["target_harvest_validated"] is False


def test_preflight_reports_confirmed_protocol_coverage_ready(tmp_path: Path):
    workbook = _write_workbook(tmp_path / "harvest.xlsx")
    protocols = pd.DataFrame(
        [
            {
                "protocol_id": f"PROTOCOL-{season_id}",
                "season_id": season_id,
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": planting_code,
                "timezone": "Asia/Shanghai",
                "effective_start_datetime": start,
                "effective_end_datetime": None,
                "protocol_status": "confirmed",
                "pick_weekdays": "0,2,4",
                "pick_local_hour": 8,
                "confirmation_source": "site_manager_signed_schedule",
                "source_record_id": f"SOURCE-{season_id}",
            }
            for season_id, planting_code, start in (
                ("s1", "P1", "2025-03-01"),
                ("s2", "P2", "2026-03-01"),
            )
        ]
    )
    with pd.ExcelWriter(
        workbook, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        protocols.to_excel(writer, sheet_name="HarvestProtocol", index=False)

    report = run_preflight(workbook_path=workbook, output_root=tmp_path / "out")

    assert report["management_protocol_coverage_ready"] is True
    assert all(
        not row["blocking_reasons"]
        for row in report["management_protocol_coverage"]["season_readiness"]
    )
