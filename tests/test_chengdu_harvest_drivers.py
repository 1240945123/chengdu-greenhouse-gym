from __future__ import annotations

import pandas as pd
import pytest
import yaml

from experiments.crop.build_chengdu_harvest_drivers import build_harvest_driver_bundle
from processing.chengdu_harvest_drivers import (
    build_harvest_drivers_from_state_trajectory,
)


def _trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-02 07:00:00+08:00",
                    "2026-03-02 09:00:00+08:00",
                    "2026-03-02 10:00:00+08:00",
                ]
            ),
            "standing_dry_kg_m2": [0.100, 0.110, 0.105],
            "native_harvested_dry_kg_m2": [0.0, 0.0, 0.002],
            "air_temperature_c": [20.0, 21.0, 22.0],
        }
    )


def _protocol_periods(*, second_status: str = "confirmed", gap: bool = False):
    second_start = pd.Timestamp("2026-03-02 09:30:00+08:00") if gap else pd.Timestamp(
        "2026-03-02 09:00:00+08:00"
    )
    return pd.DataFrame(
        [
            {
                "protocol_id": "P-A",
                "season_id": "2026_spring",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": "P-2026",
                "timezone": "Asia/Shanghai",
                "effective_start": pd.Timestamp("2026-03-01 00:00:00+08:00"),
                "effective_end": pd.Timestamp("2026-03-02 09:00:00+08:00"),
                "protocol_status": "confirmed",
                "pick_weekdays": (0,),
                "pick_local_hour": 8,
                "confirmation_source": "site_manager_signed_schedule",
                "source_record_id": "R-A",
            },
            {
                "protocol_id": "P-B",
                "season_id": "2026_spring",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": "P-2026",
                "timezone": "Asia/Shanghai",
                "effective_start": second_start,
                "effective_end": pd.NaT,
                "protocol_status": second_status,
                "pick_weekdays": (0,),
                "pick_local_hour": 10,
                "confirmation_source": (
                    "site_manager_signed_schedule"
                    if second_status == "confirmed"
                    else "researcher_assumption"
                ),
                "source_record_id": "R-B",
            },
        ]
    )


def test_state_trajectory_builder_preserves_mass_and_uses_declared_pick_protocol():
    drivers, audit = build_harvest_drivers_from_state_trajectory(
        _trajectory(),
        season_id="2026_spring",
        greenhouse_id=63,
        greenhouse_code="GH-PIDU",
        planting_code="P-2026",
        initial_fruit_maturity_fraction=0.25,
        protocol_status="declared",
        pick_weekdays=[0],
        pick_local_hour=8,
        season_complete=True,
        driver_model_status="accepted_target_crop_model",
        air_temperature_source="observed_indoor",
    )

    assert drivers["timestamp"].tolist() == _trajectory()["timestamp"].tolist()[1:]
    assert drivers["net_fruit_dry_matter_change_kg_m2"].tolist() == pytest.approx(
        [0.010, -0.003]
    )
    assert drivers["initial_fruit_dry_matter_kg_m2"].tolist() == [0.1, 0.0]
    assert drivers["initial_fruit_maturity_fraction"].tolist() == [0.25, 0.0]
    assert drivers["dt_hours"].tolist() == [2.0, 1.0]
    assert drivers["pick"].tolist() == [True, False]
    assert drivers["pick_source"].eq("management_protocol").all()
    assert drivers["driver_model_status"].eq("accepted_target_crop_model").all()
    assert drivers["fruit_change_source"].eq(
        "greenlight_fruit_state_balance"
    ).all()
    assert audit["state_row_count"] == 3
    assert audit["driver_row_count"] == 2
    assert audit["pick_permission_count"] == 1
    assert audit["initial_fruit_dry_matter_kg_m2"] == pytest.approx(0.1)
    assert audit["native_harvest_added_back_kg_m2"] == pytest.approx(0.002)
    assert audit["net_fruit_dry_matter_change_kg_m2"] == pytest.approx(0.007)
    assert audit["mass_accounting_residual_kg_m2"] == pytest.approx(0.0, abs=1e-12)
    assert audit["calibration_eligible"] is True
    assert audit["blocking_reasons"] == []


def test_missing_management_protocol_builds_explicitly_ineligible_no_pick_driver():
    drivers, audit = build_harvest_drivers_from_state_trajectory(
        _trajectory(),
        season_id="partial",
        greenhouse_id=63,
        greenhouse_code="GH-PIDU",
        planting_code="P-2026",
        initial_fruit_maturity_fraction=0.0,
        protocol_status="missing",
        pick_weekdays=None,
        pick_local_hour=None,
        season_complete=False,
        driver_model_status="diagnostic_not_final",
        air_temperature_source="observed_indoor",
    )

    assert not drivers["pick"].any()
    assert drivers["pick_source"].eq("management_protocol_missing").all()
    assert audit["calibration_eligible"] is False
    assert set(audit["blocking_reasons"]) == {
        "management_protocol_missing",
        "season_incomplete",
        "crop_model_not_accepted",
    }


def test_period_protocol_changes_fully_cover_driver_horizon():
    drivers, audit = build_harvest_drivers_from_state_trajectory(
        _trajectory(),
        season_id="2026_spring",
        greenhouse_id=63,
        greenhouse_code="GH-PIDU",
        planting_code="P-2026",
        initial_fruit_maturity_fraction=0.0,
        protocol_status="missing",
        pick_weekdays=None,
        pick_local_hour=None,
        protocol_periods=_protocol_periods(),
        season_complete=True,
        driver_model_status="accepted_target_crop_model",
        air_temperature_source="observed_indoor",
    )

    assert drivers["pick"].tolist() == [True, True]
    assert drivers["pick_source"].eq("management_protocol").all()
    assert audit["protocol_coverage_fraction"] == 1.0
    assert audit["confirmed_protocol_interval_count"] == 2
    assert audit["protocol_ids"] == ["P-A", "P-B"]
    assert audit["calibration_eligible"] is True


@pytest.mark.parametrize(
    ("periods", "expected_source", "expected_blocker"),
    [
        (
            _protocol_periods(second_status="provisional"),
            "provisional_management_protocol",
            "management_protocol_not_confirmed",
        ),
        (
            _protocol_periods(gap=True),
            "management_protocol_missing",
            "management_protocol_missing",
        ),
    ],
)
def test_provisional_or_gapped_protocol_periods_remain_ineligible(
    periods: pd.DataFrame,
    expected_source: str,
    expected_blocker: str,
):
    drivers, audit = build_harvest_drivers_from_state_trajectory(
        _trajectory(),
        season_id="2026_spring",
        greenhouse_id=63,
        greenhouse_code="GH-PIDU",
        planting_code="P-2026",
        initial_fruit_maturity_fraction=0.0,
        protocol_status="missing",
        pick_weekdays=None,
        pick_local_hour=None,
        protocol_periods=periods,
        season_complete=True,
        driver_model_status="accepted_target_crop_model",
        air_temperature_source="observed_indoor",
    )

    assert expected_source in set(drivers["pick_source"])
    assert expected_blocker in audit["blocking_reasons"]
    assert audit["calibration_eligible"] is False


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.assign(
                timestamp=[frame.loc[0, "timestamp"]] * len(frame)
            ),
            "unique",
        ),
        (
            lambda frame: frame.assign(
                native_harvested_dry_kg_m2=[0.001, 0.0, 0.002]
            ),
            "first",
        ),
        (
            lambda frame: frame.assign(
                timestamp=pd.to_datetime(
                    [
                        "2026-03-01 00:00:00+08:00",
                        "2026-03-02 01:00:00+08:00",
                        "2026-03-02 02:00:00+08:00",
                    ]
                )
            ),
            "24",
        ),
    ],
)
def test_state_trajectory_builder_rejects_unauditable_intervals(mutator, message):
    with pytest.raises(ValueError, match=message):
        build_harvest_drivers_from_state_trajectory(
            mutator(_trajectory()),
            season_id="2026_spring",
            greenhouse_id=63,
            greenhouse_code="GH-PIDU",
            planting_code="P-2026",
            initial_fruit_maturity_fraction=0.0,
            protocol_status="declared",
            pick_weekdays=[0, 3],
            pick_local_hour=8,
            season_complete=True,
            driver_model_status="accepted_target_crop_model",
            air_temperature_source="observed_indoor",
        )


def test_manifest_builder_combines_strictly_chronological_seasons(tmp_path):
    first = _trajectory()
    second = _trajectory().copy()
    second["timestamp"] = second["timestamp"] + pd.DateOffset(years=1)
    first_path = tmp_path / "season_1.csv"
    second_path = tmp_path / "season_2.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    manifest_path = tmp_path / "manifest.yml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "timezone": "Asia/Shanghai",
                "maximum_interval_hours": 24,
                "management_protocol": {
                    "status": "declared",
                    "pick_weekdays": [0],
                    "pick_local_hour": 8,
                },
                "seasons": [
                    {
                        "season_id": "s1",
                        "trajectory_path": str(first_path),
                        "greenhouse_id": 63,
                        "greenhouse_code": "GH-PIDU",
                        "planting_code": "P1",
                        "initial_fruit_maturity_fraction": 0.0,
                        "season_complete": True,
                        "driver_model_status": "accepted_target_crop_model",
                        "air_temperature_source": "observed_indoor",
                    },
                    {
                        "season_id": "s2",
                        "trajectory_path": str(second_path),
                        "greenhouse_id": 63,
                        "greenhouse_code": "GH-PIDU",
                        "planting_code": "P2",
                        "initial_fruit_maturity_fraction": 0.0,
                        "season_complete": True,
                        "driver_model_status": "accepted_target_crop_model",
                        "air_temperature_source": "observed_indoor",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output_csv = tmp_path / "drivers.csv"
    output_audit = tmp_path / "audit.json"

    audit = build_harvest_driver_bundle(
        manifest_path=manifest_path,
        output_csv_path=output_csv,
        output_audit_path=output_audit,
    )

    combined = pd.read_csv(output_csv)
    assert combined["season_id"].tolist() == ["s1", "s1", "s2", "s2"]
    assert combined.groupby("season_id")["planting_code"].first().to_dict() == {
        "s1": "P1",
        "s2": "P2",
    }
    assert audit["season_count"] == 2
    assert audit["driver_row_count"] == 4
    assert audit["all_seasons_calibration_eligible"] is True
    assert audit["blocking_reasons"] == []
    assert output_audit.exists()


def test_manifest_builder_reads_per_season_protocols_from_workbook(tmp_path):
    first = _trajectory()
    second = _trajectory().copy()
    second["timestamp"] = second["timestamp"] + pd.DateOffset(years=1)
    first_path = tmp_path / "season_1.csv"
    second_path = tmp_path / "season_2.csv"
    first.to_csv(first_path, index=False)
    second.to_csv(second_path, index=False)
    seasons = []
    protocol_rows = []
    for season_id, planting, trajectory_path, start in (
        ("s1", "P1", first_path, pd.Timestamp("2026-03-01 00:00:00")),
        ("s2", "P2", second_path, pd.Timestamp("2027-03-01 00:00:00")),
    ):
        seasons.append(
            {
                "season_id": season_id,
                "trajectory_path": str(trajectory_path),
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": planting,
                "initial_fruit_maturity_fraction": 0.0,
                "season_complete": True,
                "driver_model_status": "accepted_target_crop_model",
                "air_temperature_source": "observed_indoor",
            }
        )
        protocol_rows.append(
            {
                "protocol_id": f"PROTOCOL-{season_id}",
                "season_id": season_id,
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": planting,
                "timezone": "Asia/Shanghai",
                "effective_start_datetime": start,
                "effective_end_datetime": None,
                "protocol_status": "confirmed",
                "pick_weekdays": "0",
                "pick_local_hour": 8,
                "confirmation_source": "site_manager_signed_schedule",
                "source_record_id": f"SOURCE-{season_id}",
                "notes": "",
            }
        )
    manifest_path = tmp_path / "manifest.yml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "timezone": "Asia/Shanghai",
                "maximum_interval_hours": 24,
                "management_protocol": {"status": "missing"},
                "seasons": seasons,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    workbook = tmp_path / "protocols.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(protocol_rows).to_excel(
            writer, sheet_name="HarvestProtocol", index=False
        )

    audit = build_harvest_driver_bundle(
        manifest_path=manifest_path,
        protocol_workbook_path=workbook,
        output_csv_path=tmp_path / "drivers.csv",
        output_audit_path=tmp_path / "audit.json",
    )

    combined = pd.read_csv(tmp_path / "drivers.csv")
    assert combined["pick_source"].eq("management_protocol").all()
    assert audit["protocol_workbook_path"] == str(workbook.resolve())
    assert audit["all_seasons_calibration_eligible"] is True
    assert all(season["protocol_coverage_fraction"] == 1.0 for season in audit["seasons"])
