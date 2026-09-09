from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import experiments.crop.run_chengdu_harvest_pipeline as harvest_pipeline
from experiments.crop.harvest_package_validation import (
    assess_harvest_calibration_package,
)
from experiments.crop.validate_chengdu_harvest_package import run_package_preflight


TARGET_CODE = "GH202602061448266452514"


def _package_frames(
    *,
    season_count: int = 3,
    second_complete: bool = True,
    second_driver_status: str = "accepted_target_crop_model",
    greenhouse_code: str = "GH-PIDU",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_parts = []
    event_rows = []
    for index in range(season_count):
        season_id = f"s{index + 1}"
        start = pd.Timestamp(year=2024 + index, month=3, day=1, hour=8)
        timestamps = pd.date_range(start, periods=100, freq="D")
        complete = second_complete if index == 1 else True
        driver_status = second_driver_status if index == 1 else "accepted_target_crop_model"
        driver_parts.append(
            pd.DataFrame(
                {
                    "season_id": season_id,
                    "timestamp": timestamps,
                    "net_fruit_dry_matter_change_kg_m2": 0.01,
                    "air_temperature_c": 20.0,
                    "dt_hours": 24.0,
                    "pick": [
                        timestamp.weekday() in {0, 2, 4}
                        for timestamp in timestamps
                    ],
                    "pick_source": "management_protocol",
                    "driver_model_status": driver_status,
                    "season_complete": complete,
                    "fruit_change_source": "greenlight_fruit_state_balance",
                    "air_temperature_source": "observed_indoor",
                    "greenhouse_id": 63,
                    "greenhouse_code": greenhouse_code,
                    "planting_code": f"P-{season_id}",
                    "initial_fruit_dry_matter_kg_m2": [0.0] * 100,
                    "initial_fruit_maturity_fraction": [0.0] * 100,
                }
            )
        )
        for event_index, timestamp in enumerate(timestamps[60:100:5], start=1):
            event_rows.append(
                {
                    "harvest_event_id": f"H-{season_id}-{event_index}",
                    "season_id": season_id,
                    "timestamp": timestamp,
                    "greenhouse_id": 63,
                    "greenhouse_code": greenhouse_code,
                    "planting_code": f"P-{season_id}",
                    "target_eligible": True,
                    "fresh_kg_m2": 0.1,
                    "season_complete_evidence": complete,
                }
            )
    return pd.concat(driver_parts, ignore_index=True), pd.DataFrame(event_rows)


def _protocols_for_drivers(
    drivers: pd.DataFrame,
    *,
    status: str = "confirmed",
) -> pd.DataFrame:
    rows = []
    for season_id, season in drivers.groupby("season_id", sort=False):
        start = pd.Timestamp(season["timestamp"].min())
        start -= pd.Timedelta(hours=float(season["dt_hours"].iloc[0]))
        if start.tzinfo is None:
            start = start.tz_localize("Asia/Shanghai")
        rows.append(
            {
                "protocol_id": f"PROTOCOL-{season_id}",
                "season_id": str(season_id),
                "greenhouse_id": int(season["greenhouse_id"].iloc[0]),
                "greenhouse_code": str(season["greenhouse_code"].iloc[0]),
                "planting_code": str(season["planting_code"].iloc[0]),
                "effective_start": start,
                "effective_end": pd.NaT,
                "protocol_status": status,
                "pick_weekdays": (0, 2, 4),
                "pick_local_hour": 8,
                "confirmation_source": "site_manager_signed_schedule",
            }
        )
    return pd.DataFrame(rows)


def test_package_assessment_returns_three_season_chronological_split():
    drivers, events = _package_frames()

    report = assess_harvest_calibration_package(
        drivers,
        events,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["qualifying_complete_seasons"] == ["s1", "s2", "s3"]
    assert report["calibration_package_ready"] is True
    assert report["independent_validation_package_ready"] is True
    assert report["split"] == {
        "train_seasons": ["s1"],
        "validation_seasons": ["s2"],
        "test_seasons": ["s3"],
    }
    assert all(not row["blocking_reasons"] for row in report["season_readiness"])


def test_package_assessment_rejects_pick_flags_inconsistent_with_protocol():
    drivers, events = _package_frames(season_count=2)
    drivers["pick"] = False

    report = assess_harvest_calibration_package(
        drivers,
        events,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        protocols=_protocols_for_drivers(drivers),
    )

    assert report["independent_validation_package_ready"] is False
    assert all(
        "management_protocol_pick_mismatch" in row["blocking_reasons"]
        for row in report["season_readiness"]
    )


def test_package_assessment_retains_ongoing_and_diagnostic_model_blockers():
    drivers, events = _package_frames(
        season_count=2,
        second_complete=False,
        second_driver_status="diagnostic_not_final",
    )

    report = assess_harvest_calibration_package(
        drivers,
        events,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["qualifying_complete_seasons"] == ["s1"]
    assert report["calibration_package_ready"] is True
    assert report["independent_validation_package_ready"] is False
    second = next(row for row in report["season_readiness"] if row["season_id"] == "s2")
    assert "season_incomplete" in second["blocking_reasons"]
    assert "crop_model_not_accepted" in second["blocking_reasons"]
    assert report["target_harvest_validated"] is False


def test_package_assessment_rejects_overlapping_qualifying_seasons():
    drivers, events = _package_frames(season_count=2)
    first_driver_start = drivers.loc[drivers["season_id"].eq("s1"), "timestamp"].min()
    second_driver_start = drivers.loc[drivers["season_id"].eq("s2"), "timestamp"].min()
    shift = second_driver_start - (first_driver_start + pd.Timedelta(days=30))
    drivers.loc[drivers["season_id"].eq("s2"), "timestamp"] -= shift
    events.loc[events["season_id"].eq("s2"), "timestamp"] -= shift

    report = assess_harvest_calibration_package(
        drivers,
        events,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["independent_validation_package_ready"] is False
    assert any(
        "season_overlap" in row["blocking_reasons"]
        for row in report["season_readiness"]
    )


def test_production_calibration_requires_shared_package_gate_before_fitting(monkeypatch):
    drivers, events = _package_frames(season_count=2)
    monkeypatch.setattr(
        harvest_pipeline,
        "assess_harvest_calibration_package",
        lambda *args, **kwargs: {
            "independent_validation_package_ready": False,
            "blocking_reasons": ["s2:crop_model_not_accepted"],
        },
        raising=False,
    )
    monkeypatch.setattr(
        harvest_pipeline,
        "bootstrap_harvest_parameter_estimates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fitting started before package gate")
        ),
    )

    with pytest.raises(ValueError, match="not ready for independent validation"):
        harvest_pipeline.calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=_protocols_for_drivers(drivers),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
            maturity_grid_size=5,
            parameter_bootstrap_samples=2,
            bootstrap_samples=2,
        )


def test_production_calibration_rejects_provisional_protocol_before_fitting(monkeypatch):
    drivers, events = _package_frames(season_count=2)
    monkeypatch.setattr(
        harvest_pipeline,
        "bootstrap_harvest_parameter_estimates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fitting started before protocol gate")
        ),
    )

    with pytest.raises(ValueError, match="management_protocol_not_confirmed"):
        harvest_pipeline.calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=_protocols_for_drivers(drivers, status="provisional"),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
            maturity_grid_size=5,
            parameter_bootstrap_samples=2,
            bootstrap_samples=2,
        )


def test_production_calibration_rejects_invalid_driver_scenario_before_fitting(
    monkeypatch,
):
    drivers, events = _package_frames(season_count=2)
    test_drivers = drivers[drivers["season_id"].eq("s2")]
    scenarios = []
    for scenario_id in ("scenario_a", "scenario_b"):
        scenario = test_drivers.copy()
        scenario["driver_scenario_id"] = scenario_id
        scenario["driver_uncertainty_source"] = (
            "greenhouse_climate_model_parameter_uncertainty"
        )
        scenarios.append(scenario)
    invalid = pd.concat(scenarios, ignore_index=True)
    invalid.loc[0, "pick"] = not bool(invalid.loc[0, "pick"])
    monkeypatch.setattr(
        harvest_pipeline,
        "bootstrap_harvest_parameter_estimates",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fitting started before driver-scenario gate")
        ),
    )

    with pytest.raises(ValueError, match="pick mismatch"):
        harvest_pipeline.calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=_protocols_for_drivers(drivers),
            uncertainty_driver_scenarios=invalid,
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
            maturity_grid_size=5,
            parameter_bootstrap_samples=2,
            bootstrap_samples=2,
        )


def test_package_preflight_loads_real_workbook_and_driver_csv(tmp_path: Path):
    drivers, events = _package_frames(greenhouse_code=TARGET_CODE)
    workbook = tmp_path / "harvest.xlsx"
    driver_path = tmp_path / "drivers.csv"
    output = tmp_path / "out"
    event_input = events.assign(
        harvest_datetime=events["timestamp"],
        cultivar="Sainirui",
        harvested_fresh_kg=events["fresh_kg_m2"] * 192.0,
        harvested_area_m2=192.0,
        batch_id=events["harvest_event_id"].map(lambda value: f"B-{value}"),
        source_record_id=events["harvest_event_id"].map(lambda value: f"R-{value}"),
    )
    season_rows = []
    for season_id, season in drivers.groupby("season_id", sort=False):
        season_rows.append(
            {
                "season_id": season_id,
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": season["planting_code"].iloc[0],
                "cultivar": "Sainirui",
                "season_start_datetime": season["timestamp"].min(),
                "season_end_datetime": season["timestamp"].max(),
                "completion_status": "complete",
                "completion_source": "crop_clearance_record",
                "source_record_id": f"SEASON-{season_id}",
            }
        )
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        event_input.to_excel(writer, sheet_name="HarvestEvents", index=False)
        pd.DataFrame(season_rows).to_excel(
            writer, sheet_name="HarvestSeasons", index=False
        )
        pd.DataFrame(
            [
                {
                    "protocol_id": f"PROTOCOL-{row['season_id']}",
                    "season_id": row["season_id"],
                    "greenhouse_id": 63,
                    "greenhouse_code": TARGET_CODE,
                    "planting_code": row["planting_code"],
                    "timezone": "Asia/Shanghai",
                    "effective_start_datetime": row["season_start_datetime"]
                    - pd.Timedelta(days=1),
                    "effective_end_datetime": None,
                    "protocol_status": "confirmed",
                    "pick_weekdays": "0,2,4",
                    "pick_local_hour": 8,
                    "confirmation_source": "site_manager_signed_schedule",
                    "source_record_id": f"PROTOCOL-SOURCE-{row['season_id']}",
                }
                for row in season_rows
            ]
        ).to_excel(writer, sheet_name="HarvestProtocol", index=False)
    drivers.to_csv(driver_path, index=False)

    report = run_package_preflight(
        workbook_path=workbook,
        drivers_path=driver_path,
        output_root=output,
    )

    assert report["qualifying_complete_seasons"] == ["s1", "s2", "s3"]
    assert report["independent_validation_package_ready"] is True
    assert report["management_protocol_coverage_ready"] is True
    assert report["harvest_parameters_fitted"] is False
    assert report["target_harvest_validated"] is False
    assert (output / "harvest_calibration_package_preflight.json").exists()
