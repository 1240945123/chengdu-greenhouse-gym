from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from experiments.crop.build_chengdu_harvest_driver_scenarios import (
    build_harvest_driver_scenario_bundle,
)
from processing.chengdu_harvest_drivers import (
    build_harvest_drivers_from_state_trajectory,
)
from processing.chengdu_harvest_protocols import load_harvest_protocols


TARGET_CODE = "GH202602061448266452514"


def _trajectory(*, final_mass: float = 0.120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-03-02 07:00:00+08:00",
                    "2026-03-02 09:00:00+08:00",
                    "2026-03-02 10:00:00+08:00",
                ]
            ),
            "standing_dry_kg_m2": [0.100, 0.110, final_mass],
            "native_harvested_dry_kg_m2": [0.0, 0.0, 0.0],
            "air_temperature_c": [20.0, 21.0, 22.0],
        }
    )


def _write_protocol(path: Path, *, status: str = "confirmed") -> Path:
    frame = pd.DataFrame(
        [
            {
                "protocol_id": "PROTOCOL-S1",
                "season_id": "s1",
                "greenhouse_id": 63,
                "greenhouse_code": TARGET_CODE,
                "planting_code": "P1",
                "timezone": "Asia/Shanghai",
                "effective_start_datetime": "2026-03-01 00:00:00",
                "effective_end_datetime": None,
                "protocol_status": status,
                "pick_weekdays": "0",
                "pick_local_hour": 8,
                "confirmation_source": (
                    "site_manager_signed_schedule"
                    if status == "confirmed"
                    else "researcher_assumption"
                ),
                "source_record_id": "SOURCE-S1",
                "notes": "",
            }
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="HarvestProtocol", index=False)
    return path


def _write_bundle_inputs(
    tmp_path: Path,
    *,
    season_complete: bool = True,
    driver_model_status: str = "accepted_target_crop_model",
    protocol_status: str = "confirmed",
) -> tuple[Path, Path]:
    protocol_path = _write_protocol(
        tmp_path / "harvest.xlsx", status=protocol_status
    )
    periods = load_harvest_protocols(
        protocol_path,
        target_greenhouse_ids=[63],
        target_greenhouse_codes=[TARGET_CODE],
    )
    point, _ = build_harvest_drivers_from_state_trajectory(
        _trajectory(final_mass=0.115),
        season_id="s1",
        greenhouse_id=63,
        greenhouse_code=TARGET_CODE,
        planting_code="P1",
        initial_fruit_maturity_fraction=0.0,
        protocol_status="missing",
        pick_weekdays=None,
        pick_local_hour=None,
        protocol_periods=periods,
        season_complete=True,
        driver_model_status="accepted_target_crop_model",
        air_temperature_source="greenlight_simulated_indoor",
    )
    point_path = tmp_path / "point.csv"
    point.to_csv(point_path, index=False)
    scenario_rows = []
    for scenario_id, source, final_mass in (
        (
            "climate_low",
            "greenhouse_climate_model_parameter_uncertainty",
            0.112,
        ),
        ("weather_high", "outdoor_weather_scenario_uncertainty", 0.128),
    ):
        trajectory_path = tmp_path / f"{scenario_id}.csv"
        _trajectory(final_mass=final_mass).to_csv(trajectory_path, index=False)
        scenario_rows.append(
            {
                "driver_scenario_id": scenario_id,
                "driver_uncertainty_source": source,
                "trajectory_path": str(trajectory_path),
            }
        )
    manifest = {
        "timezone": "Asia/Shanghai",
        "maximum_interval_hours": 24,
        "point_driver_path": str(point_path),
        "season": {
            "season_id": "s1",
            "greenhouse_id": 63,
            "greenhouse_code": TARGET_CODE,
            "planting_code": "P1",
            "initial_fruit_maturity_fraction": 0.0,
            "season_complete": season_complete,
            "driver_model_status": driver_model_status,
            "air_temperature_source": "greenlight_simulated_indoor",
        },
        "scenarios": scenario_rows,
    }
    manifest_path = tmp_path / "scenario_manifest.yml"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return manifest_path, protocol_path


def test_builds_validated_driver_scenario_bundle_from_state_trajectories(
    tmp_path: Path,
):
    manifest, protocol = _write_bundle_inputs(tmp_path)
    output_csv = tmp_path / "scenarios.csv"
    output_audit = tmp_path / "scenario_audit.json"

    audit = build_harvest_driver_scenario_bundle(
        manifest_path=manifest,
        protocol_workbook_path=protocol,
        output_csv_path=output_csv,
        output_audit_path=output_audit,
    )

    scenarios = pd.read_csv(output_csv)
    assert len(scenarios) == 4
    assert scenarios["driver_scenario_id"].drop_duplicates().tolist() == [
        "climate_low",
        "weather_high",
    ]
    assert scenarios["pick_source"].eq("management_protocol").all()
    assert scenarios.groupby("driver_scenario_id")[
        "net_fruit_dry_matter_change_kg_m2"
    ].sum().nunique() == 2
    assert audit["scenario_count"] == 2
    assert audit["all_scenarios_calibration_eligible"] is True
    assert len(audit["scenario_build_audits"]) == 2
    assert output_audit.exists()


@pytest.mark.parametrize(
    ("season_complete", "driver_model_status", "protocol_status", "message"),
    [
        (False, "accepted_target_crop_model", "confirmed", "complete"),
        (True, "diagnostic_not_final", "confirmed", "accepted_target_crop_model"),
        (True, "accepted_target_crop_model", "provisional", "not_confirmed"),
    ],
)
def test_rejects_ineligible_scenario_evidence_without_writing_outputs(
    tmp_path: Path,
    season_complete: bool,
    driver_model_status: str,
    protocol_status: str,
    message: str,
):
    manifest, protocol = _write_bundle_inputs(
        tmp_path,
        season_complete=season_complete,
        driver_model_status=driver_model_status,
        protocol_status=protocol_status,
    )
    output_csv = tmp_path / "out" / "scenarios.csv"
    output_audit = tmp_path / "out" / "audit.json"

    with pytest.raises(ValueError, match=message):
        build_harvest_driver_scenario_bundle(
            manifest_path=manifest,
            protocol_workbook_path=protocol,
            output_csv_path=output_csv,
            output_audit_path=output_audit,
        )

    assert not output_csv.exists()
    assert not output_audit.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("unknown_source", "unsupported.*uncertainty source"),
    ],
)
def test_rejects_invalid_scenario_manifest_without_writing_outputs(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    manifest_path, protocol = _write_bundle_inputs(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        manifest["scenarios"][1]["driver_scenario_id"] = manifest["scenarios"][0][
            "driver_scenario_id"
        ]
    else:
        manifest["scenarios"][0]["driver_uncertainty_source"] = "unknown_noise"
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    output_csv = tmp_path / "out" / "scenarios.csv"
    output_audit = tmp_path / "out" / "audit.json"

    with pytest.raises(ValueError, match=message):
        build_harvest_driver_scenario_bundle(
            manifest_path=manifest_path,
            protocol_workbook_path=protocol,
            output_csv_path=output_csv,
            output_audit_path=output_audit,
        )

    assert not output_csv.exists()
    assert not output_audit.exists()


def test_rejects_scenario_timestamp_mismatch_without_writing_outputs(tmp_path: Path):
    manifest_path, protocol = _write_bundle_inputs(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    second_path = Path(manifest["scenarios"][1]["trajectory_path"])
    shortened = pd.read_csv(second_path).iloc[:-1]
    shortened.to_csv(second_path, index=False)
    output_csv = tmp_path / "out" / "scenarios.csv"
    output_audit = tmp_path / "out" / "audit.json"

    with pytest.raises(ValueError, match="keys do not match"):
        build_harvest_driver_scenario_bundle(
            manifest_path=manifest_path,
            protocol_workbook_path=protocol,
            output_csv_path=output_csv,
            output_audit_path=output_audit,
        )

    assert not output_csv.exists()
    assert not output_audit.exists()
