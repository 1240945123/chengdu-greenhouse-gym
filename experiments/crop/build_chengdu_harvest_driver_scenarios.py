from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiments.crop.calibrate_harvest_model import (
    DRIVER_UNCERTAINTY_SOURCES,
    validate_uncertainty_driver_scenarios,
)
from processing.chengdu_harvest_drivers import (
    build_harvest_drivers_from_state_trajectory,
)
from processing.chengdu_harvest_protocols import load_harvest_protocols


_SEASON_FIELDS = {
    "season_id",
    "greenhouse_id",
    "greenhouse_code",
    "planting_code",
    "initial_fruit_maturity_fraction",
    "season_complete",
    "driver_model_status",
    "air_temperature_source",
}
_SCENARIO_FIELDS = {
    "driver_scenario_id",
    "driver_uncertainty_source",
    "trajectory_path",
}


def build_harvest_driver_scenario_bundle(
    *,
    manifest_path: str | Path,
    protocol_workbook_path: str | Path,
    output_csv_path: str | Path,
    output_audit_path: str | Path,
) -> dict[str, Any]:
    """Build aligned uncertainty drivers from traceable GreenLight states."""
    manifest_source = Path(manifest_path)
    manifest = yaml.safe_load(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("driver scenario manifest must be a mapping")
    season = manifest.get("season")
    scenarios = manifest.get("scenarios")
    if not isinstance(season, dict):
        raise ValueError("driver scenario manifest season must be a mapping")
    if missing := sorted(_SEASON_FIELDS - set(season)):
        raise ValueError(f"driver scenario season missing fields: {', '.join(missing)}")
    if not isinstance(scenarios, list) or len(scenarios) < 2:
        raise ValueError("driver scenario manifest requires at least two scenarios")
    if "point_driver_path" not in manifest:
        raise ValueError("driver scenario manifest requires point_driver_path")

    if season["season_complete"] is not True:
        raise ValueError("driver scenario season must be complete")
    if str(season["driver_model_status"]) != "accepted_target_crop_model":
        raise ValueError("driver scenario crop model must be accepted_target_crop_model")
    scenario_ids: list[str] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"driver scenario {index} must be a mapping")
        if missing := sorted(_SCENARIO_FIELDS - set(scenario)):
            raise ValueError(
                f"driver scenario {index} missing fields: {', '.join(missing)}"
            )
        scenario_id = str(scenario["driver_scenario_id"]).strip()
        if not scenario_id:
            raise ValueError("driver scenario IDs must be present")
        scenario_ids.append(scenario_id)
        source = str(scenario["driver_uncertainty_source"]).strip()
        if source not in DRIVER_UNCERTAINTY_SOURCES:
            raise ValueError(f"unsupported driver uncertainty source: {source}")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("duplicate driver_scenario_id values are not allowed")

    timezone = str(manifest.get("timezone", "Asia/Shanghai"))
    maximum_interval_hours = float(manifest.get("maximum_interval_hours", 24.0))
    point_path = _resolve_path(manifest["point_driver_path"], manifest_source.parent)
    point_drivers = pd.read_csv(point_path)
    protocol_path = Path(protocol_workbook_path)
    protocols = load_harvest_protocols(
        protocol_path,
        target_greenhouse_ids=[int(season["greenhouse_id"])],
        target_greenhouse_codes=[str(season["greenhouse_code"])],
        timezone=timezone,
    )
    season_protocols = protocols[
        protocols["season_id"].astype(str).eq(str(season["season_id"]))
    ].copy()

    scenario_parts: list[pd.DataFrame] = []
    scenario_audits: list[dict[str, Any]] = []
    for scenario in scenarios:
        scenario_id = str(scenario["driver_scenario_id"]).strip()
        source = str(scenario["driver_uncertainty_source"]).strip()
        trajectory_path = _resolve_path(
            scenario["trajectory_path"], manifest_source.parent
        )
        trajectory = pd.read_csv(trajectory_path)
        drivers, audit = build_harvest_drivers_from_state_trajectory(
            trajectory,
            season_id=str(season["season_id"]),
            greenhouse_id=int(season["greenhouse_id"]),
            greenhouse_code=str(season["greenhouse_code"]),
            planting_code=str(season["planting_code"]),
            initial_fruit_maturity_fraction=float(
                season["initial_fruit_maturity_fraction"]
            ),
            protocol_status="missing",
            pick_weekdays=None,
            pick_local_hour=None,
            protocol_periods=season_protocols,
            season_complete=True,
            driver_model_status="accepted_target_crop_model",
            air_temperature_source=str(season["air_temperature_source"]),
            timezone=timezone,
            maximum_interval_hours=maximum_interval_hours,
        )
        if not audit["calibration_eligible"]:
            reasons = ", ".join(map(str, audit["blocking_reasons"]))
            raise ValueError(f"driver scenario {scenario_id} is ineligible: {reasons}")
        drivers["driver_scenario_id"] = scenario_id
        drivers["driver_uncertainty_source"] = source
        scenario_parts.append(drivers)
        scenario_audits.append(
            {
                **audit,
                "driver_scenario_id": scenario_id,
                "driver_uncertainty_source": source,
                "trajectory_path": str(trajectory_path.resolve()),
            }
        )

    combined = pd.concat(scenario_parts, ignore_index=True)
    canonical, alignment_audit = validate_uncertainty_driver_scenarios(
        point_drivers,
        combined,
    )
    output_csv = Path(output_csv_path)
    output_audit = Path(output_audit_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_csv, index=False, encoding="utf-8-sig")
    audit = {
        "manifest_path": str(manifest_source.resolve()),
        "protocol_workbook_path": str(protocol_path.resolve()),
        "point_driver_path": str(point_path.resolve()),
        "timezone": timezone,
        "scenario_count": int(alignment_audit["scenario_count"]),
        "scenario_ids": alignment_audit["scenario_ids"],
        "uncertainty_sources": alignment_audit["uncertainty_sources"],
        "scenario_row_count": int(len(canonical)),
        "scenario_sampling_policy": alignment_audit["scenario_sampling_policy"],
        "all_scenarios_calibration_eligible": True,
        "protocol_ids": sorted(set(protocols["protocol_id"].astype(str))),
        "protocol_confirmation_sources": sorted(
            set(protocols["confirmation_source"].astype(str))
        ),
        "scenario_build_audits": scenario_audits,
        "alignment_audit": alignment_audit,
        "artifacts": {
            "scenario_driver_csv": str(output_csv.resolve()),
            "scenario_audit_json": str(output_audit.resolve()),
        },
    }
    output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def _resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validated Chengdu harvest uncertainty driver scenarios"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--protocol-workbook", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-audit", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_harvest_driver_scenario_bundle(
        manifest_path=args.manifest,
        protocol_workbook_path=args.protocol_workbook,
        output_csv_path=args.output_csv,
        output_audit_path=args.output_audit,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
