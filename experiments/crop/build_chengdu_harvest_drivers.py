from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from processing.chengdu_harvest_drivers import (
    DRIVER_COLUMNS,
    build_harvest_drivers_from_state_trajectory,
)
from processing.chengdu_harvest_protocols import load_harvest_protocols


def build_harvest_driver_bundle(
    *,
    manifest_path: str | Path,
    output_csv_path: str | Path,
    output_audit_path: str | Path,
    protocol_workbook_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest_source = Path(manifest_path)
    manifest = yaml.safe_load(manifest_source.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("harvest driver manifest must be a mapping")
    protocol = manifest.get("management_protocol")
    seasons = manifest.get("seasons")
    if not isinstance(protocol, dict):
        raise ValueError("manifest management_protocol must be a mapping")
    if not isinstance(seasons, list) or not seasons:
        raise ValueError("manifest seasons must be a non-empty list")
    timezone = str(manifest.get("timezone", "Asia/Shanghai"))
    maximum_interval_hours = float(manifest.get("maximum_interval_hours", 24.0))
    protocol_status = str(protocol.get("status", ""))
    pick_weekdays = protocol.get("pick_weekdays")
    pick_local_hour = protocol.get("pick_local_hour")

    required_season_fields = {
        "season_id",
        "trajectory_path",
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "initial_fruit_maturity_fraction",
        "season_complete",
        "driver_model_status",
        "air_temperature_source",
    }
    protocols: pd.DataFrame | None = None
    if protocol_workbook_path is not None:
        protocol_ids = []
        protocol_codes = []
        for index, season in enumerate(seasons):
            if not isinstance(season, dict):
                raise ValueError(f"manifest season {index} must be a mapping")
            if missing := sorted(required_season_fields - set(season)):
                raise ValueError(
                    f"manifest season {index} missing fields: {', '.join(missing)}"
                )
            protocol_ids.append(int(season["greenhouse_id"]))
            protocol_codes.append(str(season["greenhouse_code"]))
        protocols = load_harvest_protocols(
            protocol_workbook_path,
            target_greenhouse_ids=protocol_ids,
            target_greenhouse_codes=protocol_codes,
            timezone=timezone,
        )
    driver_parts: list[pd.DataFrame] = []
    season_audits: list[dict[str, Any]] = []
    seen_seasons: set[str] = set()
    previous_end: pd.Timestamp | None = None
    for index, season in enumerate(seasons):
        if not isinstance(season, dict):
            raise ValueError(f"manifest season {index} must be a mapping")
        if missing := sorted(required_season_fields - set(season)):
            raise ValueError(
                f"manifest season {index} missing fields: {', '.join(missing)}"
            )
        season_id = str(season["season_id"])
        if season_id in seen_seasons:
            raise ValueError(f"duplicate manifest season_id: {season_id}")
        seen_seasons.add(season_id)
        trajectory_path = Path(str(season["trajectory_path"]))
        if not trajectory_path.is_absolute():
            trajectory_path = Path.cwd() / trajectory_path
        trajectory = pd.read_csv(trajectory_path)
        season_protocols = (
            protocols[protocols["season_id"].astype(str).eq(season_id)].copy()
            if protocols is not None
            else None
        )
        drivers, audit = build_harvest_drivers_from_state_trajectory(
            trajectory,
            season_id=season_id,
            greenhouse_id=int(season["greenhouse_id"]),
            greenhouse_code=str(season["greenhouse_code"]),
            planting_code=str(season["planting_code"]),
            initial_fruit_maturity_fraction=float(
                season["initial_fruit_maturity_fraction"]
            ),
            protocol_status=protocol_status,
            pick_weekdays=pick_weekdays,
            pick_local_hour=pick_local_hour,
            season_complete=season["season_complete"],
            driver_model_status=str(season["driver_model_status"]),
            air_temperature_source=str(season["air_temperature_source"]),
            timezone=timezone,
            maximum_interval_hours=maximum_interval_hours,
            protocol_periods=season_protocols,
        )
        start = pd.Timestamp(audit["state_start"])
        end = pd.Timestamp(audit["state_end"])
        if previous_end is not None and start <= previous_end:
            raise ValueError("manifest seasons must be strictly chronological and disjoint")
        previous_end = end
        audit["trajectory_path"] = str(trajectory_path.resolve())
        driver_parts.append(drivers)
        season_audits.append(audit)

    combined = pd.concat(driver_parts, ignore_index=True)[DRIVER_COLUMNS]
    blockers = sorted(
        {
            reason
            for audit in season_audits
            for reason in audit["blocking_reasons"]
        }
    )
    bundle_audit = {
        "manifest_path": str(manifest_source.resolve()),
        "protocol_workbook_path": (
            str(Path(protocol_workbook_path).resolve())
            if protocol_workbook_path is not None
            else None
        ),
        "timezone": timezone,
        "season_count": int(len(season_audits)),
        "driver_row_count": int(len(combined)),
        "all_seasons_calibration_eligible": not blockers,
        "blocking_reasons": blockers,
        "seasons": season_audits,
    }
    output_csv = Path(output_csv_path)
    output_audit = Path(output_audit_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_audit.write_text(
        json.dumps(bundle_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return bundle_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auditable Chengdu harvest drivers from crop-state trajectories"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--protocol-workbook")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    audit = build_harvest_driver_bundle(
        manifest_path=args.manifest,
        output_csv_path=args.output_csv,
        output_audit_path=args.output_audit,
        protocol_workbook_path=args.protocol_workbook,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
