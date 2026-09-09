from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from experiments.crop.harvest_package_validation import (
    assess_harvest_calibration_package,
)
from processing.chengdu_harvest_events import load_harvest_events
from processing.chengdu_harvest_protocols import (
    HARVEST_PROTOCOL_COLUMNS,
    load_harvest_protocols,
)
from processing.chengdu_harvest_seasons import (
    bind_harvest_events_to_seasons,
    load_harvest_seasons,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CONFIG = (
    REPOSITORY_ROOT / "configs" / "datasets" / "chengdu_agri_greenhouse_001.yml"
)
DEFAULT_HARVEST_CONFIG = (
    REPOSITORY_ROOT / "configs" / "crops" / "chengdu_tomato_harvest.yml"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "chengdu_agri_greenhouse_001"
    / "harvest_model"
    / "package_preflight"
)


def run_package_preflight(
    *,
    workbook_path: str | Path,
    drivers_path: str | Path,
    output_root: str | Path,
    dataset_config_path: str | Path = DEFAULT_DATASET_CONFIG,
    harvest_config_path: str | Path = DEFAULT_HARVEST_CONFIG,
) -> dict[str, Any]:
    dataset = _load_yaml(dataset_config_path)
    harvest = _load_yaml(harvest_config_path)
    crop_config = dataset["crop_observations"]
    target_site = harvest["target_site"]
    target_ids = [int(value) for value in crop_config["target_greenhouse_ids"]]
    target_codes = [str(value) for value in crop_config["target_greenhouse_codes"]]
    target_id = int(target_site["greenhouse_id"])
    target_code = str(target_site["greenhouse_code"])
    target_area_m2 = float(target_site["greenhouse_area_m2"])
    if target_id not in target_ids or target_code not in target_codes:
        raise ValueError("dataset and harvest target greenhouse identities do not agree")

    events = load_harvest_events(
        workbook_path,
        target_greenhouse_ids=target_ids,
        target_greenhouse_codes=target_codes,
        normalization_areas_m2={target_code: target_area_m2},
        source_sites=crop_config.get("source_sites", {}),
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
    )
    seasons = load_harvest_seasons(
        workbook_path,
        target_greenhouse_ids=target_ids,
        target_greenhouse_codes=target_codes,
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
    )
    bound_events, season_audit = bind_harvest_events_to_seasons(events, seasons)
    drivers = pd.read_csv(drivers_path)
    try:
        protocols = load_harvest_protocols(
            workbook_path,
            target_greenhouse_ids=target_ids,
            target_greenhouse_codes=target_codes,
            timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
        )
        protocol_sheet_present = True
    except ValueError as exc:
        if "HarvestProtocol sheet" not in str(exc):
            raise
        protocols = pd.DataFrame(columns=HARVEST_PROTOCOL_COLUMNS)
        protocol_sheet_present = False
    assessment = assess_harvest_calibration_package(
        drivers,
        bound_events,
        target_greenhouse_id=target_id,
        target_greenhouse_code=target_code,
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
        protocols=protocols,
    )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "harvest_calibration_package_preflight.json"
    assessment.update(
        {
            "workbook_path": str(Path(workbook_path).resolve()),
            "drivers_path": str(Path(drivers_path).resolve()),
            "target_normalization_area_m2": target_area_m2,
            "batch_semantics": "greenhouse_calendar_day_total",
            "season_manifest_audit": season_audit,
            "management_protocol_sheet_present": protocol_sheet_present,
            "artifact": str(report_path.resolve()),
        }
    )
    report_path.write_text(
        json.dumps(assessment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return assessment


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight Chengdu measured harvest and GreenLight drivers."
    )
    parser.add_argument("--harvest-workbook", required=True, type=Path)
    parser.add_argument("--drivers", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--harvest-config", type=Path, default=DEFAULT_HARVEST_CONFIG)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_package_preflight(
        workbook_path=args.harvest_workbook,
        drivers_path=args.drivers,
        output_root=args.output_root,
        dataset_config_path=args.dataset_config,
        harvest_config_path=args.harvest_config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
