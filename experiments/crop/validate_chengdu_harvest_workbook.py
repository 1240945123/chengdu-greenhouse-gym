from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from processing.chengdu_harvest_events import load_harvest_events
from processing.chengdu_harvest_protocols import (
    HARVEST_PROTOCOL_COLUMNS,
    assess_harvest_protocol_coverage,
    load_harvest_protocols,
)
from processing.chengdu_harvest_seasons import (
    bind_harvest_events_to_seasons,
    load_harvest_seasons,
)
from processing.chengdu_measured_harvest import build_measured_harvest_artifacts


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
    / "workbook_preflight"
)


def run_preflight(
    *,
    workbook_path: str | Path,
    output_root: str | Path,
    dataset_config_path: str | Path = DEFAULT_DATASET_CONFIG,
    harvest_config_path: str | Path = DEFAULT_HARVEST_CONFIG,
    minimum_events: int = 8,
    minimum_span_days: float = 30.0,
) -> dict[str, Any]:
    """Validate measured harvest coverage without claiming model acceptance."""
    if minimum_events < 1:
        raise ValueError("minimum_events must be at least one")
    if minimum_span_days < 0.0:
        raise ValueError("minimum_span_days must be non-negative")

    dataset = _load_yaml(dataset_config_path)
    harvest = _load_yaml(harvest_config_path)
    crop_observations = dataset["crop_observations"]
    target_site = harvest["target_site"]
    target_ids = [int(value) for value in crop_observations["target_greenhouse_ids"]]
    target_codes = [str(value) for value in crop_observations["target_greenhouse_codes"]]
    target_code = str(target_site["greenhouse_code"])
    target_area_m2 = float(target_site["greenhouse_area_m2"])
    if int(target_site["greenhouse_id"]) not in target_ids or target_code not in target_codes:
        raise ValueError("dataset and harvest target greenhouse identities do not agree")

    events = load_harvest_events(
        workbook_path,
        target_greenhouse_ids=target_ids,
        target_greenhouse_codes=target_codes,
        normalization_areas_m2={target_code: target_area_m2},
        source_sites=crop_observations.get("source_sites", {}),
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
    )
    seasons = load_harvest_seasons(
        workbook_path,
        target_greenhouse_ids=target_ids,
        target_greenhouse_codes=target_codes,
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
    )
    bound_events, binding_audit = bind_harvest_events_to_seasons(events, seasons)
    measured = build_measured_harvest_artifacts(bound_events)

    warnings: list[str] = []
    try:
        protocols = load_harvest_protocols(
            workbook_path,
            target_greenhouse_ids=target_ids,
            target_greenhouse_codes=target_codes,
            timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
        )
    except ValueError as exc:
        if "HarvestProtocol sheet" not in str(exc):
            raise
        protocols = pd.DataFrame(columns=HARVEST_PROTOCOL_COLUMNS)
        warnings.append("management_protocol_sheet_missing")
    event_ends = bound_events.groupby("season_id")["timestamp"].max()
    protocol_horizons = seasons[seasons["season_id"].isin(event_ends.index)][
        ["season_id", "greenhouse_id", "greenhouse_code", "planting_code", "season_start"]
    ].copy()
    protocol_horizons["horizon_start"] = protocol_horizons.pop("season_start")
    protocol_horizons["horizon_end"] = protocol_horizons["season_id"].map(event_ends)
    protocol_coverage = assess_harvest_protocol_coverage(
        protocols,
        protocol_horizons,
        timezone=dataset.get("source", {}).get("timezone", "Asia/Shanghai"),
    )
    for row in protocol_coverage["season_readiness"]:
        for reason in row["blocking_reasons"]:
            warnings.append(f"{reason}:{row['season_id']}")
    qualifying: list[str] = []
    for summary in measured["season_summary"]:
        season_id = str(summary["season_id"])
        event_count = int(summary["physical_event_count"])
        if int(summary["batch_id_recorded_count"]) < event_count:
            warnings.append(
                f"batch_id_missing:{season_id}:"
                f"{event_count - int(summary['batch_id_recorded_count'])}"
            )
        if int(summary["source_record_id_recorded_count"]) < event_count:
            warnings.append(
                f"source_record_id_missing:{season_id}:"
                f"{event_count - int(summary['source_record_id_recorded_count'])}"
            )
        if not bool(summary["season_complete_evidence"]):
            warnings.append(f"ongoing_season_excluded:{season_id}")
            continue
        if event_count < minimum_events or float(summary["event_span_days"]) < minimum_span_days:
            warnings.append(f"complete_season_insufficient_coverage:{season_id}")
            continue
        qualifying.append(season_id)

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    event_path = output / "measured_harvest_event_trajectory.csv"
    daily_path = output / "measured_daily_harvest_trajectory.csv"
    report_path = output / "harvest_workbook_preflight.json"
    measured["event_trajectory"].to_csv(event_path, index=False, encoding="utf-8-sig")
    measured["daily_trajectory"].to_csv(daily_path, index=False, encoding="utf-8-sig")

    report: dict[str, Any] = {
        "scope": "measured_harvest_coverage_only",
        "workbook_path": str(Path(workbook_path).resolve()),
        "target_greenhouse_ids": target_ids,
        "target_greenhouse_codes": target_codes,
        "target_normalization_area_m2": target_area_m2,
        "target_harvest_event_count": int(len(bound_events)),
        "season_count": int(binding_audit["season_count"]),
        "complete_seasons": list(binding_audit["complete_seasons"]),
        "ongoing_seasons": list(binding_audit["ongoing_seasons"]),
        "minimum_events_per_complete_season": int(minimum_events),
        "minimum_event_span_days": float(minimum_span_days),
        "qualifying_complete_seasons": qualifying,
        "measurement_calibration_coverage_ready": len(qualifying) >= 1,
        "measurement_independent_validation_coverage_ready": len(qualifying) >= 2,
        "management_protocol_coverage_ready": protocol_coverage[
            "all_horizons_confirmed_and_covered"
        ],
        "management_protocol_coverage": protocol_coverage,
        "crop_driver_model_accepted": False,
        "harvest_parameters_fitted": False,
        "target_harvest_validated": False,
        "batch_semantics": measured["batch_semantics"],
        "season_summary": measured["season_summary"],
        "warnings": warnings,
        "artifacts": {
            "event_trajectory_csv": str(event_path.resolve()),
            "daily_trajectory_csv": str(daily_path.resolve()),
            "preflight_report_json": str(report_path.resolve()),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight Chengdu measured tomato harvest workbook coverage."
    )
    parser.add_argument("--harvest-workbook", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--harvest-config", type=Path, default=DEFAULT_HARVEST_CONFIG)
    parser.add_argument("--minimum-events", type=int, default=8)
    parser.add_argument("--minimum-span-days", type=float, default=30.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_preflight(
        workbook_path=args.harvest_workbook,
        output_root=args.output_root,
        dataset_config_path=args.dataset_config,
        harvest_config_path=args.harvest_config,
        minimum_events=args.minimum_events,
        minimum_span_days=args.minimum_span_days,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
