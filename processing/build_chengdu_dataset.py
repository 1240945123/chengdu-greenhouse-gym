from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd
import yaml

from processing.chengdu_control_states import (
    DEVICE_COLUMNS,
    derive_hourly_controls,
    reconstruct_minute_controls,
)
from processing.chengdu_sensor_stream import build_minute_sensor_table, read_sensor_files
from processing.chengdu_sql_controls import extract_control_events


def build_dataset(
    sql_path: str | Path,
    sensor_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, object],
) -> dict[str, Path]:
    """Build canonical Chengdu minute/hour datasets and an audit report."""
    source = _mapping(config, "source")
    processing = _mapping(config, "processing")
    ranges = _mapping(config, "sensor_ranges")
    greenhouse_id = int(source.get("greenhouse_id", 63))
    chunksize = int(processing.get("chunksize", 250_000))
    interpolation_limit = int(processing.get("interpolation_limit_minutes", 5))

    events = extract_control_events(sql_path)
    clean_sensor, sensor_read_quality = read_sensor_files(
        sensor_root,
        greenhouse_id=greenhouse_id,
        ranges=ranges,
        chunksize=chunksize,
    )
    minute_sensor, sensor_minute_quality = build_minute_sensor_table(
        clean_sensor,
        interpolation_limit=interpolation_limit,
    )
    if minute_sensor.empty:
        raise ValueError("No target greenhouse sensor minutes were produced")

    minute_index = pd.DatetimeIndex(pd.to_datetime(minute_sensor["timestamp"]))
    minute_controls = reconstruct_minute_controls(events, minute_index)
    hourly_controls = derive_hourly_controls(minute_controls)
    hourly_sensor = derive_hourly_dataset(minute_sensor)

    aligned_minute = minute_sensor.merge(
        minute_controls,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    aligned_hourly = hourly_sensor.merge(
        hourly_controls,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    root = Path(output_root)
    control_dir = root / "controls"
    aligned_dir = root / "aligned"
    report_dir = root / "reports"
    for directory in [control_dir, aligned_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    outputs = {
        "control_events": control_dir / "control_events.csv",
        "controls_1min": control_dir / "controls_1min.csv",
        "controls_1h": control_dir / "controls_1h.csv",
        "aligned_1min": aligned_dir / "greenhouse_1min.csv",
        "aligned_1h": aligned_dir / "greenhouse_1h.csv",
        "quality_report": report_dir / "alignment_quality.json",
    }
    events.to_csv(outputs["control_events"], index=False)
    minute_controls.to_csv(outputs["controls_1min"], index=False)
    hourly_controls.to_csv(outputs["controls_1h"], index=False)
    aligned_minute.to_csv(outputs["aligned_1min"], index=False)
    aligned_hourly.to_csv(outputs["aligned_1h"], index=False)

    report = _quality_report(
        events=events,
        minute_controls=minute_controls,
        sensor_read_quality=sensor_read_quality,
        sensor_minute_quality=sensor_minute_quality,
        aligned_minute=aligned_minute,
        aligned_hourly=aligned_hourly,
        greenhouse_id=greenhouse_id,
        ranges=ranges,
        interpolation_limit=interpolation_limit,
    )
    outputs["quality_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return outputs


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Dataset config section {key!r} must be a mapping")
    return value


def derive_hourly_dataset(minute_sensor: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sensor values and observation flags on an hourly grid."""
    if "timestamp" not in minute_sensor.columns:
        raise ValueError("Minute sensor table is missing timestamp")
    minute = minute_sensor.copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"], errors="coerce")
    if minute["timestamp"].isna().any():
        raise ValueError("Minute sensor table contains invalid timestamps")
    if minute["timestamp"].duplicated().any():
        raise ValueError("Minute sensor timestamps must be unique")
    value_columns = [
        column
        for column in minute.columns
        if column != "timestamp"
        and not column.endswith("_observed")
        and not column.endswith("_interpolated")
    ]
    flag_columns = [
        column
        for column in minute.columns
        if column.endswith("_observed") or column.endswith("_interpolated")
    ]
    hour_key = minute["timestamp"].dt.floor("h").rename("timestamp")
    hourly = minute[[*value_columns, *flag_columns]].groupby(hour_key, sort=True).mean().reset_index()
    hourly = hourly.rename(columns={column: f"{column}_fraction" for column in flag_columns})
    return hourly


def _quality_report(
    events: pd.DataFrame,
    minute_controls: pd.DataFrame,
    sensor_read_quality: Mapping[str, object],
    sensor_minute_quality: Mapping[str, object],
    aligned_minute: pd.DataFrame,
    aligned_hourly: pd.DataFrame,
    greenhouse_id: int,
    ranges: Mapping[str, object],
    interpolation_limit: int,
) -> dict[str, object]:
    ordered = events.sort_values(["controller_id", "timestamp", "log_id"], kind="stable")
    repeated = ordered["action"].eq(ordered.groupby("controller_id")["action"].shift())
    event_minutes = ordered.assign(event_minute=ordered["timestamp"].dt.floor("min"))
    conflicts = (
        event_minutes.groupby(["controller_id", "event_minute"])["action"].nunique().gt(1).sum()
    )
    by_device: dict[str, dict[str, int]] = {}
    device_coverage: dict[str, dict[str, object]] = {}
    for controller_id, group in ordered.groupby("controller_id", sort=True):
        by_device[str(int(controller_id))] = {
            str(action): int(count) for action, count in group["action"].value_counts().items()
        }
    for controller_id, device in zip(range(43, 52), DEVICE_COLUMNS):
        group = ordered.loc[ordered["controller_id"].eq(controller_id)]
        device_coverage[str(controller_id)] = {
            "first": None if group.empty else group["timestamp"].min().isoformat(),
            "last": None if group.empty else group["timestamp"].max().isoformat(),
            "unknown_fraction": float((~minute_controls[f"{device}_known"]).mean()),
            "uncertain_fraction": float(minute_controls[f"{device}_uncertain"].mean()),
        }

    sensor_start = pd.Timestamp(aligned_minute["timestamp"].min())
    sensor_end = pd.Timestamp(aligned_minute["timestamp"].max())
    event_start = events["timestamp"].min()
    event_end = events["timestamp"].max()
    overlap_start = max(sensor_start, event_start)
    overlap_end = min(sensor_end, event_end)
    variable_quality = sensor_minute_quality.get("variables", {})
    interpolated_count = sum(
        int(metrics.get("interpolated", 0))
        for metrics in variable_quality.values()
        if isinstance(metrics, Mapping)
    )
    possible_sensor_values = int(sensor_minute_quality.get("rows", 0)) * len(variable_quality)
    overall_interpolated_fraction = (
        float(interpolated_count / possible_sensor_values) if possible_sensor_values else 0.0
    )
    key_columns = [
        column for column in ["air_temperature", "relative_humidity"] if column in aligned_minute
    ]
    complete_air_temp_rh_fraction = (
        float(aligned_minute[key_columns].notna().all(axis=1).mean())
        if len(key_columns) == 2
        else 0.0
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "greenhouse_id": greenhouse_id,
        "processing": {
            "interpolation_limit_minutes": interpolation_limit,
            "sensor_ranges": ranges,
        },
        "controls": {
            "target_events": int(len(events)),
            "by_device": by_device,
            "exact_duplicates": int(events.duplicated().sum()),
            "repeated_commands": int(repeated.sum()),
            "same_minute_conflicts": int(conflicts),
            "device_coverage": device_coverage,
        },
        "sensors": {
            "read": dict(sensor_read_quality),
            "minute": dict(sensor_minute_quality),
            "overall_interpolated_fraction": overall_interpolated_fraction,
        },
        "overlap": {
            "sensor_start": sensor_start.isoformat(),
            "sensor_end": sensor_end.isoformat(),
            "control_start": event_start.isoformat(),
            "control_end": event_end.isoformat(),
            "start": overlap_start.isoformat(),
            "end": overlap_end.isoformat(),
        },
        "outputs": {
            "aligned_1min_rows": int(len(aligned_minute)),
            "aligned_1h_rows": int(len(aligned_hourly)),
            "complete_air_temp_rh_fraction": complete_air_temp_rh_fraction,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned Chengdu greenhouse datasets.")
    parser.add_argument("--sql", required=True, help="External MySQL backup path")
    parser.add_argument("--sensor_root", required=True, help="Root containing daily sensor CSV files")
    parser.add_argument("--dataset_config", required=True, help="Dataset YAML configuration")
    parser.add_argument("--output_root", required=True, help="Processed output root")
    args = parser.parse_args()

    with Path(args.dataset_config).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    outputs = build_dataset(args.sql, args.sensor_root, args.output_root, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
