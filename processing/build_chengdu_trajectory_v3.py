from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from processing.build_chengdu_trajectory_v2 import _build_valid_transitions, _validate_source
from processing.chengdu_trajectory import split_trajectory_dataframe


SCHEMA_VERSION = "chengdu_trajectory_v3"
PHYSICAL_DEVICE_COLUMNS = [
    "roof_window",
    "fan",
    "wet_pad_pump",
    "wet_pad_roll_film",
    "roof_thermal_screen",
    "side_thermal_screen",
    "external_shade",
    "uPad",
]
SENSOR_PATTERN = re.compile(r"^air_temperature__e\d+__s\d+$")
INDOOR_EQUIPMENT_IDS = (3038, 3044)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _indoor_temperature_channels(source: pd.DataFrame) -> list[str]:
    return [
        column
        for column in source.columns
        if SENSOR_PATTERN.match(column)
        and any(f"__e{equipment_id}__" in column for equipment_id in INDOOR_EQUIPMENT_IDS)
    ]


def _sensor_quality(source: pd.DataFrame) -> pd.DataFrame:
    channels = _indoor_temperature_channels(source)
    if not channels:
        raise ValueError("V3 source has no channel-level indoor temperature measurements")
    values = source[channels].apply(pd.to_numeric, errors="coerce")
    count = values.notna().sum(axis=1)
    disagreement = values.max(axis=1, skipna=True) - values.min(axis=1, skipna=True)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(source["timestamp"]),
            "indoor_temperature_sensor_count": count.astype(int),
            "indoor_temperature_sensor_disagreement_c": disagreement.fillna(float("inf")),
            "indoor_temperature_single_sensor": count.eq(1),
        }
    )


def build_trajectory_v3_bundle(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    max_sensor_disagreement_c: float = 3.0,
) -> dict:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    source = pd.read_csv(source_path)
    missing = [column for column in PHYSICAL_DEVICE_COLUMNS if column not in source]
    if missing:
        raise ValueError(f"Missing V3 physical device columns: {', '.join(missing)}")
    _validate_source(source)
    trajectory, exclusions = _build_valid_transitions(source)

    physical = source[["timestamp", *PHYSICAL_DEVICE_COLUMNS]].copy()
    physical["timestamp"] = pd.to_datetime(physical["timestamp"])
    physical = physical.rename(columns={"roof_window": "uRoofVent", "fan": "uFan"})
    quality = _sensor_quality(source)
    trajectory["timestamp"] = pd.to_datetime(trajectory["timestamp"])
    trajectory = trajectory.merge(physical, on="timestamp", how="left", validate="one_to_one")
    trajectory = trajectory.merge(quality, on="timestamp", how="left", validate="one_to_one")
    trajectory["identification_eligible"] = (
        trajectory["indoor_temperature_sensor_count"].ge(1)
        & trajectory["indoor_temperature_sensor_disagreement_c"].le(max_sensor_disagreement_c)
        & trajectory.get("roof_window_known_fraction", 1.0).ge(0.95)
        & trajectory.get("fan_known_fraction", 1.0).ge(0.95)
        & trajectory.get("roof_window_uncertain_fraction", 0.0).le(0.05)
        & trajectory.get("fan_uncertain_fraction", 0.0).le(0.05)
    )

    splits = split_trajectory_dataframe(trajectory, train_fraction=train_fraction, val_fraction=val_fraction)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_hashes = {}
    split_ranges = {}
    for name, split in splits.items():
        path = output_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        split_hashes[name] = _sha256(path)
        split_ranges[name] = {
            "start": None if split.empty else str(split.iloc[0]["timestamp"]),
            "end": None if split.empty else str(split.iloc[-1]["next_timestamp"]),
        }

    device_support = {}
    for device in PHYSICAL_DEVICE_COLUMNS:
        values = pd.to_numeric(source[device], errors="coerce")
        device_support[device] = {
            "observed_rows": int(values.notna().sum()),
            "active_rows": int(values.fillna(0.0).gt(0.01).sum()),
            "active_fraction": float(values.fillna(0.0).gt(0.01).mean()),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path.as_posix(),
        "source_sha256": _sha256(source_path),
        "source_rows": int(len(source)),
        "candidate_transitions": max(0, int(len(source) - 1)),
        "excluded_transitions": exclusions,
        "valid_transitions": int(len(trajectory)),
        "split_policy": {"kind": "chronological", "train_fraction": train_fraction, "val_fraction": val_fraction},
        "split_rows": {name: int(len(split)) for name, split in splits.items()},
        "split_ranges": split_ranges,
        "split_sha256": split_hashes,
        "identification_policy": {
            "max_sensor_disagreement_c": float(max_sensor_disagreement_c),
            "minimum_control_known_fraction": 0.95,
            "maximum_control_uncertain_fraction": 0.05,
            "wet_pad_efficiency_identifiable": False,
        },
        "identification_eligible_rows": {
            name: int(split["identification_eligible"].sum()) for name, split in splits.items()
        },
        "temperature_sensor_channels": _indoor_temperature_channels(source),
        "device_support": device_support,
        "outdoor_disturbance_policy": "measured_required_no_indoor_fallback",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited Chengdu physical-control trajectory V3 artifacts.")
    parser.add_argument("--input", default="data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv")
    parser.add_argument("--output_dir", default="data/processed/chengdu_agri/greenhouse_001/trajectories/v3")
    args = parser.parse_args()
    print(json.dumps(build_trajectory_v3_bundle(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
