from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from processing.build_chengdu_trajectory_v2 import _build_valid_transitions, _validate_source
from processing.build_chengdu_trajectory_v3 import (
    INDOOR_EQUIPMENT_IDS,
    PHYSICAL_DEVICE_COLUMNS,
)
from processing.chengdu_sensor_quality import (
    derive_hourly_variable_quality,
    discover_sensor_channels,
)
from processing.chengdu_trajectory import split_trajectory_dataframe


SCHEMA_VERSION = "chengdu_trajectory_v4_quality_aware"
QUALITY_POLICY = {
    "minimum_observed_fraction": 0.8,
    "temperature_high_disagreement_c": 1.0,
    "temperature_medium_disagreement_c": 3.0,
    "humidity_high_disagreement_pct": 6.0,
    "humidity_medium_disagreement_pct": 10.0,
    "multi_sensor_weight": 1.0,
    "single_sensor_weight": 0.6,
    "low_quality_weight": 0.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _heat_regime(temperature: pd.Series) -> pd.Series:
    values = pd.to_numeric(temperature, errors="raise")
    return pd.Series(
        np.select(
            [values >= 40.0, values >= 35.0],
            ["extreme", "high"],
            default="normal",
        ),
        index=temperature.index,
        dtype=object,
    )


def _quality_counts(frame: pd.DataFrame) -> dict[str, int]:
    weight = pd.to_numeric(frame["transition_quality_weight"], errors="raise")
    return {
        "rows": int(len(frame)),
        "high_weight": int(weight.eq(1.0).sum()),
        "medium_weight": int(weight.gt(0.0).sum() - weight.eq(1.0).sum()),
        "low_weight": int(weight.eq(0.0).sum()),
        "normal": int(frame["target_heat_regime"].eq("normal").sum()),
        "high": int(frame["target_heat_regime"].eq("high").sum()),
        "extreme": int(frame["target_heat_regime"].eq("extreme").sum()),
        "extreme_with_physical_support": int(
            (
                frame["target_heat_regime"].eq("extreme")
                & frame["extreme_heat_physical_support"].astype(bool)
            ).sum()
        ),
        "identification_eligible": int(frame["identification_eligible"].astype(bool).sum()),
    }


def build_trajectory_v4_bundle(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    source = pd.read_csv(source_path)
    missing = [column for column in PHYSICAL_DEVICE_COLUMNS if column not in source]
    if missing:
        raise ValueError(f"Missing V4 physical device columns: {', '.join(missing)}")
    _validate_source(source)
    trajectory, exclusions = _build_valid_transitions(source)

    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    temperature_quality = derive_hourly_variable_quality(
        source,
        variable="air_temperature",
        indoor_equipment_ids=INDOOR_EQUIPMENT_IDS,
        coverage_threshold=QUALITY_POLICY["minimum_observed_fraction"],
        high_disagreement=QUALITY_POLICY["temperature_high_disagreement_c"],
        medium_disagreement=QUALITY_POLICY["temperature_medium_disagreement_c"],
    )
    humidity_quality = derive_hourly_variable_quality(
        source,
        variable="relative_humidity",
        indoor_equipment_ids=INDOOR_EQUIPMENT_IDS,
        coverage_threshold=QUALITY_POLICY["minimum_observed_fraction"],
        high_disagreement=QUALITY_POLICY["humidity_high_disagreement_pct"],
        medium_disagreement=QUALITY_POLICY["humidity_medium_disagreement_pct"],
    ).drop(columns="timestamp")
    quality = pd.concat(
        [temperature_quality.reset_index(drop=True), humidity_quality.reset_index(drop=True)],
        axis=1,
    )

    physical = source[["timestamp", *PHYSICAL_DEVICE_COLUMNS]].copy()
    physical = physical.rename(columns={"roof_window": "uRoofVent", "fan": "uFan"})
    current_metadata = physical.merge(quality, on="timestamp", how="left", validate="one_to_one")
    next_quality = quality.rename(
        columns={
            column: "next_timestamp" if column == "timestamp" else f"next_{column}"
            for column in quality.columns
        }
    )
    heat = pd.DataFrame(
        {
            "next_timestamp": source["timestamp"],
            "target_heat_regime": _heat_regime(source["air_temperature"]),
            "extreme_heat_physical_support": (
                pd.to_numeric(source["air_temperature"], errors="coerce").ge(40.0)
                & pd.to_numeric(source["outdoor_air_temperature"], errors="coerce").ge(30.0)
                & pd.to_numeric(source["global_radiation"], errors="coerce").ge(500.0)
            ),
        }
    )

    trajectory["timestamp"] = pd.to_datetime(trajectory["timestamp"])
    trajectory["next_timestamp"] = pd.to_datetime(trajectory["next_timestamp"])
    trajectory = trajectory.merge(current_metadata, on="timestamp", how="left", validate="one_to_one")
    trajectory = trajectory.merge(next_quality, on="next_timestamp", how="left", validate="one_to_one")
    trajectory = trajectory.merge(heat, on="next_timestamp", how="left", validate="one_to_one")
    weight_columns = [
        "air_temperature_quality_weight",
        "relative_humidity_quality_weight",
        "next_air_temperature_quality_weight",
        "next_relative_humidity_quality_weight",
    ]
    trajectory["transition_quality_weight"] = trajectory[weight_columns].min(axis=1)
    trajectory["identification_eligible"] = (
        trajectory["transition_quality_weight"].gt(0.0)
        & trajectory.get("roof_window_known_fraction", 1.0).ge(0.95)
        & trajectory.get("fan_known_fraction", 1.0).ge(0.95)
        & trajectory.get("roof_window_uncertain_fraction", 0.0).le(0.05)
        & trajectory.get("fan_uncertain_fraction", 0.0).le(0.05)
    )

    splits = split_trajectory_dataframe(
        trajectory,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
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

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path.as_posix(),
        "source_sha256": _sha256(source_path),
        "source_rows": int(len(source)),
        "valid_transitions": int(len(trajectory)),
        "excluded_transitions": exclusions,
        "split_policy": {
            "kind": "chronological",
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
        },
        "split_rows": {name: int(len(split)) for name, split in splits.items()},
        "split_ranges": split_ranges,
        "split_sha256": split_hashes,
        "quality_policy": QUALITY_POLICY,
        "quality_counts": {
            "all": _quality_counts(trajectory),
            **{name: _quality_counts(split) for name, split in splits.items()},
        },
        "temperature_sensor_channels": discover_sensor_channels(
            source,
            variable="air_temperature",
            indoor_equipment_ids=INDOOR_EQUIPMENT_IDS,
        ),
        "humidity_sensor_channels": discover_sensor_channels(
            source,
            variable="relative_humidity",
            indoor_equipment_ids=INDOOR_EQUIPMENT_IDS,
        ),
        "extreme_heat_policy": "retained_and_labeled_not_filtered",
        "canonical_observation_policy": "unchanged_from_aligned_source",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build quality-aware Chengdu trajectory V4.")
    parser.add_argument(
        "--input",
        default="data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v4_quality_aware",
    )
    args = parser.parse_args()
    print(json.dumps(build_trajectory_v4_bundle(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
