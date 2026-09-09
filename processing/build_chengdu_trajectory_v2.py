from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from processing.chengdu_trajectory import CONTROL_COLUMNS, split_trajectory_dataframe


SCHEMA_VERSION = "chengdu_trajectory_v2"
REQUIRED_CURRENT_COLUMNS = [
    "air_temperature",
    "relative_humidity",
    "outdoor_air_temperature",
    "outdoor_relative_humidity",
    "global_radiation",
    "wind_speed",
    *CONTROL_COLUMNS,
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(frame: pd.DataFrame, name: str, default: float | pd.Series) -> pd.Series:
    if name not in frame:
        if isinstance(default, pd.Series):
            return default.astype(float)
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).astype(float)


def _validate_source(frame: pd.DataFrame) -> None:
    expected = ["timestamp", *REQUIRED_CURRENT_COLUMNS]
    missing = [name for name in expected if name not in frame]
    if missing:
        raise ValueError(f"Missing V2 trajectory source columns: {', '.join(missing)}")

    finite = frame[["air_temperature", "relative_humidity", "outdoor_air_temperature", "outdoor_relative_humidity"]]
    finite = finite.dropna()
    if not finite.empty:
        temp_identical = np.allclose(finite["air_temperature"], finite["outdoor_air_temperature"])
        rh_identical = np.allclose(finite["relative_humidity"], finite["outdoor_relative_humidity"])
        if temp_identical and rh_identical:
            raise ValueError("Outdoor temperature and humidity are identical to indoor series")


def _build_valid_transitions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.sort_values("timestamp").reset_index(drop=True)
    current = data.iloc[:-1].reset_index(drop=True)
    following = data.iloc[1:].reset_index(drop=True)

    missing_measurement = current[REQUIRED_CURRENT_COLUMNS].isna().any(axis=1)
    missing_measurement |= following[["air_temperature", "relative_humidity"]].isna().any(axis=1)
    one_hour = (following["timestamp"] - current["timestamp"]) == pd.Timedelta(hours=1)
    invalid_timestamp_or_gap = ~one_hour & ~missing_measurement
    valid = ~missing_measurement & one_hour

    cur = current.loc[valid].reset_index(drop=True)
    nxt = following.loc[valid].reset_index(drop=True)
    indoor_temp = _numeric(cur, "air_temperature", 20.0)
    indoor_rh = _numeric(cur, "relative_humidity", 70.0)
    co2 = _numeric(cur, "co2_concentration", 650.0)
    illumination = _numeric(cur, "illumination", 0.0)
    soil_temp = _numeric(cur, "soil_temperature", indoor_temp)

    trajectory = pd.DataFrame(
        {
            "timestamp": cur["timestamp"],
            "next_timestamp": nxt["timestamp"],
            "x_air_temperature": indoor_temp,
            "x_relative_humidity": indoor_rh,
            "x_co2_concentration": co2,
            "x_illumination": illumination,
            "x_soil_temperature": soil_temp,
            "x_soil_humidity": _numeric(cur, "soil_humidity", 0.0),
            **{name: _numeric(cur, name, 0.0) for name in CONTROL_COLUMNS},
            "d_global_radiation": _numeric(cur, "global_radiation", 0.0),
            "d_wind_speed": _numeric(cur, "wind_speed", 0.0),
            "d_air_pressure": _numeric(cur, "air_pressure", 0.0),
            "d_air_temperature": _numeric(cur, "outdoor_air_temperature", 20.0),
            "d_relative_humidity": _numeric(cur, "outdoor_relative_humidity", 70.0),
            "next_x_air_temperature": _numeric(nxt, "air_temperature", 20.0),
            "next_x_relative_humidity": _numeric(nxt, "relative_humidity", 70.0),
            "next_x_co2_concentration": _numeric(nxt, "co2_concentration", co2),
            "next_x_illumination": _numeric(nxt, "illumination", illumination),
        }
    )

    quality_columns = [
        name for name in cur.columns if name.endswith("_known_fraction") or name.endswith("_uncertain_fraction")
    ]
    for name in quality_columns:
        trajectory[name] = _numeric(cur, name, 0.0)

    exclusions = {
        "missing_required_measurement": int(missing_measurement.sum()),
        "non_hourly_or_invalid_timestamp": int(invalid_timestamp_or_gap.sum()),
    }
    return trajectory, exclusions


def build_trajectory_v2_bundle(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict:
    source = Path(source_path)
    output = Path(output_dir)
    frame = pd.read_csv(source)
    _validate_source(frame)
    trajectory, exclusions = _build_valid_transitions(frame)
    splits = split_trajectory_dataframe(trajectory, train_fraction=train_fraction, val_fraction=val_fraction)

    output.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    split_ranges: dict[str, dict[str, str | None]] = {}
    for name, split in splits.items():
        path = output / f"{name}.csv"
        split.to_csv(path, index=False)
        split_hashes[name] = _sha256(path)
        split_ranges[name] = {
            "start": None if split.empty else str(split.iloc[0]["timestamp"]),
            "end": None if split.empty else str(split.iloc[-1]["next_timestamp"]),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_path": source.as_posix(),
        "source_sha256": _sha256(source),
        "source_rows": int(len(frame)),
        "candidate_transitions": max(0, int(len(frame) - 1)),
        "excluded_transitions": exclusions,
        "valid_transitions": int(len(trajectory)),
        "split_policy": {"kind": "chronological", "train_fraction": train_fraction, "val_fraction": val_fraction},
        "split_rows": {name: int(len(split)) for name, split in splits.items()},
        "split_ranges": split_ranges,
        "split_sha256": split_hashes,
        "outdoor_disturbance_policy": "measured_required_no_indoor_fallback",
        "required_current_columns": REQUIRED_CURRENT_COLUMNS,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audited Chengdu measured-disturbance trajectory V2 artifacts.")
    parser.add_argument(
        "--input",
        default="data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v2",
    )
    args = parser.parse_args()
    print(json.dumps(build_trajectory_v2_bundle(args.input, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
