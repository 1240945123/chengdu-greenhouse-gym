from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from processing.chengdu_trajectory import split_trajectory_dataframe


SCHEMA_VERSION = "chengdu_trajectory_v5_cultivation_window"
SPLIT_NAMES = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_range(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty:
        return {"timestamp_min": None, "next_timestamp_max": None}
    return {
        "timestamp_min": str(pd.to_datetime(frame["timestamp"]).min()),
        "next_timestamp_max": str(pd.to_datetime(frame["next_timestamp"]).max()),
    }


def build_trajectory_v5_bundle(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    cultivation_end_exclusive: str = "2026-07-12 00:00",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict:
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    manifest_path = source_path / "manifest.json"
    required_paths = [manifest_path, *(source_path / f"{name}.csv" for name in SPLIT_NAMES)]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing V4 source files: {', '.join(missing)}")

    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [pd.read_csv(source_path / f"{name}.csv") for name in SPLIT_NAMES]
    source = pd.concat(frames, ignore_index=True)
    required_columns = {"timestamp", "next_timestamp"}
    if not required_columns.issubset(source.columns):
        raise ValueError("V4 trajectory must contain timestamp and next_timestamp")

    timestamps = pd.to_datetime(source["timestamp"], errors="raise")
    next_timestamps = pd.to_datetime(source["next_timestamp"], errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("V4 trajectory contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("V4 trajectory is not chronological")

    cutoff = pd.Timestamp(cultivation_end_exclusive)
    retained = source.loc[next_timestamps < cutoff].reset_index(drop=True)
    if retained.empty:
        raise ValueError("Cultivation cutoff excludes every source transition")

    splits = split_trajectory_dataframe(
        retained,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    split_hashes: dict[str, str] = {}
    for name, frame in splits.items():
        path = output_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        split_hashes[name] = _sha256(path)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": source_manifest.get("schema_version"),
        "source_dir": source_path.resolve().as_posix(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_split_sha256": {
            name: _sha256(source_path / f"{name}.csv") for name in SPLIT_NAMES
        },
        "cultivation_end_exclusive": str(cutoff),
        "cutoff_reason": "tomato_cultivation_ended_and_crop_was_fully_harvested",
        "cutoff_policy": "retain transitions whose next_timestamp is before cultivation_end_exclusive",
        "value_policy": "all retained values and columns are unchanged from V4",
        "source_rows": int(len(source)),
        "retained_rows": int(len(retained)),
        "post_cultivation_excluded_rows": int(len(source) - len(retained)),
        "split_fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": 1.0 - train_fraction - val_fraction,
        },
        "split_rows": {name: int(len(frame)) for name, frame in splits.items()},
        "split_time_ranges": {name: _time_range(frame) for name, frame in splits.items()},
        "split_sha256": split_hashes,
    }
    if "target_heat_regime" in retained.columns:
        manifest["target_heat_regime_rows"] = {
            str(key): int(value)
            for key, value in retained["target_heat_regime"].value_counts().items()
        }

    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a cultivation-window Chengdu trajectory bundle from immutable V4 data."
    )
    parser.add_argument(
        "--source_dir",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v4_quality_aware",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v5_cultivation_window",
    )
    parser.add_argument("--cultivation_end_exclusive", default="2026-07-12 00:00")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    args = parser.parse_args()
    manifest = build_trajectory_v5_bundle(
        args.source_dir,
        args.output_dir,
        cultivation_end_exclusive=args.cultivation_end_exclusive,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
