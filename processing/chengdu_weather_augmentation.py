from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from processing.chengdu_weather import OUTPUT_COLUMNS

JOIN_BLEND_HOURS = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source(source: pd.DataFrame) -> None:
    missing = sorted(set(OUTPUT_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError(f"Missing GreenLight weather columns: {missing}")
    numeric = source[OUTPUT_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Source weather contains non-finite values")
    if len(source) < 24:
        raise ValueError("Source weather must contain at least one complete day")
    if not np.allclose(np.diff(source["time"].to_numpy(dtype=float)), 3600.0):
        raise ValueError("Source weather must be continuous hourly data")


def _smooth_daily_values(values: np.ndarray, hours: int) -> np.ndarray:
    centers = np.arange(len(values), dtype=float) * 24.0 + 12.0
    return np.interp(np.arange(hours, dtype=float), centers, values)


def _blend_block_joins(
    weather: pd.DataFrame,
    block_hours: int,
    blend_hours: int = JOIN_BLEND_HOURS,
) -> None:
    columns = [
        "global radiation",
        "wind speed",
        "air temperature",
        "sky temperature",
        "RH",
    ]
    for boundary in range(block_hours, len(weather), block_hours):
        count = min(blend_hours, len(weather) - boundary)
        if count < 1:
            continue
        previous = weather.loc[boundary - 1, columns].to_numpy(dtype=float)
        target_index = min(boundary + count, len(weather) - 1)
        target = weather.loc[target_index, columns].to_numpy(dtype=float)
        weights = np.arange(1, count + 1, dtype=float)[:, None] / (count + 1)
        weather.loc[boundary:boundary + count - 1, columns] = (
            previous + weights * (target - previous)
        )


def generate_synthetic_weather(
    source_weather: pd.DataFrame,
    *,
    seed: int,
    days: int = 110,
    block_days: int = 3,
    source_start_day: int = 0,
    source_days: int | None = None,
    perturb: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate auditable weather with multivariate whole-day block bootstrap."""
    _validate_source(source_weather)
    if days < 1 or block_days < 1:
        raise ValueError("days and block_days must be positive")
    complete_days = len(source_weather) // 24
    if source_start_day < 0 or source_start_day >= complete_days:
        raise ValueError("source_start_day is outside the complete source days")
    selected_days = complete_days - source_start_day if source_days is None else source_days
    if selected_days < 1 or source_start_day + selected_days > complete_days:
        raise ValueError("source_days is outside the complete source days")
    if block_days > selected_days:
        raise ValueError("block_days exceeds complete source days")
    source = source_weather.iloc[
        source_start_day * 24:(source_start_day + selected_days) * 24
    ].reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    blocks: list[pd.DataFrame] = []
    starts: list[int] = []
    collected_days = 0
    while collected_days < days:
        start = int(rng.integers(0, selected_days - block_days + 1))
        starts.append(start)
        blocks.append(source.iloc[start * 24:(start + block_days) * 24].copy())
        collected_days += block_days
    generated = pd.concat(blocks, ignore_index=True).iloc[: days * 24].copy()
    blend_hours = JOIN_BLEND_HOURS
    _blend_block_joins(generated, block_days * 24, blend_hours)

    perturbation = {
        "temperature_offset_limit_deg_c": 2.0,
        "humidity_offset_limit_percent": 5.0,
        "radiation_scale_bounds": [0.75, 1.25],
        "wind_scale_bounds": [0.7, 1.3],
    }
    if perturb:
        temp_daily = np.clip(rng.normal(0.0, 0.8, days), -2.0, 2.0)
        rh_daily = np.clip(-1.5 * temp_daily + rng.normal(0.0, 1.5, days), -5.0, 5.0)
        radiation_daily = np.clip(rng.lognormal(0.0, 0.08, days), 0.75, 1.25)
        wind_daily = np.clip(rng.lognormal(0.0, 0.12, days), 0.7, 1.3)
        hours = len(generated)
        temp_offset = _smooth_daily_values(temp_daily, hours)
        rh_offset = _smooth_daily_values(rh_daily, hours)
        radiation_scale = _smooth_daily_values(radiation_daily, hours)
        wind_scale = _smooth_daily_values(wind_daily, hours)
        generated["air temperature"] = np.clip(
            generated["air temperature"].to_numpy(dtype=float) + temp_offset,
            -30.0,
            60.0,
        )
        generated["sky temperature"] = np.clip(
            generated["sky temperature"].to_numpy(dtype=float) + temp_offset,
            -40.0,
            60.0,
        )
        generated["RH"] = np.clip(
            generated["RH"].to_numpy(dtype=float) + rh_offset,
            0.0,
            100.0,
        )
        generated["global radiation"] = np.maximum(
            0.0,
            generated["global radiation"].to_numpy(dtype=float) * radiation_scale,
        )
        generated["wind speed"] = np.maximum(
            0.0,
            generated["wind speed"].to_numpy(dtype=float) * wind_scale,
        )

    calendar_start = float(source_weather["time"].iloc[source_start_day * 24])
    generated["time"] = calendar_start + np.arange(len(generated), dtype=float) * 3600.0
    generated["day number"] = 1 + (generated["time"] // 86400.0).astype(int)
    generated = generated[OUTPUT_COLUMNS].reset_index(drop=True)
    _validate_source(generated)
    metadata = {
        "method": "multivariate_moving_block_bootstrap",
        "seed": int(seed),
        "days": int(days),
        "block_days": int(block_days),
        "join_blend_hours": int(blend_hours),
        "source_complete_days": int(complete_days),
        "source_start_day": int(source_start_day),
        "source_days": int(selected_days),
        "source_block_start_days": starts,
        "source_block_absolute_start_days": [source_start_day + start for start in starts],
        "perturb": bool(perturb),
        "perturbation": perturbation,
    }
    return generated, metadata


def write_synthetic_weather_ensemble(
    source_path: str | Path,
    *,
    output_root: str | Path,
    location: str,
    years: Iterable[int],
    seeds: Iterable[int],
    days: int = 110,
    block_days: int = 3,
    source_start_day: int = 0,
    source_days: int | None = None,
) -> Path:
    source_file = Path(source_path)
    output_root_path = Path(output_root)
    year_values = tuple(int(year) for year in years)
    seed_values = tuple(int(seed) for seed in seeds)
    if len(year_values) != len(seed_values) or not year_values:
        raise ValueError("years and seeds must have the same non-zero length")
    if len(set(year_values)) != len(year_values):
        raise ValueError("Synthetic weather years must be unique")
    source = pd.read_csv(source_file)
    location_dir = output_root_path / location
    location_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for year, seed in zip(year_values, seed_values):
        generated, metadata = generate_synthetic_weather(
            source,
            seed=seed,
            days=days,
            block_days=block_days,
            source_start_day=source_start_day,
            source_days=source_days,
        )
        output_path = location_dir / f"{year}.csv"
        generated.to_csv(output_path, index=False)
        outputs.append({
            "year": year,
            "seed": seed,
            "path": output_path.relative_to(output_root_path).as_posix(),
            "sha256": _sha256(output_path),
            "source_block_start_days": metadata["source_block_start_days"],
            "source_block_absolute_start_days": metadata["source_block_absolute_start_days"],
        })
    manifest = {
        "method": "multivariate_moving_block_bootstrap",
        "source_path": str(source_file),
        "source_sha256": _sha256(source_file),
        "location": str(location),
        "days": int(days),
        "block_days": int(block_days),
        "join_blend_hours": int(JOIN_BLEND_HOURS),
        "source_start_day": int(source_start_day),
        "source_days": int(
            len(source) // 24 - source_start_day if source_days is None else source_days
        ),
        "perturbation": generate_synthetic_weather(
            source,
            seed=seed_values[0],
            days=days,
            block_days=block_days,
            source_start_day=source_start_day,
            source_days=source_days,
        )[1]["perturbation"],
        "outputs": outputs,
    }
    manifest_path = output_root_path / "synthetic_weather_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest_path


def _parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate block-bootstrap Chengdu weather years")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--location", default="Chengdu")
    parser.add_argument("--years", default="3000,3001,3002,3003,3004,3005,3006,3007,3008,3009")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--days", type=int, default=110)
    parser.add_argument("--block_days", type=int, default=3)
    parser.add_argument("--source_start_day", type=int, default=0)
    parser.add_argument("--source_days", type=int)
    args = parser.parse_args()
    manifest = write_synthetic_weather_ensemble(
        args.source,
        output_root=args.output_root,
        location=args.location,
        years=_parse_ints(args.years),
        seeds=_parse_ints(args.seeds),
        days=args.days,
        block_days=args.block_days,
        source_start_day=args.source_start_day,
        source_days=args.source_days,
    )
    print(f"Saved synthetic weather manifest to {manifest}")


if __name__ == "__main__":
    main()
