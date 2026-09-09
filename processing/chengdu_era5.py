from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import sleep
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests

from processing.chengdu_weather import convert_weather_dataframe


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "shortwave_radiation",
    "wind_speed_10m",
    "precipitation",
    "soil_temperature_0_to_7cm",
    "soil_temperature_28_to_100cm",
)
EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "shortwave_radiation": "W/m²",
    "wind_speed_10m": "m/s",
    "precipitation": "mm",
    "soil_temperature_0_to_7cm": "°C",
    "soil_temperature_28_to_100cm": "°C",
}
CANONICAL_NAMES = {
    "temperature_2m": "air_temperature",
    "relative_humidity_2m": "relative_humidity",
    "shortwave_radiation": "global_radiation",
    "wind_speed_10m": "wind_speed",
}
SEASON_DATES = {
    "spring": ("03-01", "06-28"),
    "autumn": ("08-15", "12-12"),
}


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive_request(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    return {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Shanghai",
        "models": "era5",
        "wind_speed_unit": "ms",
    }


def parse_archive_payload(payload: Mapping[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, Mapping) or not isinstance(units, Mapping):
        raise ValueError("ERA5 payload is missing hourly data or units")
    required = ("time", *HOURLY_VARIABLES)
    missing = [name for name in required if name not in hourly]
    if missing:
        raise ValueError(f"ERA5 payload is missing variables: {missing}")
    lengths = {name: len(hourly[name]) for name in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"ERA5 hourly arrays must have equal length: {lengths}")
    for name, expected in EXPECTED_UNITS.items():
        if units.get(name) != expected:
            raise ValueError(
                f"Unexpected ERA5 unit for {name}: {units.get(name)!r}; expected {expected!r}"
            )

    frame = pd.DataFrame({name: hourly[name] for name in required})
    frame = frame.rename(columns={"time": "timestamp", **CANONICAL_NAMES})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    numeric_columns = [name for name in frame.columns if name != "timestamp"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("ERA5 hourly data contains missing or non-numeric values")
    if not frame["timestamp"].is_monotonic_increasing or frame["timestamp"].duplicated().any():
        raise ValueError("ERA5 timestamps must be increasing and unique")
    if len(frame) > 1 and not frame["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("ERA5 timestamps must form a continuous hourly grid")
    if not frame["relative_humidity"].between(0, 100).all():
        raise ValueError("ERA5 relative humidity is outside 0-100%")
    if (frame[["global_radiation", "wind_speed", "precipitation"]] < 0).any().any():
        raise ValueError("ERA5 radiation, wind, and precipitation must be non-negative")
    return frame


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def download_archive_period(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    raw_json_path: str | Path,
    timeout_seconds: float = 60.0,
    retries: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = build_archive_request(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )
    response = None
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < retries:
                sleep(2 ** attempt)
    if response is None:
        raise RuntimeError(f"ERA5 request failed after {retries} attempts") from last_error
    payload = response.json()
    frame = parse_archive_payload(payload)
    path = Path(raw_json_path)
    _write_json_atomic(path, payload)
    metadata = {
        "endpoint": ARCHIVE_URL,
        "request": params,
        "request_url": response.url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(path),
        "raw_sha256": sha256_path(path),
        "api": "Open-Meteo Historical Weather API",
        "model": "ERA5",
    }
    return frame, metadata


def to_greenlight_weather(frame: pd.DataFrame) -> pd.DataFrame:
    precipitation = pd.to_numeric(frame["precipitation"], errors="coerce")
    if precipitation.isna().any() or (precipitation < 0.0).any():
        raise ValueError("ERA5 precipitation must be finite and non-negative")
    canonical = pd.DataFrame({
        "timestamp": pd.to_datetime(frame["timestamp"]),
        "global_radiation": pd.to_numeric(frame["global_radiation"]),
        "wind_speed": pd.to_numeric(frame["wind_speed"]),
        "air_temperature": pd.to_numeric(frame["air_temperature"]),
        "sky_temperature": pd.to_numeric(frame["air_temperature"]) - 6.0,
        "co2_concentration": 400.0,
        "relative_humidity": pd.to_numeric(frame["relative_humidity"]),
    })
    converted = convert_weather_dataframe(canonical)
    converted["precipitation accumulation"] = precipitation.to_numpy(dtype=float)
    return converted


def fit_local_bias_correction(
    era5: pd.DataFrame,
    observed: pd.DataFrame,
) -> dict[str, Any]:
    columns = ["air_temperature", "relative_humidity", "global_radiation", "wind_speed"]
    left = era5[["timestamp", *columns]].copy()
    right = observed[["timestamp", *columns]].copy()
    joined = left.merge(right, on="timestamp", suffixes=("_era5", "_observed"))
    if len(joined) < 24:
        raise ValueError("At least 24 overlapping hourly samples are required for correction")
    temp_diff = joined["air_temperature_observed"] - joined["air_temperature_era5"]
    rh_diff = joined["relative_humidity_observed"] - joined["relative_humidity_era5"]
    daylight = joined["global_radiation_era5"] > 1.0
    radiation_denominator = joined.loc[daylight, "global_radiation_era5"].sum()
    radiation_scale = (
        joined.loc[daylight, "global_radiation_observed"].sum() / radiation_denominator
        if radiation_denominator > 0 else 1.0
    )
    wind_denominator = joined["wind_speed_era5"].median()
    wind_scale = (
        joined["wind_speed_observed"].median() / wind_denominator
        if wind_denominator > 0 else 1.0
    )
    grouped = joined.assign(
        month=joined["timestamp"].dt.month,
        hour=joined["timestamp"].dt.hour,
        temp_diff=temp_diff,
        rh_diff=rh_diff,
    ).groupby(["month", "hour"])
    month_hour: dict[str, dict[str, float]] = {}
    for (month, hour), group in grouped:
        if len(group) >= 3:
            month_hour[f"{month:02d}-{hour:02d}"] = {
                "temperature_offset": float(group["temp_diff"].median()),
                "rh_offset": float(group["rh_diff"].median()),
                "samples": int(len(group)),
            }
    return {
        "method": "robust_overlap_bias_correction",
        "fit_samples": int(len(joined)),
        "fit_start": joined["timestamp"].min().isoformat(),
        "fit_end": joined["timestamp"].max().isoformat(),
        "temperature_offset_global": float(np.clip(temp_diff.median(), -8.0, 8.0)),
        "rh_offset_global": float(np.clip(rh_diff.median(), -20.0, 20.0)),
        "radiation_scale_global": float(np.clip(radiation_scale, 0.5, 2.0)),
        "wind_scale_global": float(np.clip(wind_scale, 0.25, 4.0)),
        "month_hour_offsets": month_hour,
        "unsupported_month_policy": "global_robust_correction",
    }


def apply_local_bias_correction(
    frame: pd.DataFrame,
    correction: Mapping[str, Any],
) -> pd.DataFrame:
    corrected = frame.copy()
    enabled = correction.get("enabled_variables", {})
    timestamps = pd.to_datetime(corrected["timestamp"])
    month_hour = correction.get("month_hour_offsets", {})
    temp_global = float(correction["temperature_offset_global"])
    rh_global = float(correction["rh_offset_global"])
    temp_offsets = []
    rh_offsets = []
    for timestamp in timestamps:
        values = month_hour.get(f"{timestamp.month:02d}-{timestamp.hour:02d}", {})
        temp_offsets.append(float(values.get("temperature_offset", temp_global)))
        rh_offsets.append(float(values.get("rh_offset", rh_global)))
    if enabled.get("air_temperature", True):
        corrected["air_temperature"] = np.clip(
            corrected["air_temperature"].to_numpy(float) + np.asarray(temp_offsets), -30, 60
        )
    if enabled.get("relative_humidity", True):
        corrected["relative_humidity"] = np.clip(
            corrected["relative_humidity"].to_numpy(float) + np.asarray(rh_offsets), 0, 100
        )
    if enabled.get("global_radiation", True):
        corrected["global_radiation"] = np.maximum(
            0.0,
            corrected["global_radiation"].to_numpy(float)
            * float(correction["radiation_scale_global"]),
        )
    if enabled.get("wind_speed", True):
        corrected["wind_speed"] = np.maximum(
            0.0,
            corrected["wind_speed"].to_numpy(float) * float(correction["wind_scale_global"]),
        )
    return corrected


def extract_season(frame: pd.DataFrame, year: int, season: str) -> pd.DataFrame:
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}")
    start_suffix, end_suffix = SEASON_DATES[season]
    start = pd.Timestamp(f"{year}-{start_suffix} 00:00:00")
    end = pd.Timestamp(f"{year}-{end_suffix} 23:00:00")
    timestamps = pd.to_datetime(frame["timestamp"])
    selected = frame.loc[timestamps.between(start, end)].copy().reset_index(drop=True)
    if len(selected) != 120 * 24:
        raise ValueError(f"{year}_{season} must contain 2880 hourly rows, got {len(selected)}")
    if selected["timestamp"].iloc[0] != start or selected["timestamp"].iloc[-1] != end:
        raise ValueError(f"{year}_{season} has incorrect boundaries")
    return selected


def build_six_seasons(years: Mapping[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if set(years) != {2023, 2024, 2025}:
        raise ValueError("Six-season build requires full years 2023, 2024, and 2025")
    return {
        f"{year}_{season}": extract_season(years[year], year, season)
        for year in (2023, 2024, 2025)
        for season in ("spring", "autumn")
    }


def _observed_outdoor_frame(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(source["timestamp"]),
        "air_temperature": pd.to_numeric(source["outdoor_air_temperature"]),
        "relative_humidity": pd.to_numeric(source["outdoor_relative_humidity"]),
        "global_radiation": pd.to_numeric(source["global_radiation"]),
        "wind_speed": pd.to_numeric(source["wind_speed"]),
    })


def correction_validation_metrics(
    era5: pd.DataFrame,
    observed: pd.DataFrame,
    correction: Mapping[str, Any],
) -> dict[str, Any]:
    columns = ["air_temperature", "relative_humidity", "global_radiation", "wind_speed"]
    corrected = apply_local_bias_correction(era5, correction)
    merged_raw = era5[["timestamp", *columns]].merge(
        observed[["timestamp", *columns]], on="timestamp", suffixes=("_pred", "_true")
    )
    merged_corrected = corrected[["timestamp", *columns]].merge(
        observed[["timestamp", *columns]], on="timestamp", suffixes=("_pred", "_true")
    )
    return {
        "samples": int(len(merged_raw)),
        "start": merged_raw["timestamp"].min().isoformat(),
        "end": merged_raw["timestamp"].max().isoformat(),
        "raw_mae": {
            column: float((merged_raw[f"{column}_pred"] - merged_raw[f"{column}_true"]).abs().mean())
            for column in columns
        },
        "corrected_mae": {
            column: float((merged_corrected[f"{column}_pred"] - merged_corrected[f"{column}_true"]).abs().mean())
            for column in columns
        },
    }


def generate_dataset(
    *,
    latitude: float,
    longitude: float,
    external_root: str | Path,
    processed_root: str | Path,
    observed_path: str | Path,
) -> Path:
    external = Path(external_root)
    processed = Path(processed_root)
    raw_dir = external / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    full_years: dict[int, pd.DataFrame] = {}
    requests_metadata: list[dict[str, Any]] = []
    for year in (2023, 2024, 2025):
        frame, metadata = download_archive_period(
            latitude=latitude,
            longitude=longitude,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            raw_json_path=raw_dir / f"era5_{year}.json",
        )
        expected = 8784 if year == 2024 else 8760
        if len(frame) != expected:
            raise ValueError(f"ERA5 year {year} has {len(frame)} rows; expected {expected}")
        full_years[year] = frame
        requests_metadata.append(metadata)
    calibration, calibration_meta = download_archive_period(
        latitude=latitude,
        longitude=longitude,
        start_date="2026-04-01",
        end_date="2026-07-20",
        raw_json_path=raw_dir / "era5_2026_calibration.json",
    )
    requests_metadata.append(calibration_meta)
    observed = _observed_outdoor_frame(Path(observed_path))
    fit_end = observed["timestamp"].min() + pd.Timedelta(days=70)
    fit_observed = observed.loc[observed["timestamp"] < fit_end].copy()
    validation_observed = observed.loc[observed["timestamp"] >= fit_end].copy()
    correction = fit_local_bias_correction(calibration, fit_observed)
    candidate_validation = correction_validation_metrics(
        calibration.loc[calibration["timestamp"] >= fit_end].copy(),
        validation_observed,
        correction,
    )
    correction["enabled_variables"] = {
        variable: candidate_validation["corrected_mae"][variable]
        < candidate_validation["raw_mae"][variable]
        for variable in (
            "air_temperature", "relative_humidity", "global_radiation", "wind_speed"
        )
    }
    validation = correction_validation_metrics(
        calibration.loc[calibration["timestamp"] >= fit_end].copy(),
        validation_observed,
        correction,
    )
    validation["candidate_corrected_mae"] = candidate_validation["corrected_mae"]
    _write_json_atomic(processed / "bias_correction.json", correction)

    corrected_years = {
        year: apply_local_bias_correction(frame, correction)
        for year, frame in full_years.items()
    }
    full_dir = processed / "full_years" / "Chengdu"
    full_dir.mkdir(parents=True, exist_ok=True)
    processed_outputs: list[dict[str, Any]] = []
    for year, frame in corrected_years.items():
        output = full_dir / f"{year}.csv"
        to_greenlight_weather(frame).to_csv(output, index=False)
        processed_outputs.append({
            "kind": "full_year_greenlight",
            "year": year,
            "path": output.relative_to(processed).as_posix(),
            "rows": len(frame),
            "sha256": sha256_path(output),
        })
    seasons = build_six_seasons(corrected_years)
    season_dir = processed / "seasons"
    season_dir.mkdir(parents=True, exist_ok=True)
    roles = {
        "2023_spring": "train", "2023_autumn": "train",
        "2024_spring": "train", "2024_autumn": "train",
        "2025_spring": "validation", "2025_autumn": "test",
    }
    for name, frame in seasons.items():
        output = season_dir / f"{name}.csv"
        frame.to_csv(output, index=False)
        processed_outputs.append({
            "kind": "season_canonical",
            "season": name,
            "role": roles[name],
            "path": output.relative_to(processed).as_posix(),
            "rows": len(frame),
            "sha256": sha256_path(output),
        })

    quality = {
        "full_year_rows": {str(year): len(frame) for year, frame in corrected_years.items()},
        "season_rows": {name: len(frame) for name, frame in seasons.items()},
        "season_count": len(seasons),
        "total_season_rows": int(sum(len(frame) for frame in seasons.values())),
        "correction_fit_samples": correction["fit_samples"],
        "local_holdout_validation": validation,
        "physical_bounds_valid": True,
    }
    _write_json_atomic(processed / "quality_report.json", quality)
    manifest = {
        "dataset": "Chengdu greenhouse ERA5 outdoor forcing",
        "source_type": "reanalysis_not_site_observation",
        "coordinate": {
            "latitude": latitude,
            "longitude": longitude,
            "precision": "Pidu district centroid fallback",
        },
        "timezone": "Asia/Shanghai",
        "years": [2023, 2024, 2025],
        "season_definitions": SEASON_DATES,
        "requests": requests_metadata,
        "outputs": processed_outputs,
        "bias_correction_path": "bias_correction.json",
        "quality_report_path": "quality_report.json",
    }
    manifest_path = external / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build Chengdu ERA5 seasons")
    parser.add_argument("--latitude", type=float, default=30.7980107)
    parser.add_argument("--longitude", type=float, default=103.8986134)
    parser.add_argument(
        "--external_root",
        default="data/external/weather/era5/chengdu_greenhouse_001",
    )
    parser.add_argument(
        "--processed_root",
        default="data/processed/chengdu_agri/greenhouse_001/weather_era5",
    )
    parser.add_argument(
        "--observed_path",
        default="data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv",
    )
    args = parser.parse_args()
    manifest = generate_dataset(
        latitude=args.latitude,
        longitude=args.longitude,
        external_root=args.external_root,
        processed_root=args.processed_root,
        observed_path=args.observed_path,
    )
    print(f"Saved ERA5 manifest to {manifest}")


if __name__ == "__main__":
    main()
