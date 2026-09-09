from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "global_radiation",
    "wind_speed",
    "air_temperature",
    "sky_temperature",
    "co2_concentration",
    "relative_humidity",
}

WIDE_COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time"],
    "global_radiation": ["global_radiation", "solar_radiation_mean", "photo_radiation_mean"],
    "wind_speed": ["wind_speed", "wind_speed_mean"],
    "air_temperature": ["air_temperature", "air_temperature_mean"],
    "sky_temperature": ["sky_temperature"],
    "co2_concentration": ["co2_concentration", "co2_mean"],
    "relative_humidity": ["relative_humidity", "air_humidity_mean"],
}


OUTPUT_COLUMNS = [
    "time",
    "global radiation",
    "wind speed",
    "air temperature",
    "sky temperature",
    "CO2 concentration",
    "day number",
    "RH",
]


def _validate_columns(df: pd.DataFrame):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required weather columns: {missing}")


def _canonical_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    canonical = pd.DataFrame()
    for target, candidates in WIDE_COLUMN_ALIASES.items():
        source = next((name for name in candidates if name in df.columns), None)
        if source is not None:
            canonical[target] = df[source]

    if "sky_temperature" not in canonical and "air_temperature" in canonical:
        canonical["sky_temperature"] = pd.to_numeric(canonical["air_temperature"], errors="coerce") - 6.0

    _validate_columns(canonical)
    return canonical


def convert_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = _canonical_weather_dataframe(df)
    timestamps = pd.to_datetime(data["timestamp"])
    if timestamps.isna().any():
        raise ValueError("timestamp column contains invalid values")
    if timestamps.dt.year.nunique() != 1:
        raise ValueError("A GreenLight weather file must contain one calendar year")
    seconds = (
        (timestamps.dt.dayofyear - 1) * 86400
        + timestamps.dt.hour * 3600
        + timestamps.dt.minute * 60
        + timestamps.dt.second
        + timestamps.dt.microsecond / 1_000_000
    )

    converted = pd.DataFrame(
        {
            "time": seconds.astype(float),
            "global radiation": data["global_radiation"].astype(float),
            "wind speed": data["wind_speed"].astype(float),
            "air temperature": data["air_temperature"].astype(float),
            "sky temperature": data["sky_temperature"].astype(float),
            "CO2 concentration": data["co2_concentration"].astype(float),
            "day number": timestamps.dt.dayofyear.astype(int),
            "RH": data["relative_humidity"].astype(float),
        }
    )
    return converted[OUTPUT_COLUMNS]


def convert_aligned_weather_dataframe(
    df: pd.DataFrame,
    *,
    outdoor_co2_ppm: float = 400.0,
) -> pd.DataFrame:
    """Convert the audited aligned table using explicit outdoor weather fields."""
    required = {
        "timestamp",
        "global_radiation",
        "wind_speed",
        "outdoor_air_temperature",
        "outdoor_relative_humidity",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing aligned outdoor weather columns: {missing}")
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("timestamp column contains invalid values")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Aligned weather timestamps must be increasing and unique")
    if len(timestamps) > 1:
        intervals = timestamps.diff().dropna().dt.total_seconds().to_numpy(dtype=float)
        if not (intervals == 3600.0).all():
            raise ValueError("Aligned weather must use a continuous hourly grid")

    canonical = pd.DataFrame(
        {
            "timestamp": timestamps,
            "global_radiation": pd.to_numeric(df["global_radiation"], errors="coerce"),
            "wind_speed": pd.to_numeric(df["wind_speed"], errors="coerce"),
            "air_temperature": pd.to_numeric(
                df["outdoor_air_temperature"], errors="coerce"
            ),
            "co2_concentration": float(outdoor_co2_ppm),
            "relative_humidity": pd.to_numeric(
                df["outdoor_relative_humidity"], errors="coerce"
            ),
        }
    )
    canonical["sky_temperature"] = canonical["air_temperature"] - 6.0
    if canonical.isna().any().any():
        raise ValueError("Aligned outdoor weather contains missing or non-numeric values")
    if (canonical["global_radiation"] < 0.0).any() or (canonical["wind_speed"] < 0.0).any():
        raise ValueError("Radiation and wind speed must be non-negative")
    if not canonical["relative_humidity"].between(0.0, 100.0).all():
        raise ValueError("Outdoor relative humidity must be within 0-100%")
    if not canonical["air_temperature"].between(-30.0, 60.0).all():
        raise ValueError("Outdoor air temperature is outside the configured physical range")
    return convert_weather_dataframe(canonical)


def write_greenlight_weather_csv(
    df: pd.DataFrame,
    output_root: str | Path,
    location: str,
    year: int,
) -> Path:
    converted = convert_weather_dataframe(df)
    output_dir = Path(output_root) / location
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{int(year)}.csv"
    converted.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert Chengdu greenhouse raw weather CSV to GreenLight weather format.")
    parser.add_argument("--input", required=True, help="Raw weather CSV with schema columns from data/schemas/weather_schema.yml")
    parser.add_argument("--output_root", default="data/processed/chengdu_agri/greenhouse_001/weather/")
    parser.add_argument("--location", default="Chengdu")
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    output_path = write_greenlight_weather_csv(
        raw,
        output_root=args.output_root,
        location=args.location,
        year=args.year,
    )
    print(f"Saved GreenLight weather CSV to {output_path}")


if __name__ == "__main__":
    main()
