from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CONTROL_COLUMNS = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in df.columns), None)


def _series_or_default(df: pd.DataFrame, candidates: list[str], default: float = 0.0) -> pd.Series:
    column = _first_existing(df, candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def build_trajectory_dataframe(
    df: pd.DataFrame,
    *,
    require_measured_outdoor: bool = True,
) -> pd.DataFrame:
    data = df.copy()
    timestamp_col = _first_existing(data, ["timestamp", "time"])
    if timestamp_col is None:
        raise ValueError("Missing timestamp/time column")
    timestamps = pd.to_datetime(data[timestamp_col])
    order = timestamps.argsort()
    data = data.iloc[order].reset_index(drop=True)
    timestamps = timestamps.iloc[order].reset_index(drop=True)

    air_temp = _series_or_default(data, ["air_temperature", "air_temperature_mean"])
    rh = _series_or_default(data, ["relative_humidity", "air_humidity_mean"])
    outdoor_temp_candidates = ["outdoor_air_temperature", "outdoor_air_temperature_mean"]
    outdoor_rh_candidates = ["outdoor_relative_humidity", "outdoor_air_humidity_mean"]
    missing_outdoor = []
    if _first_existing(data, outdoor_temp_candidates) is None:
        missing_outdoor.append("outdoor_air_temperature")
    if _first_existing(data, outdoor_rh_candidates) is None:
        missing_outdoor.append("outdoor_relative_humidity")
    if require_measured_outdoor and missing_outdoor:
        raise ValueError(f"Missing measured outdoor disturbance columns: {', '.join(missing_outdoor)}")

    outdoor_air_temp = (
        _series_or_default(data, outdoor_temp_candidates)
        if "outdoor_air_temperature" not in missing_outdoor
        else air_temp.copy()
    )
    outdoor_rh = (
        _series_or_default(data, outdoor_rh_candidates)
        if "outdoor_relative_humidity" not in missing_outdoor
        else rh.copy()
    )
    co2 = _series_or_default(data, ["co2_concentration", "co2_mean"])
    radiation = _series_or_default(data, ["global_radiation", "solar_radiation_mean", "photo_radiation_mean"])
    wind = _series_or_default(data, ["wind_speed", "wind_speed_mean"])
    illumination = _series_or_default(data, ["illumination", "illumination_mean"])
    pressure = _series_or_default(data, ["air_pressure", "air_pressure_mean"])
    soil_temp = _series_or_default(data, ["soil_temperature", "soil_temperature_mean"])
    soil_humidity = _series_or_default(data, ["soil_humidity", "soil_humidity_mean"])

    n = len(data) - 1
    if n < 1:
        raise ValueError("Need at least two rows to build one-step trajectories")

    trajectory = pd.DataFrame(
        {
            "timestamp": timestamps.iloc[:n].values,
            "next_timestamp": timestamps.iloc[1:n + 1].values,
            "x_air_temperature": air_temp.iloc[:n].values,
            "x_relative_humidity": rh.iloc[:n].values,
            "x_co2_concentration": co2.iloc[:n].values,
            "x_illumination": illumination.iloc[:n].values,
            "x_soil_temperature": soil_temp.iloc[:n].values,
            "x_soil_humidity": soil_humidity.iloc[:n].values,
            "d_global_radiation": radiation.iloc[:n].values,
            "d_wind_speed": wind.iloc[:n].values,
            "d_air_pressure": pressure.iloc[:n].values,
            "d_air_temperature": outdoor_air_temp.iloc[:n].values,
            "d_relative_humidity": outdoor_rh.iloc[:n].values,
            "next_x_air_temperature": air_temp.iloc[1:n + 1].values,
            "next_x_relative_humidity": rh.iloc[1:n + 1].values,
            "next_x_co2_concentration": co2.iloc[1:n + 1].values,
            "next_x_illumination": illumination.iloc[1:n + 1].values,
        }
    )

    for control in CONTROL_COLUMNS:
        trajectory[control] = _series_or_default(data, [control, control.lower(), f"{control}_mean"], default=0.0).iloc[:n].values

    ordered_columns = [
        "timestamp",
        "next_timestamp",
        "x_air_temperature",
        "x_relative_humidity",
        "x_co2_concentration",
        "x_illumination",
        "x_soil_temperature",
        "x_soil_humidity",
        *CONTROL_COLUMNS,
        "d_global_radiation",
        "d_wind_speed",
        "d_air_pressure",
        "d_air_temperature",
        "d_relative_humidity",
        "next_x_air_temperature",
        "next_x_relative_humidity",
        "next_x_co2_concentration",
        "next_x_illumination",
    ]
    return trajectory[ordered_columns].dropna().reset_index(drop=True)


def split_trajectory_dataframe(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    ordered = df.sort_values("timestamp").reset_index(drop=True) if "timestamp" in df.columns else df.reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    return {
        "train": ordered.iloc[:train_end].reset_index(drop=True),
        "val": ordered.iloc[train_end:val_end].reset_index(drop=True),
        "test": ordered.iloc[val_end:].reset_index(drop=True),
    }


def write_trajectory_splits(
    df: pd.DataFrame,
    output_dir: str | Path,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    require_measured_outdoor: bool = True,
) -> dict[str, Path]:
    trajectory = build_trajectory_dataframe(df, require_measured_outdoor=require_measured_outdoor)
    splits = split_trajectory_dataframe(trajectory, train_fraction=train_fraction, val_fraction=val_fraction)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, split_df in splits.items():
        path = output_path / f"{name}.csv"
        split_df.to_csv(path, index=False)
        paths[name] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build Chengdu greenhouse one-step trajectory train/val/test CSV files.")
    parser.add_argument("--input", required=True, help="Raw or hourly-wide greenhouse CSV")
    parser.add_argument("--output_dir", default="data/processed/chengdu_agri/greenhouse_001/trajectories/")
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    paths = write_trajectory_splits(
        raw,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
