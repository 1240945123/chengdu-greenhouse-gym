from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


SENSOR_INPUT_COLUMNS = [
    "_id",
    "devCode",
    "paramId",
    "timeString",
    "paramName",
    "paramType",
    "code",
    "paramCode",
    "paramNo",
    "paramVal",
    "booleanIndoor",
    "paramUnit",
    "ctime",
    "ip",
    "topic",
    "plantingAreaId",
    "equipmentSensorId",
    "equipmentId",
    "greenhouseId",
    "extraFieldsJson",
]

READ_COLUMNS = [
    "timeString",
    "paramType",
    "paramVal",
    "booleanIndoor",
    "equipmentSensorId",
    "equipmentId",
    "greenhouseId",
]

PARAMETER_NAMES = {
    "AIR_TEMPERATURE": "air_temperature",
    "AIR_HUMIDITY": "relative_humidity",
    "CO2": "co2_concentration",
    "ILLUMINATION": "illumination",
    "PHOTO_ELECTRIC_RADIATION": "global_radiation",
    "WIND_SPEED": "wind_speed",
    "SOIL_TEMPERATURE": "soil_temperature",
    "SOIL_HUMIDITY": "soil_humidity",
    "SOIL_CONDUCTIVITY": "soil_conductivity",
    "YUGEN_IBT": "infrared_cavity_temperature",
    "YUGEN_ITT": "canopy_temperature",
}


def clean_sensor_chunk(
    raw: pd.DataFrame,
    greenhouse_id: int,
    ranges: Mapping[str, tuple[float, float] | list[float]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Filter one raw chunk and normalize its sensor identity and value fields."""
    missing = set(READ_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing raw sensor columns: {sorted(missing)}")

    quality = {
        "input_rows": int(len(raw)),
        "target_rows": 0,
        "invalid_timestamp": 0,
        "invalid_numeric": 0,
        "invalid_range": 0,
    }
    greenhouse_values = pd.to_numeric(raw["greenhouseId"], errors="coerce")
    target = raw.loc[greenhouse_values.eq(int(greenhouse_id)), READ_COLUMNS].copy()
    quality["target_rows"] = int(len(target))

    timestamp_text = target["timeString"].astype("string").str.replace(
        r"(\d{2}:\d{2}:\d{2}):(\d{3})$", r"\1.\2", regex=True
    )
    target["timestamp"] = pd.to_datetime(timestamp_text, errors="coerce")
    quality["invalid_timestamp"] = int(target["timestamp"].isna().sum())

    target["value"] = pd.to_numeric(target["paramVal"], errors="coerce")
    quality["invalid_numeric"] = int(target["value"].isna().sum())
    target["param_type"] = target["paramType"].astype("string").str.strip().str.upper()
    target["equipment_id"] = pd.to_numeric(target["equipmentId"], errors="coerce").astype("Int64")
    target["sensor_id"] = pd.to_numeric(target["equipmentSensorId"], errors="coerce").astype("Int64")
    target["boolean_indoor"] = _boolean_series(target["booleanIndoor"])

    invalid_range = pd.Series(False, index=target.index)
    for param_type, bounds in ranges.items():
        lower, upper = float(bounds[0]), float(bounds[1])
        selected = target["param_type"].eq(str(param_type).upper()) & target["value"].notna()
        invalid_range |= selected & ~target["value"].between(lower, upper, inclusive="both")
    quality["invalid_range"] = int(invalid_range.sum())
    target.loc[invalid_range, "value"] = np.nan

    clean = target[
        ["timestamp", "param_type", "value", "boolean_indoor", "equipment_id", "sensor_id"]
    ].reset_index(drop=True)
    return clean, quality


def _boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map({"true": True, "1": True, "false": False, "0": False})
    return mapped.astype("boolean")


def read_sensor_files(
    sensor_root: str | Path,
    greenhouse_id: int,
    ranges: Mapping[str, tuple[float, float] | list[float]],
    chunksize: int = 250_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read daily CSV files in chunks and retain only the target greenhouse rows."""
    paths = sorted(Path(sensor_root).glob("**/*.csv"))
    if not paths:
        raise ValueError(f"No sensor CSV files found in {sensor_root}")

    frames: list[pd.DataFrame] = []
    totals: defaultdict[str, int] = defaultdict(int)
    for path in paths:
        try:
            chunks = pd.read_csv(path, usecols=READ_COLUMNS, chunksize=chunksize, low_memory=False)
            for chunk in chunks:
                clean, quality = clean_sensor_chunk(chunk, greenhouse_id=greenhouse_id, ranges=ranges)
                frames.append(clean)
                for key, value in quality.items():
                    totals[key] += int(value)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Failed to read sensor file {path}: {exc}") from exc

    quality_report: dict[str, object] = {"files": len(paths), **dict(totals)}
    if not frames:
        return _empty_clean_frame(), quality_report
    return pd.concat(frames, ignore_index=True), quality_report


def _empty_clean_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "param_type": pd.Series(dtype="string"),
            "value": pd.Series(dtype="float64"),
            "boolean_indoor": pd.Series(dtype="boolean"),
            "equipment_id": pd.Series(dtype="Int64"),
            "sensor_id": pd.Series(dtype="Int64"),
        }
    )


def build_minute_sensor_table(
    clean_long: pd.DataFrame,
    interpolation_limit: int = 5,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Pivot cleaned sensor rows to a complete minute grid with short-gap flags."""
    required = {"timestamp", "param_type", "value", "boolean_indoor", "equipment_id", "sensor_id"}
    missing = required.difference(clean_long.columns)
    if missing:
        raise ValueError(f"Missing cleaned sensor columns: {sorted(missing)}")
    if interpolation_limit < 0:
        raise ValueError("interpolation_limit must be non-negative")

    clean = clean_long.copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
    clean = clean.dropna(subset=["timestamp", "param_type", "equipment_id", "sensor_id"])
    if clean.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")}), {"rows": 0, "variables": {}}

    clean["minute"] = clean["timestamp"].dt.floor("min")
    clean["base_name"] = clean["param_type"].map(PARAMETER_NAMES).fillna(
        clean["param_type"].str.lower()
    )
    clean["channel"] = (
        clean["base_name"]
        + "__e"
        + clean["equipment_id"].astype("int64").astype(str)
        + "__s"
        + clean["sensor_id"].astype("int64").astype(str)
    )

    channel_values = clean.pivot_table(
        index="minute",
        columns="channel",
        values="value",
        aggfunc="median",
    )
    minute_index = pd.date_range(channel_values.index.min(), channel_values.index.max(), freq="min")
    values = channel_values.reindex(minute_index).sort_index()

    summary = _summary_variables(clean, minute_index)
    for column in summary.columns:
        values[column] = summary[column]

    values = values.reindex(sorted(values.columns), axis=1)
    result_columns: dict[str, np.ndarray] = {}
    variable_quality: dict[str, dict[str, int]] = {}
    for column in values.columns:
        series = pd.to_numeric(values[column], errors="coerce")
        observed = series.notna()
        filled, interpolated = _interpolate_short_internal_gaps(series, interpolation_limit)
        result_columns[column] = filled.to_numpy(dtype=float)
        result_columns[f"{column}_observed"] = observed.to_numpy(dtype=bool)
        result_columns[f"{column}_interpolated"] = interpolated.to_numpy(dtype=bool)
        variable_quality[column] = {
            "observed": int(observed.sum()),
            "interpolated": int(interpolated.sum()),
            "missing": int(filled.isna().sum()),
            "longest_missing_gap": _longest_true_run(filled.isna()),
        }

    result = pd.DataFrame({"timestamp": minute_index, **result_columns})
    return result, {"rows": int(len(result)), "variables": variable_quality}


def _summary_variables(clean: pd.DataFrame, minute_index: pd.DatetimeIndex) -> pd.DataFrame:
    summaries = pd.DataFrame(index=minute_index)
    definitions = {
        "air_temperature": ("AIR_TEMPERATURE", True),
        "relative_humidity": ("AIR_HUMIDITY", True),
        "co2_concentration": ("CO2", True),
        "illumination": ("ILLUMINATION", True),
        "outdoor_air_temperature": ("AIR_TEMPERATURE", False),
        "outdoor_relative_humidity": ("AIR_HUMIDITY", False),
        "global_radiation": ("PHOTO_ELECTRIC_RADIATION", False),
        "wind_speed": ("WIND_SPEED", False),
        "soil_temperature": ("SOIL_TEMPERATURE", True),
        "soil_humidity": ("SOIL_HUMIDITY", True),
        "soil_conductivity": ("SOIL_CONDUCTIVITY", True),
        "canopy_temperature": ("YUGEN_ITT", True),
    }
    for name, (param_type, indoor) in definitions.items():
        selected = clean.loc[
            clean["param_type"].eq(param_type) & clean["boolean_indoor"].eq(indoor)
        ]
        if selected.empty:
            continue
        summaries[name] = selected.groupby("minute")["value"].median().reindex(minute_index)
    return summaries


def _interpolate_short_internal_gaps(
    series: pd.Series,
    limit: int,
) -> tuple[pd.Series, pd.Series]:
    missing = series.isna()
    if limit == 0 or not missing.any():
        return series.copy(), pd.Series(False, index=series.index)
    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    bounded = series.notna().cummax() & series[::-1].notna().cummax()[::-1]
    eligible = missing & run_length.le(limit) & bounded
    candidate = series.interpolate(method="time", limit_area="inside")
    filled = series.copy()
    filled.loc[eligible] = candidate.loc[eligible]
    return filled, eligible & filled.notna()


def _longest_true_run(mask: pd.Series) -> int:
    if mask.empty or not mask.any():
        return 0
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    return int(mask.groupby(groups).sum().max())
