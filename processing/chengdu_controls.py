from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from processing.chengdu_control_states import derive_hourly_controls


CONTROL_COLUMNS = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"]

DEVICE_CONTROL_MAP = {
    "CO2发生器": "uCO2",
    "顶保温": "uThScr",
    "四周保温": "uThScr",
    "顶窗": "uVent",
    "湿帘风机": "uVent",
    "湿帘水泵": "uVent",
    "湿帘卷膜机": "uVent",
    "补光灯": "uLamp",
    "外遮阳": "uBlScr",
}

ACTION_STATE_MAP = {
    "TURN_ON": 1.0,
    "TURN_OFF": 0.0,
}


def _time_column(df: pd.DataFrame) -> str:
    for column in ["time", "timestamp"]:
        if column in df.columns:
            return column
    raise ValueError("Missing time/timestamp column")


def normalize_device_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {"操作时间", "设备类型", "操作类型"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing device event columns: {sorted(missing)}")

    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(events["操作时间"], errors="coerce", format="mixed"),
            "device": events["设备类型"].astype(str),
            "action": events["操作类型"].astype(str),
        }
    )
    normalized["control"] = normalized["device"].map(DEVICE_CONTROL_MAP)
    normalized["state"] = normalized["action"].map(ACTION_STATE_MAP)
    return normalized.dropna(subset=["timestamp", "control", "state"]).sort_values("timestamp").reset_index(drop=True)


def build_hourly_controls(events: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    normalized = normalize_device_events(events)
    hourly_timestamps = pd.DatetimeIndex(pd.to_datetime(hourly_index)).floor("h")
    controls = pd.DataFrame({"timestamp": hourly_timestamps})
    for control in CONTROL_COLUMNS:
        controls[control] = 0.0

    if normalized.empty:
        return controls

    for (_device, control), device_events in normalized.groupby(["device", "control"]):
        duty = _hourly_duty_cycle(device_events[["timestamp", "state"]], hourly_timestamps)
        controls[control] = np.maximum(controls[control].to_numpy(dtype=float), duty)

    # Several physical devices contribute to one GreenLight actuator; keep
    # the actuator bounded even when repeated event rows exist.
    for control in CONTROL_COLUMNS:
        controls[control] = np.clip(controls[control].astype(float), 0.0, 1.0)
    return controls


def _hourly_duty_cycle(device_events: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> np.ndarray:
    if len(hourly_index) == 0:
        return np.array([], dtype=float)

    events = device_events.dropna(subset=["timestamp", "state"]).sort_values("timestamp")
    if events.empty:
        return np.zeros(len(hourly_index), dtype=float)

    hour_starts = pd.DatetimeIndex(hourly_index)
    hour_ends = hour_starts + pd.Timedelta(hours=1)
    values = np.zeros(len(hour_starts), dtype=float)

    current_state = 0.0
    cursor = hour_starts[0]
    for _, event in events.iterrows():
        event_time = pd.Timestamp(event["timestamp"])
        if event_time > cursor and current_state > 0.0:
            _accumulate_overlap(values, hour_starts, hour_ends, cursor, event_time, current_state)
        current_state = float(event["state"])
        cursor = max(event_time, cursor)

    final_end = hour_ends[-1]
    if cursor < final_end and current_state > 0.0:
        _accumulate_overlap(values, hour_starts, hour_ends, cursor, final_end, current_state)

    return np.clip(values / 3600.0, 0.0, 1.0)


def _accumulate_overlap(
    values: np.ndarray,
    hour_starts: pd.DatetimeIndex,
    hour_ends: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    state: float,
) -> None:
    for idx, (hour_start, hour_end) in enumerate(zip(hour_starts, hour_ends)):
        overlap_start = max(start, hour_start)
        overlap_end = min(end, hour_end)
        if overlap_end <= overlap_start:
            continue
        values[idx] += (overlap_end - overlap_start).total_seconds() * state


def merge_controls_with_hourly_wide(hourly: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    time_col = _time_column(hourly)
    merged = hourly.copy()
    timestamps = pd.to_datetime(merged[time_col], errors="coerce")
    controls = build_hourly_controls(events, pd.DatetimeIndex(timestamps))
    for control in CONTROL_COLUMNS:
        merged[control] = controls[control].to_numpy(dtype=float)
    return merged


def build_hourly_controls_from_states(states: pd.DataFrame) -> pd.DataFrame:
    """Convert canonical minute device states to hourly Chengdu controls."""
    return derive_hourly_controls(states)


def read_device_event_csvs(input_dir: str | Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(input_dir).glob("*.csv")):
        try:
            frame = pd.read_csv(path, encoding="gbk")
        except UnicodeDecodeError:
            frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No device CSV files found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Merge Chengdu device event CSV files into hourly wide greenhouse data.")
    parser.add_argument("--hourly_wide_csv", required=True)
    parser.add_argument("--device_csv_dir", required=True)
    parser.add_argument("--output_csv", default="data/processed/chengdu_agri/greenhouse_001/trajectories/hourly_wide_with_controls.csv")
    args = parser.parse_args()

    hourly = pd.read_csv(args.hourly_wide_csv)
    events = read_device_event_csvs(args.device_csv_dir)
    merged = merge_controls_with_hourly_wide(hourly, events)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
