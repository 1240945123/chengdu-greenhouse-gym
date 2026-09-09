from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.reports.evaluate_chengdu_physics import evaluate_one_step_predictions, load_parameter_vector


ERROR_COLUMNS = ["temperature_error", "relative_humidity_error"]


def _autocorrelation(values: pd.Series, lag: int) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size <= lag:
        return None
    left = clean[:-lag]
    right = clean[lag:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _group_metrics(frame: pd.DataFrame, minimum_group_size: int) -> dict[str, Any]:
    if len(frame) < minimum_group_size:
        return {"status": "insufficient", "rows": int(len(frame))}
    result: dict[str, Any] = {"status": "ok", "rows": int(len(frame))}
    for column in ERROR_COLUMNS:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        result[column] = {
            "mae": float(values.abs().mean()),
            "rmse": float(np.sqrt(np.mean(values.to_numpy(dtype=float) ** 2))),
            "bias": float(values.mean()),
        }
    return result


def residual_diagnostics(frame: pd.DataFrame, minimum_group_size: int = 12) -> dict[str, Any]:
    required = {"timestamp", "d_global_radiation", "uRoofVent", "uFan", *ERROR_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing residual diagnostic columns: {missing}")
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="raise")
    roof = pd.to_numeric(data["uRoofVent"], errors="coerce").fillna(0.0).gt(0.05)
    fan = pd.to_numeric(data["uFan"], errors="coerce").fillna(0.0).gt(0.05)
    radiation = pd.to_numeric(data["d_global_radiation"], errors="coerce").fillna(0.0)

    autocorrelation = {
        column: {f"lag_{lag}": _autocorrelation(data[column], lag) for lag in (1, 6, 24)}
        for column in ERROR_COLUMNS
    }
    day_night = {
        "day": _group_metrics(data.loc[radiation.gt(5.0)], minimum_group_size),
        "night": _group_metrics(data.loc[~radiation.gt(5.0)], minimum_group_size),
    }
    masks = {
        "both_off": ~roof & ~fan,
        "roof_only": roof & ~fan,
        "fan_only": ~roof & fan,
        "both_on": roof & fan,
    }
    control_regimes = {
        name: _group_metrics(data.loc[mask], minimum_group_size) for name, mask in masks.items()
    }
    return {
        "rows": int(len(data)),
        "minimum_group_size": int(minimum_group_size),
        "autocorrelation": autocorrelation,
        "day_night": day_night,
        "control_regimes": control_regimes,
    }


def build_residual_predictions(
    trajectory: pd.DataFrame,
    parameter_vector: np.ndarray,
    model_backend: str,
) -> pd.DataFrame:
    _metrics, predictions = evaluate_one_step_predictions(
        trajectory,
        parameter_vector=parameter_vector,
        model_backend=model_backend,
    )
    predictions["temperature_error"] = (
        predictions["pred_air_temperature"] - predictions["next_x_air_temperature"]
    )
    predictions["relative_humidity_error"] = (
        predictions["pred_relative_humidity"] - predictions["next_x_relative_humidity"]
    )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Chengdu physics validation residual regimes.")
    parser.add_argument("--trajectory_csv", default="data/processed/chengdu_agri/greenhouse_001/trajectories/v3/val.csv")
    parser.add_argument("--params_json", default="results/chengdu_agri_greenhouse_001/physics_v3_identification/selected_params.json")
    parser.add_argument("--save_dir", default="results/chengdu_agri_greenhouse_001/physics_v3_identification/residuals")
    parser.add_argument("--minimum_group_size", type=int, default=12)
    parser.add_argument("--model_backend", default="ChengduPhysicsV3")
    args = parser.parse_args()

    trajectory = pd.read_csv(args.trajectory_csv)
    params = load_parameter_vector(args.params_json)
    predictions = build_residual_predictions(
        trajectory,
        parameter_vector=params,
        model_backend=args.model_backend,
    )
    diagnostics = residual_diagnostics(predictions, minimum_group_size=args.minimum_group_size)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "one_step_predictions.csv", index=False)
    (output / "residual_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
