from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESIDUAL_FEATURES = [
    "pred_relative_humidity",
    "pred_air_temperature",
    "x_relative_humidity",
    "x_air_temperature",
    "uVent",
    "uBlScr",
    "uThScr",
    "d_global_radiation",
    "d_wind_speed",
]
DEFAULT_TARGETS = ["air_temperature", "relative_humidity"]

BASE_RESIDUAL_FEATURES = [
    "pred_air_temperature",
    "pred_relative_humidity",
    "d_air_temperature",
    "d_relative_humidity",
    "d_global_radiation",
    "d_wind_speed",
    "uRoofVent",
    "uFan",
    "uBlScr",
    "uThScr",
]
THERMAL_DYNAMICS_FEATURES = [
    "solar_trapping",
    "ventilation_heat_exchange",
    "fan_cooling_demand",
    "pad_evaporative_potential",
    "hot_solar_load",
]
FEATURE_SETS = {
    "base": BASE_RESIDUAL_FEATURES,
    "periodic": BASE_RESIDUAL_FEATURES + ["hour_sin", "hour_cos"],
    "periodic_interactions": BASE_RESIDUAL_FEATURES
    + [
        "hour_sin",
        "hour_cos",
        "temperature_outdoor_gap",
        "humidity_outdoor_gap",
        "radiation_hour_sin",
        "radiation_hour_cos",
    ],
    "thermal_dynamics": BASE_RESIDUAL_FEATURES
    + [
        "hour_sin",
        "hour_cos",
        "temperature_outdoor_gap",
        "humidity_outdoor_gap",
        "radiation_hour_sin",
        "radiation_hour_cos",
        "uPad",
    ]
    + THERMAL_DYNAMICS_FEATURES,
}
PROHIBITED_FEATURE_PREFIXES = ("x_", "next_x_", "true_", "corrected_")


def validate_residual_features(feature_columns: list[str]) -> None:
    if not feature_columns:
        raise ValueError("Residual feature list cannot be empty")
    duplicates = sorted({name for name in feature_columns if feature_columns.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate residual features: {duplicates}")
    prohibited = [name for name in feature_columns if name.startswith(PROHIBITED_FEATURE_PREFIXES)]
    if prohibited:
        raise ValueError(f"Residual feature leakage is prohibited: {prohibited}")


def build_residual_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    validate_residual_features(feature_columns)
    values = frame.copy()
    if any(name in feature_columns for name in ("hour_sin", "hour_cos", "radiation_hour_sin", "radiation_hour_cos")):
        if "timestamp" not in values.columns:
            raise ValueError("Missing residual correction feature source: timestamp")
        timestamp = pd.to_datetime(values["timestamp"], errors="raise")
        hour = timestamp.dt.hour.to_numpy(dtype=float) + timestamp.dt.minute.to_numpy(dtype=float) / 60.0
        values["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        values["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    if "temperature_outdoor_gap" in feature_columns:
        values["temperature_outdoor_gap"] = values["pred_air_temperature"] - values["d_air_temperature"]
    if "humidity_outdoor_gap" in feature_columns:
        values["humidity_outdoor_gap"] = values["pred_relative_humidity"] - values["d_relative_humidity"]
    if "radiation_hour_sin" in feature_columns:
        values["radiation_hour_sin"] = values["d_global_radiation"] * values["hour_sin"]
    if "radiation_hour_cos" in feature_columns:
        values["radiation_hour_cos"] = values["d_global_radiation"] * values["hour_cos"]
    if any(name in feature_columns for name in THERMAL_DYNAMICS_FEATURES):
        required = [
            "pred_air_temperature",
            "d_air_temperature",
            "d_relative_humidity",
            "d_global_radiation",
            "d_wind_speed",
            "uRoofVent",
            "uFan",
            "uPad",
        ]
        missing_sources = [column for column in required if column not in values.columns]
        if missing_sources:
            raise ValueError(
                f"Missing thermal dynamics feature sources: {missing_sources}"
            )
        numeric = values[required].apply(pd.to_numeric, errors="raise").astype(float)
        roof = numeric["uRoofVent"].clip(0.0, 1.0)
        fan = numeric["uFan"].clip(0.0, 1.0)
        pad = numeric["uPad"].clip(0.0, 1.0)
        radiation = numeric["d_global_radiation"].clip(lower=0.0)
        temperature_gap = (
            numeric["pred_air_temperature"] - numeric["d_air_temperature"]
        )
        positive_gap = temperature_gap.clip(lower=0.0)
        humidity_deficit = (1.0 - numeric["d_relative_humidity"] / 100.0).clip(
            0.0, 1.0
        )
        values["solar_trapping"] = radiation * (1.0 - roof) * (1.0 - fan) * (1.0 - pad)
        values["ventilation_heat_exchange"] = (
            roof * numeric["d_wind_speed"].clip(lower=0.0) * temperature_gap
        )
        values["fan_cooling_demand"] = fan * positive_gap
        values["pad_evaporative_potential"] = pad * positive_gap * humidity_deficit
        values["hot_solar_load"] = radiation * (
            numeric["d_air_temperature"] - 25.0
        ).clip(lower=0.0)
    missing = [column for column in feature_columns if column not in values.columns]
    if missing:
        raise ValueError(f"Missing residual correction features: {missing}")
    result = values[feature_columns].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Residual features must be finite")
    return result


def fit_ridge_residual_corrector(
    predictions: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
    target_names: list[str] | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> dict:
    if alpha < 0.0 or not np.isfinite(alpha):
        raise ValueError("Ridge alpha must be finite and nonnegative")
    targets = target_names or DEFAULT_TARGETS
    features = build_residual_features(predictions, feature_columns)
    weighted_fit = sample_weight is not None
    weights = (
        np.ones(len(features), dtype=float)
        if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )
    if (
        weights.ndim != 1
        or len(weights) != len(features)
        or not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or float(weights.sum()) <= 0.0
    ):
        raise ValueError("Ridge sample weights must be finite, nonnegative, and have positive sum")
    feature_values = features.to_numpy(dtype=float)
    means = np.average(feature_values, axis=0, weights=weights)
    variance = np.average(np.square(feature_values - means), axis=0, weights=weights)
    scales = np.sqrt(variance)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (feature_values - means) / scales
    design = np.c_[np.ones(len(standardized)), standardized]
    sqrt_weight = np.sqrt(weights)
    weighted_design = design * sqrt_weight[:, None]
    penalty = np.diag(np.r_[0.0, np.full(standardized.shape[1], float(alpha))])
    lhs = weighted_design.T @ weighted_design + penalty

    coefficients = {}
    for target in targets:
        pred_column = f"pred_{target}"
        next_column = f"next_x_{target}"
        if pred_column not in predictions.columns or next_column not in predictions.columns:
            raise ValueError(f"Missing residual target columns: {pred_column}, {next_column}")
        residual = (
            predictions[next_column].to_numpy(dtype=float)
            - predictions[pred_column].to_numpy(dtype=float)
        )
        weighted_residual = residual * sqrt_weight
        coefficients[target] = np.linalg.solve(
            lhs, weighted_design.T @ weighted_residual
        ).tolist()

    return {
        "schema_version": "ridge_residual_v1",
        "feature_columns": list(feature_columns),
        "feature_means": means.tolist(),
        "feature_scales": scales.tolist(),
        "target_names": list(targets),
        "coefficients": coefficients,
        "alpha": float(alpha),
        "weighted_fit": weighted_fit,
        "weight_sum": float(weights.sum()),
        "positive_weight_rows": int(np.count_nonzero(weights > 0.0)),
    }


def apply_ridge_residual_corrector(
    predictions: pd.DataFrame,
    model: dict,
    *,
    gain: float | None = None,
) -> pd.DataFrame:
    applied_gain = float(model.get("gain", 1.0) if gain is None else gain)
    if not 0.0 <= applied_gain <= 1.0:
        raise ValueError("Residual feedback gain must be between zero and one")
    features = build_residual_features(predictions, list(model["feature_columns"]))
    means = np.asarray(model["feature_means"], dtype=float)
    scales = np.asarray(model["feature_scales"], dtype=float)
    design = np.c_[np.ones(len(features)), (features.to_numpy(dtype=float) - means) / scales]
    corrected = predictions.copy()
    fallback = np.zeros(len(corrected), dtype=bool)
    target_gains = model.get("target_gains", {})
    for target in model["target_names"]:
        target_gain = float(target_gains.get(target, 1.0))
        if not 0.0 <= target_gain <= 1.0:
            raise ValueError("Residual target gain must be between zero and one")
        coefficients = np.asarray(model["coefficients"][target], dtype=float)
        residual = design @ coefficients
        invalid = ~np.isfinite(residual)
        fallback |= invalid
        residual[invalid] = 0.0
        lower, upper = (-10.0, 60.0) if target == "air_temperature" else (0.0, 100.0)
        corrected[f"pred_{target}"] = np.clip(
            predictions[f"pred_{target}"].to_numpy(dtype=float)
            + applied_gain * target_gain * residual,
            lower,
            upper,
        )
    corrected["residual_fallback"] = fallback
    return corrected


def calibrate_residual_uncertainty(
    calibration: pd.DataFrame,
    model: dict,
    *,
    gain: float,
    quantiles: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> dict:
    if not quantiles or any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("Uncertainty quantiles must be inside (0, 1)")
    result = deepcopy(model)
    result["gain"] = float(gain)
    corrected = apply_ridge_residual_corrector(calibration, result)
    uncertainty = {}
    for target in result["target_names"]:
        error = np.abs(
            corrected[f"pred_{target}"].to_numpy(dtype=float)
            - calibration[f"next_x_{target}"].to_numpy(dtype=float)
        )
        uncertainty[target] = {
            f"q{int(round(100 * quantile))}": float(np.quantile(error, quantile, method="higher"))
            for quantile in quantiles
        }
    result["uncertainty"] = uncertainty
    result["calibration_rows"] = int(len(calibration))
    return result


def _design_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing residual correction features: {missing}")
    return np.c_[np.ones(len(frame)), frame[feature_columns].to_numpy(dtype=float)]


def fit_linear_residual_corrector(
    predictions: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_names: list[str] | None = None,
) -> dict:
    features = feature_columns or DEFAULT_RESIDUAL_FEATURES
    targets = target_names or DEFAULT_TARGETS
    x = _design_matrix(predictions, features)

    coefficients = {}
    for target in targets:
        next_column = f"next_x_{target}"
        if next_column not in predictions.columns:
            raise ValueError(f"Missing target column: {next_column}")
        y = predictions[next_column].to_numpy(dtype=float)
        coefficients[target] = np.linalg.lstsq(x, y, rcond=None)[0].tolist()

    return {
        "feature_columns": features,
        "target_names": targets,
        "coefficients": coefficients,
    }


def evaluate_corrected_predictions(predictions: pd.DataFrame, model: dict) -> tuple[dict[str, float], pd.DataFrame]:
    features = list(model["feature_columns"])
    targets = list(model["target_names"])
    x = _design_matrix(predictions, features)
    corrected = predictions.copy()

    maes = []
    mses = []
    metrics: dict[str, float] = {"num_samples": float(len(corrected))}
    for target in targets:
        coef = np.asarray(model["coefficients"][target], dtype=float)
        corrected_column = f"corrected_{target}"
        next_column = f"next_x_{target}"
        corrected[corrected_column] = x @ coef
        error = corrected[corrected_column].astype(float) - corrected[next_column].astype(float)
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(np.square(error)))
        metrics[f"{target}_mae"] = mae
        metrics[f"{target}_mse"] = mse
        maes.append(mae)
        mses.append(mse)

    metrics["mean_mae"] = float(np.mean(maes))
    metrics["mean_mse"] = float(np.mean(mses))
    return metrics, corrected


def save_residual_model(model: dict, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return output


def load_residual_model(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate a linear residual correction for ChengduPhysics predictions.")
    parser.add_argument("--train_predictions_csv", required=True)
    parser.add_argument("--test_predictions_csv", required=True)
    parser.add_argument("--save_dir", default="results/chengdu_agri_greenhouse_001/physics_residual_correction/")
    args = parser.parse_args()

    train_predictions = pd.read_csv(args.train_predictions_csv)
    test_predictions = pd.read_csv(args.test_predictions_csv)
    model = fit_linear_residual_corrector(train_predictions)
    metrics, corrected = evaluate_corrected_predictions(test_predictions, model)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_residual_model(model, save_dir / "linear_residual_model.json")
    corrected.to_csv(save_dir / "corrected_test_predictions.csv", index=False)
    (save_dir / "linear_residual_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
