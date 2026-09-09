from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.reports.chengdu_residual_correction import (
    FEATURE_SETS,
    apply_ridge_residual_corrector,
    calibrate_residual_uncertainty,
    fit_ridge_residual_corrector,
)
from experiments.reports.evaluate_chengdu_multistep import (
    build_physics_predictor,
    evaluate_multistep_rollouts,
)
from experiments.reports.evaluate_chengdu_physics import (
    evaluate_one_step_predictions,
    load_parameter_vector,
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reproducibility_code_hashes() -> dict[str, str]:
    reports = Path(__file__).parent
    return {
        "selector": _sha256(Path(__file__)),
        "residual_correction": _sha256(reports / "chengdu_residual_correction.py"),
        "multistep_evaluator": _sha256(reports / "evaluate_chengdu_multistep.py"),
    }


def chronological_fit_calibration_split(
    frame: pd.DataFrame, *, fit_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be inside (0, 1)")
    if len(frame) < 2:
        raise ValueError("At least two rows are required for blocked splitting")
    if "timestamp" in frame.columns:
        timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
        if not timestamp.is_monotonic_increasing:
            raise ValueError("Training predictions must be chronological")
    split = min(len(frame) - 1, max(1, int(np.floor(len(frame) * fit_fraction))))
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def resolve_fit_sample_weights(
    frame: pd.DataFrame,
    column: str | None,
    *,
    heat_regime_weights: dict[str, float] | None = None,
) -> np.ndarray | None:
    if column is None and heat_regime_weights is None:
        return None
    weights = np.ones(len(frame), dtype=float)
    if column is not None:
        if column not in frame.columns:
            raise ValueError(f"Missing configured sample-weight column: {column}")
        weights *= pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if heat_regime_weights is not None:
        if "target_heat_regime" not in frame.columns:
            raise ValueError("Missing target_heat_regime for configured heat weights")
        mapped = frame["target_heat_regime"].map(heat_regime_weights)
        if mapped.isna().any():
            unknown = sorted(frame.loc[mapped.isna(), "target_heat_regime"].astype(str).unique())
            raise ValueError(f"Missing configured heat-regime weights: {unknown}")
        weights *= pd.to_numeric(mapped, errors="raise").to_numpy(dtype=float)
    return weights


def rank_base_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    eligible = [item for item in candidates if np.isfinite(float(item["calibration_score"]))]
    eligible.sort(key=lambda item: (float(item["calibration_score"]), int(item["candidate_id"])))
    if not eligible:
        raise RuntimeError("No finite grey-box base candidate")
    return eligible[:top_k]


def build_validation_candidates(
    base_candidates: list[dict[str, Any]], gains: list[float]
) -> list[dict[str, Any]]:
    if not base_candidates:
        raise ValueError("At least one base candidate is required")
    candidates = []
    for base in base_candidates:
        for gain in gains:
            if not 0.0 < float(gain) <= 1.0:
                raise ValueError("Hybrid validation gains must be inside (0, 1]")
            model = deepcopy(base["model"])
            model["gain"] = float(gain)
            candidates.append(
                {
                    "candidate_kind": "hybrid",
                    "base_candidate_id": int(base["candidate_id"]),
                    "gain": float(gain),
                    "model": model,
                }
            )
    fallback_model = deepcopy(base_candidates[0]["model"])
    fallback_model["gain"] = 0.0
    candidates.append(
        {
            "candidate_kind": "physical_fallback",
            "base_candidate_id": int(base_candidates[0]["candidate_id"]),
            "gain": 0.0,
            "model": fallback_model,
        }
    )
    return candidates


def select_validation_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in candidates if bool(item.get("physical_envelope_pass", False))]
    if not eligible:
        raise RuntimeError("No grey-box candidate passed the validation physical envelope")
    return min(eligible, key=lambda item: (float(item["score"]), int(item["candidate_id"])))


def _normalized_one_step_score(frame: pd.DataFrame, model: dict) -> float:
    corrected = apply_ridge_residual_corrector(frame, model)
    temperature_mae = float(
        np.mean(np.abs(corrected["pred_air_temperature"] - frame["next_x_air_temperature"]))
    )
    humidity_mae = float(
        np.mean(np.abs(corrected["pred_relative_humidity"] - frame["next_x_relative_humidity"]))
    )
    return float(np.mean([temperature_mae / 2.0, humidity_mae / 10.0]))


def _multistep_score(metrics: dict[str, float], horizons: list[int]) -> float:
    terms = []
    for horizon in horizons:
        terms.extend(
            [
                float(metrics[f"horizon_{horizon}_air_temperature_mae"]) / 2.0,
                float(metrics[f"horizon_{horizon}_relative_humidity_mae"]) / 10.0,
            ]
        )
    return float(np.mean(terms))


def select_from_config(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    output_path = Path(output_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_path = Path(config["train_csv"])
    validation_path = Path(config["validation_csv"])
    params_path = Path(config["v4_params_json"])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    params = load_parameter_vector(params_path)
    if params is None or len(params) != 225:
        raise ValueError("A 225-value V4 parameter artifact is required")

    _one_step_metrics, train_predictions = evaluate_one_step_predictions(
        train,
        parameter_vector=params,
        model_backend="ChengduPhysicsV4",
    )
    fit, calibration = chronological_fit_calibration_split(
        train_predictions, fit_fraction=float(config["fit_fraction"])
    )
    sample_weight_column = config.get("sample_weight_column")
    heat_regime_weights = config.get("heat_regime_weights")
    fit_sample_weight = resolve_fit_sample_weights(
        fit,
        sample_weight_column,
        heat_regime_weights=heat_regime_weights,
    )
    base_candidates = []
    candidate_id = 0
    for feature_set_name in config["feature_sets"]:
        features = FEATURE_SETS[str(feature_set_name)]
        for alpha in config["alphas"]:
            model = fit_ridge_residual_corrector(
                fit,
                features,
                alpha=float(alpha),
                sample_weight=fit_sample_weight,
            )
            model["feature_set"] = str(feature_set_name)
            model["fit_rows"] = int(len(fit))
            model["sample_weight_column"] = sample_weight_column
            model["heat_regime_weights"] = heat_regime_weights
            calibrated = calibrate_residual_uncertainty(calibration, model, gain=1.0)
            base_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "feature_set": str(feature_set_name),
                    "alpha": float(alpha),
                    "calibration_score": _normalized_one_step_score(calibration, calibrated),
                    "model": calibrated,
                }
            )
            candidate_id += 1
    base_finalists = rank_base_candidates(base_candidates, int(config["base_top_k"]))
    validation_candidates = build_validation_candidates(
        base_finalists, [float(value) for value in config["gains"]]
    )
    horizons = [int(value) for value in config["validation_horizons"]]
    max_horizon = max(horizons)
    starts = list(
        range(
            0,
            max(0, len(validation) - max_horizon + 1),
            int(config["validation_stride"]),
        )
    )
    evaluated = []
    for validation_id, candidate in enumerate(validation_candidates):
        model = calibrate_residual_uncertainty(
            calibration, candidate["model"], gain=float(candidate["gain"])
        )
        predictor = build_physics_predictor(
            params,
            residual_model=model,
            model_backend="ChengduPhysicsV4",
        )
        metrics, _rollouts = evaluate_multistep_rollouts(
            validation,
            predictor=predictor,
            horizons=horizons,
            start_indices=starts,
        )
        score = _multistep_score(metrics, [24, 72]) if metrics["physical_envelope_pass"] else 1e9
        evaluated.append(
            {
                **candidate,
                "candidate_id": validation_id,
                "model": model,
                **metrics,
                "score": score,
                "residual_fallback_count": int(predictor.residual_fallback_count),
            }
        )
        print(f"validation candidate {validation_id + 1}/{len(validation_candidates)} score={score:.6f}")
    selected = select_validation_candidate(evaluated)
    selected_model = deepcopy(selected["model"])
    selected_model.update(
        {
            "candidate_kind": selected["candidate_kind"],
            "selection_audit": {
                "fit_role": "train_prefix_only",
                "calibration_role": "train_suffix_only",
                "validation_role": "model_selection_only",
                "test_role": "unopened",
                "fit_rows": int(len(fit)),
                "calibration_rows": int(len(calibration)),
                "sample_weight_column": sample_weight_column,
                "fit_weight_sum": None
                if fit_sample_weight is None
                else float(fit_sample_weight.sum()),
                "heat_regime_weights": heat_regime_weights,
                "base_candidates": base_candidates,
                "validation_candidates": evaluated,
                "selected_candidate_id": int(selected["candidate_id"]),
            },
            "hashes": {
                "train_csv": _sha256(train_path),
                "validation_csv": _sha256(validation_path),
                "v4_params_json": _sha256(params_path),
                "config": _sha256(config_path),
                "code": reproducibility_code_hashes(),
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_predictions.to_csv(output_path.parent / "train_one_step_predictions.csv", index=False)
    output_path.write_text(json.dumps(selected_model, indent=2, allow_nan=False), encoding="utf-8")
    return selected_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a leakage-safe V4 grey-box residual model.")
    parser.add_argument("--config", default="configs/models/chengdu_v4_greybox_residual.yml")
    parser.add_argument(
        "--output",
        default="results/chengdu_agri_greenhouse_001/physics_v4_greybox_residual/selected_model.json",
    )
    args = parser.parse_args()
    model = select_from_config(args.config, args.output)
    print(
        json.dumps(
            {
                "feature_set": model.get("feature_set"),
                "alpha": model.get("alpha"),
                "gain": model.get("gain"),
                "candidate_kind": model.get("candidate_kind"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
