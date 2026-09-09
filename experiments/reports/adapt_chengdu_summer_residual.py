from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.reports.chengdu_residual_correction import (
    FEATURE_SETS,
    apply_ridge_residual_corrector,
    calibrate_residual_uncertainty,
    fit_ridge_residual_corrector,
    save_residual_model,
)
from experiments.reports.evaluate_chengdu_physics import (
    evaluate_one_step_predictions,
    load_parameter_vector,
)


ADAPTATION_CONTRACT = {
    "feature_set": "thermal_dynamics",
    "alpha": 0.1,
    "gain": 1.0,
    "heat_regime_weights": {"normal": 1.0, "high": 5.0, "extreme": 5.0},
}
HUMIDITY_GAIN_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_adaptation_weights(frame: pd.DataFrame) -> np.ndarray:
    if "target_heat_regime" not in frame.columns:
        raise ValueError("Missing target_heat_regime for online adaptation weights")
    weights = frame["target_heat_regime"].map(
        ADAPTATION_CONTRACT["heat_regime_weights"]
    )
    if weights.isna().any():
        unknown = sorted(frame.loc[weights.isna(), "target_heat_regime"].astype(str).unique())
        raise ValueError(f"Unknown adaptation heat regimes: {unknown}")
    return weights.to_numpy(dtype=float)


def select_calibration_target_gain(
    calibration: pd.DataFrame,
    model: dict,
    *,
    target: str,
    candidates: tuple[float, ...],
) -> dict[str, object]:
    if target not in model["target_names"]:
        raise ValueError(f"Unknown residual target for gain selection: {target}")
    target_column = f"next_x_{target}"
    if target_column not in calibration.columns:
        raise ValueError(f"Missing gain calibration target: {target_column}")
    records = []
    for gain in candidates:
        value = float(gain)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Calibration target gains must be inside [0, 1]")
        candidate_model = deepcopy(model)
        candidate_model["target_gains"] = {
            **candidate_model.get("target_gains", {}),
            target: value,
        }
        corrected = apply_ridge_residual_corrector(
            calibration, candidate_model, gain=1.0
        )
        error = (
            corrected[f"pred_{target}"].to_numpy(dtype=float)
            - calibration[target_column].to_numpy(dtype=float)
        )
        records.append({"gain": value, "mae": float(np.mean(np.abs(error)))})
    selected = min(records, key=lambda item: (item["mae"], item["gain"]))
    return {
        "target": target,
        "selected_gain": float(selected["gain"]),
        "selection_metric": "one_step_mae",
        "candidates": records,
    }


def _ordered_by_target(frame: pd.DataFrame, role: str) -> pd.DataFrame:
    if "next_timestamp" not in frame.columns:
        raise ValueError(f"Missing next_timestamp in {role}")
    result = frame.copy()
    result["next_timestamp"] = pd.to_datetime(result["next_timestamp"], errors="raise")
    result = result.sort_values("next_timestamp", kind="stable").reset_index(drop=True)
    if result["next_timestamp"].duplicated().any():
        raise ValueError(f"Duplicate target timestamps in {role}")
    return result


def build_online_roles(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    replay: pd.DataFrame,
    *,
    fit_end: str | pd.Timestamp,
    evaluation_start: str | pd.Timestamp,
) -> Mapping[str, pd.DataFrame]:
    fit_boundary = pd.Timestamp(fit_end)
    evaluation_boundary = pd.Timestamp(evaluation_start)
    if fit_boundary >= evaluation_boundary:
        raise ValueError("fit_end must precede evaluation_start")

    train = _ordered_by_target(train, "train")
    validation = _ordered_by_target(validation, "validation")
    replay = _ordered_by_target(replay, "replay")
    base = pd.concat([train, validation], ignore_index=True).sort_values(
        "next_timestamp", kind="stable"
    )
    if base.empty or base["next_timestamp"].ge(fit_boundary).any():
        raise ValueError("base history crosses online fit boundary")

    replay_fit = replay[replay["next_timestamp"].lt(fit_boundary)]
    calibration = replay[
        replay["next_timestamp"].ge(fit_boundary)
        & replay["next_timestamp"].lt(evaluation_boundary)
    ].copy()
    evaluation = replay[replay["next_timestamp"].ge(evaluation_boundary)].copy()
    fit = pd.concat([base, replay_fit], ignore_index=True).sort_values(
        "next_timestamp", kind="stable"
    )
    roles = {
        "fit": fit.reset_index(drop=True),
        "calibration": calibration.reset_index(drop=True),
        "evaluation": evaluation.reset_index(drop=True),
    }
    if any(frame.empty for frame in roles.values()):
        raise ValueError("Online fit, calibration, and evaluation roles must be non-empty")
    return roles


def build_role_audit(roles: Mapping[str, pd.DataFrame]) -> dict[str, object]:
    expected = ("fit", "calibration", "evaluation")
    if any(role not in roles or roles[role].empty for role in expected):
        raise ValueError("Role audit requires non-empty fit, calibration, and evaluation data")
    audit: dict[str, object] = {}
    for role in expected:
        target = pd.to_datetime(roles[role]["next_timestamp"], errors="raise")
        audit[role] = {
            "rows": int(len(target)),
            "target_start": target.min().isoformat(),
            "target_end": target.max().isoformat(),
        }
    audit["coefficient_fit_role"] = "fit_only"
    audit["calibration_updates_coefficients"] = False
    return audit


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_online_adaptation_bundle(
    train_path: str | Path,
    validation_path: str | Path,
    replay_path: str | Path,
    params_path: str | Path,
    output_dir: str | Path,
    *,
    fit_end: str = "2026-07-13 00:00",
    evaluation_start: str = "2026-07-14 00:00",
) -> dict[str, object]:
    paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "replay": Path(replay_path),
        "params": Path(params_path),
    }
    roles = build_online_roles(
        pd.read_csv(paths["train"]),
        pd.read_csv(paths["validation"]),
        pd.read_csv(paths["replay"]),
        fit_end=fit_end,
        evaluation_start=evaluation_start,
    )
    params = load_parameter_vector(paths["params"])
    if params is None or len(params) != 225:
        raise ValueError("Online adaptation requires a 225-value ChengduPhysicsV4 artifact")

    _fit_metrics, fit_predictions = evaluate_one_step_predictions(
        roles["fit"],
        parameter_vector=params,
        model_backend="ChengduPhysicsV4",
    )
    _calibration_metrics, calibration_predictions = evaluate_one_step_predictions(
        roles["calibration"],
        parameter_vector=params,
        model_backend="ChengduPhysicsV4",
    )
    model = fit_ridge_residual_corrector(
        fit_predictions,
        FEATURE_SETS[ADAPTATION_CONTRACT["feature_set"]],
        alpha=float(ADAPTATION_CONTRACT["alpha"]),
        sample_weight=build_adaptation_weights(fit_predictions),
    )
    model["feature_set"] = ADAPTATION_CONTRACT["feature_set"]
    model["heat_regime_weights"] = ADAPTATION_CONTRACT["heat_regime_weights"]
    humidity_gain_selection = select_calibration_target_gain(
        calibration_predictions,
        model,
        target="relative_humidity",
        candidates=HUMIDITY_GAIN_CANDIDATES,
    )
    model["target_gains"] = {
        "air_temperature": 1.0,
        "relative_humidity": humidity_gain_selection["selected_gain"],
    }
    model["target_gain_selection"] = humidity_gain_selection
    model = calibrate_residual_uncertainty(
        calibration_predictions,
        model,
        gain=float(ADAPTATION_CONTRACT["gain"]),
    )
    role_audit = build_role_audit(roles)
    model["online_adaptation_audit"] = {
        **role_audit,
        "fit_end": pd.Timestamp(fit_end).isoformat(),
        "evaluation_start": pd.Timestamp(evaluation_start).isoformat(),
        "evidence_scope": "retrospective_prequential_replay_not_independent_holdout",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_residual_model(model, output_dir / "adapted_model.json")
    evaluation_path = output_dir / "evaluation_trajectory.csv"
    calibration_path = output_dir / "calibration_one_step_predictions.csv"
    roles["evaluation"].to_csv(evaluation_path, index=False)
    calibration_predictions.to_csv(calibration_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": "chengdu_online_summer_adaptation_v1",
        "contract": ADAPTATION_CONTRACT,
        "target_gain_selection": humidity_gain_selection,
        "roles": role_audit,
        "fit_end": pd.Timestamp(fit_end).isoformat(),
        "evaluation_start": pd.Timestamp(evaluation_start).isoformat(),
        "fit_weight_sum": float(model["weight_sum"]),
        "source_paths": {name: path.as_posix() for name, path in paths.items()},
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "artifacts": {
            "model": model_path.as_posix(),
            "evaluation_trajectory": evaluation_path.as_posix(),
            "calibration_predictions": calibration_path.as_posix(),
        },
        "artifact_sha256": {
            "model": _sha256(model_path),
            "evaluation_trajectory": _sha256(evaluation_path),
            "calibration_predictions": _sha256(calibration_path),
        },
        "evidence_scope": "retrospective_prequential_replay_not_independent_holdout",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chronological Chengdu summer adaptation model.")
    parser.add_argument(
        "--train",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v4_quality_aware/train.csv",
    )
    parser.add_argument(
        "--validation",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v4_quality_aware/val.csv",
    )
    parser.add_argument(
        "--replay",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v4_quality_aware/test.csv",
    )
    parser.add_argument(
        "--params",
        default="results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json",
    )
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/physics_v4_summer_online_adaptation",
    )
    parser.add_argument("--fit_end", default="2026-07-13 00:00")
    parser.add_argument("--evaluation_start", default="2026-07-14 00:00")
    args = parser.parse_args()
    print(
        json.dumps(
            build_online_adaptation_bundle(
                args.train,
                args.validation,
                args.replay,
                args.params,
                args.output_dir,
                fit_end=args.fit_end,
                evaluation_start=args.evaluation_start,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
