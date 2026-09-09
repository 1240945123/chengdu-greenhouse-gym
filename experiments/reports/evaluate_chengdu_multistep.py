from __future__ import annotations

import argparse
from importlib import import_module
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiments.reports.chengdu_residual_correction import apply_ridge_residual_corrector, load_residual_model
from experiments.reports.evaluate_chengdu_physics import (
    CONTROL_SCHEMAS,
    load_parameter_vector,
    predict_one_step,
    row_to_model_inputs,
)
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres, vaporPres2rh


TARGETS = ["air_temperature", "relative_humidity"]


def _apply_residual_model(
    row: pd.Series,
    prediction: dict[str, float],
    residual_model: dict | None,
    *,
    gain_scale: float = 1.0,
) -> dict[str, float]:
    if residual_model is None:
        return prediction
    if residual_model.get("schema_version") == "ridge_residual_v1":
        if not 0.0 <= gain_scale <= 1.0:
            raise ValueError("Residual gain scale must be between zero and one")
        scaled_model = dict(residual_model)
        scaled_model["gain"] = float(residual_model.get("gain", 1.0)) * float(gain_scale)
        values = pd.DataFrame([{**row.to_dict(), **prediction}])
        try:
            corrected_frame = apply_ridge_residual_corrector(values, scaled_model)
        except (KeyError, TypeError, ValueError, OverflowError):
            return {**prediction, "residual_fallback": True}
        corrected = dict(prediction)
        for target in residual_model["target_names"]:
            corrected[f"pred_{target}"] = float(corrected_frame.iloc[0][f"pred_{target}"])
        corrected["residual_fallback"] = bool(corrected_frame.iloc[0]["residual_fallback"])
        return corrected
    values = {**row.to_dict(), **prediction}
    features = [1.0] + [float(values[column]) for column in residual_model["feature_columns"]]
    x = np.asarray(features, dtype=float)
    corrected = dict(prediction)
    for target in residual_model["target_names"]:
        coef = np.asarray(residual_model["coefficients"][target], dtype=float)
        corrected[f"pred_{target}"] = float(x @ coef)
    return corrected


def evaluate_multistep_rollouts(
    trajectories: pd.DataFrame,
    predictor: Callable[[pd.Series], dict[str, float]],
    horizons: list[int] | None = None,
    start_indices: list[int] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    horizon_values = horizons or [1, 6, 12, 24]
    max_horizon = max(horizon_values)
    if start_indices is None:
        start_indices = list(range(0, max(0, len(trajectories) - max_horizon + 1)))

    rows = []
    for start in start_indices:
        state = (
            predictor.initialize_state(trajectories.iloc[start])
            if hasattr(predictor, "initialize_state")
            else trajectories.iloc[start].copy()
        )
        for step in range(1, max_horizon + 1):
            source_idx = start + step - 1
            if source_idx >= len(trajectories):
                break
            if step > 1 and {"timestamp", "next_timestamp"}.issubset(trajectories.columns):
                previous_next = pd.Timestamp(trajectories.iloc[source_idx - 1]["next_timestamp"])
                current_time = pd.Timestamp(trajectories.iloc[source_idx]["timestamp"])
                if previous_next != current_time:
                    break
            source = trajectories.iloc[source_idx].copy()
            if hasattr(predictor, "predict_state"):
                prediction, state = predictor.predict_state(state, source)
            else:
                source["x_air_temperature"] = state["x_air_temperature"]
                source["x_relative_humidity"] = state["x_relative_humidity"]
                prediction = predictor(source)
                state["x_air_temperature"] = prediction["pred_air_temperature"]
                state["x_relative_humidity"] = prediction["pred_relative_humidity"]
            truth = trajectories.iloc[source_idx]
            rows.append(
                {
                    "start_index": start,
                    "horizon": step,
                    "pred_air_temperature": prediction["pred_air_temperature"],
                    "pred_relative_humidity": prediction["pred_relative_humidity"],
                    "true_air_temperature": truth["next_x_air_temperature"],
                    "true_relative_humidity": truth["next_x_relative_humidity"],
                }
            )

    rollouts = pd.DataFrame(rows)
    metrics: dict[str, float] = {"num_rollout_rows": float(len(rollouts)), "num_start_indices": float(len(start_indices))}
    for horizon in horizon_values:
        horizon_rows = rollouts[rollouts["horizon"] == horizon]
        if horizon_rows.empty:
            continue
        maes = []
        mses = []
        for target in TARGETS:
            error = horizon_rows[f"pred_{target}"].astype(float) - horizon_rows[f"true_{target}"].astype(float)
            mae = float(np.mean(np.abs(error)))
            mse = float(np.mean(np.square(error)))
            metrics[f"horizon_{horizon}_{target}_mae"] = mae
            metrics[f"horizon_{horizon}_{target}_mse"] = mse
            metrics[f"horizon_{horizon}_{target}_rmse"] = float(np.sqrt(mse))
            metrics[f"horizon_{horizon}_{target}_bias"] = float(np.mean(error))
            metrics[f"horizon_{horizon}_{target}_pred_min"] = float(horizon_rows[f"pred_{target}"].min())
            metrics[f"horizon_{horizon}_{target}_pred_max"] = float(horizon_rows[f"pred_{target}"].max())
            metrics[f"horizon_{horizon}_{target}_true_min"] = float(horizon_rows[f"true_{target}"].min())
            metrics[f"horizon_{horizon}_{target}_true_max"] = float(horizon_rows[f"true_{target}"].max())
            maes.append(mae)
            mses.append(mse)
        metrics[f"horizon_{horizon}_mean_mae"] = float(np.mean(maes))
        metrics[f"horizon_{horizon}_mean_mse"] = float(np.mean(mses))
    if rollouts.empty:
        metrics["all_predictions_finite"] = False
        metrics["physical_envelope_pass"] = False
    else:
        predicted = rollouts[["pred_air_temperature", "pred_relative_humidity"]].to_numpy(dtype=float)
        metrics["all_predictions_finite"] = bool(np.isfinite(predicted).all())
        metrics["physical_envelope_pass"] = bool(
            metrics["all_predictions_finite"]
            and rollouts["pred_air_temperature"].between(-10.0, 60.0).all()
            and rollouts["pred_relative_humidity"].between(0.0, 100.0).all()
        )
    return metrics, rollouts


def persistence_predictor(row: pd.Series) -> dict[str, float]:
    return {
        "pred_air_temperature": float(row["x_air_temperature"]),
        "pred_relative_humidity": float(row["x_relative_humidity"]),
    }


def outdoor_following_predictor(row: pd.Series) -> dict[str, float]:
    return {
        "pred_air_temperature": float(row["d_air_temperature"]),
        "pred_relative_humidity": float(row.get("d_relative_humidity", row["x_relative_humidity"])),
    }


class StatefulPhysicsPredictor:
    def __init__(
        self,
        parameter_vector: np.ndarray,
        residual_model: dict | None = None,
        dt: float = 3600.0,
        model_backend: str = "ChengduPhysics",
    ):
        self.params = np.asarray(parameter_vector, dtype=float)
        self.residual_model = residual_model
        self.model_backend = model_backend
        self.residual_fallback_count = 0
        self.residual_gain_scale = min(1.0, float(dt) / 3600.0)
        define_model = import_module(f"glassgym.models.{model_backend}.utils").define_model
        self.integrator = define_model(
            nx=28,
            nu=len(CONTROL_SCHEMAS[model_backend]),
            nd=10,
            n_params=len(self.params),
            dt=dt,
        )

    def initialize_state(self, row: pd.Series) -> np.ndarray:
        state, _u, _d = row_to_model_inputs(row, model_backend=self.model_backend)
        return state

    def predict_state(self, state: np.ndarray, row: pd.Series) -> tuple[dict[str, float], np.ndarray]:
        import casadi as ca

        _row_state, control, disturbance = row_to_model_inputs(
            row,
            model_backend=self.model_backend,
        )
        p_dyn = ca.vertcat(ca.DM(disturbance), ca.DM(self.params))
        result = self.integrator(x0=ca.DM(state), u=ca.DM(control), p=p_dyn)
        next_state = result["xf"].full().flatten()
        prediction = {
            "pred_air_temperature": float(next_state[2]),
            "pred_relative_humidity": float(vaporPres2rh(next_state[2], next_state[15])),
            "pred_co2_concentration": float(next_state[0]),
        }
        prediction = _apply_residual_model(
            row,
            prediction,
            self.residual_model,
            gain_scale=self.residual_gain_scale,
        )
        residual_fallback = bool(prediction.pop("residual_fallback", False))
        if residual_fallback:
            self.residual_fallback_count += 1
        if self.residual_model is not None:
            next_state[2] = prediction["pred_air_temperature"]
            next_state[15] = vaporDens2pres(
                next_state[2],
                rh2vaporDens(next_state[2], prediction["pred_relative_humidity"]),
            )
        return prediction, next_state

    def __call__(self, row: pd.Series) -> dict[str, float]:
        prediction, _state = self.predict_state(self.initialize_state(row), row)
        return prediction


def build_physics_predictor(
    parameter_vector: np.ndarray,
    residual_model: dict | None = None,
    dt: float = 3600.0,
    model_backend: str = "ChengduPhysics",
):
    return StatefulPhysicsPredictor(
        parameter_vector,
        residual_model=residual_model,
        dt=dt,
        model_backend=model_backend,
    )


def _parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-step Chengdu greenhouse temperature/RH rollouts.")
    parser.add_argument("--trajectory_csv", default="data/processed/chengdu_agri/greenhouse_001/trajectories/test.csv")
    parser.add_argument("--params_json", required=True)
    parser.add_argument("--residual_model_json", default=None)
    parser.add_argument("--model_backend", default="ChengduPhysics")
    parser.add_argument("--horizons", default="1,6,12,24")
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--save_dir", default="results/chengdu_agri_greenhouse_001/multistep/")
    args = parser.parse_args()

    trajectories = pd.read_csv(args.trajectory_csv)
    horizons = _parse_int_list(args.horizons)
    max_horizon = max(horizons)
    start_indices = list(range(0, max(0, len(trajectories) - max_horizon), max(1, args.stride)))
    params = load_parameter_vector(args.params_json)
    residual_model = load_residual_model(args.residual_model_json) if args.residual_model_json else None
    predictor = build_physics_predictor(
        params,
        residual_model=residual_model,
        model_backend=args.model_backend,
    )
    metrics, rollouts = evaluate_multistep_rollouts(
        trajectories,
        predictor=predictor,
        horizons=horizons,
        start_indices=start_indices,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "chengdu_multistep_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rollouts.to_csv(save_dir / "chengdu_multistep_rollouts.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
