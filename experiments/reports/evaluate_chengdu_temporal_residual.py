from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.predictors.train_chengdu_temporal_residual import build_temporal_model
from experiments.reports.chengdu_residual_correction import (
    build_residual_features,
    load_residual_model,
)
from experiments.reports.evaluate_chengdu_multistep import (
    build_physics_predictor,
    evaluate_multistep_rollouts,
    persistence_predictor,
)
from experiments.reports.evaluate_chengdu_physics import (
    evaluate_one_step_predictions,
    load_parameter_vector,
)
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres
from glassgym.models.residual.temporal import ArrayStandardizer


TARGETS = ("air_temperature", "relative_humidity")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_warmup_indices(
    timestamps: pd.Series,
    *,
    origin: int,
    lookback: int,
) -> list[int]:
    parsed = pd.to_datetime(timestamps, errors="raise")
    if origin < 0 or origin >= len(parsed):
        raise IndexError("origin is outside timestamp series")
    indices: list[int] = []
    current = int(origin)
    while current > 0 and len(indices) < int(lookback) - 1:
        previous = current - 1
        if parsed.iloc[current] - parsed.iloc[previous] != pd.Timedelta(hours=1):
            break
        indices.append(previous)
        current = previous
    return sorted(indices)


def write_temperature_humidity_to_physical_state(
    state: np.ndarray,
    *,
    air_temperature: float,
    relative_humidity: float,
) -> np.ndarray:
    state[2] = float(air_temperature)
    state[15] = vaporDens2pres(
        state[2],
        rh2vaporDens(state[2], float(relative_humidity)),
    )
    return state


class TemporalResidualEnsemble:
    def __init__(self, artifact_paths: list[str | Path]):
        if not artifact_paths:
            raise ValueError("Temporal ensemble requires at least one artifact")
        self.models = []
        self.artifact_paths = [Path(path) for path in artifact_paths]
        reference: dict[str, Any] | None = None
        for path in self.artifact_paths:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            if artifact.get("schema_version") != "chengdu_temporal_residual_v1":
                raise ValueError(f"Unsupported temporal artifact: {path}")
            if reference is None:
                reference = artifact
            else:
                for field in ("feature_columns", "lookback", "input_scaler", "target_scaler"):
                    if artifact[field] != reference[field]:
                        raise ValueError(f"Temporal ensemble artifacts disagree on {field}")
            model = build_temporal_model(
                artifact["architecture"],
                input_dim=int(artifact["model_config"]["input_dim"]),
                hidden_dim=int(artifact["model_config"]["hidden_dim"]),
            )
            model.load_state_dict(artifact["model_state_dict"])
            model.eval()
            self.models.append(model)
        assert reference is not None
        self.feature_columns = list(reference["feature_columns"])
        self.lookback = int(reference["lookback"])
        self.input_scaler = ArrayStandardizer.from_dict(reference["input_scaler"])
        self.target_scaler = ArrayStandardizer.from_dict(reference["target_scaler"])

    def predict_residual(self, context: pd.DataFrame) -> np.ndarray:
        if len(context) != self.lookback:
            raise ValueError(f"Expected {self.lookback} context rows, got {len(context)}")
        features = build_residual_features(context, self.feature_columns).to_numpy(dtype=np.float32)
        inputs = torch.as_tensor(
            self.input_scaler.transform(features)[None, :, :],
            dtype=torch.float32,
        )
        predictions = []
        with torch.no_grad():
            for model in self.models:
                normalized = model(inputs).cpu().numpy()
                predictions.append(self.target_scaler.inverse_transform(normalized)[0])
        return np.mean(np.asarray(predictions, dtype=np.float32), axis=0)


def _rollout_metrics(
    rollouts: pd.DataFrame,
    *,
    horizons: list[int],
    requested_starts: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_rollout_rows": int(len(rollouts)),
        "num_requested_starts": int(requested_starts),
    }
    for horizon in horizons:
        rows = rollouts[rollouts["horizon"] == horizon]
        metrics[f"horizon_{horizon}_completed_starts"] = int(len(rows))
        for target in TARGETS:
            error = rows[f"pred_{target}"].to_numpy(dtype=float) - rows[f"true_{target}"].to_numpy(dtype=float)
            if len(error) == 0:
                continue
            metrics[f"horizon_{horizon}_{target}_mae"] = float(np.mean(np.abs(error)))
            metrics[f"horizon_{horizon}_{target}_rmse"] = float(np.sqrt(np.mean(np.square(error))))
            metrics[f"horizon_{horizon}_{target}_bias"] = float(np.mean(error))
    if rollouts.empty:
        metrics["all_predictions_finite"] = False
        metrics["physical_envelope_pass"] = False
    else:
        predictions = rollouts[["pred_air_temperature", "pred_relative_humidity"]].to_numpy(dtype=float)
        metrics["all_predictions_finite"] = bool(np.isfinite(predictions).all())
        metrics["physical_envelope_pass"] = bool(
            metrics["all_predictions_finite"]
            and rollouts["pred_air_temperature"].between(-10.0, 60.0).all()
            and rollouts["pred_relative_humidity"].between(0.0, 100.0).all()
        )
    return metrics


def evaluate_temporal_rollouts(
    trajectories: pd.DataFrame,
    *,
    one_step_v4_predictions: pd.DataFrame,
    parameter_vector: np.ndarray,
    ensemble: TemporalResidualEnsemble,
    horizons: list[int],
    start_indices: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    max_horizon = max(horizons)
    rows: list[dict[str, float | int]] = []
    timestamps = pd.to_datetime(trajectories["timestamp"], errors="raise")
    predictor = build_physics_predictor(parameter_vector, model_backend="ChengduPhysicsV4")

    for start in start_indices:
        warmup = causal_warmup_indices(timestamps, origin=start, lookback=ensemble.lookback)
        if len(warmup) != ensemble.lookback - 1:
            continue
        context_rows = [one_step_v4_predictions.iloc[index].to_dict() for index in warmup]
        state = predictor.initialize_state(trajectories.iloc[start])
        for step in range(1, max_horizon + 1):
            source_index = start + step - 1
            if source_index >= len(trajectories):
                break
            if step > 1 and timestamps.iloc[source_index] - timestamps.iloc[source_index - 1] != pd.Timedelta(hours=1):
                break
            source = trajectories.iloc[source_index]
            physical_prediction, next_state = predictor.predict_state(state, source)
            token = {**source.to_dict(), **physical_prediction}
            context_rows.append(token)
            context_rows = context_rows[-ensemble.lookback :]
            residual = ensemble.predict_residual(pd.DataFrame(context_rows))
            temperature = float(np.clip(physical_prediction["pred_air_temperature"] + residual[0], -10.0, 60.0))
            humidity = float(np.clip(physical_prediction["pred_relative_humidity"] + residual[1], 0.0, 100.0))
            state = write_temperature_humidity_to_physical_state(
                next_state,
                air_temperature=temperature,
                relative_humidity=humidity,
            )
            rows.append(
                {
                    "start_index": int(start),
                    "horizon": int(step),
                    "pred_air_temperature": temperature,
                    "pred_relative_humidity": humidity,
                    "true_air_temperature": float(source["next_x_air_temperature"]),
                    "true_relative_humidity": float(source["next_x_relative_humidity"]),
                }
            )
    rollouts = pd.DataFrame(rows)
    return _rollout_metrics(rollouts, horizons=horizons, requested_starts=len(start_indices)), rollouts


def _augment_baseline_metrics(
    metrics: dict[str, Any],
    rollouts: pd.DataFrame,
    *,
    horizons: list[int],
    requested_starts: int,
) -> dict[str, Any]:
    standardized = _rollout_metrics(rollouts, horizons=horizons, requested_starts=requested_starts)
    return {**metrics, **standardized}


def evaluate_from_config(
    config_path: str | Path,
    artifact_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path)
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validation_path = Path(config["validation_csv"])
    params_path = Path(config["v4_params_json"])
    ridge_path = Path(config["ridge_model_json"])
    trajectories = pd.read_csv(validation_path)
    params = load_parameter_vector(params_path)
    if params is None:
        raise ValueError("Missing V4 parameter vector")
    horizons = [int(value) for value in config["evaluation"]["horizons"]]
    lookback = int(config["lookback"])
    max_horizon = max(horizons)
    stride = int(config["evaluation"]["stride"])
    starts = list(range(lookback - 1, max(lookback - 1, len(trajectories) - max_horizon + 1), stride))

    _one_step_metrics, one_step_predictions = evaluate_one_step_predictions(
        trajectories,
        parameter_vector=params,
        model_backend="ChengduPhysicsV4",
    )
    results: dict[str, Any] = {}
    rollouts_to_save: list[pd.DataFrame] = []

    baseline_predictors = {
        "persistence": persistence_predictor,
        "v4_physics": build_physics_predictor(params, model_backend="ChengduPhysicsV4"),
        "ridge_residual": build_physics_predictor(
            params,
            residual_model=load_residual_model(ridge_path),
            model_backend="ChengduPhysicsV4",
        ),
    }
    for name, predictor in baseline_predictors.items():
        metrics, rollouts = evaluate_multistep_rollouts(
            trajectories,
            predictor=predictor,
            horizons=horizons,
            start_indices=starts,
        )
        results[name] = _augment_baseline_metrics(
            metrics,
            rollouts,
            horizons=horizons,
            requested_starts=len(starts),
        )
        rollouts["model"] = name
        rollouts_to_save.append(rollouts)

    training_summary = json.loads((artifact_dir / "training_summary.json").read_text(encoding="utf-8"))
    for record in training_summary["candidates"]:
        name = f"{record['architecture']}_seed_{record['seed']}"
        ensemble = TemporalResidualEnsemble([artifact_dir / record["artifact"]])
        metrics, rollouts = evaluate_temporal_rollouts(
            trajectories,
            one_step_v4_predictions=one_step_predictions,
            parameter_vector=params,
            ensemble=ensemble,
            horizons=horizons,
            start_indices=starts,
        )
        results[name] = {**metrics, "parameter_count": record["parameter_count"]}
        rollouts["model"] = name
        rollouts_to_save.append(rollouts)

    for architecture in config["architectures"]:
        records = [
            record for record in training_summary["candidates"]
            if record["architecture"] == architecture
        ]
        name = f"{architecture}_ensemble"
        ensemble = TemporalResidualEnsemble([artifact_dir / record["artifact"] for record in records])
        metrics, rollouts = evaluate_temporal_rollouts(
            trajectories,
            one_step_v4_predictions=one_step_predictions,
            parameter_vector=params,
            ensemble=ensemble,
            horizons=horizons,
            start_indices=starts,
        )
        results[name] = {**metrics, "ensemble_seeds": [record["seed"] for record in records]}
        rollouts["model"] = name
        rollouts_to_save.append(rollouts)

    selected_name = f"{training_summary['selected']['architecture']}_ensemble"
    selected = results[selected_name]
    ridge = results["ridge_residual"]
    comparisons = {}
    for horizon in (24, 72):
        for target in TARGETS:
            key = f"horizon_{horizon}_{target}_mae"
            comparisons[key] = bool(selected[key] < ridge[key])
    gate = {
        "selected_model": selected_name,
        "physical_envelope_pass": bool(selected["physical_envelope_pass"]),
        "beats_ridge_each_target_at_24_and_72": bool(all(comparisons.values())),
        "individual_comparisons": comparisons,
    }
    gate["pass"] = bool(gate["physical_envelope_pass"] and gate["beats_ridge_each_target_at_24_and_72"])
    report = {
        "schema_version": "chengdu_temporal_residual_validation_v1",
        "data_role": "validation_reporting_only",
        "test_role": "unused_already_opened",
        "common_start_indices": starts,
        "metrics": results,
        "gate": gate,
        "hashes": {
            "validation_csv": _sha256(validation_path),
            "v4_params_json": _sha256(params_path),
            "ridge_model_json": _sha256(ridge_path),
            "training_summary": _sha256(artifact_dir / "training_summary.json"),
            "evaluation_code": _sha256(Path(__file__)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.concat(rollouts_to_save, ignore_index=True).to_csv(
        output_dir / "validation_rollouts.csv",
        index=False,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chengdu temporal residual candidates.")
    parser.add_argument("--config", default="configs/models/chengdu_temporal_residual.yml")
    parser.add_argument(
        "--artifact_dir",
        default="results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual",
    )
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual",
    )
    args = parser.parse_args()
    report = evaluate_from_config(args.config, args.artifact_dir, args.output_dir)
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
