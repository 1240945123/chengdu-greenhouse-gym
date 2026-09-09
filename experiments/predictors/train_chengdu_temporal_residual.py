from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from experiments.reports.chengdu_residual_correction import FEATURE_SETS
from experiments.reports.evaluate_chengdu_physics import (
    evaluate_one_step_predictions,
    load_parameter_vector,
)
from experiments.reports.select_chengdu_v4_greybox_residual import (
    chronological_fit_calibration_split,
)
from glassgym.models.residual.temporal import (
    ArrayStandardizer,
    LSTMResidualModel,
    TCNResidualModel,
    build_temporal_residual_sequences,
)


@dataclass
class TemporalTrainingResult:
    model: nn.Module
    best_epoch: int
    initial_calibration_mse: float
    best_calibration_mse: float
    calibration_predictions: np.ndarray
    epochs_ran: int


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def build_temporal_model(
    architecture: str,
    *,
    input_dim: int,
    hidden_dim: int,
) -> nn.Module:
    models = {"lstm": LSTMResidualModel, "tcn": TCNResidualModel}
    if architecture not in models:
        raise ValueError(f"Unsupported temporal residual architecture: {architecture}")
    return models[architecture](input_dim=input_dim, hidden_dim=hidden_dim)


def _predict_numpy(model: nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(inputs).cpu().numpy().astype(np.float32)


def train_single_temporal_model(
    *,
    architecture: str,
    fit_inputs: np.ndarray,
    fit_targets: np.ndarray,
    calibration_inputs: np.ndarray,
    calibration_targets: np.ndarray,
    hidden_dim: int,
    seed: int,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> TemporalTrainingResult:
    _set_seed(int(seed))
    fit_x = torch.as_tensor(fit_inputs, dtype=torch.float32)
    fit_y = torch.as_tensor(fit_targets, dtype=torch.float32)
    calibration_x = torch.as_tensor(calibration_inputs, dtype=torch.float32)
    calibration_y = torch.as_tensor(calibration_targets, dtype=torch.float32)
    if len(fit_x) == 0 or len(calibration_x) == 0:
        raise ValueError("Fit and calibration sequences must both be non-empty")

    model = build_temporal_model(
        architecture,
        input_dim=int(fit_x.shape[-1]),
        hidden_dim=int(hidden_dim),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(int(seed))

    model.eval()
    with torch.no_grad():
        initial_mse = float(criterion(model(calibration_x), calibration_y))
    best_mse = initial_mse
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    stale_epochs = 0
    epochs_ran = 0

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        order = torch.randperm(len(fit_x), generator=generator)
        for start in range(0, len(order), int(batch_size)):
            indices = order[start : start + int(batch_size)]
            optimizer.zero_grad()
            loss = criterion(model(fit_x[indices]), fit_y[indices])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            calibration_mse = float(criterion(model(calibration_x), calibration_y))
        epochs_ran = epoch
        if calibration_mse < best_mse - 1e-8:
            best_mse = calibration_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break

    model.load_state_dict(best_state)
    predictions = _predict_numpy(model, calibration_x)
    return TemporalTrainingResult(
        model=model,
        best_epoch=int(best_epoch),
        initial_calibration_mse=float(initial_mse),
        best_calibration_mse=float(best_mse),
        calibration_predictions=predictions,
        epochs_ran=int(epochs_ran),
    )


def select_architecture_by_median_score(
    records: list[dict[str, Any]], *, required_seeds: list[int]
) -> dict[str, Any]:
    if not records:
        raise ValueError("Architecture selection requires candidate records")
    required = sorted(int(seed) for seed in required_seeds)
    summaries = []
    for architecture in sorted({str(record["architecture"]) for record in records}):
        candidates = [record for record in records if record["architecture"] == architecture]
        seeds = sorted(int(record["seed"]) for record in candidates)
        if seeds != required:
            raise ValueError(f"Architecture {architecture} does not contain every required seed")
        scores = np.asarray([record["calibration_score"] for record in candidates], dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Architecture {architecture} has nonfinite calibration score")
        summaries.append(
            {
                "architecture": architecture,
                "median_calibration_score": float(np.median(scores)),
                "mean_calibration_score": float(np.mean(scores)),
                "seed_scores": {str(record["seed"]): float(record["calibration_score"]) for record in candidates},
            }
        )
    return min(
        summaries,
        key=lambda item: (item["median_calibration_score"], item["mean_calibration_score"], item["architecture"]),
    )


def _calibration_metrics(
    predictions_normalized: np.ndarray,
    targets_normalized: np.ndarray,
    target_scaler: ArrayStandardizer,
) -> dict[str, float]:
    predictions = target_scaler.inverse_transform(predictions_normalized)
    targets = target_scaler.inverse_transform(targets_normalized)
    errors = predictions - targets
    temperature_mae = float(np.mean(np.abs(errors[:, 0])))
    humidity_mae = float(np.mean(np.abs(errors[:, 1])))
    return {
        "air_temperature_residual_mae": temperature_mae,
        "relative_humidity_residual_mae": humidity_mae,
        "calibration_score": float(np.mean([temperature_mae / 2.0, humidity_mae / 10.0])),
    }


def train_from_config(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_path = Path(config["train_csv"])
    params_path = Path(config["v4_params_json"])
    trajectory = pd.read_csv(train_path)
    params = load_parameter_vector(params_path)
    if params is None or len(params) != 225:
        raise ValueError("A 225-value ChengduPhysicsV4 parameter artifact is required")

    _metrics, predictions = evaluate_one_step_predictions(
        trajectory,
        parameter_vector=params,
        model_backend="ChengduPhysicsV4",
    )
    fit_frame, calibration_frame = chronological_fit_calibration_split(
        predictions,
        fit_fraction=float(config["fit_fraction"]),
    )
    feature_columns = FEATURE_SETS[str(config["feature_set"])]
    lookback = int(config["lookback"])
    fit_sequences = build_temporal_residual_sequences(
        fit_frame,
        feature_columns=feature_columns,
        lookback=lookback,
    )
    calibration_sequences = build_temporal_residual_sequences(
        calibration_frame,
        feature_columns=feature_columns,
        lookback=lookback,
    )
    input_scaler = ArrayStandardizer.fit(fit_sequences.inputs)
    target_scaler = ArrayStandardizer.fit(fit_sequences.targets)
    fit_inputs = input_scaler.transform(fit_sequences.inputs)
    fit_targets = target_scaler.transform(fit_sequences.targets)
    calibration_inputs = input_scaler.transform(calibration_sequences.inputs)
    calibration_targets = target_scaler.transform(calibration_sequences.targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    training = config["training"]
    for architecture in config["architectures"]:
        for seed in config["seeds"]:
            result = train_single_temporal_model(
                architecture=str(architecture),
                fit_inputs=fit_inputs,
                fit_targets=fit_targets,
                calibration_inputs=calibration_inputs,
                calibration_targets=calibration_targets,
                hidden_dim=int(config["hidden_dim"]),
                seed=int(seed),
                learning_rate=float(training["learning_rate"]),
                batch_size=int(training["batch_size"]),
                max_epochs=int(training["max_epochs"]),
                patience=int(training["patience"]),
            )
            metrics = _calibration_metrics(
                result.calibration_predictions,
                calibration_targets,
                target_scaler,
            )
            artifact_name = f"{architecture}_seed_{seed}.pt"
            artifact_path = output_dir / artifact_name
            artifact = {
                "schema_version": "chengdu_temporal_residual_v1",
                "architecture": str(architecture),
                "seed": int(seed),
                "model_config": {
                    "input_dim": len(feature_columns),
                    "hidden_dim": int(config["hidden_dim"]),
                },
                "model_state_dict": result.model.state_dict(),
                "feature_columns": feature_columns,
                "lookback": lookback,
                "input_scaler": input_scaler.to_dict(),
                "target_scaler": target_scaler.to_dict(),
                "data_roles": {
                    "fit": "train_prefix_only",
                    "calibration": "train_suffix_only",
                    "validation": "reporting_only",
                    "test": "unused_already_opened",
                },
            }
            torch.save(artifact, artifact_path)
            record = {
                "architecture": str(architecture),
                "seed": int(seed),
                "artifact": artifact_name,
                "parameter_count": int(sum(parameter.numel() for parameter in result.model.parameters())),
                "fit_sequences": int(len(fit_sequences.inputs)),
                "calibration_sequences": int(len(calibration_sequences.inputs)),
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                "initial_calibration_mse_normalized": result.initial_calibration_mse,
                "best_calibration_mse_normalized": result.best_calibration_mse,
                **metrics,
            }
            records.append(record)
            print(json.dumps(record, allow_nan=False))

    selected = select_architecture_by_median_score(
        records,
        required_seeds=[int(seed) for seed in config["seeds"]],
    )
    summary = {
        "schema_version": "chengdu_temporal_residual_selection_v1",
        "selected": selected,
        "candidates": records,
        "feature_columns": feature_columns,
        "lookback": lookback,
        "fit_rows": int(len(fit_frame)),
        "calibration_rows": int(len(calibration_frame)),
        "hashes": {
            "train_csv": _sha256(train_path),
            "v4_params_json": _sha256(params_path),
            "config": _sha256(config_path),
            "training_code": _sha256(Path(__file__)),
        },
        "test_role": "unused_already_opened",
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "train_v4_one_step_predictions.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chengdu V4 temporal residual candidates.")
    parser.add_argument(
        "--config",
        default="configs/models/chengdu_temporal_residual.yml",
    )
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/physics_v4_temporal_residual",
    )
    args = parser.parse_args()
    summary = train_from_config(args.config, args.output_dir)
    print(json.dumps(summary["selected"], indent=2))


if __name__ == "__main__":
    main()

