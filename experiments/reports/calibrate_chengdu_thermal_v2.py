from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from experiments.reports.evaluate_chengdu_multistep import build_physics_predictor, evaluate_multistep_rollouts
from experiments.reports.evaluate_chengdu_physics import default_parameter_vector, evaluate_one_step_predictions


def fit_and_select_candidates(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    candidates: dict[str, Any],
    *,
    evaluator: Callable[[pd.DataFrame, Any, str], dict[str, Any]],
    shortlist_size: int,
) -> tuple[Any, dict[str, Any]]:
    fit_results = {
        candidate_id: dict(evaluator(train, candidate, "fit"))
        for candidate_id, candidate in candidates.items()
    }
    ranked = sorted(
        candidates,
        key=lambda candidate_id: (
            not bool(fit_results[candidate_id].get("physical_envelope_pass", True)),
            float(fit_results[candidate_id]["score"]),
            candidate_id,
        ),
    )
    shortlist = ranked[: max(1, min(int(shortlist_size), len(ranked)))]
    selection_results = {
        candidate_id: dict(evaluator(validation, candidates[candidate_id], "selection"))
        for candidate_id in shortlist
    }
    valid = [
        candidate_id
        for candidate_id in shortlist
        if bool(selection_results[candidate_id].get("physical_envelope_pass", False))
    ]
    if not valid:
        raise RuntimeError("No shortlisted candidate passed the validation physical-envelope gate")
    selected_id = min(valid, key=lambda candidate_id: (float(selection_results[candidate_id]["score"]), candidate_id))
    audit = {
        "fit_role": "train_only",
        "selection_role": "validation_only",
        "test_role": "unopened",
        "shortlist_candidate_ids": shortlist,
        "selected_candidate_id": selected_id,
        "rejected_candidate_ids": [candidate_id for candidate_id in shortlist if candidate_id not in valid],
        "fit_results": fit_results,
        "selection_results": selection_results,
    }
    return candidates[selected_id], audit


def _candidate_vectors(search: dict[str, list[float]]) -> dict[str, np.ndarray]:
    indices = [int(index) for index in search]
    vectors: dict[str, np.ndarray] = {}
    for sequence, values in enumerate(itertools.product(*(search[str(index)] for index in indices))):
        vector = default_parameter_vector()
        for index, value in zip(indices, values):
            vector[index] = float(value)
        candidate_id = f"candidate_{sequence:03d}_" + "_".join(f"p{index}-{value:g}" for index, value in zip(indices, values))
        vectors[candidate_id] = vector
    return vectors


def _fit_metrics(data: pd.DataFrame, params: np.ndarray, model_backend: str) -> dict[str, Any]:
    starts = list(range(0, len(data), 3))
    metrics, _ = evaluate_multistep_rollouts(
        data,
        predictor=build_physics_predictor(params, model_backend=model_backend),
        horizons=[1],
        start_indices=starts,
    )
    score = metrics["horizon_1_air_temperature_mae"] / 2.0 + metrics["horizon_1_relative_humidity_mae"] / 10.0
    return {"score": float(score), "physical_envelope_pass": True, **metrics}


def _selection_metrics(data: pd.DataFrame, params: np.ndarray, model_backend: str) -> dict[str, Any]:
    horizons = [6, 24, 72]
    starts = list(range(0, max(0, len(data) - max(horizons) + 1), 12))
    metrics, _ = evaluate_multistep_rollouts(
        data,
        predictor=build_physics_predictor(params, model_backend=model_backend),
        horizons=horizons,
        start_indices=starts,
    )
    normalized = [
        metrics[f"horizon_{horizon}_air_temperature_mae"] / 2.0
        + metrics[f"horizon_{horizon}_relative_humidity_mae"] / 10.0
        for horizon in horizons
        if f"horizon_{horizon}_air_temperature_mae" in metrics
    ]
    return {"score": float(np.mean(normalized)), **metrics}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free Chengdu thermal candidate calibration and selection.")
    parser.add_argument("--config", default="configs/models/chengdu_thermal_validation_v2.yml")
    parser.add_argument("--output", default="results/chengdu_agri_greenhouse_001/thermal_validation_v2/selected_params.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_path = Path(config["train_csv"])
    validation_path = Path(config["validation_csv"])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    candidates = _candidate_vectors(config["search_space"])
    model_backend = str(config.get("model_backend", "ChengduPhysics"))

    def evaluator(data: pd.DataFrame, params: np.ndarray, role: str) -> dict[str, Any]:
        return (
            _fit_metrics(data, params, model_backend)
            if role == "fit"
            else _selection_metrics(data, params, model_backend)
        )

    selected, audit = fit_and_select_candidates(
        train,
        validation,
        candidates,
        evaluator=evaluator,
        shortlist_size=int(config["shortlist_size"]),
    )
    indices = [int(index) for index in config["search_space"]]
    payload = {
        "schema_version": "chengdu_thermal_calibration_v2",
        "model_backend": model_backend,
        "multipliers": {str(index): float(selected[index]) for index in indices},
        "data_sha256": {"train": _sha256(train_path), "validation": _sha256(validation_path)},
        "config_sha256": _sha256(config_path),
        "audit": audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": payload["multipliers"], "candidate_id": audit["selected_candidate_id"]}, indent=2))


if __name__ == "__main__":
    main()
