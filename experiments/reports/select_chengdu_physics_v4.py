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
from experiments.reports.evaluate_chengdu_physics import load_parameter_vector


REFERENCE_EXTENSION = [0.0, 1.0, 0.0, 1.0, 0.0]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for candidate in candidates:
        key = tuple(float(value) for value in candidate["extension"])
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def generate_thermal_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["thermal_grid"]
    candidates = [{"candidate_kind": "v3_reference", "extension": REFERENCE_EXTENSION.copy()}]
    for longwave, floor_capacity, latent in itertools.product(
        grid["longwave_scale"], grid["floor_capacity_scale"], grid["latent_heat_scale"]
    ):
        candidates.append(
            {
                "candidate_kind": "thermal",
                "extension": [float(longwave), float(floor_capacity), 0.0, 1.0, float(latent)],
            }
        )
    return _unique_candidates(candidates)


def generate_moisture_candidates(
    thermal_finalists: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grid = config["moisture_grid"]
    candidates = []
    for thermal in thermal_finalists:
        extension = thermal["extension"]
        for rate, capacity in itertools.product(
            grid["moisture_buffer_rate_s"], grid["moisture_buffer_capacity_ratio"]
        ):
            values = [
                float(extension[0]),
                float(extension[1]),
                float(rate),
                float(capacity),
                float(extension[4]),
            ]
            kind = "v3_reference" if values == REFERENCE_EXTENSION else "combined"
            candidates.append({"candidate_kind": kind, "extension": values})
    candidates.append({"candidate_kind": "v3_reference", "extension": REFERENCE_EXTENSION.copy()})
    return _unique_candidates(candidates)


def rank_train_candidates(evaluated: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    eligible = [item for item in evaluated if bool(item.get("physical_envelope_pass", False))]
    eligible.sort(key=lambda item: (float(item.get("score", float("inf"))), tuple(item["extension"])))
    reference = next((item for item in eligible if item.get("candidate_kind") == "v3_reference"), None)
    if reference is None:
        raise RuntimeError("The V3 reference candidate did not pass the training physical envelope")
    selected = eligible[:top_k]
    if reference not in selected:
        selected = selected[: max(0, top_k - 1)] + [reference]
    return selected


def select_validation_candidate(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [item for item in evaluated if bool(item.get("physical_envelope_pass", False))]
    if not eligible:
        raise RuntimeError("No V4 candidate passed the validation physical envelope")
    return min(eligible, key=lambda item: (float(item["score"]), int(item["candidate_id"])))


def _parameter_vector(base: np.ndarray, extension: list[float]) -> np.ndarray:
    if len(base) != 220:
        raise ValueError(f"Expected 220 V3 parameters, received {len(base)}")
    if len(extension) != 5:
        raise ValueError("V4 extension must contain five parameters")
    return np.r_[base, np.asarray(extension, dtype=float)]


def _selection_score(metrics: dict[str, float], horizons: list[int]) -> float:
    terms = []
    for horizon in horizons:
        terms.extend(
            [
                float(metrics[f"horizon_{horizon}_air_temperature_mae"]) / 2.0,
                float(metrics[f"horizon_{horizon}_relative_humidity_mae"]) / 10.0,
            ]
        )
    return float(np.mean(terms))


def evaluate_candidate(
    frame: pd.DataFrame,
    candidate: dict[str, Any],
    base_params: np.ndarray,
    *,
    horizons: list[int],
    stride: int,
) -> dict[str, Any]:
    max_horizon = max(horizons)
    starts = list(range(0, max(0, len(frame) - max_horizon + 1), max(1, stride)))
    try:
        params = _parameter_vector(base_params, candidate["extension"])
        metrics, _rows = evaluate_multistep_rollouts(
            frame,
            predictor=build_physics_predictor(params, model_backend="ChengduPhysicsV4"),
            horizons=horizons,
            start_indices=starts,
        )
        score = _selection_score(metrics, horizons) if metrics.get("physical_envelope_pass") else float("inf")
        return {**candidate, **metrics, "score": score}
    except Exception as exc:
        return {
            **candidate,
            "score": float("inf"),
            "all_predictions_finite": False,
            "physical_envelope_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _evaluate_many(
    frame: pd.DataFrame,
    candidates: list[dict[str, Any]],
    base_params: np.ndarray,
    *,
    horizons: list[int],
    stride: int,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    results = []
    for index, candidate in enumerate(candidates):
        result = evaluate_candidate(
            frame, candidate, base_params, horizons=horizons, stride=stride
        )
        result["candidate_id"] = index
        results.append(result)
        progress(f"candidate {index + 1}/{len(candidates)} score={result['score']:.6f}")
    return results


def select_from_config(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    output_path = Path(output_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_path = Path(config["train_csv"])
    validation_path = Path(config["validation_csv"])
    v3_params_path = Path(config["v3_params_json"])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    base_params = load_parameter_vector(v3_params_path)
    if base_params is None:
        raise ValueError("V3 parameter artifact is required")

    train_horizons = [int(value) for value in config["train_horizons"]]
    thermal = _evaluate_many(
        train,
        generate_thermal_candidates(config),
        base_params,
        horizons=train_horizons,
        stride=int(config["train_stride"]),
    )
    thermal_finalists = rank_train_candidates(thermal, int(config["thermal_top_k"]))
    moisture = _evaluate_many(
        train,
        generate_moisture_candidates(thermal_finalists, config),
        base_params,
        horizons=train_horizons,
        stride=int(config["train_stride"]),
    )
    train_finalists = rank_train_candidates(
        _unique_candidates(thermal_finalists + moisture), int(config["validation_top_k"])
    )

    validation_horizons = [int(value) for value in config["validation_horizons"]]
    validation_results = _evaluate_many(
        validation,
        train_finalists,
        base_params,
        horizons=validation_horizons,
        stride=int(config["validation_stride"]),
    )
    selected = select_validation_candidate(validation_results)
    selected_params = _parameter_vector(base_params, selected["extension"])
    artifact = {
        "schema_version": "chengdu-physics-v4-daily-balance-selection",
        "model_backend": "ChengduPhysicsV4",
        "num_params": 225,
        "multipliers": {str(index): float(selected_params[index]) for index in range(208, 225)},
        "extension_parameter_names": [
            "longwave_scale",
            "floor_capacity_scale",
            "moisture_buffer_rate_s",
            "moisture_buffer_capacity_ratio",
            "latent_heat_scale",
        ],
        "selected_candidate": selected,
        "audit": {
            "train_role": "candidate_ranking_only",
            "validation_role": "final_selection_only",
            "test_role": "unopened",
            "thermal_train_candidates": thermal,
            "moisture_train_candidates": moisture,
            "validation_candidates": validation_results,
        },
        "hashes": {
            "train_csv": _sha256(train_path),
            "validation_csv": _sha256(validation_path),
            "v3_params_json": _sha256(v3_params_path),
            "config": _sha256(config_path),
            "v4_ode": _sha256(Path(__file__).parents[2] / "glassgym/models/ChengduPhysicsV4/ode.py"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Select ChengduPhysicsV4 using train and validation only.")
    parser.add_argument("--config", default="configs/models/chengdu_physics_v4_selection.yml")
    parser.add_argument(
        "--output",
        default="results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json",
    )
    args = parser.parse_args()
    artifact = select_from_config(args.config, args.output)
    print(json.dumps({"selected_candidate": artifact["selected_candidate"]}, indent=2))


if __name__ == "__main__":
    main()

