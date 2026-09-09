from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiments.reports.evaluate_chengdu_physics import default_parameter_vector, evaluate_one_step_predictions


DEFAULT_SEARCH_SPACE = {
    208: [1.0, 2.0, 3.0, 4.0],
    209: [0.3, 0.6, 1.0],
    210: [1.0, 1.6, 2.5, 3.5],
    211: [1.0, 1.6, 2.5, 4.0, 6.0],
    212: [0.6, 1.0],
    213: [1.0],
    214: [0.3, 0.6, 1.0, 1.6],
    215: [1.0, 1.6, 2.5, 4.0, 6.0, 8.0],
}


def _iter_candidate_vectors(search_space: dict[int, list[float]], base_params: np.ndarray):
    keys = list(search_space.keys())
    for values in itertools.product(*(search_space[key] for key in keys)):
        params = base_params.copy()
        for key, value in zip(keys, values):
            params[int(key)] = float(value)
        yield params


def _serializable_multipliers(params: np.ndarray, keys: list[int]) -> dict[str, float]:
    return {str(key): float(params[key]) for key in keys}


def calibrate_parameter_multipliers(
    trajectories: pd.DataFrame,
    search_space: dict[int, list[float]] | None = None,
    evaluator: Callable[[pd.DataFrame, np.ndarray], dict[str, float]] | None = None,
    output_path: str | Path | None = None,
    max_passes: int = 2,
) -> tuple[np.ndarray, dict[str, float]]:
    space = search_space or DEFAULT_SEARCH_SPACE
    keys = list(space.keys())
    base_params = default_parameter_vector()
    evaluate = evaluator or (lambda data, params: evaluate_one_step_predictions(data, parameter_vector=params)[0])

    best_params = base_params.copy()
    best_metrics = dict(evaluate(trajectories, best_params))
    best_score = float(best_metrics["mean_mae"])

    for _ in range(max_passes):
        improved = False
        for key in keys:
            local_best_params = best_params.copy()
            local_best_metrics = best_metrics
            local_best_score = best_score
            for value in space[key]:
                candidate = best_params.copy()
                candidate[int(key)] = float(value)
                metrics = evaluate(trajectories, candidate)
                score = float(metrics["mean_mae"])
                if score < local_best_score:
                    local_best_score = score
                    local_best_params = candidate
                    local_best_metrics = dict(metrics)
            if local_best_score < best_score:
                best_score = local_best_score
                best_params = local_best_params.copy()
                best_metrics = dict(local_best_metrics)
                improved = True
        if not improved:
            break

    if output_path is not None:
        payload = {
            "multipliers": _serializable_multipliers(best_params, keys),
            "metrics": best_metrics,
            "searched_parameters": keys,
        }
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return best_params, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Grid-search ChengduPhysics calibration multipliers on real trajectories.")
    parser.add_argument("--trajectory_csv", default="data/processed/chengdu_agri/greenhouse_001/trajectories/train.csv")
    parser.add_argument("--sample_limit", type=int, default=240)
    parser.add_argument("--save_dir", default="results/chengdu_agri_greenhouse_001/physics_calibration/")
    args = parser.parse_args()

    trajectories = pd.read_csv(args.trajectory_csv)
    if args.sample_limit is not None:
        trajectories = trajectories.head(args.sample_limit)

    save_dir = Path(args.save_dir)
    output_path = save_dir / "chengdu_physics_calibrated_params.json"
    _params, metrics = calibrate_parameter_multipliers(trajectories, output_path=output_path)
    print(json.dumps({"output_path": str(output_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
