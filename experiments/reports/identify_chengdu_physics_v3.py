from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import casadi as ca
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from experiments.reports.evaluate_chengdu_multistep import build_physics_predictor, evaluate_multistep_rollouts
from experiments.reports.evaluate_chengdu_physics import predict_one_step
from glassgym.configs.default_params import init_default_params
from glassgym.models.ChengduPhysicsV3.utils import define_model


def validate_parameter_specs(specs: dict[str, dict[str, Any]]) -> None:
    indices = set()
    for name, spec in specs.items():
        index = int(spec["index"])
        if index in indices:
            raise ValueError(f"Duplicate parameter index {index}")
        indices.add(index)
        if "fixed" in spec:
            if bool(spec.get("identifiable", False)):
                raise ValueError(f"Fixed parameter {name} cannot be marked identifiable")
            continue
        lower = float(spec["lower"])
        upper = float(spec["upper"])
        initial = float(spec["initial"])
        if lower < 0.0 or lower >= upper:
            raise ValueError(f"Parameter {name} lower bound must be nonnegative and below upper bound")
        if not lower <= initial <= upper:
            raise ValueError(f"Parameter {name} initial value must be inside bounds")


def split_parameter_specs(specs: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[int, float]]:
    validate_parameter_specs(specs)
    free = {name: spec for name, spec in specs.items() if "fixed" not in spec}
    fixed = {int(spec["index"]): float(spec["fixed"]) for spec in specs.values() if "fixed" in spec}
    return free, fixed


def soft_l1_cost(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=float)
    return float(np.sum(2.0 * (np.sqrt(1.0 + values * values) - 1.0)))


def fit_bounded_soft_l1(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    *,
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int,
):
    return least_squares(
        residual_fn,
        start,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        diff_step=1e-4,
        x_scale="jac",
        max_nfev=max_nfev,
    )


def finite_difference_sensitivity(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    names: list[str],
    relative_step: float = 1e-4,
) -> dict[str, Any]:
    center = np.asarray(values, dtype=float)
    baseline = np.asarray(residual_fn(center), dtype=float)
    jacobian = np.empty((baseline.size, center.size), dtype=float)
    for column in range(center.size):
        step = relative_step * max(1.0, abs(center[column]))
        perturbed = center.copy()
        perturbed[column] += step
        jacobian[:, column] = (np.asarray(residual_fn(perturbed), dtype=float) - baseline) / step
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    norms = np.linalg.norm(jacobian, axis=0)
    threshold = max(float(norms.max(initial=0.0)) * 1e-8, 1e-12)
    weak = [name for name, norm in zip(names, norms) if float(norm) <= threshold]
    positive = singular_values[singular_values > max(float(singular_values.max(initial=0.0)) * 1e-10, 1e-12)]
    condition = float("inf") if positive.size < 2 else float(positive.max() / positive.min())
    return {
        "rank": int(np.linalg.matrix_rank(jacobian)),
        "shape": [int(value) for value in jacobian.shape],
        "singular_values": singular_values.tolist(),
        "condition_number_nonzero": condition,
        "parameter_sensitivity_norms": {name: float(norm) for name, norm in zip(names, norms)},
        "weak_parameter_names": weak,
    }


def identify_and_select(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    starts: list[np.ndarray],
    *,
    reference_start: np.ndarray | None = None,
    fit_fn: Callable[[pd.DataFrame, np.ndarray], dict[str, Any]],
    validation_fn: Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fitted = []
    if reference_start is not None:
        fitted.append(
            {
                "values": np.asarray(reference_start, dtype=float).copy(),
                "candidate_kind": "unfitted_physical_reference",
                "success": True,
                "status": 0,
                "nfev": 0,
                "message": "Reference candidate retained for validation comparison",
            }
        )
    for start in starts:
        candidate = fit_fn(train, start)
        candidate.setdefault("candidate_kind", "robust_one_step_fit")
        fitted.append(candidate)
    evaluated = []
    for candidate_id, candidate in enumerate(fitted):
        metrics = validation_fn(validation, candidate)
        evaluated.append({"candidate_id": candidate_id, "candidate": candidate, "validation": metrics})
    valid = [item for item in evaluated if bool(item["validation"].get("physical_envelope_pass", False))]
    if not valid:
        raise RuntimeError("No V3 identification start passed validation physical gates")
    selected = min(valid, key=lambda item: (float(item["validation"]["score"]), int(item["candidate_id"])))
    audit = {
        "fit_role": "train_only",
        "selection_role": "validation_only",
        "test_role": "unopened",
        "candidates": evaluated,
        "selected_candidate_id": int(selected["candidate_id"]),
    }
    return selected["candidate"], audit


class V3ResidualEvaluator:
    def __init__(
        self,
        data: pd.DataFrame,
        free_specs: dict[str, dict[str, Any]],
        fixed: dict[int, float],
        sample_limit: int,
    ):
        eligible = data.loc[data["identification_eligible"].astype(bool)].reset_index(drop=True)
        if eligible.empty:
            raise ValueError("No identification-eligible training rows")
        if len(eligible) > sample_limit:
            positions = np.linspace(0, len(eligible) - 1, sample_limit, dtype=int)
            eligible = eligible.iloc[positions].reset_index(drop=True)
        self.data = eligible
        self.free_specs = free_specs
        self.fixed = fixed
        self.integrator = define_model(nx=28, nu=8, nd=10, n_params=220, dt=3600.0)

    def parameter_vector(self, values: np.ndarray) -> np.ndarray:
        params = np.asarray(init_default_params(220), dtype=float)
        for index, value in self.fixed.items():
            params[index] = value
        for value, spec in zip(values, self.free_specs.values()):
            params[int(spec["index"])] = float(value)
        return params

    def __call__(self, values: np.ndarray) -> np.ndarray:
        params = self.parameter_vector(values)
        residuals = []
        for _, row in self.data.iterrows():
            prediction = predict_one_step(
                row,
                integrator=self.integrator,
                params=params,
                model_backend="ChengduPhysicsV3",
            )
            residuals.extend(
                [
                    (prediction["pred_air_temperature"] - float(row["next_x_air_temperature"])) / 2.0,
                    (prediction["pred_relative_humidity"] - float(row["next_x_relative_humidity"])) / 10.0,
                ]
            )
        result = np.asarray(residuals, dtype=float)
        if not np.isfinite(result).all():
            return np.full_like(result, 1e6)
        return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify roof/fan-separated ChengduPhysicsV3 parameters.")
    parser.add_argument("--config", default="configs/models/chengdu_physics_v3_identification.yml")
    parser.add_argument("--output", default="results/chengdu_agri_greenhouse_001/physics_v3_identification/selected_params.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specs = config["parameters"]
    free_specs, fixed = split_parameter_specs(specs)
    names = list(free_specs)
    lower = np.array([float(spec["lower"]) for spec in free_specs.values()])
    upper = np.array([float(spec["upper"]) for spec in free_specs.values()])
    initial = np.array([float(spec["initial"]) for spec in free_specs.values()])
    factors = [float(value) for value in config["multistart_factors"]]
    starts = [np.clip(initial * factor, lower, upper) for factor in factors]
    train_path = Path(config["train_csv"])
    validation_path = Path(config["validation_csv"])
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    def fit_fn(data: pd.DataFrame, start: np.ndarray) -> dict[str, Any]:
        evaluator = V3ResidualEvaluator(data, free_specs, fixed, int(config["fit_sample_limit"]))
        result = fit_bounded_soft_l1(
            evaluator,
            start=start,
            lower=lower,
            upper=upper,
            max_nfev=int(config["max_nfev"]),
        )
        return {
            "values": result.x,
            "fit_cost": float(result.cost),
            "success": bool(result.success),
            "status": int(result.status),
            "nfev": int(result.nfev),
            "message": str(result.message),
        }

    def validation_fn(data: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
        evaluator = V3ResidualEvaluator(train, free_specs, fixed, 1)
        params = evaluator.parameter_vector(np.asarray(candidate["values"], dtype=float))
        horizons = [6, 24, 72]
        starts_idx = list(range(0, max(0, len(data) - 72 + 1), int(config["validation_stride"])))
        metrics, _ = evaluate_multistep_rollouts(
            data,
            predictor=build_physics_predictor(params, model_backend="ChengduPhysicsV3"),
            horizons=horizons,
            start_indices=starts_idx,
        )
        scores = [
            metrics[f"horizon_{h}_air_temperature_mae"] / 2.0
            + metrics[f"horizon_{h}_relative_humidity_mae"] / 10.0
            for h in horizons
            if f"horizon_{h}_air_temperature_mae" in metrics
        ]
        return {"score": float(np.mean(scores)), **metrics}

    selected, audit = identify_and_select(
        train,
        validation,
        starts,
        reference_start=initial,
        fit_fn=fit_fn,
        validation_fn=validation_fn,
    )
    sensitivity_evaluator = V3ResidualEvaluator(train, free_specs, fixed, int(config["sensitivity_sample_limit"]))
    sensitivity = finite_difference_sensitivity(
        sensitivity_evaluator,
        np.asarray(selected["values"], dtype=float),
        names,
    )
    selected_params = sensitivity_evaluator.parameter_vector(np.asarray(selected["values"], dtype=float))
    output_payload = {
        "schema_version": "chengdu_physics_v3_identification",
        "model_backend": "ChengduPhysicsV3",
        "num_params": 220,
        "multipliers": {str(index): float(selected_params[index]) for index in range(208, 220)},
        "free_parameter_names": names,
        "fixed_non_identifiable": {
            name: {"index": int(spec["index"]), "value": float(spec["fixed"])}
            for name, spec in specs.items()
            if "fixed" in spec
        },
        "sensitivity": sensitivity,
        "audit": audit,
        "data_sha256": {"train": _sha256(train_path), "validation": _sha256(validation_path)},
        "config_sha256": _sha256(config_path),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(output_payload), indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "selected": output_payload["multipliers"], "sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
