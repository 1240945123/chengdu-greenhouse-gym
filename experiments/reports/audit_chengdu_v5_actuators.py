from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ACTUATORS = ("uRoofVent", "uFan", "uPad", "uLamp", "uThScr", "uBlScr")
DEFAULT_COOLING_ACTUATORS = ("uRoofVent", "uFan", "uPad")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cultivation_window(
    frame: pd.DataFrame,
    cultivation_end_exclusive: str,
) -> None:
    if "next_timestamp" not in frame.columns:
        raise ValueError("Trajectory is missing next_timestamp")
    target_times = pd.to_datetime(frame["next_timestamp"], errors="raise")
    cutoff = pd.Timestamp(cultivation_end_exclusive)
    if (target_times >= cutoff).any():
        raise ValueError("Trajectory contains targets outside the tomato cultivation window")


def summarize_actuator_support(
    frame: pd.DataFrame,
    actuators: Sequence[str],
    *,
    min_active_rows: int = 30,
    min_switches: int = 20,
    min_unique_levels: int = 3,
    active_threshold: float = 0.01,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for actuator in actuators:
        if actuator not in frame.columns:
            raise ValueError(f"Trajectory is missing actuator column: {actuator}")
        values = pd.to_numeric(frame[actuator], errors="raise").astype(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Actuator contains non-finite values: {actuator}")
        rounded = values.round(3)
        active_rows = int((values > active_threshold).sum())
        switch_count = int(rounded.diff().abs().fillna(0.0).gt(1e-6).sum())
        unique_levels = int(rounded.nunique())
        rows.append(
            {
                "actuator": actuator,
                "rows": int(len(values)),
                "active_rows": active_rows,
                "active_fraction": float(active_rows / len(values)) if len(values) else 0.0,
                "switch_count": switch_count,
                "rounded_unique_levels": unique_levels,
                "minimum": float(values.min()) if len(values) else np.nan,
                "p05": float(values.quantile(0.05)) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "p95": float(values.quantile(0.95)) if len(values) else np.nan,
                "maximum": float(values.max()) if len(values) else np.nan,
                "empirical_support_pass": bool(
                    active_rows >= min_active_rows
                    and switch_count >= min_switches
                    and unique_levels >= min_unique_levels
                ),
            }
        )
    return pd.DataFrame(rows)


def probe_actuator_response(
    frame: pd.DataFrame,
    *,
    actuator: str,
    low_level: float,
    high_level: float,
    predictor: Callable[[pd.Series], dict[str, float]],
) -> dict[str, Any]:
    if actuator not in frame.columns:
        raise ValueError(f"Trajectory is missing actuator column: {actuator}")
    if not 0.0 <= float(low_level) < float(high_level) <= 1.0:
        raise ValueError("Counterfactual levels must satisfy 0 <= low < high <= 1")
    temperature_deltas = []
    humidity_deltas = []
    for _, source in frame.iterrows():
        low = source.copy()
        high = source.copy()
        low[actuator] = float(low_level)
        high[actuator] = float(high_level)
        low_prediction = predictor(low)
        high_prediction = predictor(high)
        temperature_deltas.append(
            float(high_prediction["pred_air_temperature"])
            - float(low_prediction["pred_air_temperature"])
        )
        humidity_deltas.append(
            float(high_prediction["pred_relative_humidity"])
            - float(low_prediction["pred_relative_humidity"])
        )

    temperature = np.asarray(temperature_deltas, dtype=float)
    humidity = np.asarray(humidity_deltas, dtype=float)
    if not np.isfinite(temperature).all() or not np.isfinite(humidity).all():
        raise ValueError("Counterfactual predictor returned non-finite responses")
    return {
        "actuator": actuator,
        "low_level": float(low_level),
        "high_level": float(high_level),
        "sample_rows": int(len(frame)),
        "air_temperature_delta_q25": float(np.quantile(temperature, 0.25)),
        "air_temperature_delta_median": float(np.median(temperature)),
        "air_temperature_delta_q75": float(np.quantile(temperature, 0.75)),
        "relative_humidity_delta_q25": float(np.quantile(humidity, 0.25)),
        "relative_humidity_delta_median": float(np.median(humidity)),
        "relative_humidity_delta_q75": float(np.quantile(humidity, 0.75)),
        "cooling_response_fraction": float(np.mean(temperature < 0.0)),
    }


def _fixed_sample(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, maximum_rows, dtype=int)
    return frame.iloc[indices].copy()


def build_actuator_audit_bundle(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    predictor: Callable[[pd.Series], dict[str, float]],
    actuators: Sequence[str] = DEFAULT_ACTUATORS,
    cooling_actuators: Sequence[str] = DEFAULT_COOLING_ACTUATORS,
    cultivation_end_exclusive: str = "2026-07-12 00:00",
    min_active_rows: int = 30,
    min_switches: int = 20,
    min_unique_levels: int = 3,
    maximum_probe_rows: int = 128,
    model_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    trajectory_path = Path(trajectory_path)
    output_dir = Path(output_dir)
    frame = pd.read_csv(trajectory_path)
    validate_cultivation_window(frame, cultivation_end_exclusive)
    support = summarize_actuator_support(
        frame,
        actuators,
        min_active_rows=min_active_rows,
        min_switches=min_switches,
        min_unique_levels=min_unique_levels,
    )
    support_by_actuator = support.set_index("actuator")
    cooling_demand = frame.loc[
        pd.to_numeric(frame["x_air_temperature"], errors="raise")
        - pd.to_numeric(frame["d_air_temperature"], errors="raise")
        >= 2.0
    ]
    probe_frame = _fixed_sample(cooling_demand, maximum_probe_rows)

    responses = []
    recommended = []
    for actuator in cooling_actuators:
        if actuator not in support_by_actuator.index:
            continue
        support_row = support_by_actuator.loc[actuator]
        if not bool(support_row["empirical_support_pass"]) or probe_frame.empty:
            continue
        low_level = float(support_row["p05"])
        high_level = float(support_row["p95"])
        if high_level <= low_level:
            continue
        response = probe_actuator_response(
            probe_frame,
            actuator=actuator,
            low_level=low_level,
            high_level=high_level,
            predictor=predictor,
        )
        response["directional_response_pass"] = bool(
            response["air_temperature_delta_median"] < 0.0
            and response["cooling_response_fraction"] >= 0.60
        )
        responses.append(response)
        if response["directional_response_pass"]:
            recommended.append(actuator)

    response_frame = pd.DataFrame(responses)
    output_dir.mkdir(parents=True, exist_ok=True)
    support.to_csv(output_dir / "actuator_support.csv", index=False)
    response_frame.to_csv(output_dir / "counterfactual_responses.csv", index=False)
    audit = {
        "schema_version": "chengdu_v5_actuator_identifiability_v1",
        "cultivation_end_exclusive": str(pd.Timestamp(cultivation_end_exclusive)),
        "trajectory_rows": int(len(frame)),
        "cooling_demand_rows": int(len(cooling_demand)),
        "probe_rows": int(len(probe_frame)),
        "support_thresholds": {
            "min_active_rows": int(min_active_rows),
            "min_switches": int(min_switches),
            "min_unique_levels": int(min_unique_levels),
        },
        "directional_thresholds": {
            "air_temperature_delta_median_max": 0.0,
            "cooling_response_fraction_min": 0.60,
        },
        "trajectory_sha256": _sha256(trajectory_path),
        "model_sha256": dict(sorted((model_hashes or {}).items())),
        "recommended_action_space": recommended,
        "unsupported_for_optimization": [
            actuator for actuator in actuators if actuator not in recommended
        ],
        "interpretation": (
            "model_directional_sanity_check_not_observational_causal_effect"
        ),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    support_lines = [
        f"- {row.actuator}: active={int(row.active_rows)}/{int(row.rows)}, "
        f"switches={int(row.switch_count)}, levels={int(row.rounded_unique_levels)}, "
        f"support={bool(row.empirical_support_pass)}"
        for row in support.itertuples(index=False)
    ]
    response_lines = [
        f"- {row.actuator}: median dT={row.air_temperature_delta_median:.4f} C, "
        f"cooling fraction={row.cooling_response_fraction:.3f}, "
        f"direction pass={bool(row.directional_response_pass)}"
        for row in response_frame.itertuples(index=False)
    ]
    markdown = "\n".join(
        [
            "# V5 actuator identifiability audit",
            "",
            f"- Valid cultivation rows: {len(frame)}",
            f"- Cooling-demand rows: {len(cooling_demand)}",
            f"- Counterfactual probe rows: {len(probe_frame)}",
            f"- Recommended action space: {', '.join(recommended) if recommended else 'none'}",
            "",
            "## Empirical support",
            "",
            *support_lines,
            "",
            "## Local model response",
            "",
            *(response_lines or ["- No cooling actuator had enough support for probing."]),
            "",
            "This is a local model sanity check, not a causal effect estimate. Actions not",
            "recommended here must remain fixed or be excluded from controller optimization.",
        ]
    )
    (output_dir / "audit.md").write_text(markdown + "\n", encoding="utf-8")
    return audit


def main() -> None:
    from experiments.reports.chengdu_residual_correction import load_residual_model
    from experiments.reports.evaluate_chengdu_multistep import build_physics_predictor
    from experiments.reports.evaluate_chengdu_physics import load_parameter_vector

    parser = argparse.ArgumentParser(description="Audit V5 actuator support and local climate response.")
    parser.add_argument(
        "--trajectory",
        default="data/processed/chengdu_agri/greenhouse_001/trajectories/v5_cultivation_window/train.csv",
    )
    parser.add_argument(
        "--params",
        default="results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json",
    )
    parser.add_argument(
        "--residual_model",
        default="results/chengdu_agri_greenhouse_001/physics_v5_cultivation_greybox_residual/selected_model.json",
    )
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/physics_v5_cultivation_greybox_residual/actuator_audit",
    )
    args = parser.parse_args()
    params_path = Path(args.params)
    model_path = Path(args.residual_model)
    params = load_parameter_vector(params_path)
    if params is None or len(params) != 225:
        raise ValueError("Actuator audit requires a 225-value ChengduPhysicsV4 artifact")
    predictor = build_physics_predictor(
        params,
        residual_model=load_residual_model(model_path),
        model_backend="ChengduPhysicsV4",
    )
    audit = build_actuator_audit_bundle(
        args.trajectory,
        args.output_dir,
        predictor=predictor,
        model_hashes={"params": _sha256(params_path), "residual_model": _sha256(model_path)},
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
