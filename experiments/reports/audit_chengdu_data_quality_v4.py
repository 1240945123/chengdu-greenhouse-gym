from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("air_temperature", "relative_humidity")


def _quality_tier(weights: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(weights, errors="raise")
    return pd.Series(
        np.select(
            [numeric.eq(1.0), numeric.gt(0.0)],
            ["high", "medium"],
            default="low",
        ),
        index=weights.index,
        dtype=object,
    )


def attach_rollout_quality(
    rollouts: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> pd.DataFrame:
    required_rollout = {"start_index", "horizon"}
    required_trajectory = {
        "transition_quality_weight",
        "target_heat_regime",
        "extreme_heat_physical_support",
    }
    missing_rollout = required_rollout.difference(rollouts.columns)
    missing_trajectory = required_trajectory.difference(trajectory.columns)
    if missing_rollout or missing_trajectory:
        missing = sorted(missing_rollout | missing_trajectory)
        raise ValueError(f"Missing quality audit columns: {', '.join(missing)}")

    attached = rollouts.copy()
    target_index = (
        pd.to_numeric(attached["start_index"], errors="raise").astype(int)
        + pd.to_numeric(attached["horizon"], errors="raise").astype(int)
        - 1
    )
    if target_index.lt(0).any() or target_index.ge(len(trajectory)).any():
        raise ValueError("Rollout target index is outside trajectory split")

    metadata = trajectory.reset_index(drop=True).iloc[target_index.to_numpy()].reset_index(drop=True)
    attached = attached.reset_index(drop=True)
    attached["target_transition_index"] = target_index.reset_index(drop=True)
    attached["target_quality_weight"] = metadata["transition_quality_weight"]
    attached["target_quality_tier"] = _quality_tier(attached["target_quality_weight"])
    attached["target_heat_regime"] = metadata["target_heat_regime"]
    attached["extreme_heat_physical_support"] = metadata[
        "extreme_heat_physical_support"
    ].astype(bool)
    return attached


def summarize_stratified_rollouts(
    attached_rollouts: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper: str | list[str] = (
        group_columns[0] if len(group_columns) == 1 else list(group_columns)
    )
    for keys, group in attached_rollouts.groupby(grouper, dropna=False, sort=True):
        key_values = (keys,) if len(group_columns) == 1 else tuple(keys)
        row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
        row["count"] = int(len(group))
        for target in TARGETS:
            error = (
                pd.to_numeric(group[f"pred_{target}"], errors="raise")
                - pd.to_numeric(group[f"true_{target}"], errors="raise")
            )
            row[f"{target}_mae"] = float(error.abs().mean())
            row[f"{target}_rmse"] = float(np.sqrt(np.square(error).mean()))
            row[f"{target}_bias"] = float(error.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _regime_metrics(frame: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {"rows": int(len(frame))}
    normalized = []
    for target, scale in (("air_temperature", 2.0), ("relative_humidity", 10.0)):
        error = (
            pd.to_numeric(frame[f"pred_{target}"], errors="raise")
            - pd.to_numeric(frame[f"true_{target}"], errors="raise")
        )
        mae = float(error.abs().mean())
        result[f"{target}_mae"] = mae
        result[f"{target}_rmse"] = float(np.sqrt(np.square(error).mean()))
        result[f"{target}_bias"] = float(error.mean())
        normalized.append(mae / scale)
    result["normalized_mae"] = float(np.mean(normalized))
    return result


def compare_regime_candidate(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    normal_tolerance: float = 0.02,
    selection_horizons: Sequence[int] = (24, 72),
) -> dict[str, object]:
    if normal_tolerance < 0.0:
        raise ValueError("Normal-regime tolerance must be nonnegative")
    alignment_columns = [
        "horizon",
        "target_heat_regime",
        "true_air_temperature",
        "true_relative_humidity",
    ]
    if len(reference) != len(candidate) or not reference[alignment_columns].reset_index(
        drop=True
    ).equals(candidate[alignment_columns].reset_index(drop=True)):
        raise ValueError("Reference and candidate rollout targets are not aligned")

    selected_reference = reference[reference["horizon"].isin(selection_horizons)]
    selected_candidate = candidate[candidate["horizon"].isin(selection_horizons)]
    normal_reference = reference[reference["target_heat_regime"].eq("normal")]
    normal_candidate = candidate[candidate["target_heat_regime"].eq("normal")]
    high_reference = reference[~reference["target_heat_regime"].eq("normal")]
    high_candidate = candidate[~candidate["target_heat_regime"].eq("normal")]
    if selected_reference.empty or normal_reference.empty or high_reference.empty:
        raise ValueError("Dual-regime comparison requires selection, normal, and high rows")

    metrics = {
        "reference": {
            "aggregate_24_72": _regime_metrics(selected_reference),
            "normal": _regime_metrics(normal_reference),
            "high_temperature": _regime_metrics(high_reference),
        },
        "candidate": {
            "aggregate_24_72": _regime_metrics(selected_candidate),
            "normal": _regime_metrics(normal_candidate),
            "high_temperature": _regime_metrics(high_candidate),
        },
    }
    reference_metrics = metrics["reference"]
    candidate_metrics = metrics["candidate"]
    aggregate_improved = (
        candidate_metrics["aggregate_24_72"]["normalized_mae"]
        < reference_metrics["aggregate_24_72"]["normalized_mae"]
    )
    high_temperature_improved = all(
        candidate_metrics["high_temperature"][f"{target}_mae"]
        < reference_metrics["high_temperature"][f"{target}_mae"]
        for target in TARGETS
    )
    normal_regime_within_tolerance = (
        candidate_metrics["normal"]["normalized_mae"]
        <= reference_metrics["normal"]["normalized_mae"] * (1.0 + normal_tolerance)
    )
    predicted = candidate[
        ["pred_air_temperature", "pred_relative_humidity"]
    ].to_numpy(dtype=float)
    physical_envelope_pass = bool(
        np.isfinite(predicted).all()
        and candidate["pred_air_temperature"].between(-10.0, 60.0).all()
        and candidate["pred_relative_humidity"].between(0.0, 100.0).all()
    )
    return {
        "schema_version": "chengdu_dual_regime_gate_v1",
        "selection_horizons": list(selection_horizons),
        "normal_tolerance": float(normal_tolerance),
        **metrics,
        "aggregate_improved": bool(aggregate_improved),
        "high_temperature_improved": bool(high_temperature_improved),
        "normal_regime_within_tolerance": bool(normal_regime_within_tolerance),
        "physical_envelope_pass": physical_envelope_pass,
        "promote": bool(
            aggregate_improved
            and high_temperature_improved
            and normal_regime_within_tolerance
            and physical_envelope_pass
        ),
    }


def _count_values(values: pd.Series, order: Sequence[str]) -> dict[str, int]:
    counts = values.value_counts()
    return {name: int(counts.get(name, 0)) for name in order}


def build_quality_audit_bundle(
    trajectory_path: str | Path,
    rollout_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    trajectory_path = Path(trajectory_path)
    rollout_path = Path(rollout_path)
    output_dir = Path(output_dir)
    trajectory = pd.read_csv(trajectory_path)
    rollouts = pd.read_csv(rollout_path)
    attached = attach_rollout_quality(rollouts, trajectory)

    reports = {
        "metrics_by_quality.csv": summarize_stratified_rollouts(
            attached, ["target_quality_tier"]
        ),
        "metrics_by_heat_regime.csv": summarize_stratified_rollouts(
            attached, ["target_heat_regime"]
        ),
        "metrics_by_horizon_quality_heat.csv": summarize_stratified_rollouts(
            attached,
            ["horizon", "target_quality_tier", "target_heat_regime"],
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    attached.to_csv(output_dir / "rollouts_with_quality.csv", index=False)
    for filename, report in reports.items():
        report.to_csv(output_dir / filename, index=False)

    trajectory_tier = _quality_tier(trajectory["transition_quality_weight"])
    audit: dict[str, object] = {
        "schema_version": "chengdu_data_quality_audit_v1",
        "trajectory_path": trajectory_path.as_posix(),
        "rollout_path": rollout_path.as_posix(),
        "trajectory_rows": int(len(trajectory)),
        "rollout_rows": int(len(attached)),
        "quality_counts": _count_values(trajectory_tier, ("high", "medium", "low")),
        "heat_regime_counts": _count_values(
            trajectory["target_heat_regime"], ("normal", "high", "extreme")
        ),
        "extreme_heat_physically_supported": int(
            (
                trajectory["target_heat_regime"].eq("extreme")
                & trajectory["extreme_heat_physical_support"].astype(bool)
            ).sum()
        ),
        "report_files": list(reports),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    quality = audit["quality_counts"]
    heat = audit["heat_regime_counts"]
    markdown = "\n".join(
        [
            "# Quality-aware data audit",
            "",
            f"- Trajectory rows: {audit['trajectory_rows']}",
            f"- Rollout rows: {audit['rollout_rows']}",
            f"- Quality tiers: high={quality['high']}, medium={quality['medium']}, low={quality['low']}",
            f"- Heat regimes: normal={heat['normal']}, high={heat['high']}, extreme={heat['extreme']}",
            f"- Physically supported extreme heat rows: {audit['extreme_heat_physically_supported']}",
            "",
            "Original observations are retained. Quality labels are diagnostic metadata, not replacements.",
        ]
    )
    (output_dir / "audit.md").write_text(markdown + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit quality-aware Chengdu rollouts.")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_quality_audit_bundle(args.trajectory, args.rollouts, args.output_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
