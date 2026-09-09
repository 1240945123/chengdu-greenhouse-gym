from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.controllers.benchmark_protocol import PROJECT_ROOT
from experiments.crop.audit_regional_controller_yield import dataframe_to_markdown_table


def _envelope_status(value: float, low: float, high: float, season_id: str) -> str:
    if not str(season_id).endswith("_spring"):
        return "not_applicable"
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "inside"


def build_stress_audit(
    priors: dict[str, Any],
    scenarios: pd.DataFrame,
    seasons: pd.DataFrame,
    v2_seasons: pd.DataFrame,
    *,
    expected_scenario_count: int,
    expected_controller_episode_count: int,
    expected_stress_scenarios: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    base_keys = ["algorithm", "seed", "season_id", "cohort_draw"]
    paired = scenarios.groupby(base_keys)["heat_stress_scenario"].nunique()
    paired_complete = bool(
        len(paired) > 0 and paired.eq(int(expected_stress_scenarios)).all()
    )
    controller_episode_count = int(
        scenarios[["algorithm", "seed", "season_id"]].drop_duplicates().shape[0]
    )
    max_balance = float(scenarios["dry_matter_balance_error_kg_m2"].abs().max())

    draw_harvest_violation_count = 0
    max_draw_harvest_reversal = 0.0
    retention_violation_count = 0
    stress_loss_violation_count = 0
    season_yield_violation_count = 0
    required_names = {"sensitive", "central", "tolerant"}
    if required_names.issubset(set(scenarios["heat_stress_scenario"].astype(str))):
        yield_pivot = scenarios.pivot(
            index=base_keys,
            columns="heat_stress_scenario",
            values="partial_yield_kg_m2",
        )
        draw_reversals = np.maximum(
            yield_pivot["sensitive"] - yield_pivot["central"],
            yield_pivot["central"] - yield_pivot["tolerant"],
        )
        draw_harvest_violation_count = int((draw_reversals > 1e-12).sum())
        max_draw_harvest_reversal = float(max(0.0, draw_reversals.max()))
        retention_pivot = scenarios.pivot(
            index=base_keys,
            columns="heat_stress_scenario",
            values="mean_reproductive_heat_retention",
        )
        retention_violation_count = int(
            (
                (retention_pivot["sensitive"] > retention_pivot["central"] + 1e-12)
                | (retention_pivot["central"] > retention_pivot["tolerant"] + 1e-12)
            ).sum()
        )
        loss_pivot = scenarios.pivot(
            index=base_keys,
            columns="heat_stress_scenario",
            values="reproductive_heat_stress_loss_dry_matter_kg_m2",
        )
        stress_loss_violation_count = int(
            (
                (loss_pivot["sensitive"] + 1e-12 < loss_pivot["central"])
                | (loss_pivot["central"] + 1e-12 < loss_pivot["tolerant"])
            ).sum()
        )
        season_pivot = seasons.pivot(
            index=["algorithm", "season_id"],
            columns="heat_stress_scenario",
            values="projected_harvest_kg_m2",
        )
        season_yield_violation_count = int(
            (
                (season_pivot["sensitive"] > season_pivot["central"] + 1e-12)
                | (season_pivot["central"] > season_pivot["tolerant"] + 1e-12)
            ).sum()
        )
    reproductive_monotonic = bool(
        retention_violation_count == 0 and stress_loss_violation_count == 0
    )
    season_yield_monotonic = bool(season_yield_violation_count == 0)

    central = seasons.loc[seasons["heat_stress_scenario"].eq("central")].copy()
    envelope = priors["spring_partial_yield_envelope"]
    low = float(envelope["low_kg_m2"])
    high = float(envelope["high_kg_m2"])
    target_day = float(priors["maturity_thermal_time"]["target_first_harvest_day"])
    central["spring_envelope_status"] = [
        _envelope_status(value, low, high, season_id)
        for value, season_id in zip(
            central["projected_harvest_kg_m2"], central["season_id"], strict=True
        )
    ]
    central["first_harvest_abs_error_days"] = (
        central["projected_first_harvest_day"] - target_day
    ).abs()

    records: list[dict[str, Any]] = []
    for algorithm, group in central.groupby("algorithm", sort=True):
        spring = group.loc[group["spring_envelope_status"].ne("not_applicable")]
        records.append(
            {
                "algorithm": str(algorithm),
                "central_projected_yield_mean_kg_m2": float(
                    group["projected_harvest_kg_m2"].mean()
                ),
                "central_first_harvest_mae_days": float(
                    group["first_harvest_abs_error_days"].mean()
                ),
                "central_spring_envelope_coverage_fraction": float(
                    spring["spring_envelope_status"].eq("inside").mean()
                ),
                "central_mean_reproductive_heat_retention": float(
                    group["mean_reproductive_heat_retention"].mean()
                ),
                "central_reproductive_heat_stress_loss_mean_kg_m2": float(
                    group["reproductive_heat_stress_loss_mean_kg_m2"].mean()
                ),
                "target_eligible": False,
            }
        )
    by_algorithm = pd.DataFrame(records)

    comparison = central.merge(
        v2_seasons[
            [
                "algorithm",
                "season_id",
                "projected_harvest_kg_m2",
                "projected_first_harvest_day",
            ]
        ].rename(
            columns={
                "projected_harvest_kg_m2": "v2_projected_harvest_kg_m2",
                "projected_first_harvest_day": "v2_projected_first_harvest_day",
            }
        ),
        on=["algorithm", "season_id"],
        how="inner",
        validate="one_to_one",
    )
    comparison["yield_change_from_v2_fraction"] = (
        comparison["projected_harvest_kg_m2"]
        / comparison["v2_projected_harvest_kg_m2"]
        - 1.0
    )
    complete = bool(
        len(scenarios) == int(expected_scenario_count)
        and controller_episode_count == int(expected_controller_episode_count)
        and paired_complete
        and reproductive_monotonic
        and season_yield_monotonic
        and max_balance < 1e-9
        and len(comparison) == len(central)
    )
    audit = {
        "complete": complete,
        "scenario_count": int(len(scenarios)),
        "expected_scenario_count": int(expected_scenario_count),
        "controller_episode_count": controller_episode_count,
        "expected_controller_episode_count": int(expected_controller_episode_count),
        "paired_stress_scenarios_complete": paired_complete,
        "yield_monotonic_with_heat_tolerance": season_yield_monotonic,
        "season_aggregate_yield_monotonic_with_heat_tolerance": season_yield_monotonic,
        "season_aggregate_yield_monotonicity_violation_count": season_yield_violation_count,
        "reproductive_retention_monotonic_with_heat_tolerance": reproductive_monotonic,
        "retention_monotonicity_violation_count": retention_violation_count,
        "stress_loss_monotonicity_violation_count": stress_loss_violation_count,
        "draw_harvest_monotonicity_violation_count": draw_harvest_violation_count,
        "maximum_draw_harvest_reversal_kg_m2": max_draw_harvest_reversal,
        "maximum_abs_dry_matter_balance_error_kg_m2": max_balance,
        "spring_partial_yield_envelope_kg_m2": {"low": low, "high": high},
        "target_first_harvest_day": target_day,
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
        "target_harvest_validated": False,
        "accuracy_scope": "regional_heat_stress_sensitivity_not_target_validation",
    }
    return audit, by_algorithm, comparison


def run_audit(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = PROJECT_ROOT / config["controller_benchmark_root"]
    prefix = str(config.get("artifact_prefix", "regional_v3_"))
    priors = json.loads(
        (PROJECT_ROOT / config["regional_prior_path"]).read_text(encoding="utf-8")
    )
    scenarios = pd.read_csv(root / f"{prefix}stress_yield_scenarios.csv")
    seasons = pd.read_csv(root / f"{prefix}stress_yield_by_season.csv")
    v2 = pd.read_csv(root / "regional_v2_yield_reasonableness_by_season.csv")
    stress_count = len(
        config["reproductive_heat_stress"]["floor_retention_by_scenario"]
    )
    metrics = pd.read_csv(root / "episode_metrics.csv")
    draws = int(config["draws_per_episode"])
    audit, by_algorithm, comparison = build_stress_audit(
        priors,
        scenarios,
        seasons,
        v2,
        expected_scenario_count=len(metrics) * draws * stress_count,
        expected_controller_episode_count=len(metrics),
        expected_stress_scenarios=stress_count,
    )
    by_algorithm.to_csv(root / f"{prefix}stress_reasonableness_by_algorithm.csv", index=False)
    comparison.to_csv(root / f"{prefix}vs_v2_by_season.csv", index=False)
    output = root / f"{prefix}stress_reasonableness.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Chengdu Reproductive Heat-Stress Audit",
        "",
        f"- Complete: `{audit['complete']}` ({audit['controller_episode_count']} controller episodes, {audit['scenario_count']} scenarios)",
        f"- Paired stress scenarios complete: `{audit['paired_stress_scenarios_complete']}`",
        f"- Season aggregate yield monotonic with heat tolerance: `{audit['season_aggregate_yield_monotonic_with_heat_tolerance']}`",
        f"- Reproductive retention and loss monotonic: `{audit['reproductive_retention_monotonic_with_heat_tolerance']}`",
        f"- Finite-window draw harvest reversals: `{audit['draw_harvest_monotonicity_violation_count']}` (maximum `{audit['maximum_draw_harvest_reversal_kg_m2']:.4f} kg/m2`)",
        f"- Maximum dry-matter balance error: `{audit['maximum_abs_dry_matter_balance_error_kg_m2']:.3e} kg/m2`",
        "- Target greenhouse harvest validated: `False`",
        "",
        "## Central Scenario",
        "",
        dataframe_to_markdown_table(by_algorithm, float_digits=4),
        "",
        "## Interpretation",
        "",
        "The v3 central scenario applies a literature-constrained reproductive temperature sensitivity to the unchanged v2 controller trajectories. Sensitive and tolerant scenarios are sensitivity bounds, not confidence intervals. Humidity and VPD remain diagnostic-only, and the result is not target-site harvest validation.",
    ]
    (root / f"{prefix}stress_reasonableness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/crops/chengdu_reproductive_heat_stress.yml"
    )
    args = parser.parse_args()
    print(run_audit(args.config))


if __name__ == "__main__":
    main()
