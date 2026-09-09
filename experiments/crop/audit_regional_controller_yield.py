from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

from experiments.controllers.benchmark_protocol import PROJECT_ROOT


def dataframe_to_markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{int(float_digits)}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def compare_yield_transfer_versions(
    original: pd.DataFrame, regional: pd.DataFrame
) -> pd.DataFrame:
    old = original[["algorithm", "projected_harvest_mean_kg_m2"]].rename(
        columns={"projected_harvest_mean_kg_m2": "wur_transfer_mean_kg_m2"}
    )
    new = regional[["algorithm", "projected_yield_mean_kg_m2"]].rename(
        columns={"projected_yield_mean_kg_m2": "regional_transfer_mean_kg_m2"}
    )
    merged = old.merge(new, on="algorithm", how="inner", validate="one_to_one")
    if len(merged) != len(old) or len(merged) != len(new):
        raise ValueError("yield transfer versions have different algorithm sets")
    merged["absolute_change_kg_m2"] = (
        merged["regional_transfer_mean_kg_m2"] - merged["wur_transfer_mean_kg_m2"]
    )
    merged["relative_change_fraction"] = (
        merged["absolute_change_kg_m2"] / merged["wur_transfer_mean_kg_m2"]
    )
    return merged.sort_values("algorithm", kind="stable").reset_index(drop=True)


def compare_regional_audit_versions(
    v1: pd.DataFrame, v2: pd.DataFrame
) -> pd.DataFrame:
    metrics = [
        "projected_yield_mean_kg_m2",
        "first_harvest_mae_days",
        "spring_envelope_coverage_fraction",
    ]
    old = v1[["algorithm", *metrics]].rename(
        columns={name: f"{name}_v1" for name in metrics}
    )
    new = v2[["algorithm", *metrics]].rename(
        columns={name: f"{name}_v2" for name in metrics}
    )
    merged = old.merge(new, on="algorithm", how="inner", validate="one_to_one")
    if len(merged) != len(old) or len(merged) != len(new):
        raise ValueError("regional audit versions have different algorithm sets")
    merged["yield_change_fraction"] = (
        merged["projected_yield_mean_kg_m2_v2"]
        / merged["projected_yield_mean_kg_m2_v1"]
        - 1.0
    )
    merged["first_harvest_mae_improvement_days"] = (
        merged["first_harvest_mae_days_v1"]
        - merged["first_harvest_mae_days_v2"]
    )
    merged["spring_envelope_coverage_change"] = (
        merged["spring_envelope_coverage_fraction_v2"]
        - merged["spring_envelope_coverage_fraction_v1"]
    )
    return merged.sort_values("algorithm", kind="stable").reset_index(drop=True)


def _spring_envelope_status(value: float, low: float, high: float, season_id: str) -> str:
    if not str(season_id).endswith("_spring"):
        return "not_applicable"
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "inside"


def _interval_status(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "missing"
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "inside"


def _paired_yield_tests(seasons: pd.DataFrame) -> dict[str, Any]:
    pivot = seasons.pivot(index="season_id", columns="algorithm", values="projected_harvest_kg_m2")
    pivot = pivot.dropna(axis=0, how="any")
    algorithms = sorted(pivot.columns.astype(str))
    result: dict[str, Any] = {
        "paired_season_count": int(len(pivot)),
        "algorithms": algorithms,
        "friedman_statistic": None,
        "friedman_p_value": None,
        "baseline_pairwise_wilcoxon_holm": [],
    }
    if len(pivot) < 2 or len(algorithms) < 3:
        return result
    friedman = friedmanchisquare(*(pivot[name].to_numpy(dtype=float) for name in algorithms))
    result["friedman_statistic"] = float(friedman.statistic)
    result["friedman_p_value"] = float(friedman.pvalue)
    raw: list[dict[str, Any]] = []
    for algorithm in algorithms:
        if algorithm == "baseline":
            continue
        test = wilcoxon(
            pivot[algorithm].to_numpy(dtype=float),
            pivot["baseline"].to_numpy(dtype=float),
            alternative="two-sided",
            zero_method="wilcox",
        )
        raw.append(
            {
                "algorithm": algorithm,
                "statistic": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    order = sorted(range(len(raw)), key=lambda index: raw[index]["p_value"])
    adjusted = [1.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, raw[index]["p_value"] * (len(raw) - rank))
        running = max(running, candidate)
        adjusted[index] = running
    for item, value in zip(raw, adjusted, strict=True):
        item["holm_adjusted_p_value"] = float(value)
    result["baseline_pairwise_wilcoxon_holm"] = raw
    return result


def build_regional_reasonableness_audit(
    priors: dict[str, Any],
    scenarios: pd.DataFrame,
    seasons: pd.DataFrame,
    *,
    expected_scenario_count: int,
    expected_episode_count: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    envelope = priors["spring_partial_yield_envelope"]
    low = float(envelope["low_kg_m2"])
    high = float(envelope["high_kg_m2"])
    target_day = float(priors["maturity_thermal_time"]["target_first_harvest_day"])
    onset_days = priors["maturity_thermal_time"].get(
        "onset_days", [target_day - 5.0, target_day, target_day + 5.0]
    )
    timing_low = float(min(onset_days))
    timing_high = float(max(onset_days))
    by_season = seasons.copy()
    by_season["first_harvest_error_days"] = (
        by_season["projected_first_harvest_day"] - target_day
    )
    by_season["first_harvest_abs_error_days"] = by_season[
        "first_harvest_error_days"
    ].abs()
    by_season["first_harvest_window_status"] = [
        _interval_status(float(value), timing_low, timing_high)
        for value in by_season["projected_first_harvest_day"]
    ]
    by_season["spring_envelope_status"] = [
        _spring_envelope_status(value, low, high, season_id)
        for value, season_id in zip(
            by_season["projected_harvest_kg_m2"],
            by_season["season_id"],
            strict=True,
        )
    ]
    records: list[dict[str, Any]] = []
    for algorithm, group in by_season.groupby("algorithm", sort=True):
        spring = group.loc[group["spring_envelope_status"].ne("not_applicable")]
        records.append(
            {
                "algorithm": str(algorithm),
                "season_count": int(len(group)),
                "projected_yield_mean_kg_m2": float(group["projected_harvest_kg_m2"].mean()),
                "projected_yield_std_kg_m2": float(group["projected_harvest_kg_m2"].std(ddof=1)),
                "first_harvest_bias_days": float(group["first_harvest_error_days"].mean()),
                "first_harvest_mae_days": float(group["first_harvest_abs_error_days"].mean()),
                "first_harvest_window_inside_count": int(
                    group["first_harvest_window_status"].eq("inside").sum()
                ),
                "first_harvest_window_coverage_fraction": float(
                    group["first_harvest_window_status"].eq("inside").mean()
                ),
                "spring_season_count": int(len(spring)),
                "spring_envelope_inside_count": int(spring["spring_envelope_status"].eq("inside").sum()),
                "spring_envelope_coverage_fraction": float(
                    spring["spring_envelope_status"].eq("inside").mean()
                ),
                "target_eligible": False,
            }
        )
    by_algorithm = pd.DataFrame(records)
    episode_count = int(
        scenarios[["algorithm", "seed", "season_id"]].drop_duplicates().shape[0]
    )
    max_balance = float(scenarios["dry_matter_balance_error_kg_m2"].abs().max())
    complete = bool(
        len(scenarios) == int(expected_scenario_count)
        and episode_count == int(expected_episode_count)
        and max_balance < 1e-9
    )
    audit = {
        "complete": complete,
        "scenario_count": int(len(scenarios)),
        "expected_scenario_count": int(expected_scenario_count),
        "episode_count": episode_count,
        "expected_episode_count": int(expected_episode_count),
        "max_abs_mass_balance_error_kg_m2": max_balance,
        "target_first_harvest_day": target_day,
        "first_harvest_window_days": {"low": timing_low, "high": timing_high},
        "spring_partial_yield_envelope_kg_m2": {"low": low, "high": high},
        "paired_yield_statistics": _paired_yield_tests(by_season),
        "evidence_class": "external_transfer_prior",
        "target_eligible": False,
        "target_harvest_validated": False,
        "accuracy_scope": "regional_plausibility_not_pidu_harvest_accuracy",
        "blocking_reasons": list(priors.get("limitations", [])),
    }
    return audit, by_algorithm, by_season


def run_audit(
    controller_root: str | Path,
    prior_path: str | Path,
    *,
    artifact_prefix: str = "regional_",
) -> Path:
    root = Path(controller_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    prior = Path(prior_path)
    if not prior.is_absolute():
        prior = PROJECT_ROOT / prior
    priors = json.loads(prior.read_text(encoding="utf-8"))
    scenarios = pd.read_csv(root / f"{artifact_prefix}cohort_yield_scenarios.csv")
    seasons = pd.read_csv(root / f"{artifact_prefix}cohort_yield_by_season.csv")
    audit, by_algorithm, by_season = build_regional_reasonableness_audit(
        priors,
        scenarios,
        seasons,
        expected_scenario_count=78 * 16,
        expected_episode_count=78,
    )
    comparison_path = root / "comparison_by_season.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)[
            ["algorithm", "season_id", "cumulative_reward"]
        ]
        aligned = by_season.merge(comparison, on=["algorithm", "season_id"], how="inner")
        correlation = spearmanr(
            aligned["cumulative_reward"].to_numpy(dtype=float),
            aligned["projected_harvest_kg_m2"].to_numpy(dtype=float),
        )
        audit["reward_yield_spearman"] = {
            "pair_count": int(len(aligned)),
            "rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
        }
    by_algorithm.to_csv(
        root / f"{artifact_prefix}yield_reasonableness_by_algorithm.csv", index=False
    )
    by_season.to_csv(
        root / f"{artifact_prefix}yield_reasonableness_by_season.csv", index=False
    )
    v1_audit_path = root / "regional_yield_reasonableness_by_algorithm.csv"
    if artifact_prefix != "regional_" and v1_audit_path.exists():
        version_comparison = compare_regional_audit_versions(
            pd.read_csv(v1_audit_path), by_algorithm
        )
        version_comparison.to_csv(
            root / f"{artifact_prefix}vs_v1_by_algorithm.csv", index=False
        )
        audit["v1_v2_comparison_available"] = True
    original_path = root / "cohort_yield_by_algorithm.csv"
    sensitivity = None
    if original_path.exists():
        sensitivity = compare_yield_transfer_versions(
            pd.read_csv(original_path), by_algorithm
        )
        sensitivity.to_csv(
            root / f"{artifact_prefix}yield_transfer_sensitivity.csv", index=False
        )
        audit["yield_transfer_relative_change_range"] = {
            "minimum": float(sensitivity["relative_change_fraction"].min()),
            "maximum": float(sensitivity["relative_change_fraction"].max()),
        }
    audit_path = root / f"{artifact_prefix}yield_reasonableness.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Chengdu Regional Yield Transfer Audit",
        "",
        f"- Complete: `{audit['complete']}` ({audit['episode_count']} episodes, {audit['scenario_count']} scenarios)",
        f"- Maximum dry-matter balance error: `{audit['max_abs_mass_balance_error_kg_m2']:.3e} kg/m2`",
        f"- Regional first-harvest target: `{audit['target_first_harvest_day']:.0f} crop days`",
        f"- First-harvest acceptance window: `{audit['first_harvest_window_days']['low']:.0f}-{audit['first_harvest_window_days']['high']:.0f} crop days`",
        f"- Spring 120-day plausibility envelope: `{audit['spring_partial_yield_envelope_kg_m2']['low']:.3f}-{audit['spring_partial_yield_envelope_kg_m2']['high']:.3f} kg/m2`",
        "- Target Pidu harvest validated: `False`",
        "",
        "## Algorithms",
        "",
        dataframe_to_markdown_table(by_algorithm, float_digits=4),
        "",
    ]
    if sensitivity is not None:
        lines.extend(
            [
                "## Transfer Sensitivity",
                "",
                dataframe_to_markdown_table(sensitivity, float_digits=4),
                "",
            ]
        )
    lines.extend(
        [
        "## Interpretation",
        "",
        "The timing prior is regionally constrained, while production remains a mass-conserving GreenLight cohort projection. Spring envelope coverage is a plausibility check, not measured target-site accuracy. Autumn has no matched regional yield source and is intentionally not scored against the spring envelope.",
        ]
    )
    (root / f"{artifact_prefix}yield_reasonableness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller-root",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/six_season_120d_guarded_v2",
    )
    parser.add_argument(
        "--prior-path",
        default="data/processed/external/chengdu_regional_tomato/regional_transfer_priors.json",
    )
    parser.add_argument("--artifact-prefix", default="regional_")
    args = parser.parse_args()
    print(
        run_audit(
            args.controller_root,
            args.prior_path,
            artifact_prefix=args.artifact_prefix,
        )
    )


if __name__ == "__main__":
    main()
