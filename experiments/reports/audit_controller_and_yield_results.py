from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_ALGORITHMS = {"baseline", "pid", "mpc", "ppo", "sac"}
SEVERITY_ORDER = {"critical": 0, "major": 1, "warning": 2, "info": 3}


def _all_runs_complete(manifest: dict[str, Any]) -> bool:
    runs = manifest.get("runs", {})
    if not REQUIRED_ALGORITHMS.issubset(runs):
        return False
    statuses = [
        payload.get("status")
        for algorithm_runs in runs.values()
        for payload in algorithm_runs.values()
    ]
    return bool(statuses) and all(status == "complete" for status in statuses)


def _finding(code: str, severity: str, message: str) -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def audit_results(
    *,
    comparison: pd.DataFrame,
    episodes: pd.DataFrame,
    manifest: dict[str, Any],
    readiness: dict[str, Any],
    plausibility: dict[str, Any],
    episode_days: int,
) -> dict[str, Any]:
    algorithms = set(comparison["algorithm"].astype(str))
    training_complete = _all_runs_complete(manifest)
    unique_scenarios = int(len(episodes[["growth_year", "start_day"]].drop_duplicates()))
    climate_evaluable = training_complete and REQUIRED_ALGORITHMS.issubset(algorithms)
    yield_evaluable = int(episode_days) >= 90 and unique_scenarios >= 6
    target_validated = bool(readiness.get("target_validated", False))
    target_events = int(
        readiness.get("harvest_data", {}).get("target_harvest_event_count", 0)
    )
    season_count = int(plausibility.get("season_count", 0))
    scenario_count = int(plausibility.get("scenario_count", 0))
    mass_balance_error = float(
        plausibility.get("quality_gates", {}).get(
            "maximum_abs_mass_balance_error_kg_m2", float("nan")
        )
    )

    findings: list[dict[str, str]] = []
    if not yield_evaluable:
        findings.append(
            _finding(
                "SHORT_HORIZON_YIELD",
                "critical",
                f"The controller benchmark uses {episode_days} days and "
                f"{unique_scenarios} test scenario; complete-season yield is not evaluable.",
            )
        )
    legacy = pd.to_numeric(
        comparison.get("simulated_fresh_fruit_production_kg_m2_mean"), errors="coerce"
    )
    if legacy.notna().any() and (legacy < 0.0).any():
        findings.append(
            _finding(
                "AMBIGUOUS_NEGATIVE_NET_PRODUCTION",
                "major",
                "The legacy simulated-production field is negative because it is a short-window "
                "fruit inventory balance, not physically negative harvested production.",
            )
        )
    joint = pd.to_numeric(comparison.get("joint_comfort_fraction_mean"), errors="coerce")
    if joint.notna().any() and float(joint.max()) < 0.50:
        findings.append(
            _finding(
                "LOW_JOINT_COMFORT",
                "major",
                f"The best joint temperature-humidity comfort fraction is only {joint.max():.3f}.",
            )
        )
    safety = pd.to_numeric(
        comparison.get("safety_intervention_fraction_mean"), errors="coerce"
    )
    if safety.notna().any() and float(safety.min()) > 0.20:
        findings.append(
            _finding(
                "HIGH_SAFETY_INTERVENTION",
                "major",
                f"Every controller is projected by the safety layer on at least {safety.min():.1%} of steps.",
            )
        )
    if target_events == 0 or not target_validated:
        findings.append(
            _finding(
                "MISSING_TARGET_HARVEST",
                "critical",
                "No dated Pidu harvest events are available; target-site yield accuracy cannot be claimed.",
            )
        )
    if unique_scenarios == 1:
        findings.append(
            _finding(
                "SINGLE_TEST_WEATHER_SCENARIO",
                "major",
                "All reported controller scores use one fixed 2025 autumn test window.",
            )
        )
    findings.sort(key=lambda item: (SEVERITY_ORDER[item["severity"]], item["code"]))

    if target_validated and yield_evaluable:
        verdict = "target_yield_validated"
        calibration_level = "target_calibrated_and_validated"
    else:
        verdict = "control_results_preliminary_yield_not_target_validated"
        calibration_level = (
            "external_transfer_prior" if season_count >= 6 and scenario_count > 0 else "uncalibrated"
        )

    ranking = (
        comparison.sort_values("reward_mean", ascending=False)[
            [
                column
                for column in (
                    "algorithm",
                    "reward_mean",
                    "temperature_mae_mean",
                    "humidity_mae_mean",
                    "joint_comfort_fraction_mean",
                    "safety_intervention_fraction_mean",
                )
                if column in comparison.columns
            ]
        ]
        .to_dict(orient="records")
    )
    return {
        "schema_version": "controller-yield-reasonableness-v1",
        "overall_verdict": verdict,
        "controller_experiment": {
            "training_complete": training_complete,
            "climate_comparison_evaluable": climate_evaluable,
            "yield_comparison_evaluable": yield_evaluable,
            "episode_days": int(episode_days),
            "unique_test_scenarios": unique_scenarios,
            "ranking_scope": "short_horizon_climate_reward_only",
            "reward_ranking": ranking,
        },
        "crop_model": {
            "target_validated": target_validated,
            "target_harvest_event_count": target_events,
            "calibration_level": calibration_level,
            "simulated_season_count": season_count,
            "simulated_scenario_count": scenario_count,
            "maximum_abs_mass_balance_error_kg_m2": mass_balance_error,
            "accuracy_scope": "plausibility_only_not_pidu_harvest_accuracy",
        },
        "findings": findings,
        "claim_policy": {
            "allowed": [
                "SAC has the highest reward on the fixed four-day 2025 autumn climate-control test.",
                "The six-season crop ensemble is a WUR-informed Chengdu simulated prior.",
            ],
            "not_allowed": [
                "SAC increases real Pidu tomato yield.",
                "The crop model accurately predicts Pidu harvest yield.",
            ],
        },
    }


def build_calibration_summary(
    process_prior: dict[str, Any], readiness: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in (
        "maturity_thermal_time_deg_day",
        "fruit_dry_matter_fraction",
        "median_interpick_days",
    ):
        values = process_prior["metrics"][metric]
        rows.append(
            {
                "metric": metric,
                "estimate": float(values["estimate"]),
                "ci95_low": float(values["ci95_low"]),
                "ci95_high": float(values["ci95_high"]),
                "evidence_scope": "external_observed_wur",
                "target_harvest_validated": False,
            }
        )
    standing_wmape = readiness.get("standing_crop_validation", {}).get(
        "validation_wmape"
    )
    rows.append(
        {
            "metric": "pidu_standing_crop_validation_wmape",
            "estimate": float(standing_wmape) if standing_wmape is not None else float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "evidence_scope": "target_standing_crop_not_harvest",
            "target_harvest_validated": bool(readiness.get("target_validated", False)),
        }
    )
    return pd.DataFrame(rows)


def _markdown(audit: dict[str, Any]) -> str:
    controller = audit["controller_experiment"]
    crop = audit["crop_model"]
    lines = [
        "# Controller and Yield Result Reasonableness Audit",
        "",
        f"Overall verdict: `{audit['overall_verdict']}`.",
        "",
        "## Scope",
        "",
        f"- Controller training complete: `{controller['training_complete']}`",
        f"- Climate comparison evaluable: `{controller['climate_comparison_evaluable']}`",
        f"- Complete-season yield comparison evaluable: `{controller['yield_comparison_evaluable']}`",
        f"- Test horizon: {controller['episode_days']} days across {controller['unique_test_scenarios']} scenario(s)",
        f"- Target harvest validated: `{crop['target_validated']}`",
        f"- Calibration level: `{crop['calibration_level']}`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(
        f"- **{item['severity'].upper()} {item['code']}**: {item['message']}"
        for item in audit["findings"]
    )
    lines.extend(["", "## Reward Ranking", "", "| Algorithm | Reward | Temp MAE | RH MAE |", "|---|---:|---:|---:|"])
    for row in controller["reward_ranking"]:
        lines.append(
            f"| {row['algorithm']} | {row.get('reward_mean', float('nan')):.3f} | "
            f"{row.get('temperature_mae_mean', float('nan')):.3f} | "
            f"{row.get('humidity_mae_mean', float('nan')):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Chengdu controller and yield results")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--plausibility", required=True)
    parser.add_argument(
        "--process-prior",
        default="data/processed/external/wur_agc2/wur_process_priors.json",
    )
    parser.add_argument("--episode-days", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark = Path(args.benchmark_dir)
    readiness = json.loads(Path(args.readiness).read_text(encoding="utf-8"))
    audit = audit_results(
        comparison=pd.read_csv(benchmark / "comparison.csv"),
        episodes=pd.read_csv(benchmark / "episode_metrics.csv"),
        manifest=json.loads((benchmark / "manifest.json").read_text(encoding="utf-8")),
        readiness=readiness,
        plausibility=json.loads(Path(args.plausibility).read_text(encoding="utf-8")),
        episode_days=args.episode_days,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result_reasonableness.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "result_reasonableness.md").write_text(_markdown(audit), encoding="utf-8")
    process_prior = json.loads(Path(args.process_prior).read_text(encoding="utf-8"))
    build_calibration_summary(process_prior, readiness).to_csv(
        output / "calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
