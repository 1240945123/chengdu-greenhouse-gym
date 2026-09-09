from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.controllers.run_v5_controller_optimization import (
    _evaluate_classical,
    _evaluate_rl,
    _mpc_policy,
    _write_json,
)
from experiments.controllers.run_v5_controller_smoke import (
    render_markdown_table,
    train_rl_checkpoints,
)
from experiments.controllers.v5_controller_optimization import (
    CONFIRMATION_SCENARIO,
    TRAINING_SCENARIOS,
    VALIDATION_SCENARIOS,
    aggregate_validation_candidates,
    assess_confirmation,
    select_best_candidate,
)
from experiments.controllers.v5_smoke_controllers import V5PIDController


SECOND_CONFIRMATION_SCENARIO = (2025, 119)


def _scenario(year_day: tuple[int, int]) -> dict:
    return {"growth_year": int(year_day[0]), "start_day": int(year_day[1])}


def build_residual_protocol_manifest(
    *, seeds: list[int], checkpoints: list[int], residual_scale: float
) -> dict:
    return {
        "schema_version": "chengdu-v5-residual-ppo-protocol-v1",
        "algorithm_label": "residual_ppo",
        "base_controller": "frozen_pid",
        "pure_ppo_first_stage_result": "failed",
        "training_scenarios": [_scenario(item) for item in TRAINING_SCENARIOS],
        "validation_scenarios": [_scenario(item) for item in VALIDATION_SCENARIOS],
        "first_confirmation_scenario": {
            **_scenario(CONFIRMATION_SCENARIO),
            "used_for_tuning": False,
            "reused_for_claim": False,
        },
        "second_confirmation_scenario": _scenario(SECOND_CONFIRMATION_SCENARIO),
        "seeds": [int(seed) for seed in seeds],
        "checkpoints": [int(value) for value in checkpoints],
        "residual_scale": float(residual_scale),
        "common_reward_unchanged": True,
        "pid_unchanged": True,
    }


def _report(
    validation_summary: pd.DataFrame,
    confirmation: pd.DataFrame,
    decisions: dict,
) -> str:
    validation_columns = [
        "candidate_id",
        "mean_cumulative_reward",
        "mean_temperature_band_mae_c",
        "mean_relative_humidity_band_mae_percent",
        "mean_total_executed_action_variation",
        "feasible",
    ]
    confirmation_columns = [
        "algorithm",
        "candidate_id",
        "cumulative_reward",
        "temperature_band_mae_c",
        "relative_humidity_band_mae_percent",
        "joint_comfort_fraction",
        "safety_intervention_fraction",
        "material_safety_intervention_fraction",
        "active_safety_intervention_fraction",
        "mean_executed_actuator_effort",
        "total_executed_action_variation",
        "fruit_state_change_mg_m2",
    ]
    decision_lines = [
        f"- `{algorithm}`: promoted=`{decision['promoted']}`, reward delta vs PID="
        f"`{decision['reward_improvement_over_pid']:.4f}`, safety metric="
        f"`{decision.get('safety_metric', 'safety_intervention_fraction')}`, "
        f"gates=`{decision['gates']}`"
        for algorithm, decision in decisions.items()
    ]
    return "\n".join(
        [
            "# V5 Residual-PPO follow-up",
            "",
            "Pure PPO remains a failed first-stage result and is not renamed or replaced.",
            "Residual-PPO uses frozen PID as the base action and learns a bounded correction.",
            "Selection uses only spring days 59, 74, and 89. The first confirmation day 104",
            "was not reused. All frozen controllers were evaluated once on fresh day 119.",
            "The legacy strict safety fraction uses the historical 1e-12 comparison and is",
            "retained for compatibility. `material_safety_intervention_fraction` and",
            "`active_safety_intervention_fraction` use a 1e-6 threshold; promotion uses the",
            "active roof-vent/fan metric while fallbacks and numerical failures remain gates.",
            "",
            "## Residual-PPO validation candidates",
            "",
            render_markdown_table(validation_summary[validation_columns], float_digits=4),
            "",
            "## Fresh unified confirmation",
            "",
            render_markdown_table(confirmation[confirmation_columns], float_digits=4),
            "",
            "## Decisions",
            "",
            *decision_lines,
            "",
            "One-day fruit-state change is not harvested yield. This stage supports a controller",
            "comparison only, not a yield-superiority claim.",
        ]
    ) + "\n"


def run_residual_optimization(
    *,
    first_stage_dir: str | Path,
    output_dir: str | Path,
    seeds: list[int],
    checkpoints: list[int],
    residual_scale: float,
    episode_days: int,
    resume: bool,
) -> dict:
    first_stage = Path(first_stage_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selections = json.loads((first_stage / "frozen_selections.json").read_text(encoding="utf-8"))
    first_decisions = json.loads((first_stage / "promotion_decisions.json").read_text(encoding="utf-8"))
    if first_decisions["ppo"]["promoted"]:
        raise ValueError("Residual-PPO follow-up is only valid after pure PPO fails")
    manifest = build_residual_protocol_manifest(
        seeds=seeds, checkpoints=checkpoints, residual_scale=residual_scale
    )
    _write_json(output / "protocol_manifest.json", manifest)

    registry = {}
    validation_rows = []
    for seed in seeds:
        training_dir = output / "training" / f"seed_{seed}"
        print(f"training residual PPO seed {seed} through {max(checkpoints)} steps", flush=True)
        train_rl_checkpoints(
            algorithm="ppo",
            output_dir=training_dir,
            checkpoints=checkpoints,
            seed=seed,
            growth_year=TRAINING_SCENARIOS[0][0],
            start_day=TRAINING_SCENARIOS[0][1],
            episode_days=episode_days,
            training_scenarios=list(TRAINING_SCENARIOS),
            residual_pid_scale=residual_scale,
            resume=resume,
        )
        for steps in checkpoints:
            candidate = {
                "algorithm": "residual_ppo",
                "model_algorithm": "ppo",
                "candidate_id": f"residual_ppo_seed{seed}_steps{steps}",
                "seed": int(seed),
                "total_timesteps": int(steps),
                "residual_pid_scale": float(residual_scale),
                "model_dir": training_dir / f"checkpoint_{steps:08d}",
            }
            registry[candidate["candidate_id"]] = candidate
            print(f"validating {candidate['candidate_id']}", flush=True)
            for scenario in VALIDATION_SCENARIOS:
                validation_rows.append(
                    _evaluate_rl(
                        output=output,
                        role="validation",
                        candidate=candidate,
                        scenario=scenario,
                        episode_days=episode_days,
                        resume=resume,
                    )
                )

    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output / "validation_metrics.csv", index=False)
    summary = aggregate_validation_candidates(validation)
    summary.insert(0, "algorithm", "residual_ppo")
    summary.to_csv(output / "validation_candidate_summary.csv", index=False)
    selected = select_best_candidate(summary)
    selected_candidate = registry[selected["candidate_id"]]
    _write_json(
        output / "frozen_selection.json",
        {
            **selected,
            "configuration": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in selected_candidate.items()
            },
        },
    )

    mpc_candidate = selections["mpc"]["configuration"]
    sac_candidate = {
        "algorithm": "sac",
        "candidate_id": selections["sac"]["candidate_id"],
        "model_dir": Path(selections["sac"]["model_dir"]),
    }
    print("opening fresh unified confirmation day 119", flush=True)
    confirmation_rows = [
        _evaluate_classical(
            output=output,
            role="confirmation2",
            candidate={"algorithm": "pid", "candidate_id": "pid"},
            policy=V5PIDController(),
            scenario=SECOND_CONFIRMATION_SCENARIO,
            episode_days=episode_days,
            resume=resume,
        ),
        _evaluate_classical(
            output=output,
            role="confirmation2",
            candidate=mpc_candidate,
            policy=_mpc_policy(mpc_candidate),
            scenario=SECOND_CONFIRMATION_SCENARIO,
            episode_days=episode_days,
            resume=resume,
        ),
        _evaluate_rl(
            output=output,
            role="confirmation2",
            candidate=sac_candidate,
            scenario=SECOND_CONFIRMATION_SCENARIO,
            episode_days=episode_days,
            resume=resume,
        ),
        _evaluate_rl(
            output=output,
            role="confirmation2",
            candidate=selected_candidate,
            scenario=SECOND_CONFIRMATION_SCENARIO,
            episode_days=episode_days,
            resume=resume,
        ),
    ]
    confirmation = pd.DataFrame(confirmation_rows)
    confirmation.to_csv(output / "confirmation_metrics.csv", index=False)
    pid = confirmation.loc[confirmation["algorithm"].eq("pid")].iloc[0].to_dict()
    decisions = {
        algorithm: assess_confirmation(
            confirmation.loc[confirmation["algorithm"].eq(algorithm)].iloc[0].to_dict(),
            pid,
        )
        for algorithm in ("mpc", "sac", "residual_ppo")
    }
    _write_json(output / "promotion_decisions.json", decisions)
    (output / "scientific_interpretation.md").write_text(
        _report(summary, confirmation, decisions), encoding="utf-8"
    )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V5 Residual-PPO follow-up.")
    parser.add_argument(
        "--first-stage-dir",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/v5_controller_optimization",
    )
    parser.add_argument(
        "--output-dir",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/v5_residual_ppo_optimization",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--episode-days", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    decisions = run_residual_optimization(
        first_stage_dir=args.first_stage_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        residual_scale=args.residual_scale,
        episode_days=args.episode_days,
        resume=args.resume,
    )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
