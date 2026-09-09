from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.controllers.run_v5_controller_smoke import (
    evaluate_classical_policy,
    evaluate_saved_rl_smoke,
    render_markdown_table,
    train_rl_checkpoints,
)
from experiments.controllers.v5_controller_optimization import (
    CONFIRMATION_SCENARIO,
    HISTORICAL_SMOKE_SCENARIO,
    TRAINING_SCENARIOS,
    VALIDATION_SCENARIOS,
    aggregate_validation_candidates,
    assess_confirmation,
    select_best_candidate,
)
from experiments.controllers.v5_smoke_controllers import V5HybridMPC, V5PIDController


def _scenario_records(scenarios):
    return [
        {"growth_year": int(year), "start_day": int(day)} for year, day in scenarios
    ]


def build_protocol_manifest(
    *, episode_days: int, seeds: list[int], checkpoints: list[int]
) -> dict:
    return {
        "schema_version": "chengdu-v5-controller-optimization-protocol-v1",
        "episode_days": int(episode_days),
        "training_scenarios": _scenario_records(TRAINING_SCENARIOS),
        "validation_scenarios": _scenario_records(VALIDATION_SCENARIOS),
        "confirmation_scenario": _scenario_records([CONFIRMATION_SCENARIO])[0],
        "historical_smoke_scenario": {
            **_scenario_records([HISTORICAL_SMOKE_SCENARIO])[0],
            "used_for_selection": False,
        },
        "rl_seeds": [int(seed) for seed in seeds],
        "rl_checkpoints": [int(value) for value in checkpoints],
        "pid_frozen": True,
        "algorithm_specific_reward": False,
    }


def build_mpc_candidates() -> list[dict]:
    profiles = (
        ("balanced", 1.0, 0.25, 0.20, 0.30),
        ("comfort", 1.5, 0.35, 0.10, 0.15),
        ("smooth", 1.0, 0.25, 0.30, 0.50),
    )
    candidates = []
    for horizon in (1, 2, 4):
        for name, temperature, humidity, effort, variation in profiles:
            levels = [0.0, 0.5, 1.0] if horizon > 1 else [0.0, 0.25, 0.5, 0.75, 1.0]
            candidates.append(
                {
                    "candidate_id": f"mpc_h{horizon}_{name}",
                    "algorithm": "mpc",
                    "horizon_steps": horizon,
                    "levels": levels,
                    "temperature_weight": temperature,
                    "humidity_weight": humidity,
                    "effort_weight": effort,
                    "variation_weight": variation,
                }
            )
    return candidates


def build_confirmation_plan(selections: dict[str, dict]) -> list[dict]:
    required = ("mpc", "ppo", "sac")
    missing = [algorithm for algorithm in required if algorithm not in selections]
    if missing:
        raise ValueError(f"Missing frozen selections: {missing}")
    return [{"algorithm": "pid", "candidate_id": "pid"}] + [
        {
            "algorithm": algorithm,
            "candidate_id": str(selections[algorithm]["candidate_id"]),
        }
        for algorithm in required
    ]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )


def _evaluate_classical(
    *,
    output: Path,
    role: str,
    candidate: dict,
    policy,
    scenario: tuple[int, int],
    episode_days: int,
    resume: bool,
) -> dict:
    year, day = scenario
    stem = f"{role}_{candidate['candidate_id']}_{year}_{day}"
    metrics_path = output / "metrics" / f"{stem}.json"
    trajectory_path = output / "trajectories" / f"{stem}.csv"
    if resume and metrics_path.exists() and trajectory_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics, trajectory = evaluate_classical_policy(
        algorithm=candidate["algorithm"],
        policy=policy,
        growth_year=year,
        start_day=day,
        episode_days=episode_days,
    )
    metrics.update(
        {
            "candidate_id": candidate["candidate_id"],
            "role": role,
            "growth_year": year,
            "start_day": day,
        }
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(trajectory_path, index=False)
    _write_json(metrics_path, metrics)
    return metrics


def _evaluate_rl(
    *,
    output: Path,
    role: str,
    candidate: dict,
    scenario: tuple[int, int],
    episode_days: int,
    resume: bool,
) -> dict:
    year, day = scenario
    stem = f"{role}_{candidate['candidate_id']}_{year}_{day}"
    metrics_path = output / "metrics" / f"{stem}.json"
    trajectory_path = output / "trajectories" / f"{stem}.csv"
    if resume and metrics_path.exists() and trajectory_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics, trajectory = evaluate_saved_rl_smoke(
        algorithm=candidate.get("model_algorithm", candidate["algorithm"]),
        model_dir=candidate["model_dir"],
        growth_year=year,
        start_day=day,
        episode_days=episode_days,
        residual_pid_scale=candidate.get("residual_pid_scale"),
    )
    metrics.update(
        {
            "algorithm": candidate["algorithm"],
            "candidate_id": candidate["candidate_id"],
            "role": role,
            "growth_year": year,
            "start_day": day,
        }
    )
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(trajectory_path, index=False)
    _write_json(metrics_path, metrics)
    return metrics


def _mpc_policy(candidate: dict) -> V5HybridMPC:
    return V5HybridMPC(
        horizon_steps=candidate["horizon_steps"],
        levels=tuple(candidate["levels"]),
        temperature_weight=candidate["temperature_weight"],
        humidity_weight=candidate["humidity_weight"],
        effort_weight=candidate["effort_weight"],
        variation_weight=candidate["variation_weight"],
    )


def _render_report(
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
        "mean_executed_actuator_effort",
        "total_executed_action_variation",
        "fruit_state_change_mg_m2",
    ]
    decision_lines = [
        f"- `{algorithm}`: promoted=`{decision['promoted']}`, reward improvement over PID="
        f"`{decision['reward_improvement_over_pid']:.4f}`, gates=`{decision['gates']}`"
        for algorithm, decision in decisions.items()
    ]
    return "\n".join(
        [
            "# V5 controller optimization result",
            "",
            "## Frozen protocol",
            "",
            "Training uses 2023-2024 scenarios; selection uses 2025 spring days 59, 74, and 89;",
            "confirmation uses 2025 spring day 104 exactly once. The historical autumn smoke test",
            "is not used for tuning. PID and the reward implementation remain frozen.",
            "",
            "## Validation candidate summary",
            "",
            render_markdown_table(validation_summary[validation_columns], float_digits=4),
            "",
            "## Untouched confirmation comparison",
            "",
            render_markdown_table(confirmation[confirmation_columns], float_digits=4),
            "",
            "## Promotion decisions",
            "",
            *decision_lines,
            "",
            "Promotion requires reward strictly greater than PID and every climate, safety, and",
            "numerical gate to pass. Fruit-state change in a one-day episode is not harvested yield",
            "and does not support a yield-superiority claim.",
        ]
    ) + "\n"


def run_optimization(
    *,
    output_dir: str | Path,
    episode_days: int,
    seeds: list[int],
    checkpoints: list[int],
    resume: bool,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_protocol_manifest(
        episode_days=episode_days, seeds=seeds, checkpoints=checkpoints
    )
    _write_json(output / "protocol_manifest.json", manifest)

    validation_rows = []
    print("evaluating frozen PID on validation scenarios", flush=True)
    for scenario in VALIDATION_SCENARIOS:
        validation_rows.append(
            _evaluate_classical(
                output=output,
                role="validation",
                candidate={"algorithm": "pid", "candidate_id": "pid"},
                policy=V5PIDController(),
                scenario=scenario,
                episode_days=episode_days,
                resume=resume,
            )
        )

    mpc_candidates = build_mpc_candidates()
    for index, candidate in enumerate(mpc_candidates, start=1):
        print(f"validating MPC candidate {index}/{len(mpc_candidates)}: {candidate['candidate_id']}", flush=True)
        for scenario in VALIDATION_SCENARIOS:
            validation_rows.append(
                _evaluate_classical(
                    output=output,
                    role="validation",
                    candidate=candidate,
                    policy=_mpc_policy(candidate),
                    scenario=scenario,
                    episode_days=episode_days,
                    resume=resume,
                )
            )

    candidate_registry = {candidate["candidate_id"]: candidate for candidate in mpc_candidates}
    for algorithm in ("ppo", "sac"):
        for seed in seeds:
            training_dir = output / "training" / algorithm / f"seed_{seed}"
            print(f"training {algorithm} seed {seed} through {max(checkpoints)} steps", flush=True)
            train_rl_checkpoints(
                algorithm=algorithm,
                output_dir=training_dir,
                checkpoints=checkpoints,
                seed=seed,
                growth_year=TRAINING_SCENARIOS[0][0],
                start_day=TRAINING_SCENARIOS[0][1],
                episode_days=episode_days,
                training_scenarios=list(TRAINING_SCENARIOS),
                resume=resume,
            )
            for steps in checkpoints:
                candidate = {
                    "algorithm": algorithm,
                    "candidate_id": f"{algorithm}_seed{seed}_steps{steps}",
                    "seed": seed,
                    "total_timesteps": steps,
                    "model_dir": training_dir / f"checkpoint_{steps:08d}",
                }
                candidate_registry[candidate["candidate_id"]] = candidate
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
    summaries = []
    selections = {}
    for algorithm in ("pid", "mpc", "ppo", "sac"):
        algorithm_rows = validation.loc[validation["algorithm"].eq(algorithm)]
        summary = aggregate_validation_candidates(algorithm_rows)
        summary.insert(0, "algorithm", algorithm)
        summaries.append(summary)
        if algorithm != "pid":
            selections[algorithm] = select_best_candidate(summary)
    validation_summary = pd.concat(summaries, ignore_index=True)
    validation_summary.to_csv(output / "validation_candidate_summary.csv", index=False)
    selection_payload = {
        algorithm: {
            **selection,
            "configuration": {
                key: value
                for key, value in candidate_registry[selection["candidate_id"]].items()
                if key != "model_dir"
            },
            "model_dir": str(candidate_registry[selection["candidate_id"]].get("model_dir", "")),
        }
        for algorithm, selection in selections.items()
    }
    _write_json(output / "frozen_selections.json", selection_payload)

    print("selections frozen; opening confirmation scenario once", flush=True)
    confirmation_rows = []
    for item in build_confirmation_plan(selections):
        algorithm = item["algorithm"]
        if algorithm == "pid":
            metrics = _evaluate_classical(
                output=output,
                role="confirmation",
                candidate=item,
                policy=V5PIDController(),
                scenario=CONFIRMATION_SCENARIO,
                episode_days=episode_days,
                resume=resume,
            )
        else:
            candidate = candidate_registry[item["candidate_id"]]
            if algorithm == "mpc":
                metrics = _evaluate_classical(
                    output=output,
                    role="confirmation",
                    candidate=candidate,
                    policy=_mpc_policy(candidate),
                    scenario=CONFIRMATION_SCENARIO,
                    episode_days=episode_days,
                    resume=resume,
                )
            else:
                metrics = _evaluate_rl(
                    output=output,
                    role="confirmation",
                    candidate=candidate,
                    scenario=CONFIRMATION_SCENARIO,
                    episode_days=episode_days,
                    resume=resume,
                )
        confirmation_rows.append(metrics)

    confirmation = pd.DataFrame(confirmation_rows)
    confirmation.to_csv(output / "confirmation_metrics.csv", index=False)
    pid_metrics = confirmation.loc[confirmation["algorithm"].eq("pid")].iloc[0].to_dict()
    decisions = {
        algorithm: assess_confirmation(
            confirmation.loc[confirmation["algorithm"].eq(algorithm)].iloc[0].to_dict(),
            pid_metrics,
        )
        for algorithm in ("mpc", "ppo", "sac")
    }
    _write_json(output / "promotion_decisions.json", decisions)
    (output / "scientific_interpretation.md").write_text(
        _render_report(validation_summary, confirmation, decisions), encoding="utf-8"
    )
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize V5 MPC, PPO, and SAC against frozen PID.")
    parser.add_argument(
        "--output-dir",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/v5_controller_optimization",
    )
    parser.add_argument("--episode-days", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    decisions = run_optimization(
        output_dir=args.output_dir,
        episode_days=args.episode_days,
        seeds=args.seeds,
        checkpoints=args.checkpoints,
        resume=args.resume,
    )
    print(json.dumps(decisions, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
