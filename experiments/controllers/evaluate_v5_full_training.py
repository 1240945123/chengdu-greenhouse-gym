from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.controllers.run_v5_controller_smoke import (
    evaluate_classical_policy,
    evaluate_saved_rl_smoke,
    summarize_smoke_trajectory,
)
from experiments.controllers.run_v5_full_training import (
    DEFAULT_OUTPUT_ROOT,
    TrainingJob,
    build_training_jobs,
)
from experiments.controllers.v5_full_training_protocol import (
    DEFAULT_CONFIG_PATH,
    load_v5_full_training_protocol,
)
from experiments.controllers.v5_smoke_controllers import (
    V5FixedBaseline,
    V5HybridMPC,
    V5PIDController,
)


SEASONS = (
    ("2023_spring", 2023, 59, "descriptive_training_weather"),
    ("2023_autumn", 2023, 226, "descriptive_training_weather"),
    ("2024_spring", 2024, 60, "descriptive_training_weather"),
    ("2024_autumn", 2024, 227, "descriptive_training_weather"),
    ("2025_spring", 2025, 59, "descriptive_validation_weather"),
    ("2025_autumn", 2025, 226, "temporal_holdout"),
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def select_validation_candidates(
    metrics: pd.DataFrame,
    *,
    expected_scenarios: Iterable[tuple[int, int]],
    active_safety_limit: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    required = {
        "role",
        "algorithm",
        "seed",
        "checkpoint",
        "growth_year",
        "start_day",
        "completed_episode",
        "numerical_failure",
        "all_values_finite",
        "crop_carbon_nonnegative",
        "residual_fallback_count",
        "active_safety_intervention_fraction",
        "cumulative_reward",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"missing validation metric columns: {', '.join(missing)}")
    expected = {(int(year), int(day)) for year, day in expected_scenarios}
    validation = metrics.loc[metrics["role"].astype(str).eq("validation")].copy()
    records: list[dict[str, object]] = []
    metric_columns = [
        column
        for column in validation.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "checkpoint", "growth_year", "start_day"}
    ]
    for (algorithm, seed, checkpoint), group in validation.groupby(
        ["algorithm", "seed", "checkpoint"], sort=True
    ):
        observed = set(
            zip(group["growth_year"].astype(int), group["start_day"].astype(int))
        )
        complete_scenarios = observed == expected and len(group) == len(expected)
        healthy = bool(
            complete_scenarios
            and group["completed_episode"].astype(bool).all()
            and not group["numerical_failure"].astype(bool).any()
            and group["all_values_finite"].astype(bool).all()
            and group["crop_carbon_nonnegative"].astype(bool).all()
            and (group["residual_fallback_count"].astype(float) == 0.0).all()
            and (
                group["active_safety_intervention_fraction"].astype(float)
                <= float(active_safety_limit)
            ).all()
        )
        record: dict[str, object] = {
            "algorithm": str(algorithm),
            "seed": int(seed),
            "checkpoint": int(checkpoint),
            "scenario_count": int(len(observed & expected)),
            "validation_complete": bool(complete_scenarios),
            "feasible": healthy,
        }
        for column in metric_columns:
            values = group[column].to_numpy(dtype=float)
            if np.isfinite(values).all():
                record[f"mean_{column}"] = float(values.mean())
        records.append(record)
    summary = pd.DataFrame(records).sort_values(
        ["algorithm", "checkpoint", "seed"]
    ).reset_index(drop=True)
    selected: dict[str, dict[str, object]] = {}
    for algorithm, group in summary.groupby("algorithm", sort=True):
        feasible = group.loc[group["feasible"].astype(bool)].sort_values(
            ["mean_cumulative_reward", "checkpoint", "seed"],
            ascending=[False, True, True],
        )
        if feasible.empty:
            continue
        selected[str(algorithm)] = feasible.iloc[0].to_dict()
    return summary, selected


def _validation_unit(
    *,
    job: TrainingJob,
    checkpoint: int,
    scenario: tuple[int, int],
    output_root: str | Path,
    episode_days: int,
) -> dict[str, object]:
    year, day = scenario
    output = Path(output_root)
    model_dir = (
        output
        / "training"
        / job.algorithm
        / f"seed_{job.seed}"
        / f"checkpoint_{int(checkpoint):08d}"
    )
    trajectory_path = (
        output
        / "validation"
        / "trajectories"
        / job.algorithm
        / f"seed_{job.seed}_step_{int(checkpoint)}_{year}_{day}.csv"
    )
    expected_steps = int(episode_days) * 96
    if trajectory_path.exists():
        trajectory = pd.read_csv(trajectory_path)
        if len(trajectory) != expected_steps:
            trajectory_path.unlink()
            trajectory = pd.DataFrame()
    else:
        trajectory = pd.DataFrame()
    if trajectory.empty:
        metrics, trajectory = evaluate_saved_rl_smoke(
            algorithm=job.sb3_algorithm,
            model_dir=model_dir,
            growth_year=year,
            start_day=day,
            episode_days=episode_days,
            residual_pid_scale=job.residual_pid_scale,
        )
        _write_csv(trajectory_path, trajectory)
    else:
        metrics = summarize_smoke_trajectory(job.algorithm, trajectory)
    metrics["algorithm"] = job.algorithm
    return {
        "role": "validation",
        "algorithm": job.algorithm,
        "seed": job.seed,
        "checkpoint": int(checkpoint),
        "growth_year": int(year),
        "start_day": int(day),
        **metrics,
    }


def run_validation_selection(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    checkpoints: Iterable[int] | None = None,
    max_workers: int = 6,
) -> dict[str, dict[str, object]]:
    protocol = load_v5_full_training_protocol(config_path)
    output = Path(output_root)
    selected_checkpoints = tuple(checkpoints or protocol.checkpoints)
    units = [
        (job, checkpoint, scenario)
        for job in build_training_jobs(protocol)
        for checkpoint in selected_checkpoints
        if (
            output
            / "training"
            / job.algorithm
            / f"seed_{job.seed}"
            / f"checkpoint_{int(checkpoint):08d}"
            / "training_metadata.json"
        ).exists()
        for scenario in protocol.validation_scenarios
    ]
    if not units:
        raise FileNotFoundError("no completed V5 full-training checkpoints found")
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(max_workers), len(units)))
    if workers == 1:
        for job, checkpoint, scenario in units:
            rows.append(
                _validation_unit(
                    job=job,
                    checkpoint=checkpoint,
                    scenario=scenario,
                    output_root=output,
                    episode_days=protocol.episode_days,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _validation_unit,
                    job=job,
                    checkpoint=checkpoint,
                    scenario=scenario,
                    output_root=output,
                    episode_days=protocol.episode_days,
                ): (job, checkpoint, scenario)
                for job, checkpoint, scenario in units
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    "validation complete: "
                    f"{row['algorithm']}/seed_{row['seed']}/step_{row['checkpoint']}/"
                    f"{row['growth_year']}_{row['start_day']}",
                    flush=True,
                )
    metrics = pd.DataFrame(rows).sort_values(
        ["algorithm", "seed", "checkpoint", "start_day"]
    )
    _write_csv(output / "validation" / "validation_metrics.csv", metrics)
    summary, selected = select_validation_candidates(
        metrics,
        expected_scenarios=protocol.validation_scenarios,
        active_safety_limit=protocol.active_safety_limit,
    )
    _write_csv(output / "validation" / "candidate_summary.csv", summary)
    for algorithm, record in selected.items():
        record["model_dir"] = str(
            output
            / "training"
            / algorithm
            / f"seed_{int(record['seed'])}"
            / f"checkpoint_{int(record['checkpoint']):08d}"
        )
        record["model_algorithm"] = "ppo" if algorithm == "residual_ppo" else algorithm
        record["residual_pid_scale"] = (
            protocol.residual_ppo_scale if algorithm == "residual_ppo" else None
        )
    _write_json(output / "frozen_rl_selections.json", selected)
    return selected


def _mpc_from_frozen_selection() -> V5HybridMPC:
    path = (
        DEFAULT_OUTPUT_ROOT.parent
        / "v5_controller_optimization"
        / "frozen_selections.json"
    )
    parameters = json.loads(path.read_text(encoding="utf-8"))["mpc"]["configuration"]
    return V5HybridMPC(
        horizon_steps=int(parameters["horizon_steps"]),
        levels=tuple(float(value) for value in parameters["levels"]),
        temperature_weight=float(parameters["temperature_weight"]),
        humidity_weight=float(parameters["humidity_weight"]),
        effort_weight=float(parameters["effort_weight"]),
        variation_weight=float(parameters["variation_weight"]),
    )


def _full_season_unit(
    *,
    algorithm: str,
    selection: dict[str, object] | None,
    season: tuple[str, int, int, str],
    output_root: str | Path,
) -> dict[str, object]:
    season_id, year, day, role = season
    output = Path(output_root)
    trajectory_path = output / "full_season" / "trajectories" / f"{algorithm}_{season_id}.csv"
    if trajectory_path.exists():
        trajectory = pd.read_csv(trajectory_path)
        if len(trajectory) != 120 * 96:
            trajectory_path.unlink()
            trajectory = pd.DataFrame()
    else:
        trajectory = pd.DataFrame()
    if trajectory.empty:
        if algorithm == "baseline":
            metrics, trajectory = evaluate_classical_policy(
                algorithm=algorithm,
                policy=V5FixedBaseline(),
                growth_year=year,
                start_day=day,
                episode_days=120,
            )
        elif algorithm == "pid":
            metrics, trajectory = evaluate_classical_policy(
                algorithm=algorithm,
                policy=V5PIDController(),
                growth_year=year,
                start_day=day,
                episode_days=120,
            )
        elif algorithm == "mpc":
            metrics, trajectory = evaluate_classical_policy(
                algorithm=algorithm,
                policy=_mpc_from_frozen_selection(),
                growth_year=year,
                start_day=day,
                episode_days=120,
            )
        else:
            if selection is None:
                raise ValueError(f"missing frozen selection for {algorithm}")
            metrics, trajectory = evaluate_saved_rl_smoke(
                algorithm=str(selection["model_algorithm"]),
                model_dir=str(selection["model_dir"]),
                growth_year=year,
                start_day=day,
                episode_days=120,
                residual_pid_scale=selection.get("residual_pid_scale"),
            )
            metrics["algorithm"] = algorithm
        _write_csv(trajectory_path, trajectory)
    else:
        metrics = summarize_smoke_trajectory(algorithm, trajectory)
    return {
        "role": role,
        "season_id": season_id,
        "growth_year": year,
        "start_day": day,
        "algorithm": algorithm,
        "seed": int(selection["seed"]) if selection else 0,
        "checkpoint": int(selection["checkpoint"]) if selection else 0,
        **metrics,
    }


def run_full_season_evaluation(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    season_ids: Iterable[str] | None = None,
    algorithms: Iterable[str] = ("baseline", "pid", "mpc", "ppo", "sac", "residual_ppo"),
    max_workers: int = 6,
) -> Path:
    output = Path(output_root)
    selections = json.loads((output / "frozen_rl_selections.json").read_text(encoding="utf-8"))
    requested_seasons = set(season_ids) if season_ids is not None else None
    seasons = tuple(
        season for season in SEASONS if requested_seasons is None or season[0] in requested_seasons
    )
    requested_algorithms = tuple(str(value).lower() for value in algorithms)
    classical = {"baseline", "pid", "mpc"}
    units = [
        (algorithm, selections.get(algorithm), season)
        for season in seasons
        for algorithm in requested_algorithms
        if algorithm in classical or selections.get(algorithm) is not None
    ]
    rows: list[dict[str, object]] = []
    workers = max(1, min(int(max_workers), len(units)))
    if workers == 1:
        for algorithm, selection, season in units:
            rows.append(
                _full_season_unit(
                    algorithm=algorithm,
                    selection=selection,
                    season=season,
                    output_root=output,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _full_season_unit,
                    algorithm=algorithm,
                    selection=selection,
                    season=season,
                    output_root=output,
                ): (algorithm, season[0])
                for algorithm, selection, season in units
            }
            for future in as_completed(futures):
                algorithm, season_id = futures[future]
                rows.append(future.result())
                print(f"full season complete: {algorithm}/{season_id}", flush=True)
    metrics = pd.DataFrame(rows).sort_values(["algorithm", "season_id"])
    _write_csv(output / "full_season" / "episode_metrics.csv", metrics)
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"seed", "checkpoint", "growth_year", "start_day"}
    ]
    aggregate = metrics.groupby("algorithm", as_index=False)[numeric].agg(["mean", "std"])
    aggregate.columns = [
        column if isinstance(column, str) else "_".join(value for value in column if value)
        for column in aggregate.columns
    ]
    _write_csv(output / "full_season" / "comparison_by_algorithm.csv", aggregate)
    return output / "full_season"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validation", "full", "all"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--seasons", nargs="+")
    parser.add_argument("--algorithms", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode in {"validation", "all"}:
        run_validation_selection(
            config_path=args.config,
            output_root=args.output_root,
            max_workers=args.max_workers,
        )
    if args.mode in {"full", "all"}:
        run_full_season_evaluation(
            output_root=args.output_root,
            season_ids=args.seasons,
            algorithms=args.algorithms or ("baseline", "pid", "mpc", "ppo", "sac", "residual_ppo"),
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
