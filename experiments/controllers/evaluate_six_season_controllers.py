from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.controllers.benchmark_metrics import summarize_episode
from experiments.controllers.benchmark_protocol import (
    BenchmarkConfig,
    ENV_CONFIG_PATH,
    PROJECT_ROOT,
    load_benchmark_config,
)
from experiments.controllers.train_benchmark_rl import evaluate_saved_agent
from experiments.controllers.tune_benchmark_classical import evaluate_classical_controller


ALGORITHMS = ("baseline", "pid", "mpc", "ppo", "sac")
DETERMINISTIC_ALGORITHMS = frozenset(("baseline", "pid", "mpc"))
DT_SECONDS = 900
NULLABLE_TEXT_COLUMNS = frozenset(
    (
        "controller_failure_kind",
        "controller_failure_message",
        "controller_fallback_failure_kind",
        "environment_failure_kind",
        "environment_failure_phase",
        "environment_failure_exception",
        "environment_failure_message",
    )
)


@dataclass(frozen=True)
class EvaluationSeason:
    season_id: str
    growth_year: int
    start_day: int
    episode_days: int = 120


@dataclass(frozen=True)
class EvaluationJob:
    algorithm: str
    seed: int
    season: EvaluationSeason

    @property
    def unit_id(self) -> str:
        return f"{self.algorithm}/seed_{self.seed}/{self.season.season_id}"


def canonical_evaluation_seasons() -> tuple[EvaluationSeason, ...]:
    return (
        EvaluationSeason("2023_spring", 2023, 59),
        EvaluationSeason("2023_autumn", 2023, 226),
        EvaluationSeason("2024_spring", 2024, 60),
        EvaluationSeason("2024_autumn", 2024, 227),
        EvaluationSeason("2025_spring", 2025, 59),
        EvaluationSeason("2025_autumn", 2025, 226),
    )


def trajectory_filename(
    algorithm: str,
    seed: int,
    season: EvaluationSeason,
) -> str:
    return f"{algorithm.lower()}_seed_{int(seed)}_{season.season_id}.csv"


def expand_evaluation_jobs(
    *,
    algorithms: Iterable[str],
    seasons: Iterable[EvaluationSeason],
    evaluation_seeds: Iterable[int],
) -> tuple[EvaluationJob, ...]:
    selected = tuple(str(value).lower() for value in algorithms)
    unsupported = sorted(set(selected) - set(ALGORITHMS))
    if unsupported:
        raise ValueError(f"unsupported algorithms: {', '.join(unsupported)}")
    seeds = tuple(int(seed) for seed in evaluation_seeds)
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    jobs: list[EvaluationJob] = []
    for season in seasons:
        for algorithm in selected:
            algorithm_seeds = (0,) if algorithm in DETERMINISTIC_ALGORITHMS else seeds
            jobs.extend(EvaluationJob(algorithm, seed, season) for seed in algorithm_seeds)
    return tuple(jobs)


def validate_complete_trajectory(
    trajectory: pd.DataFrame,
    *,
    season: EvaluationSeason,
    location: str,
    expected_steps: int,
) -> None:
    required = {
        "growth_year",
        "start_day",
        "location",
        "uBoil",
        "uCO2",
        "reward",
        "terminated",
        "truncated",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")
    if len(trajectory) != int(expected_steps):
        raise ValueError(
            f"unexpected step count: expected {int(expected_steps)}, got {len(trajectory)}"
        )
    if set(trajectory["growth_year"].astype(int)) != {season.growth_year}:
        raise ValueError("trajectory growth year does not match season")
    if set(trajectory["start_day"].astype(int)) != {season.start_day}:
        raise ValueError("trajectory start day does not match season")
    if set(trajectory["location"].astype(str)) != {str(location)}:
        raise ValueError("trajectory location does not match protocol")
    numeric = trajectory.drop(
        columns=list(NULLABLE_TEXT_COLUMNS & set(trajectory.columns))
    ).select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("trajectory contains non-finite numeric values")
    if not np.allclose(trajectory[["uBoil", "uCO2"]].to_numpy(dtype=float), 0.0):
        raise ValueError("disabled controls must remain zero")
    if not bool(trajectory["terminated"].iloc[-1]):
        raise ValueError("trajectory did not terminate at the episode boundary")
    if trajectory["truncated"].astype(bool).any():
        raise ValueError("trajectory contains a truncated step")


def can_resume_trajectory(
    path: str | Path,
    *,
    season: EvaluationSeason,
    location: str,
    expected_steps: int,
) -> bool:
    artifact = Path(path)
    if not artifact.exists():
        return False
    try:
        validate_complete_trajectory(
            pd.read_csv(artifact),
            season=season,
            location=location,
            expected_steps=expected_steps,
        )
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return True


def upsert_episode_metric(path: str | Path, record: dict[str, object]) -> None:
    artifact = Path(path)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    identity = (str(record["algorithm"]), int(record["seed"]), str(record["season_id"]))
    if artifact.exists():
        table = pd.read_csv(artifact)
        keep = ~(
            (table["algorithm"].astype(str) == identity[0])
            & (table["seed"].astype(int) == identity[1])
            & (table["season_id"].astype(str) == identity[2])
        )
        table = pd.concat([table.loc[keep], pd.DataFrame([record])], ignore_index=True)
    else:
        table = pd.DataFrame([record])
    table = table.sort_values(["algorithm", "seed", "season_id"]).reset_index(drop=True)
    temporary = artifact.with_suffix(artifact.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(artifact)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def aggregate_complete_episodes(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"algorithm", "seed", "season_id", "cumulative_reward"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise ValueError(f"missing episode metric columns: {', '.join(missing)}")
    if episodes.empty:
        raise ValueError("cannot aggregate empty episode metrics")
    numeric_metrics = [
        column
        for column in episodes.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    by_season = (
        episodes.groupby(["algorithm", "season_id"], as_index=False)[numeric_metrics]
        .mean()
        .sort_values(["algorithm", "season_id"])
        .reset_index(drop=True)
    )
    records: list[dict[str, object]] = []
    for algorithm, group in by_season.groupby("algorithm", sort=True):
        record: dict[str, object] = {
            "algorithm": str(algorithm),
            "season_count": int(len(group)),
        }
        for column in numeric_metrics:
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                continue
            record[f"{column}_mean"] = float(values.mean())
            record[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            record[f"{column}_median"] = float(np.median(values))
            if column == "cumulative_reward":
                low, high = _bootstrap_mean_interval(values)
                record[f"{column}_ci95_low"] = low
                record[f"{column}_ci95_high"] = high
        records.append(record)
    return pd.DataFrame(records), by_season


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_output_root(config: BenchmarkConfig, output_name: str) -> Path:
    name = str(output_name)
    if not name or Path(name).name != name:
        raise ValueError("output_name must be one directory name")
    benchmark_root = PROJECT_ROOT / config.output_dir
    return benchmark_root.parent / name


def _protocol_payload(
    config: BenchmarkConfig,
    jobs: tuple[EvaluationJob, ...],
    benchmark_root: Path,
) -> dict[str, object]:
    units = [job.unit_id for job in jobs]
    core: dict[str, object] = {
        "profile": config.profile,
        "dataset_id": config.dataset_id,
        "location": config.location,
        "dt_seconds": DT_SECONDS,
        "expected_steps_per_episode": 120 * 86400 // DT_SECONDS,
        "yield_interpretation": "GreenLight simulated yield; not Chengdu target-validated",
        "units": units,
        "seasons": [season.__dict__ for season in canonical_evaluation_seasons()],
        "environment_config_sha256": _sha256(ENV_CONFIG_PATH),
        "calibration_sha256": _sha256(config.calibration_path),
    }
    artifact_fingerprints: dict[str, str] = {}
    for job in jobs:
        if job.algorithm in DETERMINISTIC_ALGORITHMS:
            if job.algorithm != "baseline":
                path = benchmark_root / "tuning" / job.algorithm / "selected_params.json"
                artifact_fingerprints[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
        else:
            run_dir = benchmark_root / "training" / job.algorithm / f"seed_{job.seed}"
            for filename in ("best_model.zip", "best_vecnormalize.pkl"):
                path = run_dir / filename
                artifact_fingerprints[str(path.relative_to(PROJECT_ROOT))] = _sha256(path)
    core["controller_artifact_sha256"] = artifact_fingerprints
    encoded = json.dumps(core, sort_keys=True).encode("utf-8")
    core["protocol_sha256"] = hashlib.sha256(encoded).hexdigest()
    return core


def _initialize_manifest(
    path: Path,
    protocol: dict[str, object],
    jobs: tuple[EvaluationJob, ...],
) -> dict[str, object]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("existing manifest protocol does not match this evaluation")
        return manifest
    manifest: dict[str, object] = {
        "protocol_sha256": protocol["protocol_sha256"],
        "units": {job.unit_id: {"status": "pending"} for job in jobs},
    }
    _write_json_atomic(path, manifest)
    return manifest


def _transition_manifest(
    path: Path,
    unit_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entry: dict[str, object] = {"status": status}
    if error is not None:
        entry["error"] = error
    manifest["units"][unit_id] = entry
    _write_json_atomic(path, manifest)


def _selected_classical_params(benchmark_root: Path, algorithm: str) -> dict[str, object] | None:
    if algorithm == "baseline":
        return None
    path = benchmark_root / "tuning" / algorithm / "selected_params.json"
    if not path.exists():
        raise FileNotFoundError(f"missing selected {algorithm} parameters: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["selected_params"]


def _validate_partial_trajectory(
    trajectory: pd.DataFrame,
    *,
    job: EvaluationJob,
    location: str,
    expected_steps: int,
) -> None:
    if len(trajectory) != expected_steps:
        raise ValueError(f"smoke trajectory expected {expected_steps} rows, got {len(trajectory)}")
    if set(trajectory["growth_year"].astype(int)) != {job.season.growth_year}:
        raise ValueError("smoke trajectory growth year mismatch")
    if set(trajectory["start_day"].astype(int)) != {job.season.start_day}:
        raise ValueError("smoke trajectory start day mismatch")
    if set(trajectory["location"].astype(str)) != {location}:
        raise ValueError("smoke trajectory location mismatch")
    numeric = trajectory.drop(
        columns=list(NULLABLE_TEXT_COLUMNS & set(trajectory.columns))
    ).select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("smoke trajectory contains non-finite values")
    if not np.allclose(trajectory[["uBoil", "uCO2"]].to_numpy(dtype=float), 0.0):
        raise ValueError("disabled controls must remain zero")


def _execute_job(
    config: BenchmarkConfig,
    benchmark_root: Path,
    trajectory_dir: Path,
    job: EvaluationJob,
    max_steps: int | None,
) -> dict[str, object] | None:
    if job.algorithm in DETERMINISTIC_ALGORITHMS:
        trajectory = evaluate_classical_controller(
            config,
            algorithm=job.algorithm,
            params=_selected_classical_params(benchmark_root, job.algorithm),
            seed=job.seed,
            start_day=job.season.start_day,
            growth_year=job.season.growth_year,
            episode_days=job.season.episode_days,
            max_steps=max_steps,
        )
    else:
        run_dir = benchmark_root / "training" / job.algorithm / f"seed_{job.seed}"
        trajectory = evaluate_saved_agent(
            config,
            algorithm=job.algorithm,
            run_dir=run_dir,
            seed=job.seed,
            growth_year=job.season.growth_year,
            start_day=job.season.start_day,
            episode_days=job.season.episode_days,
            max_steps=max_steps,
        )
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    path = trajectory_dir / trajectory_filename(job.algorithm, job.seed, job.season)
    temporary = path.with_suffix(path.suffix + ".tmp")
    trajectory.to_csv(temporary, index=False)
    temporary.replace(path)
    if (
        "environment_failure" in trajectory.columns
        and trajectory["environment_failure"].astype(bool).any()
    ):
        failure = trajectory.loc[trajectory["environment_failure"].astype(bool)].iloc[-1]
        raise RuntimeError(
            "environment failure "
            f"{failure['environment_failure_kind']} at timestep "
            f"{int(failure['environment_failure_timestep'])}: "
            f"{failure['environment_failure_message']}"
        )
    if max_steps is not None:
        _validate_partial_trajectory(
            trajectory,
            job=job,
            location=config.location,
            expected_steps=max_steps,
        )
        return None
    expected_steps = job.season.episode_days * 86400 // DT_SECONDS
    validate_complete_trajectory(
        trajectory,
        season=job.season,
        location=config.location,
        expected_steps=expected_steps,
    )
    return {
        "algorithm": job.algorithm,
        "seed": job.seed,
        "season_id": job.season.season_id,
        "growth_year": job.season.growth_year,
        "start_day": job.season.start_day,
        **summarize_episode(trajectory, dt_seconds=DT_SECONDS),
    }


def _select_seasons(names: Iterable[str] | None) -> tuple[EvaluationSeason, ...]:
    seasons = canonical_evaluation_seasons()
    if names is None:
        return seasons
    requested = set(names)
    selected = tuple(season for season in seasons if season.season_id in requested)
    missing = sorted(requested - {season.season_id for season in selected})
    if missing:
        raise ValueError(f"unknown seasons: {', '.join(missing)}")
    return selected


def run_six_season_evaluation(
    *,
    profile: str = "paper",
    algorithms: Iterable[str] = ALGORITHMS,
    season_ids: Iterable[str] | None = None,
    resume: bool = True,
    smoke_steps: int | None = None,
    max_workers: int = 1,
    output_name: str = "six_season_120d_guarded_v2",
) -> Path:
    config = load_benchmark_config(profile)
    seasons = _select_seasons(season_ids)
    jobs = expand_evaluation_jobs(
        algorithms=algorithms,
        seasons=seasons,
        evaluation_seeds=config.evaluation_seeds,
    )
    benchmark_root = PROJECT_ROOT / config.output_dir
    output_root = evaluation_output_root(config, output_name)
    protocol = _protocol_payload(config, jobs, benchmark_root)
    _write_json_atomic(output_root / "resolved_protocol.json", protocol)

    if smoke_steps is not None:
        if smoke_steps <= 0:
            raise ValueError("smoke_steps must be positive")
        smoke_dir = output_root / "smoke" / "trajectories"
        for job in jobs:
            _execute_job(config, benchmark_root, smoke_dir, job, int(smoke_steps))
            print(f"smoke complete: {job.unit_id}", flush=True)
        return output_root

    manifest_path = output_root / "evaluation_manifest.json"
    manifest = _initialize_manifest(manifest_path, protocol, jobs)
    trajectory_dir = output_root / "trajectories"
    metrics_path = output_root / "episode_metrics.csv"
    pending: list[EvaluationJob] = []
    expected_steps = 120 * 86400 // DT_SECONDS
    for job in jobs:
        trajectory_path = trajectory_dir / trajectory_filename(
            job.algorithm, job.seed, job.season
        )
        reusable = resume and can_resume_trajectory(
            trajectory_path,
            season=job.season,
            location=config.location,
            expected_steps=expected_steps,
        )
        if reusable:
            trajectory = pd.read_csv(trajectory_path)
            record = {
                "algorithm": job.algorithm,
                "seed": job.seed,
                "season_id": job.season.season_id,
                "growth_year": job.season.growth_year,
                "start_day": job.season.start_day,
                **summarize_episode(trajectory, dt_seconds=DT_SECONDS),
            }
            upsert_episode_metric(metrics_path, record)
            _transition_manifest(manifest_path, job.unit_id, "complete")
            print(f"resumed: {job.unit_id}", flush=True)
        else:
            pending.append(job)
            _transition_manifest(manifest_path, job.unit_id, "running")

    failures: list[str] = []
    workers = max(1, min(int(max_workers), len(pending))) if pending else 1
    if workers == 1:
        completed = ((job, _execute_job(config, benchmark_root, trajectory_dir, job, None), None) for job in pending)
        for job, record, error in completed:
            if error is None and record is not None:
                upsert_episode_metric(metrics_path, record)
                _transition_manifest(manifest_path, job.unit_id, "complete")
                print(f"complete: {job.unit_id}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _execute_job, config, benchmark_root, trajectory_dir, job, None
                ): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    record = future.result()
                    if record is None:
                        raise RuntimeError("full evaluation returned no metrics")
                    upsert_episode_metric(metrics_path, record)
                    _transition_manifest(manifest_path, job.unit_id, "complete")
                    print(f"complete: {job.unit_id}", flush=True)
                except Exception as exc:
                    message = f"{job.unit_id}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    _transition_manifest(manifest_path, job.unit_id, "failed", error=message)
                    print(f"failed: {message}", flush=True)
    if failures:
        raise RuntimeError("evaluation failures:\n" + "\n".join(failures))

    episodes = pd.read_csv(metrics_path)
    requested_ids = {job.unit_id for job in jobs}
    complete_rows = episodes[
        episodes.apply(
            lambda row: f"{row['algorithm']}/seed_{int(row['seed'])}/{row['season_id']}" in requested_ids,
            axis=1,
        )
    ]
    by_algorithm, by_season = aggregate_complete_episodes(complete_rows)
    by_algorithm.to_csv(output_root / "comparison_by_algorithm.csv", index=False)
    by_season.to_csv(output_root / "comparison_by_season.csv", index=False)
    return output_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="paper")
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--seasons", nargs="+")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-name", default="six_season_120d_guarded_v2")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = run_six_season_evaluation(
        profile=args.profile,
        algorithms=args.algorithms,
        season_ids=args.seasons,
        resume=not args.no_resume,
        smoke_steps=args.smoke_steps,
        max_workers=args.max_workers,
        output_name=args.output_name,
    )
    print(output)


if __name__ == "__main__":
    main()
