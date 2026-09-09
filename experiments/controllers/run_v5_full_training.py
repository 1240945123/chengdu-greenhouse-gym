from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from experiments.controllers.benchmark_protocol import PROJECT_ROOT
from experiments.controllers.run_v5_controller_smoke import train_rl_checkpoints
from experiments.controllers.v5_full_training_protocol import (
    DEFAULT_CONFIG_PATH,
    V5FullTrainingProtocol,
    load_v5_full_training_protocol,
    write_resolved_protocol,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "chengdu_agri_greenhouse_001"
    / "controller_benchmark"
    / "v5_full_training"
)


@dataclass(frozen=True)
class TrainingJob:
    algorithm: str
    sb3_algorithm: str
    seed: int
    residual_pid_scale: float | None
    profile_label: str = "v5_full_training_v1"

    @property
    def unit_id(self) -> str:
        return f"{self.algorithm}/seed_{self.seed}"


def build_training_jobs(
    protocol: V5FullTrainingProtocol,
) -> tuple[TrainingJob, ...]:
    jobs: list[TrainingJob] = []
    for algorithm in ("ppo", "sac", "residual_ppo"):
        for seed in protocol.seeds:
            jobs.append(
                TrainingJob(
                    algorithm=algorithm,
                    sb3_algorithm="ppo" if algorithm == "residual_ppo" else algorithm,
                    seed=int(seed),
                    residual_pid_scale=(
                        protocol.residual_ppo_scale
                        if algorithm == "residual_ppo"
                        else None
                    ),
                )
            )
    return tuple(jobs)


def checkpoint_subset(
    protocol: V5FullTrainingProtocol,
    max_checkpoint: int,
) -> tuple[int, ...]:
    maximum = int(max_checkpoint)
    selected = tuple(value for value in protocol.checkpoints if value <= maximum)
    if protocol.minimum_full_checkpoint not in selected:
        raise ValueError("max_checkpoint must include the minimum full checkpoint")
    return selected


def _limit_torch_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_training_job(
    job: TrainingJob,
    *,
    config_path: str | Path,
    output_root: str | Path,
    checkpoints: Iterable[int],
) -> dict[str, object]:
    _limit_torch_threads()
    protocol = load_v5_full_training_protocol(config_path)
    output = Path(output_root) / "training" / job.algorithm / f"seed_{job.seed}"
    hyperparameters = protocol.algorithms[job.sb3_algorithm]
    records = train_rl_checkpoints(
        algorithm=job.sb3_algorithm,
        output_dir=output,
        checkpoints=tuple(int(value) for value in checkpoints),
        seed=job.seed,
        growth_year=protocol.training_scenarios[0][0],
        start_day=protocol.training_scenarios[0][1],
        episode_days=protocol.episode_days,
        training_scenarios=list(protocol.training_scenarios),
        residual_pid_scale=job.residual_pid_scale,
        hyperparameters=hyperparameters,
        profile_label=job.profile_label,
        resume=True,
    )
    return {
        "unit_id": job.unit_id,
        "algorithm": job.algorithm,
        "seed": job.seed,
        "checkpoints": [int(record["total_timesteps"]) for record in records],
        "reused": [bool(record["reused"]) for record in records],
    }


def run_full_training(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_checkpoint: int = 307_200,
    max_workers: int = 6,
    algorithms: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    protocol = load_v5_full_training_protocol(config_path)
    output = Path(output_root)
    write_resolved_protocol(protocol, output / "resolved_protocol.json")
    checkpoints = checkpoint_subset(protocol, max_checkpoint)
    jobs = build_training_jobs(protocol)
    if algorithms is not None:
        requested = {str(value).lower() for value in algorithms}
        jobs = tuple(job for job in jobs if job.algorithm in requested)
        missing = requested - {job.algorithm for job in jobs}
        if missing:
            raise ValueError(f"unsupported algorithms: {', '.join(sorted(missing))}")
    workers = max(1, min(int(max_workers), len(jobs)))
    results: list[dict[str, object]] = []
    if workers == 1:
        for job in jobs:
            result = run_training_job(
                job,
                config_path=config_path,
                output_root=output,
                checkpoints=checkpoints,
            )
            results.append(result)
            print(f"training complete: {job.unit_id}", flush=True)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_training_job,
                job,
                config_path=config_path,
                output_root=output,
                checkpoints=checkpoints,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(f"training complete: {job.unit_id}", flush=True)
    return sorted(results, key=lambda item: str(item["unit_id"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-checkpoint", type=int, default=307_200)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--algorithms", nargs="+", choices=("ppo", "sac", "residual_ppo")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_full_training(
        config_path=args.config,
        output_root=args.output_root,
        max_checkpoint=args.max_checkpoint,
        max_workers=args.max_workers,
        algorithms=args.algorithms,
    )


if __name__ == "__main__":
    main()
