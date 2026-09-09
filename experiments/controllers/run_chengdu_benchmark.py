from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from experiments.controllers.benchmark_metrics import summarize_episode
from experiments.controllers.benchmark_protocol import (
    DEFAULT_CONFIG_PATH,
    ENV_CONFIG_PATH,
    PROJECT_ROOT,
    BenchmarkConfig,
    load_benchmark_config,
)
from experiments.controllers.train_benchmark_rl import evaluate_saved_agent, train_rl_agent
from experiments.controllers.tune_benchmark_classical import (
    evaluate_classical_controller,
    tune_classical_controller,
)


ALGORITHMS = ("baseline", "pid", "mpc", "ppo", "sac")
DETERMINISTIC_ALGORITHMS = ("baseline", "pid", "mpc")
VALID_STATUSES = {"pending", "running", "complete", "failed"}


def evaluation_seeds_for(config: BenchmarkConfig, algorithm: str) -> tuple[int, ...]:
    return (0,) if algorithm in DETERMINISTIC_ALGORITHMS else config.evaluation_seeds


def _serializable_config(config: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["calibration_path"] = str(config.calibration_path)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _default_fingerprint_dependencies(config: BenchmarkConfig) -> list[Path]:
    era5_root = (
        PROJECT_ROOT / "data" / "processed" / "chengdu_agri" / "greenhouse_001"
        / "weather_era5"
    )
    dependencies = [
        DEFAULT_CONFIG_PATH,
        ENV_CONFIG_PATH,
        config.calibration_path,
        era5_root / "full_years" / config.location / f"{config.growth_year}.csv",
        era5_root / "full_years" / config.location / f"{config.validation_growth_year}.csv",
        era5_root / "bias_correction.json",
        era5_root / "quality_report.json",
        PROJECT_ROOT / "data" / "external" / "weather" / "era5"
        / "chengdu_greenhouse_001" / "manifest.json",
        PROJECT_ROOT / "configs" / "agents" / "rule_based.yml",
        PROJECT_ROOT / "configs" / "agents" / "pid.yml",
        PROJECT_ROOT / "configs" / "agents" / "mpc.yml",
        PROJECT_ROOT / "glassgym" / "components" / "rewards.py",
        PROJECT_ROOT / "glassgym" / "components" / "actions.py",
        PROJECT_ROOT / "glassgym" / "components" / "observations.py",
        PROJECT_ROOT / "glassgym" / "components" / "mpc.py",
        PROJECT_ROOT / "glassgym" / "components" / "pid.py",
        PROJECT_ROOT / "glassgym" / "components" / "rule_based.py",
        PROJECT_ROOT / "glassgym" / "environments" / "greenlight_env.py",
        PROJECT_ROOT / "glassgym" / "models" / "ChengduPhysics" / "ode.py",
        PROJECT_ROOT / "glassgym" / "models" / "ChengduPhysics" / "utils.py",
        PROJECT_ROOT / "experiments" / "controllers" / "benchmark_protocol.py",
        PROJECT_ROOT / "experiments" / "controllers" / "benchmark_metrics.py",
        PROJECT_ROOT / "experiments" / "controllers" / "train_benchmark_rl.py",
        PROJECT_ROOT / "experiments" / "controllers" / "tune_benchmark_classical.py",
        PROJECT_ROOT / "experiments" / "controllers" / "run_chengdu_benchmark.py",
        PROJECT_ROOT / "experiments" / "controllers" / "report_chengdu_benchmark.py",
    ]
    dependencies.extend(
        era5_root / "full_years" / config.location / f"{year}.csv"
        for year in config.training_growth_years
    )
    return dependencies


def config_fingerprint(
    config: BenchmarkConfig,
    dependency_paths: Iterable[str | Path] | None = None,
) -> str:
    digest = hashlib.sha256()
    encoded = json.dumps(
        _serializable_config(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest.update(encoded)
    dependencies = (
        _default_fingerprint_dependencies(config)
        if dependency_paths is None
        else [Path(path) for path in dependency_paths]
    )
    for dependency in sorted(dependencies, key=lambda path: str(path)):
        if not dependency.exists():
            raise FileNotFoundError(f"Fingerprint dependency is missing: {dependency}")
        digest.update(str(dependency.resolve()).encode("utf-8"))
        digest.update(dependency.read_bytes())
    return digest.hexdigest()


def validate_training_weather_provenance(
    config: BenchmarkConfig,
    *,
    manifest_path: str | Path | None = None,
    weather_root: str | Path | None = None,
) -> None:
    synthetic_years = {year for year in config.training_growth_years if year >= 3000}
    if not synthetic_years:
        return
    location_root = Path(weather_root) if weather_root is not None else (
        PROJECT_ROOT / "data" / "processed" / "chengdu_agri" / "greenhouse_001"
        / "weather" / config.location
    )
    path = Path(manifest_path) if manifest_path is not None else (
        location_root.parent / "synthetic_weather_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source_start = int(manifest["source_start_day"])
    source_days = int(manifest["source_days"])
    source_end = source_start + source_days
    if source_start < 0 or source_end > config.validation_start_day:
        raise ValueError("Synthetic weather source overlaps the validation window")

    source_path = Path(manifest["source_path"])
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["source_sha256"]:
        raise ValueError("Synthetic weather source hash does not match the manifest")

    outputs = {int(item["year"]): item for item in manifest["outputs"]}
    if set(outputs) != synthetic_years:
        raise ValueError("Synthetic weather manifest years do not match training years")
    block_days = int(manifest["block_days"])
    for year, item in outputs.items():
        starts = [int(value) for value in item["source_block_absolute_start_days"]]
        if not starts or min(starts) < source_start or max(starts) + block_days > source_end:
            raise ValueError(f"Synthetic weather year {year} samples outside its source range")
        output_path = Path(item["path"])
        if not output_path.is_absolute():
            output_path = location_root.parent / output_path
        if hashlib.sha256(output_path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Synthetic weather output hash mismatch for year {year}")


def validate_era5_weather_provenance(config: BenchmarkConfig) -> None:
    external_manifest = (
        PROJECT_ROOT / "data" / "external" / "weather" / "era5"
        / "chengdu_greenhouse_001" / "manifest.json"
    )
    processed_root = (
        PROJECT_ROOT / "data" / "processed" / "chengdu_agri" / "greenhouse_001"
        / "weather_era5"
    )
    manifest = json.loads(external_manifest.read_text(encoding="utf-8"))
    if manifest.get("source_type") != "reanalysis_not_site_observation":
        raise ValueError("ERA5 manifest must identify reanalysis data explicitly")
    required_years = set(config.training_growth_years) | {
        config.validation_growth_year,
        config.growth_year,
    }
    full_years = {
        int(item["year"]): item
        for item in manifest["outputs"]
        if item.get("kind") == "full_year_greenlight"
    }
    if not required_years.issubset(full_years):
        raise ValueError("ERA5 manifest does not contain every benchmark weather year")
    for year in required_years:
        item = full_years[year]
        path = processed_root / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"ERA5 weather hash mismatch for year {year}")
    roles = {
        item.get("season"): item.get("role")
        for item in manifest["outputs"]
        if item.get("kind") == "season_canonical"
    }
    expected_roles = {
        "2023_spring": "train", "2023_autumn": "train",
        "2024_spring": "train", "2024_autumn": "train",
        "2025_spring": "validation", "2025_autumn": "test",
    }
    if roles != expected_roles:
        raise ValueError("ERA5 season roles do not match the benchmark split")


def _pid_is_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


@contextmanager
def benchmark_lock(result_dir: str | Path):
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".benchmark.lock"
    if lock_path.exists():
        try:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            existing_pid = -1
        if existing_pid > 0 and _pid_is_alive(existing_pid):
            raise RuntimeError(
                f"Benchmark is already running in {root} with process {existing_pid}"
            )
        lock_path.unlink(missing_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        payload = json.dumps({
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")
        os.write(descriptor, payload)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def initialize_manifest(
    path: str | Path,
    config: BenchmarkConfig,
    algorithms: Iterable[str],
) -> dict[str, Any]:
    manifest_path = Path(path)
    fingerprint = config_fingerprint(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_hash") != fingerprint:
            raise ValueError("Existing manifest configuration hash does not match this run")
    else:
        manifest = {
            "profile": config.profile,
            "config_hash": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }
    for algorithm in algorithms:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown benchmark algorithm: {algorithm}")
        algorithm_runs = manifest["runs"].setdefault(algorithm, {})
        for seed in evaluation_seeds_for(config, algorithm):
            algorithm_runs.setdefault(str(seed), {"status": "pending"})
    _write_json_atomic(manifest_path, manifest)
    return manifest


def transition_manifest(
    path: str | Path,
    algorithm: str,
    seed: int,
    status: str,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid manifest status: {status}")
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["runs"][algorithm][str(seed)]
    entry["status"] = status
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    if error is None:
        entry.pop("error", None)
    else:
        entry["error"] = error
    _write_json_atomic(manifest_path, manifest)
    return manifest


def should_run(manifest: dict[str, Any], algorithm: str, seed: int, resume: bool) -> bool:
    if not resume:
        return True
    return manifest["runs"][algorithm][str(seed)]["status"] != "complete"


def training_artifacts_complete(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    metadata_path = path / "training_metadata.json"
    required = (
        path / "best_model.zip",
        path / "best_vecnormalize.pkl",
        path / "final_model.zip",
        path / "vecnormalize.pkl",
    )
    if (
        not metadata_path.exists()
        or metadata_path.stat().st_size == 0
        or not all(artifact.exists() and artifact.stat().st_size > 0 for artifact in required)
    ):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("status") == "complete"


def run_output_complete(
    root: str | Path,
    config: BenchmarkConfig,
    algorithm: str,
    seed: int,
) -> bool:
    result_root = Path(root)
    trajectory_path = result_root / "trajectories" / f"{algorithm}_seed_{seed}.csv"
    if not trajectory_path.exists():
        return False
    try:
        trajectory = pd.read_csv(trajectory_path)
        required = {
            "reward", "uBoil", "uCO2", "terminated", "truncated",
            "location", "growth_year", "start_day",
        }
        if not required.issubset(trajectory.columns):
            return False
        if len(trajectory) != config.episode_days * 24 * 4:
            return False
        if set(trajectory["location"].astype(str)) != {config.location}:
            return False
        if set(trajectory["growth_year"].astype(int)) != {config.growth_year}:
            return False
        if set(trajectory["start_day"].astype(int)) != {config.test_start_day}:
            return False
        if not np.isfinite(trajectory.select_dtypes(include=[np.number])).all().all():
            return False
        if not np.allclose(trajectory[["uBoil", "uCO2"]], 0.0):
            return False
        if not bool(trajectory["terminated"].iloc[-1]) or bool(trajectory["truncated"].any()):
            return False
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False

    metrics_path = result_root / "episode_metrics.csv"
    if not metrics_path.exists():
        return False
    try:
        metrics = pd.read_csv(metrics_path)
        matching = metrics[
            (metrics["algorithm"] == algorithm)
            & (metrics["seed"].astype(int) == int(seed))
        ]
        if len(matching) != 1 or not bool(matching["episode_complete"].iloc[0]):
            return False
        computed = summarize_episode(trajectory, dt_seconds=900)
        saved = matching.iloc[0]
        for key, value in computed.items():
            if key not in saved.index:
                return False
            if isinstance(value, bool):
                if bool(saved[key]) != value:
                    return False
            elif not np.isclose(float(saved[key]), float(value), rtol=1e-9, atol=1e-9):
                return False
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return False

    if algorithm in {"ppo", "sac"}:
        return training_artifacts_complete(
            result_root / "training" / algorithm / f"seed_{seed}"
        )
    if algorithm in {"pid", "mpc"}:
        return (result_root / "tuning" / algorithm / "selected_params.json").exists()
    return True


def _train_rl_worker(
    config: BenchmarkConfig,
    algorithm: str,
    seed: int,
    run_dir: Path,
    torch_threads: int,
) -> dict[str, Any]:
    import torch

    torch.set_num_threads(int(torch_threads))
    return train_rl_agent(
        config,
        algorithm=algorithm,
        seed=seed,
        run_dir=run_dir,
    )


def _parallel_train_seeds(
    config: BenchmarkConfig,
    algorithm: str,
    seeds: Iterable[int],
    root: Path,
    manifest_path: Path,
    *,
    resume: bool,
    max_workers: int,
) -> set[int]:
    jobs: list[tuple[int, Path]] = []
    already_available: set[int] = set()
    for seed in seeds:
        run_dir = root / "training" / algorithm / f"seed_{seed}"
        if resume and training_artifacts_complete(run_dir):
            already_available.add(int(seed))
        else:
            transition_manifest(manifest_path, algorithm, seed, "running")
            jobs.append((int(seed), run_dir))
    if not jobs:
        return already_available

    trained = set(already_available)
    workers = max(1, min(int(max_workers), len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_seed = {
            executor.submit(_train_rl_worker, config, algorithm, seed, run_dir, 2): seed
            for seed, run_dir in jobs
        }
        for future in as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                future.result()
                trained.add(seed)
            except Exception as exc:
                transition_manifest(
                    manifest_path,
                    algorithm,
                    seed,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                for pending in future_to_seed:
                    pending.cancel()
                raise
    return trained


def _load_selected_params(root: Path, algorithm: str) -> dict[str, Any]:
    path = root / "tuning" / algorithm / "selected_params.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected_params"]


def _upsert_episode_metrics(path: Path, record: dict[str, Any]) -> None:
    if path.exists():
        table = pd.read_csv(path)
        keep = ~(
            (table["algorithm"] == record["algorithm"])
            & (table["seed"].astype(int) == int(record["seed"]))
        )
        table = table.loc[keep]
        table = pd.concat([table, pd.DataFrame([record])], ignore_index=True)
    else:
        table = pd.DataFrame([record])
    table = table.sort_values(["algorithm", "seed"]).reset_index(drop=True)
    temporary = path.with_suffix(".csv.tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def _run_benchmark_unlocked(
    config: BenchmarkConfig,
    *,
    algorithms: Iterable[str] = ALGORITHMS,
    resume: bool = True,
    parallel_rl_seeds: bool = False,
    max_seed_workers: int = 3,
) -> Path:
    selected_algorithms = tuple(algorithms)
    validate_era5_weather_provenance(config)
    validate_training_weather_provenance(config)
    root = PROJECT_ROOT / config.output_dir
    root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(root / "resolved_config.json", _serializable_config(config))
    manifest_path = root / "manifest.json"
    manifest = initialize_manifest(manifest_path, config, selected_algorithms)

    for algorithm in selected_algorithms:
        if algorithm in {"pid", "mpc"}:
            selected_path = root / "tuning" / algorithm / "selected_params.json"
            if not selected_path.exists() or not resume:
                tune_classical_controller(
                    config,
                    algorithm=algorithm,
                    output_dir=selected_path.parent,
                )

        pretrained_seeds: set[int] = set()
        if algorithm in {"ppo", "sac"} and parallel_rl_seeds:
            pending_seeds = [
                seed for seed in config.seeds
                if (
                    should_run(
                        json.loads(manifest_path.read_text(encoding="utf-8")),
                        algorithm,
                        seed,
                        resume,
                    )
                    or not run_output_complete(root, config, algorithm, seed)
                )
            ]
            pretrained_seeds = _parallel_train_seeds(
                config,
                algorithm,
                pending_seeds,
                root,
                manifest_path,
                resume=resume,
                max_workers=max_seed_workers,
            )

        for seed in evaluation_seeds_for(config, algorithm):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not should_run(manifest, algorithm, seed, resume)
                and run_output_complete(root, config, algorithm, seed)
            ):
                continue
            transition_manifest(manifest_path, algorithm, seed, "running")
            try:
                if algorithm in {"ppo", "sac"}:
                    training_dir = root / "training" / algorithm / f"seed_{seed}"
                    if seed not in pretrained_seeds and not (
                        resume and training_artifacts_complete(training_dir)
                    ):
                        train_rl_agent(
                            config,
                            algorithm=algorithm,
                            seed=seed,
                            run_dir=training_dir,
                        )
                    trajectory = evaluate_saved_agent(
                        config,
                        algorithm=algorithm,
                        run_dir=training_dir,
                        seed=seed,
                    )
                else:
                    params = None if algorithm == "baseline" else _load_selected_params(root, algorithm)
                    trajectory = evaluate_classical_controller(
                        config,
                        algorithm=algorithm,
                        params=params,
                        seed=seed,
                        start_day=config.test_start_day,
                    )
                trajectory_dir = root / "trajectories"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                trajectory.to_csv(
                    trajectory_dir / f"{algorithm}_seed_{seed}.csv", index=False
                )
                metrics = summarize_episode(trajectory, dt_seconds=900)
                _upsert_episode_metrics(
                    root / "episode_metrics.csv",
                    {
                        "algorithm": algorithm,
                        "seed": int(seed),
                        "profile": config.profile,
                        "growth_year": config.growth_year,
                        "season": f"{config.growth_year}_autumn",
                        "start_day": config.test_start_day,
                        **metrics,
                    },
                )
                transition_manifest(manifest_path, algorithm, seed, "complete")
            except Exception as exc:
                transition_manifest(
                    manifest_path,
                    algorithm,
                    seed,
                    "failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

    from experiments.controllers.report_chengdu_benchmark import generate_report

    generate_report(root, config)
    return root


def run_benchmark(
    config: BenchmarkConfig,
    *,
    algorithms: Iterable[str] = ALGORITHMS,
    resume: bool = True,
    parallel_rl_seeds: bool = False,
    max_seed_workers: int = 3,
) -> Path:
    root = PROJECT_ROOT / config.output_dir
    with benchmark_lock(root):
        return _run_benchmark_unlocked(
            config,
            algorithms=algorithms,
            resume=resume,
            parallel_rl_seeds=parallel_rl_seeds,
            max_seed_workers=max_seed_workers,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified Chengdu controller benchmark")
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--parallel-rl-seeds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train independent PPO/SAC seeds concurrently (defaults on for paper)",
    )
    parser.add_argument("--max-seed-workers", type=int, default=3)
    args = parser.parse_args()
    config = load_benchmark_config(args.profile)
    output = run_benchmark(
        config,
        algorithms=args.algorithms,
        resume=args.resume,
        parallel_rl_seeds=(args.profile == "paper") if args.parallel_rl_seeds is None else args.parallel_rl_seeds,
        max_seed_workers=args.max_seed_workers,
    )
    print(f"Benchmark outputs: {output}")


if __name__ == "__main__":
    main()
