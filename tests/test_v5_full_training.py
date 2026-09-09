from pathlib import Path

from experiments.controllers.run_v5_full_training import (
    build_training_jobs,
    checkpoint_subset,
)
from experiments.controllers.v5_full_training_protocol import (
    load_v5_full_training_protocol,
)


CONFIG = Path("configs/benchmarks/chengdu_v5_full_training.yml")


def test_full_training_expands_three_algorithms_and_three_seeds():
    protocol = load_v5_full_training_protocol(CONFIG)

    jobs = build_training_jobs(protocol)

    assert len(jobs) == 9
    assert {(job.algorithm, job.seed) for job in jobs} == {
        (algorithm, seed)
        for algorithm in ("ppo", "sac", "residual_ppo")
        for seed in (0, 1, 2)
    }
    residual = [job for job in jobs if job.algorithm == "residual_ppo"]
    assert all(job.sb3_algorithm == "ppo" for job in residual)
    assert all(job.residual_pid_scale == 0.25 for job in residual)
    assert all(job.profile_label == "v5_full_training_v1" for job in jobs)


def test_checkpoint_subset_always_includes_minimum_full_budget():
    protocol = load_v5_full_training_protocol(CONFIG)

    assert checkpoint_subset(protocol, 307_200) == (20_480, 102_400, 307_200)
    assert checkpoint_subset(protocol, 614_400) == (
        20_480,
        102_400,
        307_200,
        614_400,
    )
