from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.controllers.evaluate_six_season_controllers import (
    EvaluationSeason,
    aggregate_complete_episodes,
    can_resume_trajectory,
    canonical_evaluation_seasons,
    expand_evaluation_jobs,
    evaluation_output_root,
    trajectory_filename,
    upsert_episode_metric,
    validate_complete_trajectory,
)


def _valid_trajectory(season: EvaluationSeason, steps: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "growth_year": [season.growth_year] * steps,
            "start_day": [season.start_day] * steps,
            "location": ["Chengdu_Agri_Greenhouse_001"] * steps,
            "uBoil": np.zeros(steps),
            "uCO2": np.zeros(steps),
            "reward": np.linspace(-1.0, -0.5, steps),
            "terminated": [False] * (steps - 1) + [True],
            "truncated": [False] * steps,
        }
    )


def test_canonical_evaluation_seasons_cover_three_years_and_two_seasons():
    seasons = canonical_evaluation_seasons()

    assert [season.season_id for season in seasons] == [
        "2023_spring",
        "2023_autumn",
        "2024_spring",
        "2024_autumn",
        "2025_spring",
        "2025_autumn",
    ]
    assert {season.growth_year for season in seasons} == {2023, 2024, 2025}
    assert {season.episode_days for season in seasons} == {120}


def test_trajectory_filename_includes_algorithm_seed_and_season():
    season = EvaluationSeason("2024_autumn", 2024, 227, 120)

    assert trajectory_filename("ppo", 3, season) == "ppo_seed_3_2024_autumn.csv"


def test_complete_trajectory_validation_accepts_matching_episode():
    season = EvaluationSeason("2024_spring", 2024, 60, 120)

    validate_complete_trajectory(
        _valid_trajectory(season),
        season=season,
        location="Chengdu_Agri_Greenhouse_001",
        expected_steps=4,
    )


def test_complete_trajectory_validation_ignores_empty_metadata_columns():
    season = EvaluationSeason("2024_spring", 2024, 60, 120)
    trajectory = _valid_trajectory(season).assign(
        controller_failure_kind=np.nan,
        environment_failure_kind=np.nan,
        environment_failure_phase=np.nan,
        environment_failure_exception=np.nan,
        environment_failure_message=np.nan,
    )

    validate_complete_trajectory(
        trajectory,
        season=season,
        location="Chengdu_Agri_Greenhouse_001",
        expected_steps=4,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.iloc[:-1], "step count"),
        (lambda frame: frame.assign(growth_year=2025), "growth year"),
        (lambda frame: frame.assign(uCO2=0.1), "disabled controls"),
    ],
)
def test_complete_trajectory_validation_rejects_invalid_artifacts(mutation, message):
    season = EvaluationSeason("2024_spring", 2024, 60, 120)

    with pytest.raises(ValueError, match=message):
        validate_complete_trajectory(
            mutation(_valid_trajectory(season)),
            season=season,
            location="Chengdu_Agri_Greenhouse_001",
            expected_steps=4,
        )


def test_valid_trajectory_can_resume_but_short_trajectory_cannot(tmp_path):
    season = EvaluationSeason("2024_spring", 2024, 60, 120)
    path = tmp_path / "trajectory.csv"
    _valid_trajectory(season).to_csv(path, index=False)

    assert can_resume_trajectory(
        path,
        season=season,
        location="Chengdu_Agri_Greenhouse_001",
        expected_steps=4,
    )
    assert not can_resume_trajectory(
        path,
        season=season,
        location="Chengdu_Agri_Greenhouse_001",
        expected_steps=5,
    )


def test_metric_upsert_uses_algorithm_seed_and_season_identity(tmp_path):
    path = tmp_path / "episode_metrics.csv"
    upsert_episode_metric(
        path,
        {"algorithm": "ppo", "seed": 0, "season_id": "2024_spring", "cumulative_reward": -5.0},
    )
    upsert_episode_metric(
        path,
        {"algorithm": "ppo", "seed": 0, "season_id": "2024_autumn", "cumulative_reward": -7.0},
    )
    upsert_episode_metric(
        path,
        {"algorithm": "ppo", "seed": 0, "season_id": "2024_spring", "cumulative_reward": -4.0},
    )

    table = pd.read_csv(path)
    assert len(table) == 2
    assert set(table["season_id"]) == {"2024_spring", "2024_autumn"}
    assert table.loc[table["season_id"] == "2024_spring", "cumulative_reward"].item() == -4.0


def test_primary_aggregation_first_averages_rl_seeds_within_each_season():
    rows = []
    for season_id, baseline_reward, ppo_rewards in (
        ("2024_spring", -10.0, (-8.0, -6.0)),
        ("2024_autumn", -20.0, (-14.0, -10.0)),
    ):
        rows.append({
            "algorithm": "baseline", "seed": 0, "season_id": season_id,
            "cumulative_reward": baseline_reward, "temperature_mae": 2.0,
        })
        for seed, reward in enumerate(ppo_rewards):
            rows.append({
                "algorithm": "ppo", "seed": seed, "season_id": season_id,
                "cumulative_reward": reward, "temperature_mae": 1.0 + seed,
            })

    by_algorithm, by_season = aggregate_complete_episodes(pd.DataFrame(rows))

    ppo = by_algorithm.loc[by_algorithm["algorithm"] == "ppo"].iloc[0]
    assert ppo["season_count"] == 2
    assert ppo["cumulative_reward_mean"] == pytest.approx(-9.5)
    assert len(by_season) == 4


def test_job_expansion_uses_one_classical_and_all_rl_seeds_per_season():
    seasons = canonical_evaluation_seasons()[:1]

    jobs = expand_evaluation_jobs(
        algorithms=("baseline", "pid", "mpc", "ppo", "sac"),
        seasons=seasons,
        evaluation_seeds=(0, 1, 2, 3, 4),
    )

    assert len(jobs) == 13
    assert len({job.unit_id for job in jobs}) == 13
    assert [job.seed for job in jobs if job.algorithm == "pid"] == [0]
    assert [job.seed for job in jobs if job.algorithm == "ppo"] == [0, 1, 2, 3, 4]


def test_job_expansion_rejects_unknown_algorithm():
    with pytest.raises(ValueError, match="unsupported algorithms"):
        expand_evaluation_jobs(
            algorithms=("unknown",),
            seasons=canonical_evaluation_seasons()[:1],
            evaluation_seeds=(0,),
        )


def test_evaluation_output_root_uses_explicit_protocol_name():
    from experiments.controllers.benchmark_protocol import load_benchmark_config

    config = load_benchmark_config("smoke")

    root = evaluation_output_root(config, "six_season_120d_guarded_v2")

    assert root.name == "six_season_120d_guarded_v2"
    assert root.parent.name == "controller_benchmark"
