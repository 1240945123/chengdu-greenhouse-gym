import pandas as pd
import pytest

from experiments.controllers.evaluate_v5_full_training import (
    select_validation_candidates,
)


def _row(algorithm, seed, checkpoint, start_day, reward, *, role="validation"):
    return {
        "role": role,
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "growth_year": 2025,
        "start_day": start_day,
        "completed_episode": True,
        "numerical_failure": False,
        "all_values_finite": True,
        "crop_carbon_nonnegative": True,
        "residual_fallback_count": 0,
        "active_safety_intervention_fraction": 0.01,
        "cumulative_reward": reward,
    }


def test_selection_uses_every_validation_window_and_ignores_holdout_rows():
    rows = []
    for day in (59, 73):
        rows.append(_row("ppo", 0, 102400, day, -100.0))
        rows.append(_row("ppo", 1, 102400, day, -90.0))
    rows.append(_row("ppo", 0, 102400, 226, 10_000.0, role="temporal_holdout"))

    summary, selected = select_validation_candidates(
        pd.DataFrame(rows), expected_scenarios=((2025, 59), (2025, 73))
    )

    assert len(summary) == 2
    assert selected["ppo"]["seed"] == 1
    assert selected["ppo"]["checkpoint"] == 102400
    assert selected["ppo"]["mean_cumulative_reward"] == pytest.approx(-90.0)


def test_selection_rejects_candidate_missing_a_validation_window():
    rows = [
        _row("sac", 0, 102400, 59, -80.0),
        _row("sac", 1, 102400, 59, -70.0),
        _row("sac", 1, 102400, 73, -75.0),
    ]

    summary, selected = select_validation_candidates(
        pd.DataFrame(rows), expected_scenarios=((2025, 59), (2025, 73))
    )

    incomplete = summary.loc[summary["seed"].eq(0)].iloc[0]
    assert bool(incomplete["feasible"]) is False
    assert selected["sac"]["seed"] == 1


def test_selection_rejects_unhealthy_high_reward_candidate():
    healthy = [_row("residual_ppo", 0, 307200, day, -50.0) for day in (59, 73)]
    unsafe = [_row("residual_ppo", 1, 307200, day, -10.0) for day in (59, 73)]
    for row in unsafe:
        row["active_safety_intervention_fraction"] = 0.2

    _summary, selected = select_validation_candidates(
        pd.DataFrame(healthy + unsafe),
        expected_scenarios=((2025, 59), (2025, 73)),
        active_safety_limit=0.05,
    )

    assert selected["residual_ppo"]["seed"] == 0
