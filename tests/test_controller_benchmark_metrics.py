import numpy as np
import pandas as pd
import pytest

from experiments.controllers.benchmark_protocol import load_benchmark_config
from experiments.controllers.benchmark_metrics import (
    aggregate_algorithms,
    benchmark_completeness,
    summarize_episode,
)
from experiments.controllers.report_chengdu_benchmark import generate_report


def synthetic_steps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reward": [-1.0, -2.0],
            "air_temperature": [18.0, 36.0],
            "temperature_target": [20.0, 24.0],
            "temperature_low": [16.0, 20.0],
            "temperature_high": [24.0, 28.0],
            "relative_humidity": [70.0, 90.0],
            "humidity_target": [72.5, 72.5],
            "humidity_low": [60.0, 60.0],
            "humidity_high": [85.0, 85.0],
            "uBoil": [0.0, 0.0],
            "uCO2": [0.0, 0.0],
            "uThScr": [0.0, 1.0],
            "uVent": [0.0, 0.5],
            "uLamp": [0.0, 1.0],
            "uBlScr": [0.0, 0.0],
            "proposed_uThScr": [0.0, 1.0],
            "proposed_uVent": [0.0, 1.0],
            "proposed_uLamp": [0.0, 1.0],
            "proposed_uBlScr": [0.0, 0.0],
            "safety_intervened": [False, True],
            "safety_interventions": ["", "wind_vent_closure"],
            "safety_fallback_used": [False, False],
            "safety_fallback_duration_steps": [0, 0],
            "wall_time_seconds": [0.01, 0.03],
            "controller_inference_seconds": [0.005, 0.025],
            "controller_fallback_used": [False, True],
            "controller_failure_kind": [None, "timeout"],
            "controller_fallback_duration_steps": [0, 1],
            "harvested_dry_matter_mg_m2": [62_700.0, 125_400.0],
            "allocated_fruit_dry_matter_mg_m2": [200_000.0, 100_000.0],
            "dry_matter_fraction": [0.0627, 0.0627],
            "c_buffer_mg_m2": [100.0, 110.0],
            "c_leaf_mg_m2": [90_000.0, 91_000.0],
            "c_stem_mg_m2": [250_000.0, 251_000.0],
            "c_fruit_mg_m2": [55_000.0, 56_000.0],
            "c_fruit_previous_mg_m2": [54_000.0, 55_000.0],
            "terminated": [False, True],
            "truncated": [False, False],
        }
    )


def test_episode_metrics_integrate_fifteen_minute_steps():
    metrics = summarize_episode(synthetic_steps(), dt_seconds=900)

    assert metrics["cumulative_reward"] == pytest.approx(-3.0)
    assert metrics["temperature_mae"] == pytest.approx(7.0)
    assert metrics["temperature_rmse"] == pytest.approx(np.sqrt(74.0))
    assert metrics["temperature_min"] == pytest.approx(18.0)
    assert metrics["temperature_max"] == pytest.approx(36.0)
    assert metrics["temperature_above_safety_hours"] == pytest.approx(0.25)
    assert metrics["temperature_below_safety_hours"] == pytest.approx(0.0)
    assert metrics["humidity_mae"] == pytest.approx(10.0)
    assert metrics["humidity_rmse"] == pytest.approx(12.5)
    assert metrics["humidity_min"] == pytest.approx(70.0)
    assert metrics["humidity_max"] == pytest.approx(90.0)
    assert metrics["humidity_above_safety_hours"] == pytest.approx(0.25)
    assert metrics["humidity_below_safety_hours"] == pytest.approx(0.0)
    assert metrics["temperature_violation_hours"] == pytest.approx(0.25)
    assert metrics["humidity_violation_hours"] == pytest.approx(0.25)
    assert metrics["temperature_comfort_fraction"] == pytest.approx(0.5)
    assert metrics["humidity_comfort_fraction"] == pytest.approx(0.5)
    assert metrics["joint_comfort_fraction"] == pytest.approx(0.5)
    assert metrics["mean_actuator_effort"] == pytest.approx(0.3125)
    assert metrics["uThScr_mean"] == pytest.approx(0.5)
    assert metrics["uVent_mean"] == pytest.approx(0.25)
    assert metrics["uLamp_mean"] == pytest.approx(0.5)
    assert metrics["uBlScr_mean"] == pytest.approx(0.0)
    assert metrics["total_action_variation"] == pytest.approx(2.5)
    assert metrics["switching_count"] == 3
    assert metrics["safety_intervention_fraction"] == pytest.approx(0.5)
    assert metrics["safety_intervention_steps"] == 1
    assert metrics["wind_vent_closure_steps"] == 1
    assert metrics["rain_vent_closure_steps"] == 0
    assert metrics["high_temperature_emergency_steps"] == 0
    assert metrics["rain_vent_override_high_temperature_steps"] == 0
    assert metrics["safety_fallback_steps"] == 0
    assert metrics["mean_safety_projection_l1"] == pytest.approx(0.25)
    assert metrics["total_safety_projection_l1"] == pytest.approx(0.5)
    assert metrics["mean_wall_time_seconds"] == pytest.approx(0.02)
    assert metrics["mean_wall_time_ms"] == pytest.approx(20.0)
    assert metrics["mean_controller_inference_ms"] == pytest.approx(15.0)
    assert metrics["controller_fallback_fraction"] == pytest.approx(0.5)
    assert metrics["controller_timeout_fraction"] == pytest.approx(0.5)
    assert metrics["max_controller_fallback_duration_steps"] == 1
    assert metrics["harvested_dry_matter_kg_m2"] == pytest.approx(0.1881)
    assert metrics["fresh_yield_kg_m2"] == pytest.approx(3.0)
    assert metrics["allocated_fruit_dry_matter_kg_m2"] == pytest.approx(0.3)
    assert metrics["net_fruit_production_dry_matter_kg_m2"] == pytest.approx(0.1901)
    assert metrics["simulated_fresh_fruit_production_kg_m2"] == pytest.approx(
        0.1901 / 0.0627
    )
    assert metrics["final_unharvested_fresh_fruit_kg_m2"] == pytest.approx(0.056 / 0.0627)
    assert metrics["harvest_onset_day"] == pytest.approx(0.0)
    assert metrics["peak_daily_fresh_yield_kg_m2"] == pytest.approx(3.0)
    assert metrics["final_leaf_dry_matter_kg_m2"] == pytest.approx(0.091)
    assert metrics["final_stem_dry_matter_kg_m2"] == pytest.approx(0.251)
    assert metrics["final_fruit_dry_matter_kg_m2"] == pytest.approx(0.056)
    assert metrics["episode_complete"] is True


def test_episode_metrics_reject_nonzero_disabled_controls():
    rows = synthetic_steps()
    rows.loc[0, "uBoil"] = 0.1
    with pytest.raises(ValueError, match="Heating and CO2"):
        summarize_episode(rows, dt_seconds=900)


def test_comfort_fraction_uses_reward_band_not_broad_safety_constraints():
    rows = synthetic_steps().iloc[[0]].copy()
    rows["temperature_low"] = 20.0
    rows["temperature_high"] = 28.0
    rows["terminated"] = True

    metrics = summarize_episode(rows, dt_seconds=900)

    assert metrics["temperature_violation_hours"] == pytest.approx(0.25)
    assert metrics["joint_comfort_fraction"] == pytest.approx(0.0)


def test_aggregate_reports_seed_distribution_reproducibly():
    episodes = pd.DataFrame(
        {
            "algorithm": ["pid", "pid", "sac", "sac"],
            "cumulative_reward": [-4.0, -2.0, -3.0, -3.0],
            "temperature_mae": [2.0, 1.0, 1.5, 1.5],
        }
    )

    first = aggregate_algorithms(episodes, bootstrap_samples=1000, bootstrap_seed=7)
    second = aggregate_algorithms(episodes, bootstrap_samples=1000, bootstrap_seed=7)

    assert first.loc["pid", "reward_mean"] == pytest.approx(-3.0)
    assert first.loc["pid", "reward_std"] == pytest.approx(np.sqrt(2.0))
    assert first.loc["pid", "temperature_mae_mean"] == pytest.approx(1.5)
    assert first.loc["pid", "temperature_mae_std"] == pytest.approx(np.sqrt(0.5))
    assert first.loc["sac", "reward_std"] == pytest.approx(0.0)
    assert first.loc["sac", "temperature_mae_std"] == pytest.approx(0.0)
    pd.testing.assert_frame_equal(first, second)


def test_completeness_lists_missing_algorithms_and_seeds():
    episodes = pd.DataFrame(
        {
            "algorithm": ["baseline", "pid", "mpc", "ppo", "sac"],
            "seed": [0, 0, 0, 0, 0],
            "episode_complete": [True] * 5,
        }
    )

    status = benchmark_completeness(
        episodes,
        required_algorithms=("baseline", "pid", "mpc", "ppo", "sac"),
        required_seeds=(0, 1),
    )

    assert status["paper_ready"] is False
    assert status["missing_seeds"]["ppo"] == [1]
    assert status["missing_seeds"]["baseline"] == [1]


def test_completeness_supports_single_deterministic_run_and_five_rl_seeds():
    rows = []
    for algorithm in ("baseline", "pid", "mpc"):
        rows.append({"algorithm": algorithm, "seed": 0, "episode_complete": True})
    for algorithm in ("ppo", "sac"):
        for seed in range(5):
            rows.append({"algorithm": algorithm, "seed": seed, "episode_complete": True})

    status = benchmark_completeness(
        pd.DataFrame(rows),
        required_algorithms=("baseline", "pid", "mpc", "ppo", "sac"),
        required_seeds=(0, 1, 2, 3, 4),
        deterministic_algorithms=("baseline", "pid", "mpc"),
    )

    assert status["paper_ready"] is True
    assert status["missing_seeds"] == {}


def test_report_generates_tables_and_nonempty_figures(tmp_path):
    config = load_benchmark_config("smoke")
    algorithms = ["baseline", "pid", "mpc", "ppo", "sac"]
    episodes = []
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    for algorithm_index, algorithm in enumerate(algorithms):
        for seed in config.evaluation_seeds:
            episodes.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode_complete": True,
                    "cumulative_reward": -100.0 + algorithm_index,
                    "temperature_mae": 2.0,
                    "temperature_rmse": 2.5,
                    "humidity_mae": 5.0,
                    "humidity_rmse": 6.0,
                    "temperature_comfort_fraction": 0.9,
                    "humidity_comfort_fraction": 0.85,
                    "joint_comfort_fraction": 0.8,
                    "temperature_violation_hours": 9.6,
                    "humidity_violation_hours": 14.4,
                    "mean_actuator_effort": 0.2,
                    "total_action_variation": 3.0,
                    "mean_wall_time_ms": 1.0,
                    "harvested_dry_matter_kg_m2": 0.64,
                    "fresh_yield_kg_m2": 10.2 + 0.01 * algorithm_index,
                    "simulated_fresh_fruit_production_kg_m2": 10.2 + 0.01 * algorithm_index,
                    "final_unharvested_fresh_fruit_kg_m2": 0.4,
                    "harvest_onset_day": 72.0,
                    "peak_daily_fresh_yield_kg_m2": 0.25,
                    "final_leaf_dry_matter_kg_m2": 0.11,
                    "final_stem_dry_matter_kg_m2": 0.28,
                    "final_fruit_dry_matter_kg_m2": 3.0,
                }
            )
        pd.DataFrame(
            {
                "timestep": [0, 1],
                "air_temperature": [20.0, 21.0],
                "temperature_target": [24.0, 24.0],
                "relative_humidity": [70.0, 72.0],
                "humidity_target": [72.5, 72.5],
            }
        ).to_csv(trajectory_dir / f"{algorithm}_seed_0.csv", index=False)
    pd.DataFrame(episodes).to_csv(tmp_path / "episode_metrics.csv", index=False)

    payload = generate_report(tmp_path, config)

    assert payload["paper_ready"] is False
    for filename in (
        "comparison.csv",
        "comparison.json",
        "reward_comparison.png",
        "comfort_comparison.png",
        "tracking_errors.png",
        "comfort_rates.png",
        "control_effort.png",
        "runtime_comparison.png",
        "yield_comparison.png",
        "test_trajectories.png",
        "paper_metrics.csv",
        "paper_metrics.md",
    ):
        assert (tmp_path / filename).stat().st_size > 0
    paper_metrics = pd.read_csv(tmp_path / "paper_metrics.csv")
    assert "fresh_yield_kg_m2_mean" in paper_metrics.columns
    assert "harvest_onset_day_mean" in paper_metrics.columns


def test_smoke_report_is_never_labelled_paper_ready(tmp_path):
    config = load_benchmark_config("smoke")
    pd.DataFrame(
        {
            "algorithm": ["baseline", "pid", "mpc", "ppo", "sac"],
            "seed": [0] * 5,
            "episode_complete": [True] * 5,
            "cumulative_reward": [-5.0] * 5,
        }
    ).to_csv(tmp_path / "episode_metrics.csv", index=False)

    payload = generate_report(tmp_path, config, make_plots=False)

    assert payload["paper_ready"] is False
    assert payload["completeness"]["paper_ready"] is True


def test_paper_report_rejects_metrics_without_run_artifacts(tmp_path):
    config = load_benchmark_config("paper")
    rows = []
    for algorithm in ("baseline", "pid", "mpc"):
        rows.append({
            "algorithm": algorithm,
            "seed": 0,
            "episode_complete": True,
            "cumulative_reward": -10.0,
        })
    for algorithm in ("ppo", "sac"):
        for seed in range(5):
            rows.append({
                "algorithm": algorithm,
                "seed": seed,
                "episode_complete": True,
                "cumulative_reward": -10.0,
            })
    pd.DataFrame(rows).to_csv(tmp_path / "episode_metrics.csv", index=False)

    payload = generate_report(tmp_path, config, make_plots=False)

    assert payload["completeness"]["paper_ready"] is True
    assert payload["paper_ready"] is False
    assert payload["artifact_completeness"]["complete"] is False
