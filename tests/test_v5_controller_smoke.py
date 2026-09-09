import gymnasium as gym
import json
import numpy as np
import pandas as pd
import pytest
import torch

from experiments.controllers.run_v5_controller_smoke import (
    PIDResidualActionWrapper,
    TrainingProgressCallback,
    _build_rl_model,
    _checkpoint_hyperparameters_compatible,
    build_smoke_decision,
    classify_safety_intervention,
    evaluate_classical_policy,
    evaluate_saved_rl_smoke,
    render_markdown_table,
    train_rl_checkpoints,
    train_rl_smoke,
)
from experiments.controllers.v5_smoke_controllers import V5FixedBaseline
from experiments.controllers.v5_smoke_controllers import V5PIDController
from experiments.controllers.v5_hybrid_environment import build_v5_hybrid_environment


def test_unified_classical_rollout_reports_required_metrics():
    metrics, trajectory = evaluate_classical_policy(
        algorithm="baseline",
        policy=V5FixedBaseline(),
        growth_year=2024,
        start_day=60,
        episode_days=1,
        max_steps=4,
    )

    assert len(trajectory) == 4
    assert metrics["algorithm"] == "baseline"
    assert metrics["steps"] == 4
    assert metrics["all_values_finite"] is True
    assert metrics["numerical_failure"] is False
    assert metrics["crop_carbon_nonnegative"] is True
    assert np.isfinite(metrics["mean_controller_inference_ms"])
    assert set(
        [
            "executed_uRoofVent",
            "executed_uFan",
            "proposed_uRoofVent",
            "proposed_uFan",
            "air_temperature",
            "relative_humidity",
        ]
    ).issubset(trajectory.columns)


def test_smoke_decision_requires_every_algorithm_in_both_roles():
    rows = []
    for role in ("validation", "test"):
        for index, algorithm in enumerate(("baseline", "pid", "mpc", "ppo", "sac")):
            rows.append(
                {
                    "role": role,
                    "algorithm": algorithm,
                    "completed_episode": True,
                    "numerical_failure": False,
                    "all_values_finite": True,
                    "residual_fallback_count": 0,
                    "crop_carbon_nonnegative": True,
                    "cumulative_reward": -10.0 + index,
                }
            )

    passing = build_smoke_decision(pd.DataFrame(rows))
    incomplete = build_smoke_decision(pd.DataFrame(rows[:-1]))

    assert passing["promote_to_longer_training"] is True
    assert passing["baseline_is_worst_on_test"] is True
    assert passing["yield_claim_allowed"] is False
    assert incomplete["promote_to_longer_training"] is False


def test_smoke_decision_requires_every_candidate_to_beat_validation_baseline():
    rows = []
    for role in ("validation", "test"):
        for algorithm in ("baseline", "pid", "mpc", "ppo", "sac"):
            rows.append(
                {
                    "role": role,
                    "algorithm": algorithm,
                    "completed_episode": True,
                    "numerical_failure": False,
                    "all_values_finite": True,
                    "residual_fallback_count": 0,
                    "crop_carbon_nonnegative": True,
                    "cumulative_reward": -10.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.loc[
        frame["role"].eq("validation") & frame["algorithm"].ne("baseline"),
        "cumulative_reward",
    ] = -9.0
    frame.loc[
        frame["role"].eq("validation") & frame["algorithm"].eq("ppo"),
        "cumulative_reward",
    ] = -11.0

    decision = build_smoke_decision(frame)

    assert decision["all_candidates_beat_validation_baseline"] is False
    assert decision["baseline_is_worst_on_test"] is False
    assert decision["validation_reward_improvement"]["ppo"] == pytest.approx(-1.0)
    assert decision["promote_to_longer_training"] is False


def test_smoke_markdown_renderer_has_no_optional_dependency():
    rendered = render_markdown_table(
        pd.DataFrame({"algorithm": ["PPO|SAC"], "reward": [-1.23456]}),
        float_digits=3,
    )

    assert "PPO\\|SAC" in rendered
    assert "-1.235" in rendered


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_rl_smoke_trains_saves_loads_and_evaluates(tmp_path, algorithm):
    output = tmp_path / algorithm
    metadata = train_rl_smoke(
        algorithm=algorithm,
        output_dir=output,
        total_timesteps=64,
        seed=0,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59), (2024, 227)],
    )

    assert metadata["algorithm"] == algorithm
    assert metadata["total_timesteps"] == 64
    assert metadata["training_scenarios"] == [
        {"growth_year": 2023, "start_day": 59},
        {"growth_year": 2024, "start_day": 227},
    ]
    assert (output / "model.zip").exists()
    assert (output / "vecnormalize.pkl").exists()

    metrics, trajectory = evaluate_saved_rl_smoke(
        algorithm=algorithm,
        model_dir=output,
        growth_year=2024,
        start_day=60,
        episode_days=1,
        max_steps=4,
    )
    assert len(trajectory) == 4
    assert metrics["algorithm"] == algorithm
    assert metrics["all_values_finite"] is True
    assert metrics["numerical_failure"] is False


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_rl_checkpoint_training_saves_cumulative_resumable_artifacts(tmp_path, algorithm):
    records = train_rl_checkpoints(
        algorithm=algorithm,
        output_dir=tmp_path / algorithm,
        checkpoints=[32, 64],
        seed=1,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59), (2024, 227)],
        resume=True,
    )

    assert [record["total_timesteps"] for record in records] == [32, 64]
    for steps in (32, 64):
        checkpoint = tmp_path / algorithm / f"checkpoint_{steps:08d}"
        assert (checkpoint / "model.zip").exists()
        assert (checkpoint / "vecnormalize.pkl").exists()
        assert (checkpoint / "training_metadata.json").exists()
        if algorithm == "sac":
            assert (checkpoint / "replay_buffer.pkl").exists()

    resumed = train_rl_checkpoints(
        algorithm=algorithm,
        output_dir=tmp_path / algorithm,
        checkpoints=[32, 64],
        seed=1,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59), (2024, 227)],
        resume=True,
    )
    assert [record["reused"] for record in resumed] == [True, True]


def test_checkpoint_training_records_and_enforces_paper_hyperparameters(tmp_path):
    hyperparameters = {
        "learning_rate": 3e-4,
        "n_steps": 16,
        "batch_size": 8,
        "n_epochs": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "policy_kwargs": {"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
    }
    output = tmp_path / "paper_ppo"
    first = train_rl_checkpoints(
        algorithm="ppo",
        output_dir=output,
        checkpoints=[32],
        seed=0,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59)],
        hyperparameters=hyperparameters,
        profile_label="paper",
        resume=True,
    )
    changed = {**hyperparameters, "learning_rate": 1e-4}
    second = train_rl_checkpoints(
        algorithm="ppo",
        output_dir=output,
        checkpoints=[32],
        seed=0,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59)],
        hyperparameters=changed,
        profile_label="paper",
        resume=True,
    )

    assert first[0]["profile_label"] == "paper"
    assert first[0]["hyperparameters"] == hyperparameters
    assert second[0]["reused"] is False


def test_paper_model_builder_uses_supplied_ppo_rollout_configuration():
    raw_env = build_v5_hybrid_environment(
        growth_year=2023, start_day=59, episode_days=1, dt_seconds=900
    )
    env = gym.wrappers.FlattenObservation(raw_env)
    parameters = {
        "learning_rate": 3e-4,
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 4,
        "gamma": 0.98,
        "gae_lambda": 0.94,
        "clip_range": 0.2,
        "policy_kwargs": {"net_arch": {"pi": [32, 32], "vf": [32, 32]}},
    }

    model = _build_rl_model(
        algorithm="ppo",
        env=env,
        seed=2,
        total_timesteps=1000,
        hyperparameters=parameters,
    )

    assert model.n_steps == 64
    assert model.batch_size == 32
    assert model.n_epochs == 4
    assert model.gamma == pytest.approx(0.98)
    env.close()


def test_training_progress_callback_writes_atomic_heartbeat(tmp_path):
    callback = TrainingProgressCallback(
        output_dir=tmp_path,
        interval_timesteps=1024,
        unit_id="ppo/seed_0",
        target_checkpoint=3072,
    )
    callback.model = type("Model", (), {"num_timesteps": 1024})()

    assert callback._on_step() is True

    progress = json.loads((tmp_path / "training_progress.json").read_text(encoding="utf-8"))
    assert progress["unit_id"] == "ppo/seed_0"
    assert progress["completed_timesteps"] == 1024
    assert progress["target_checkpoint"] == 3072


def test_sac_checkpoint_contract_accepts_sb3_default_policy_mutation():
    requested = {
        "learning_rate": 7e-4,
        "policy_kwargs": {"net_arch": {"pi": [64, 64], "qf": [64, 64]}},
    }
    saved = {
        "learning_rate": 7e-4,
        "policy_kwargs": {
            "net_arch": {"pi": [64, 64], "qf": [64, 64]},
            "use_sde": False,
        },
    }

    assert _checkpoint_hyperparameters_compatible(saved, requested) is True


def test_checkpoint_contract_rejects_substantive_hyperparameter_change():
    saved = {"learning_rate": 7e-4, "policy_kwargs": {"use_sde": False}}
    changed = {"learning_rate": 1e-4, "policy_kwargs": {}}

    assert _checkpoint_hyperparameters_compatible(saved, changed) is False


def test_zero_residual_action_reproduces_frozen_pid_action():
    env = build_v5_hybrid_environment(
        growth_year=2025, start_day=59, episode_days=1, dt_seconds=900
    )
    wrapped = PIDResidualActionWrapper(env, residual_scale=0.25)
    wrapped.reset(seed=0)
    expected = V5PIDController().predict_action(env)

    actual = wrapped.action(np.zeros(4, dtype=np.float32))

    assert np.allclose(actual, expected)
    wrapped.close()


def test_residual_action_is_clipped_to_common_action_space():
    env = build_v5_hybrid_environment(
        growth_year=2025, start_day=59, episode_days=1, dt_seconds=900
    )
    wrapped = PIDResidualActionWrapper(env, residual_scale=2.0)
    wrapped.reset(seed=0)

    actual = wrapped.action(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32))

    assert wrapped.action_space.contains(actual)
    wrapped.close()


def test_residual_checkpoint_metadata_prevents_scale_mismatch_reuse(tmp_path):
    output = tmp_path / "residual_ppo"
    first = train_rl_checkpoints(
        algorithm="ppo",
        output_dir=output,
        checkpoints=[32],
        seed=0,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59)],
        residual_pid_scale=0.25,
        resume=True,
    )
    mismatched = train_rl_checkpoints(
        algorithm="ppo",
        output_dir=output,
        checkpoints=[32],
        seed=0,
        growth_year=2023,
        start_day=59,
        episode_days=1,
        training_scenarios=[(2023, 59)],
        residual_pid_scale=0.5,
        resume=True,
    )

    assert first[0]["residual_pid_scale"] == pytest.approx(0.25)
    assert mismatched[0]["residual_pid_scale"] == pytest.approx(0.5)
    assert mismatched[0]["reused"] is False


def test_v5_rl_training_works_when_torch_deterministic_mode_is_already_enabled(tmp_path):
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        metadata = train_rl_smoke(
            algorithm="ppo",
            output_dir=tmp_path / "deterministic_ppo",
            total_timesteps=32,
            seed=0,
            growth_year=2023,
            start_day=59,
            episode_days=1,
            training_scenarios=[(2023, 59)],
        )
    finally:
        torch.use_deterministic_algorithms(previous)

    assert metadata["algorithm"] == "ppo"


def test_material_safety_metric_ignores_strict_floating_point_noise():
    proposed = np.zeros(8)
    executed = proposed.copy()
    executed[3] = 5e-8

    result = classify_safety_intervention(proposed, executed, reasons=("normal_slew_limit",))

    assert result["strict_safety_intervened"] is True
    assert result["material_safety_intervened"] is False
    assert result["active_safety_intervened"] is False
    assert result["safety_intervention_reasons"] == "normal_slew_limit"


def test_active_safety_metric_excludes_noncontrolled_channels():
    proposed = np.zeros(8)
    executed = proposed.copy()
    executed[4] = 0.1

    result = classify_safety_intervention(proposed, executed)

    assert result["material_safety_intervened"] is True
    assert result["active_safety_intervened"] is False
    assert result["material_projection_max"] == pytest.approx(0.1)
    assert result["active_projection_max"] == pytest.approx(0.0)


def test_active_safety_metric_detects_material_roof_or_fan_projection():
    proposed = np.zeros(8)
    executed = proposed.copy()
    executed[6] = 0.01

    result = classify_safety_intervention(proposed, executed)

    assert result["material_safety_intervened"] is True
    assert result["active_safety_intervened"] is True
    assert result["active_projection_max"] == pytest.approx(0.01)
