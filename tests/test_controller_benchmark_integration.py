from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.controllers.benchmark_protocol import load_benchmark_config
from experiments.controllers.train_benchmark_rl import (
    evaluate_saved_agent,
    train_rl_agent,
)
from experiments.controllers.tune_benchmark_classical import (
    assert_validation_only,
    evaluate_classical_controller,
    select_best_trial,
    tune_classical_controller,
)
from experiments.controllers.run_chengdu_benchmark import (
    benchmark_lock,
    config_fingerprint,
    training_artifacts_complete,
    initialize_manifest,
    should_run,
    run_output_complete,
    transition_manifest,
    validate_training_weather_provenance,
)


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_rl_agent_trains_saves_loads_and_evaluates(tmp_path, algorithm):
    config = load_benchmark_config("smoke")
    run_dir = tmp_path / algorithm
    hyperparameters = (
        {
            "learning_rate": 3e-4,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 1,
            "gamma": 0.99,
            "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
        }
        if algorithm == "ppo"
        else {
            "learning_rate": 7e-4,
            "buffer_size": 128,
            "learning_starts": 8,
            "batch_size": 16,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": {"pi": [16], "qf": [16]}},
        }
    )

    metadata = train_rl_agent(
        config,
        algorithm=algorithm,
        seed=0,
        run_dir=run_dir,
        total_timesteps=64,
        episode_days=1,
        hyperparameters=hyperparameters,
    )

    assert metadata["status"] == "complete"
    assert metadata["validation_controller_calls"] > 0
    assert metadata["validation_controller_fallback_steps"] == 0
    assert metadata["validation_controller_timeout_steps"] == 0
    assert (run_dir / "best_model.zip").exists()
    assert (run_dir / "best_vecnormalize.pkl").exists()
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()
    saved_metadata = json.loads((run_dir / "training_metadata.json").read_text())
    assert saved_metadata["algorithm"] == algorithm

    trajectory = evaluate_saved_agent(
        config,
        algorithm=algorithm,
        run_dir=run_dir,
        seed=0,
        episode_days=1,
        max_steps=8,
        growth_year=2024,
        start_day=60,
    )

    assert len(trajectory) == 8
    assert np.isfinite(trajectory["reward"]).all()
    assert np.allclose(trajectory[["uBoil", "uCO2"]], 0.0)
    assert trajectory["start_day"].nunique() == 1
    assert int(trajectory["start_day"].iloc[0]) == 60
    assert int(trajectory["growth_year"].iloc[0]) == 2024
    assert set([
        "air_temperature",
        "relative_humidity",
        "temperature_target",
        "harvested_dry_matter_mg_m2",
        "c_fruit_mg_m2",
    ]).issubset(
        trajectory.columns
    )


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_rl_training_resumes_from_an_atomic_chunk_checkpoint(tmp_path, algorithm):
    config = load_benchmark_config("smoke")
    run_dir = tmp_path / algorithm
    hyperparameters = (
        {
            "learning_rate": 3e-4,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 1,
            "gamma": 0.99,
            "policy_kwargs": {"net_arch": {"pi": [16], "vf": [16]}},
        }
        if algorithm == "ppo"
        else {
            "learning_rate": 7e-4,
            "buffer_size": 128,
            "learning_starts": 8,
            "batch_size": 16,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": {"pi": [16], "qf": [16]}},
        }
    )

    partial = train_rl_agent(
        config,
        algorithm=algorithm,
        seed=0,
        run_dir=run_dir,
        total_timesteps=64,
        episode_days=1,
        hyperparameters=hyperparameters,
        checkpoint_interval_timesteps=32,
        stop_after_checkpoint=True,
    )

    assert partial["status"] == "partial"
    assert partial["completed_timesteps"] == 32
    assert (run_dir / "resume_model.zip").exists()
    assert (run_dir / "resume_vecnormalize.pkl").exists()
    if algorithm == "sac":
        assert (run_dir / "resume_replay_buffer.pkl").exists()

    complete = train_rl_agent(
        config,
        algorithm=algorithm,
        seed=0,
        run_dir=run_dir,
        total_timesteps=64,
        episode_days=1,
        hyperparameters=hyperparameters,
        checkpoint_interval_timesteps=32,
    )

    assert complete["status"] == "complete"
    assert complete["completed_timesteps"] >= 64
    assert complete["resumed_from_timesteps"] == 32
    assert complete["validation_evaluation_timesteps"] == [32, 64]
    assert (run_dir / "final_model.zip").exists()
    assert (run_dir / "vecnormalize.pkl").exists()


def test_classical_selection_uses_reward_then_lower_action_variation():
    trials = [
        {"candidate_id": "a", "cumulative_reward": -10.0, "total_action_variation": 2.0},
        {"candidate_id": "b", "cumulative_reward": -8.0, "total_action_variation": 4.0},
        {"candidate_id": "c", "cumulative_reward": -8.0, "total_action_variation": 1.0},
    ]

    selected = select_best_trial(trials)

    assert selected["candidate_id"] == "c"


def test_classical_trials_must_use_validation_window_only():
    config = load_benchmark_config("smoke")
    assert_validation_only(
        [{"candidate_id": "a", "start_day": config.validation_start_day}],
        config,
    )
    with pytest.raises(ValueError, match="validation start day"):
        assert_validation_only(
            [{"candidate_id": "leak", "start_day": config.test_start_day}],
            config,
        )


def test_mpc_tuning_saves_the_effective_benchmark_parameters(tmp_path):
    config = load_benchmark_config("smoke")
    payload = tune_classical_controller(
        config,
        algorithm="mpc",
        output_dir=tmp_path,
        episode_days=1,
        max_steps=2,
    )

    selected = payload["selected_params"]
    assert selected["max_action_change"] == pytest.approx(0.1)
    assert selected["comfort_objective"]["temp_day_low"] == pytest.approx(20.0)
    assert set(selected["candidate_levels"]) == {"uThScr", "uVent", "uLamp", "uBlScr"}


def test_classical_default_evaluation_reaches_environment_termination():
    config = load_benchmark_config("smoke")
    trajectory = evaluate_classical_controller(
        config,
        algorithm="baseline",
        params=None,
        seed=0,
        start_day=config.test_start_day,
        episode_days=1,
    )

    assert trajectory["terminated"].iloc[-1]
    assert not trajectory["truncated"].any()
    assert len(trajectory) == 96
    assert (trajectory["harvested_dry_matter_mg_m2"] >= 0.0).all()
    assert np.isfinite(trajectory["c_fruit_mg_m2"]).all()
    assert {
        "proposed_uThScr",
        "proposed_uVent",
        "proposed_uLamp",
        "proposed_uBlScr",
        "safety_intervened",
        "safety_interventions",
        "safety_fallback_used",
        "safety_fallback_duration_steps",
    }.issubset(trajectory.columns)


def test_classical_evaluation_honors_growth_year_override():
    config = load_benchmark_config("smoke")
    trajectory = evaluate_classical_controller(
        config,
        algorithm="baseline",
        params=None,
        seed=0,
        start_day=60,
        growth_year=2024,
        episode_days=1,
        max_steps=2,
    )

    assert set(trajectory["growth_year"].astype(int)) == {2024}
    assert set(trajectory["start_day"].astype(int)) == {60}


def test_classical_trajectory_records_controller_execution_contract():
    config = load_benchmark_config("smoke")
    trajectory = evaluate_classical_controller(
        config,
        algorithm="baseline",
        params=None,
        seed=0,
        start_day=config.test_start_day,
        episode_days=1,
        max_steps=2,
    )

    assert {
        "controller_fallback_used",
        "controller_failure_kind",
        "controller_fallback_failure_kind",
        "controller_inference_seconds",
        "forecast_source_model_version",
        "forecast_error_model_id",
        "controller_model_fingerprint",
        "forecast_artifact_fingerprint",
        "forecast_fit_role",
        "forecast_select_role",
        "forecast_model_label",
        "forecast_metric_variable_indices",
        "forecast_metric_error_scales",
        "forecast_source_sha256",
        "forecast_leap_day_policy",
    }.issubset(trajectory.columns)
    assert not trajectory["controller_fallback_used"].any()
    assert set(trajectory["forecast_source_model_version"]) == {
        "historical-forecast-emulator-v2"
    }
    assert trajectory["forecast_error_model_id"].str.startswith("sha256:").all()
    assert trajectory["forecast_artifact_fingerprint"].str.len().eq(64).all()
    assert set(trajectory["forecast_fit_role"]) == {"fit"}
    assert set(trajectory["forecast_select_role"]) == {"select"}
    assert trajectory["forecast_model_label"].str.contains("not operational NWP").all()
    assert set(trajectory["forecast_leap_day_policy"]) == {
        "drop_february_29_from_select"
    }


def test_mpc_exception_uses_rule_baseline_not_hold_action(monkeypatch):
    import experiments.controllers.tune_benchmark_classical as classical

    class FailingMPC:
        def __init__(self, **kwargs):
            pass

        def reset(self):
            pass

        def predict(self, ctx):
            raise RuntimeError("forced MPC failure")

    monkeypatch.setitem(classical.CONTROLLER_CLASSES, "mpc", FailingMPC)
    config = load_benchmark_config("smoke")
    trajectory = evaluate_classical_controller(
        config, algorithm="mpc", params=None, seed=0,
        start_day=config.test_start_day, episode_days=1, max_steps=1,
    )

    assert bool(trajectory.loc[0, "controller_fallback_used"])
    assert trajectory.loc[0, "controller_failure_kind"] == "exception"
    assert pd.isna(trajectory.loc[0, "controller_fallback_failure_kind"])


def test_manifest_tracks_transitions_and_resume(tmp_path):
    config = load_benchmark_config("smoke")
    manifest_path = tmp_path / "manifest.json"
    manifest = initialize_manifest(manifest_path, config, ("ppo", "sac"))

    assert manifest["runs"]["ppo"]["0"]["status"] == "pending"
    assert should_run(manifest, "ppo", 0, resume=True) is True

    transition_manifest(manifest_path, "ppo", 0, "running")
    transition_manifest(manifest_path, "ppo", 0, "complete")
    complete = json.loads(manifest_path.read_text())
    assert should_run(complete, "ppo", 0, resume=True) is False
    assert should_run(complete, "ppo", 0, resume=False) is True


def test_manifest_rejects_stale_configuration(tmp_path):
    config = load_benchmark_config("smoke")
    manifest_path = tmp_path / "manifest.json"
    initialize_manifest(manifest_path, config, ("ppo",))
    payload = json.loads(manifest_path.read_text())
    payload["config_hash"] = "stale"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="configuration hash"):
        initialize_manifest(manifest_path, config, ("ppo",))


def test_training_reuse_requires_complete_metadata_and_all_artifacts(tmp_path):
    run_dir = tmp_path / "ppo"
    run_dir.mkdir()
    (run_dir / "training_metadata.json").write_text(json.dumps({"status": "complete"}))
    for filename in (
        "best_model.zip",
        "best_vecnormalize.pkl",
        "final_model.zip",
        "vecnormalize.pkl",
    ):
        (run_dir / filename).write_bytes(b"artifact")

    assert training_artifacts_complete(run_dir) is True
    (run_dir / "best_model.zip").write_bytes(b"")
    assert training_artifacts_complete(run_dir) is False
    (run_dir / "best_model.zip").write_bytes(b"artifact")
    (run_dir / "vecnormalize.pkl").unlink()
    assert training_artifacts_complete(run_dir) is False


def test_completed_manifest_is_not_reused_without_valid_test_trajectory(tmp_path):
    config = load_benchmark_config("smoke")
    assert run_output_complete(tmp_path, config, "ppo", 0) is False


def test_config_fingerprint_includes_dependency_contents(tmp_path):
    config = load_benchmark_config("smoke")
    dependency = tmp_path / "reward.yml"
    dependency.write_text("weight: 1")
    first = config_fingerprint(config, dependency_paths=[dependency])
    dependency.write_text("weight: 2")
    second = config_fingerprint(config, dependency_paths=[dependency])

    assert first != second


def test_training_weather_provenance_rejects_validation_leakage_and_bad_hash(tmp_path):
    config = replace(
        load_benchmark_config("smoke"),
        training_growth_years=(2026, 3000),
    )
    weather_root = tmp_path / "Chengdu"
    weather_root.mkdir()
    source = weather_root / "2026.csv"
    generated = weather_root / "3000.csv"
    source.write_text("source")
    generated.write_text("generated")
    manifest_path = tmp_path / "synthetic_weather_manifest.json"
    payload = {
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_start_day": 0,
        "source_days": config.validation_start_day + 1,
        "block_days": 3,
        "outputs": [{
            "year": 3000,
            "path": "Chengdu/3000.csv",
            "sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
            "source_block_absolute_start_days": [0],
        }],
    }
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="validation"):
        validate_training_weather_provenance(
            config, manifest_path=manifest_path, weather_root=weather_root
        )

    payload["source_days"] = config.validation_start_day
    payload["outputs"][0]["sha256"] = "bad"
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        validate_training_weather_provenance(
            config, manifest_path=manifest_path, weather_root=weather_root
        )


def test_benchmark_lock_rejects_concurrent_writer_and_cleans_up(tmp_path):
    with benchmark_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with benchmark_lock(tmp_path):
                pass

    assert not (tmp_path / ".benchmark.lock").exists()
