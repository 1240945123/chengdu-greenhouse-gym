from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.controllers.benchmark_protocol import (
    build_environment,
    full_control_target_to_action,
    load_benchmark_config,
    validate_windows,
)
import experiments.controllers.benchmark_protocol as _benchmark_protocol
from glassgym.components.weather import RandomScenarioWeatherSampler


@pytest.fixture(autouse=True)
def _use_eight_dimensional_benchmark_env(monkeypatch):
    """Sync the benchmark env to the new 8-dimensional physical control vector.

    See test_action_safety_v2 for rationale. The new ChengduClimateReward expects
    an 8-dim control vector (ALLOWED_CONTROL_INDICES=[3, 5, 6, 7]); this test-only
    patch widens the benchmark env's control vector to 8 dims so the reward can be
    evaluated. The underlying config should be updated to nu=8 for real.
    """
    original = _benchmark_protocol._load_environment_kwargs

    def patched():
        kwargs = original()
        kwargs["nu"] = 8
        kwargs["u_min"] = [0.0] * 8
        kwargs["u_max"] = [1.0] * 8
        return kwargs

    monkeypatch.setattr(_benchmark_protocol, "_load_environment_kwargs", patched)


def test_target_adapter_enforces_disabled_controls_and_slew_rate():
    current = np.array([0.0, 0.0, 0.4, 0.3, 0.0, 0.2], dtype=np.float32)
    target = np.array([1.0, 1.0, 0.8, 0.0, 1.0, 0.2], dtype=np.float32)

    action = full_control_target_to_action(current, target, delta=0.1)

    assert action.tolist() == pytest.approx([1.0, -1.0, 1.0, 0.0])


def test_target_adapter_rejects_non_six_dimensional_controls():
    with pytest.raises(ValueError, match="six controls"):
        full_control_target_to_action(np.zeros(4), np.zeros(4), delta=0.1)


def test_resolved_windows_are_chronological_and_fit_weather():
    protocol = load_benchmark_config("smoke")

    validate_windows(protocol, available_days=366)

    assert protocol.available_weather_days == 366
    assert protocol.growth_year == 2025
    assert protocol.validation_growth_year == 2025
    assert protocol.training_growth_years == (2023, 2024)
    assert {year for year, _ in protocol.training_scenarios} == {2023, 2024}
    assert (2023, 59) in protocol.training_scenarios
    assert (2023, 226) in protocol.training_scenarios
    assert (2024, 60) in protocol.training_scenarios
    assert (2024, 227) in protocol.training_scenarios
    assert protocol.validation_start_day == 59
    assert protocol.test_start_day == 226
    assert protocol.episode_days == 4
    assert protocol.seeds == (0,)


def test_window_validation_rejects_overlapping_test_episode():
    protocol = load_benchmark_config("smoke")
    with pytest.raises(ValueError, match="exceeds available weather"):
        validate_windows(protocol, available_days=109)


def test_window_validation_checks_full_training_episode_interval():
    protocol = load_benchmark_config("smoke")
    overlapping = replace(protocol, training_scenarios=((2024, 363),))

    with pytest.raises(ValueError, match="training episode exceeds"):
        validate_windows(overlapping, available_days=110)


def test_environment_exposes_four_actions_and_keeps_disabled_controls_zero():
    protocol = load_benchmark_config("smoke")
    env = build_environment(protocol, start_day=protocol.test_start_day)
    try:
        env.reset(seed=0)
        assert env.action_space.shape == (4,)
        assert env.nd == 11
        assert env.weather_data.shape[1] == 11
        assert np.all(np.isfinite(env.weather_data[:, 10]))
        assert np.all(env.weather_data[:, 10] >= 0.0)
        assert env.weather_repository.loader_version == "weather-repository-v2"
        assert env.obs["WeatherObservationsV2"].shape == (6,)
        assert env.obs["WeatherObservationsV2"][5] == pytest.approx(
            env.weather_data[0, 10]
        )
        _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert info["controls"][0] == pytest.approx(0.0)
        assert info["controls"][1] == pytest.approx(0.0)
    finally:
        env.close()


def test_environment_accepts_explicit_hourly_diagnostic_timestep():
    protocol = load_benchmark_config("smoke")
    env = build_environment(
        protocol,
        start_day=protocol.test_start_day,
        episode_days=1,
        dt_seconds=3600,
    )
    try:
        assert env.dt == pytest.approx(3600.0)
        assert env.N == 24
    finally:
        env.close()


def test_environment_interpolates_calibration_sensitivity_from_neutral():
    protocol = load_benchmark_config("smoke")
    env = build_environment(
        protocol,
        start_day=protocol.test_start_day,
        episode_days=1,
        dt_seconds=3600,
        calibration_strength=0.75,
    )
    try:
        assert env.base_p[208] == pytest.approx(1.0 + 0.75 * (4.0 - 1.0))
        assert env.base_p[209] == pytest.approx(1.0 + 0.75 * (0.6 - 1.0))
    finally:
        env.close()


def test_training_scenarios_keep_year_and_start_day_paired():
    protocol = load_benchmark_config("smoke")
    scenarios = set(protocol.training_scenarios)

    assert (2023, 60) in scenarios
    assert (2024, 59) not in scenarios
    assert (2023, 227) in scenarios
    assert (2024, 226) not in scenarios

    sampler = RandomScenarioWeatherSampler([
        {"location": "Chengdu", "growth_year": 2023, "start_day": 59},
        {"location": "Chengdu", "growth_year": 2024, "start_day": 60},
    ])
    rng = np.random.default_rng(7)
    sampled = {
        (scenario.growth_year, scenario.start_day)
        for _ in range(100)
        for scenario in [sampler.sample(rng)]
    }
    assert sampled == {(2023, 59), (2024, 60)}


def test_benchmark_paths_stay_inside_dataset_results_namespace():
    protocol = load_benchmark_config("smoke")
    expected = Path("results/chengdu_agri_greenhouse_001/controller_benchmark/smoke")
    assert protocol.output_dir.as_posix() == expected.as_posix()
