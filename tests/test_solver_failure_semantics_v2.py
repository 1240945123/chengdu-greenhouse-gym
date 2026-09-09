from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from experiments.controllers.benchmark_protocol import (
    build_environment,
    load_benchmark_config,
)
from glassgym.components.observations import BaseObservations
from glassgym.environments import greenlight_env as greenlight_env_module
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import build_env_kwargs, load_env_params


class RewardSpy:
    def __init__(self) -> None:
        self.calls = 0

    def compute_reward(self, _ctx):
        self.calls += 1
        raise AssertionError("normal reward must not run after solver failure")


class FixedReward:
    def __init__(self, value: float) -> None:
        self.value = value

    def compute_reward(self, _ctx):
        return self.value, {"reward": self.value}


class PositiveObservation(BaseObservations):
    @property
    def key(self) -> str:
        return "PositiveObservation"

    @property
    def space(self) -> spaces.Box:
        return spaces.Box(low=1.0, high=2.0, shape=(2,), dtype=np.float32)

    def compute_obs(self, _ctx):
        return np.full(2, 1.5, dtype=np.float32)


def _failing_environment():
    protocol = load_benchmark_config("smoke")
    env = build_environment(
        protocol,
        start_day=protocol.test_start_day,
        episode_days=1,
        dt_seconds=3_600,
    )
    env.reset(seed=0)
    return env


def _environment_kwargs(**overrides):
    kwargs = load_env_params("ChengduControllerBenchmark", "configs/envs/")
    kwargs, _ = build_env_kwargs(kwargs)
    kwargs.update({"season_length": 1, "dt": 3_600, **overrides})
    return kwargs


def test_numerical_solver_failure_truncates_without_normal_reward_or_observation():
    env = _failing_environment()
    reward_spy = RewardSpy()
    env.reward_fn = reward_spy
    state_before = env.x.copy()
    previous_observation = {
        key: value.copy() for key, value in env.obs.items()
    }

    def fail_solver(**_kwargs):
        raise RuntimeError("synthetic integrator failure")

    env.F = fail_solver
    try:
        obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
    finally:
        env.close()

    assert reward_spy.calls == 0
    assert not terminated
    assert truncated
    assert env.observation_space.contains(obs)
    assert all(np.all(value == 0.0) for value in obs.values())
    assert any(np.any(previous_observation[key] != 0.0) for key in obs)
    np.testing.assert_array_equal(env.x, state_before)
    assert reward == pytest.approx(-100.0 * env.N - env.failure_penalty_epsilon)
    assert reward < -100.0 * env.N
    assert info["failure"] is True
    assert info["failure_kind"] == "numerical_integration"
    assert info["failure_phase"] == "integration"
    assert info["failure_exception"] == "RuntimeError"
    assert info["failure_message"] == "synthetic integrator failure"
    assert info["failure_timestep"] == 0
    assert info["failure_remaining_steps"] == env.N
    assert info["failure_reward_lower_bound"] == pytest.approx(-100.0)
    assert info["scenario"]["location"] == env.location


def test_programming_errors_are_not_misreported_as_numerical_failures():
    env = _failing_environment()

    def fail_solver(**_kwargs):
        raise KeyError("invalid solver contract")

    env.F = fail_solver
    try:
        with pytest.raises(KeyError, match="invalid solver contract"):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    finally:
        env.close()


def test_crop_model_runtime_error_is_not_misreported_as_integrator_failure(
    monkeypatch,
):
    env = _failing_environment()

    def fail_crop_model(*_args, **_kwargs):
        raise RuntimeError("broken crop-model contract")

    monkeypatch.setattr(greenlight_env_module, "crop_fluxes", fail_crop_model)
    try:
        with pytest.raises(RuntimeError, match="broken crop-model contract"):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    finally:
        env.close()


def test_nonfinite_solver_state_uses_the_same_structured_failure_path():
    env = _failing_environment()
    reward_spy = RewardSpy()
    env.reward_fn = reward_spy
    original_solver = env.F

    def nonfinite_solver(**kwargs):
        result = original_solver(**kwargs)
        result["xf"][0] = np.nan
        return result

    env.F = nonfinite_solver
    try:
        _obs, reward, terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
    finally:
        env.close()

    assert reward_spy.calls == 0
    assert np.isfinite(reward)
    assert not terminated
    assert truncated
    assert info["failure"] is True
    assert info["failure_exception"] == "FloatingPointError"


@pytest.mark.parametrize("invalid_reward", [-100.01, 100.01, np.nan, np.inf])
def test_normal_reward_must_obey_frozen_finite_bounds(invalid_reward: float):
    env = _failing_environment()
    env.reward_fn = FixedReward(invalid_reward)
    try:
        with pytest.raises(ValueError, match="valid_reward_bounds"):
            env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    finally:
        env.close()


def test_failure_reward_uses_representable_strict_decrement():
    env = GreenLightEnv(
        **_environment_kwargs(valid_reward_bounds=(-1e16, 100.0))
    )
    env.reset(seed=0)

    def fail_solver(**_kwargs):
        raise RuntimeError("synthetic integrator failure")

    env.F = fail_solver
    try:
        _obs, reward, _terminated, _truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
    finally:
        env.close()

    assert np.isfinite(reward)
    assert reward < -1e16 * env.N


def test_reward_bounds_reject_failure_penalty_overflow():
    with pytest.raises(ValueError, match="finite failure return"):
        GreenLightEnv(
            **_environment_kwargs(
                valid_reward_bounds=(-np.finfo(np.float64).max, 100.0)
            )
        )


def test_failure_observation_respects_nonzero_box_bounds():
    env = _failing_environment()
    module = PositiveObservation(env)
    env.observation_modules = [module]
    env.observation_space = spaces.Dict({module.key: module.space})

    def fail_solver(**_kwargs):
        raise RuntimeError("synthetic integrator failure")

    env.F = fail_solver
    try:
        obs, _reward, _terminated, truncated, _info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
    finally:
        env.close()

    assert truncated
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs[module.key], 1.0)
