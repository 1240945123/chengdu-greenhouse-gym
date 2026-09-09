from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from experiments.controllers.benchmark_protocol import build_environment, load_benchmark_config
from experiments.controllers.tune_benchmark_classical import _step_context
from glassgym.components.forecast import PersistenceForecastProviderV2
from glassgym.components.observations import WeatherForecastObservationsV2
from glassgym.core.types import StepContext


def _context(disturbances: np.ndarray, forecast=None) -> StepContext:
    return StepContext(
        t=2,
        dt=900,
        Np=2,
        x_prev=np.zeros(28),
        x=np.zeros(28),
        u=np.zeros(6),
        p=np.zeros(216),
        d=disturbances,
        hour_of_day=12.0,
        day_of_year=100.0,
        forecast=forecast,
    )


def test_persistence_forecast_cannot_observe_poisoned_future_truth():
    provider = PersistenceForecastProviderV2(source_model_version="persistence-v1")
    original = np.arange(66, dtype=float).reshape(6, 11)
    poisoned = original.copy()
    poisoned[3:] = -999999.0

    first = provider.issue(original[:3], issue_timestep=2, horizon_steps=3)
    second = provider.issue(poisoned[:3], issue_timestep=2, horizon_steps=3)

    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.values, np.repeat(original[2:3], 3, axis=0))
    assert first.issue_timestep == 2
    np.testing.assert_array_equal(first.valid_timesteps, [3, 4, 5])


def test_provider_rejects_history_containing_post_issue_rows():
    provider = PersistenceForecastProviderV2()
    disturbances = np.zeros((6, 11))

    with pytest.raises(ValueError, match="issue-time history"):
        provider.issue(disturbances[:4], issue_timestep=2, horizon_steps=2)


def test_v2_forecast_observation_uses_issued_values_not_realized_future():
    provider = PersistenceForecastProviderV2()
    disturbances = np.arange(66, dtype=float).reshape(6, 11)
    issued = provider.issue(disturbances[:3], issue_timestep=2, horizon_steps=2)
    ctx = _context(disturbances, forecast=issued)
    poisoned = disturbances.copy()
    poisoned[3:, :5] = 1e9

    module = object.__new__(WeatherForecastObservationsV2)
    module.Np = 2
    module.nu = 6
    observed = module.compute_obs(replace(ctx, d=poisoned))

    np.testing.assert_array_equal(observed, issued.values[:, :5].reshape(-1))


def test_mpc_rollout_disturbance_is_independent_of_future_truth():
    from glassgym.components.mpc import LightweightMPCController

    disturbances = np.arange(66, dtype=float).reshape(6, 11)
    provider = PersistenceForecastProviderV2()
    issued = provider.issue(disturbances[:3], issue_timestep=2, horizon_steps=2)
    ctx = _context(disturbances, forecast=issued)
    poisoned = disturbances.copy()
    poisoned[3:] = 1e12
    controller = object.__new__(LightweightMPCController)

    first = controller._forecast_disturbance(ctx, horizon_index=1)
    second = controller._forecast_disturbance(replace(ctx, d=poisoned), horizon_index=1)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, issued.values[1])


def test_benchmark_step_context_contains_issue_time_forecast():
    config = load_benchmark_config("smoke")
    env = build_environment(config, config.test_start_day, episode_days=1)
    try:
        env.reset(seed=0)
        ctx = _step_context(env)

        assert ctx.forecast is not None
        assert ctx.forecast.issue_timestep == env.timestep
        assert len(ctx.forecast.values) == env.Np
        assert len(ctx.d) == env.timestep + 1
        assert ctx.d.flags.writeable is False
        assert ctx.d[-1, 7] == pytest.approx(
            ctx.d[-1, 0] * env.dt / 1_000_000.0
        )
        assert np.isfinite(ctx.forecast.values).all()
        assert ctx.forecast.source_model_version == "historical-forecast-emulator-v2"
        assert ctx.forecast.error_model_id.startswith("sha256:")
    finally:
        env.close()
