import numpy as np
import pytest

from experiments.controllers.v5_smoke_controllers import (
    V5FixedBaseline,
    V5HybridMPC,
    V5PIDController,
)
from experiments.controllers.v5_hybrid_environment import build_v5_hybrid_environment
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres


@pytest.fixture
def env():
    instance = build_v5_hybrid_environment(
        growth_year=2025, start_day=226, episode_days=1, dt_seconds=900
    )
    instance.reset(seed=0)
    instance.hour_of_day = 12.0
    return instance


def _set_climate(env, temperature, rh):
    env.x[2] = temperature
    env.x[15] = vaporDens2pres(
        temperature, rh2vaporDens(temperature, rh)
    )


def test_fixed_baseline_is_frozen_low_control(env):
    action = V5FixedBaseline().predict_action(env)
    assert np.array_equal(action, np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32))


def test_pid_increases_roof_and_fan_for_excess_temperature(env):
    _set_climate(env, 35.0, 70.0)
    controller = V5PIDController()
    action = controller.predict_action(env)
    assert action[0] > 0.0
    assert action[2] > 0.0


def test_pid_suppresses_fan_but_vents_for_high_humidity(env):
    _set_climate(env, 30.0, 98.0)
    controller = V5PIDController()
    action = controller.predict_action(env)
    assert action[0] > 0.0
    assert action[2] < 0.0


def test_mpc_returns_finite_normalized_action(env):
    _set_climate(env, 35.0, 70.0)
    action = V5HybridMPC(horizon_steps=2).predict_action(env)
    assert action.shape == (4,)
    assert np.isfinite(action).all()
    assert np.all((action >= -1.0) & (action <= 1.0))


def test_mpc_accepts_configurable_levels_and_cost_weights(env):
    controller = V5HybridMPC(
        horizon_steps=1,
        levels=(0.0, 1.0),
        temperature_weight=2.0,
        humidity_weight=0.5,
        effort_weight=0.1,
        variation_weight=0.4,
    )

    assert len(controller.candidates) == 16
    assert controller.temperature_weight == pytest.approx(2.0)
    assert controller.humidity_weight == pytest.approx(0.5)
    assert controller.effort_weight == pytest.approx(0.1)
    assert controller.variation_weight == pytest.approx(0.4)
    assert np.isfinite(controller.predict_action(env)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"levels": ()},
        {"levels": (-0.1, 0.5)},
        {"levels": (0.5, 1.1)},
        {"temperature_weight": -1.0},
        {"humidity_weight": -1.0},
        {"effort_weight": -1.0},
        {"variation_weight": -1.0},
    ],
)
def test_mpc_rejects_invalid_search_configuration(kwargs):
    with pytest.raises(ValueError):
        V5HybridMPC(**kwargs)
