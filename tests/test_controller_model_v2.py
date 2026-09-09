from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from experiments.controllers.benchmark_protocol import build_environment, load_benchmark_config
from experiments.controllers.tune_benchmark_classical import _mpc_step_context
from glassgym.components.controller_model import ControllerModelConfigV2, ControllerModelV2
from glassgym.components.mpc import LightweightMPCController
from glassgym.core.types import ControllerStepContextV2
from RL.utils import load_model_hyperparams
from glassgym.environments.utils import satVp


def test_mpc_context_exposes_only_permitted_controller_information():
    config = load_benchmark_config("smoke")
    env = build_environment(config, config.test_start_day, episode_days=1)
    try:
        env.reset(seed=0)
        original = _mpc_step_context(env)
        temperature = float(env.x[2])
        vapor_pressure = float(env.x[15])
        env.x[:] = np.linspace(-1e9, 1e9, len(env.x))
        env.x[2] = temperature
        env.x[15] = vapor_pressure
        env.p[:] = np.linspace(1e12, 2e12, len(env.p))
        poisoned = _mpc_step_context(env)
    finally:
        env.close()

    names = {field.name for field in fields(ControllerStepContextV2)}
    assert names == {
        "t", "dt", "indoor_temperature", "indoor_relative_humidity",
        "u", "forecast", "hour_of_day", "day_of_year",
    }
    assert not names.intersection({"x", "x_prev", "p", "d", "env"})
    assert original.indoor_temperature == pytest.approx(poisoned.indoor_temperature)
    assert original.indoor_relative_humidity == pytest.approx(
        poisoned.indoor_relative_humidity
    )
    np.testing.assert_array_equal(original.u, poisoned.u)
    np.testing.assert_array_equal(original.forecast.values, poisoned.forecast.values)


def test_controller_model_predictions_are_finite_bounded_and_directional():
    model = ControllerModelV2(ControllerModelConfigV2())
    weather = np.zeros(11, dtype=float)
    weather[1] = 10.0
    weather[2] = 0.7 * 1227.0
    weather[4] = 2.0
    state = np.array([28.0, 90.0])
    closed = model.predict_next(state, np.zeros(6), weather, dt_seconds=900)
    ventilated = model.predict_next(
        state, np.array([0, 0, 0, 1, 0, 0], dtype=float), weather, dt_seconds=900
    )

    assert np.isfinite(closed).all()
    assert -30.0 <= closed[0] <= 60.0
    assert 0.0 <= closed[1] <= 100.0
    assert ventilated[0] < closed[0]
    closed_vp = closed[1] / 100.0 * satVp(closed[0])
    ventilated_vp = ventilated[1] / 100.0 * satVp(ventilated[0])
    assert ventilated_vp < closed_vp


def test_controller_model_rejects_nonphysical_coefficients():
    with pytest.raises(ValueError, match="non-negative"):
        ControllerModelConfigV2(outdoor_exchange_per_hour=-0.1)


def test_controller_model_conserves_vapor_pressure_during_dry_heating():
    config = ControllerModelConfigV2(
        outdoor_exchange_per_hour=0.0,
        ventilation_exchange_per_hour=0.0,
        wind_exchange_per_hour_per_m_s=0.0,
        solar_gain_per_w_m2_hour=0.0,
        lamp_gain_per_hour=4.0,
        boiler_gain_per_hour=0.0,
        vapor_exchange_per_hour=0.0,
        vapor_ventilation_exchange_per_hour=0.0,
        vapor_source_pa_per_hour=0.0,
        lamp_drying_pa_per_hour=0.0,
        boiler_drying_pa_per_hour=0.0,
    )
    model = ControllerModelV2(config)
    weather = np.zeros(11)
    weather[1] = 20.0
    weather[2] = 1000.0
    state = np.array([20.0, 80.0])
    predicted = model.predict_next(
        state, np.array([0, 0, 0, 0, 1, 0], dtype=float), weather,
        dt_seconds=3600,
    )

    initial_vp = 0.8 * satVp(20.0)
    predicted_vp = predicted[1] / 100.0 * satVp(predicted[0])
    assert predicted[0] == pytest.approx(24.0)
    assert predicted_vp == pytest.approx(initial_vp)


def test_mpc_proposal_is_invariant_to_unavailable_plant_values():
    config = load_benchmark_config("smoke")
    env = build_environment(config, config.test_start_day, episode_days=1)
    params = load_model_hyperparams("mpc", "GreenLightEnv")
    params["horizon_steps"] = 2
    controller = LightweightMPCController(**params)
    try:
        env.reset(seed=0)
        permitted = _mpc_step_context(env)
        first = controller.predict(permitted)
        env.x[[0, 1, 3, 4, 21, 22, 23, 24, 25, 26]] = 1e15
        env.p[:] = -1e15
        second = controller.predict(_mpc_step_context(env))
    finally:
        env.close()

    np.testing.assert_array_equal(first, second)


def test_mpc_rejects_legacy_full_state_context():
    params = load_model_hyperparams("mpc", "GreenLightEnv")
    controller = LightweightMPCController(**params)
    with pytest.raises(TypeError, match="ControllerStepContextV2"):
        controller.predict(object())
