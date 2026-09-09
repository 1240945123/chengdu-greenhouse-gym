import casadi as ca
import numpy as np
import pandas as pd
import pytest

from experiments.reports.evaluate_chengdu_physics import row_to_model_inputs
from glassgym.configs.default_params import init_default_params
from glassgym.models.ChengduPhysicsV3.utils import define_model as define_v3
from glassgym.models.ChengduPhysicsV4.utils import define_model as define_v4


def _row(radiation: float = 400.0) -> pd.Series:
    return pd.Series(
        {
            "x_air_temperature": 24.0,
            "x_relative_humidity": 65.0,
            "x_co2_concentration": 650.0,
            "x_soil_temperature": 20.0,
            "d_air_temperature": 18.0,
            "d_relative_humidity": 80.0,
            "d_global_radiation": radiation,
            "d_wind_speed": 1.0,
            "d_sky_temperature": 5.0,
            "uRoofVent": 0.0,
            "uFan": 0.0,
            "uPad": 0.0,
        }
    )


def _v3_params() -> np.ndarray:
    params = np.asarray(init_default_params(220), dtype=float)
    params[208:220] = [8.0, 0.6, 2.5, 8.0, 0.6, 1.0, 0.6, 3.0, 0.2, 8.0, 0.0, 0.0]
    return params


def _v4_params(extension=(0.0, 1.0, 0.0, 1.0, 0.0)) -> np.ndarray:
    return np.r_[_v3_params(), np.asarray(extension, dtype=float)]


def _step(backend, row: pd.Series, params: np.ndarray, state_override=None) -> np.ndarray:
    state, control, disturbance = row_to_model_inputs(row, model_backend="ChengduPhysicsV4")
    if state_override is not None:
        state_override(state)
    result = backend(x0=ca.DM(state), u=ca.DM(control), p=ca.vertcat(ca.DM(disturbance), ca.DM(params)))
    return result["xf"].full().flatten()


def test_v4_requires_eight_controls():
    with pytest.raises(ValueError, match="eight control"):
        define_v4(nx=28, nu=7, nd=10, n_params=225, dt=3600.0)


def test_v4_zero_extension_matches_v3_reference():
    row = _row()
    state3, control3, disturbance3 = row_to_model_inputs(row, model_backend="ChengduPhysicsV3")
    state4, control4, disturbance4 = row_to_model_inputs(row, model_backend="ChengduPhysicsV4")
    model3 = define_v3(nx=28, nu=8, nd=10, n_params=220, dt=3600.0)
    model4 = define_v4(nx=28, nu=8, nd=10, n_params=225, dt=3600.0)
    result3 = model3(x0=state3, u=control3, p=ca.vertcat(disturbance3, _v3_params()))["xf"].full().flatten()
    result4 = model4(x0=state4, u=control4, p=ca.vertcat(disturbance4, _v4_params()))["xf"].full().flatten()

    np.testing.assert_allclose(result4[[2, 4, 5, 8, 15]], result3[[2, 4, 5, 8, 15]], atol=1e-5)


def test_longwave_and_latent_terms_cool_expected_thermal_nodes():
    row = _row(radiation=500.0)
    model = define_v4(nx=28, nu=8, nd=10, n_params=225, dt=3600.0)
    reference = _step(model, row, _v4_params())
    longwave = _step(model, row, _v4_params((1.0, 1.0, 0.0, 1.0, 0.0)))
    latent = _step(model, row, _v4_params((0.0, 1.0, 0.0, 1.0, 1.0)))

    assert longwave[5] < reference[5]
    assert latent[4] < reference[4]


def test_moisture_buffer_moves_vapor_toward_buffer_potential():
    row = _row(radiation=0.0)
    model = define_v4(nx=28, nu=8, nd=10, n_params=225, dt=3600.0)

    def humid_buffer(state):
        state[16] = state[15] + 500.0

    no_exchange = _step(model, row, _v4_params(), state_override=humid_buffer)
    exchange = _step(
        model,
        row,
        _v4_params((0.0, 1.0, 1.0 / 7200.0, 4.0, 0.0)),
        state_override=humid_buffer,
    )

    assert exchange[15] > no_exchange[15]
    assert exchange[16] < no_exchange[16]
