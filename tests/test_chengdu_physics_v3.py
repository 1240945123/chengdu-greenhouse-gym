from __future__ import annotations

import casadi as ca
import numpy as np

from glassgym.configs.default_params import init_default_params
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres


def _state() -> np.ndarray:
    x = np.zeros(28, dtype=float)
    x[0:2] = 650.0
    x[2:15] = 35.0
    x[15] = vaporDens2pres(35.0, rh2vaporDens(35.0, 70.0))
    x[16] = x[15]
    x[17:22] = 35.0
    x[23:26] = [9.5283e4, 2.5107e5, 5.5338e4]
    return x


def _params() -> np.ndarray:
    p = np.asarray(init_default_params(220), dtype=float)
    p[208:216] = [8.0, 0.6, 2.5, 8.0, 0.6, 1.0, 0.6, 3.0]
    p[216:220] = [0.2, 8.0, 0.0, 0.0]
    return p


def _derivative(control: np.ndarray, wind: float = 1.0, params: np.ndarray | None = None) -> np.ndarray:
    from glassgym.models.ChengduPhysicsV3.ode import ODE

    x = _state()
    d = np.array([0.0, 20.0, 1500.0, 420.0, wind, 14.0, 22.0, 0.0, 0.0, 0.0])
    p = _params() if params is None else params
    return ca.Function("v3_dx", [], [ODE(ca.DM(x), ca.DM(control), ca.DM(d), ca.DM(p))])()["o0"].full().ravel()


def test_v3_roof_exchange_responds_to_wind():
    control = np.zeros(8)
    control[3] = 1.0
    assert _derivative(control, wind=5.0)[2] < _derivative(control, wind=0.0)[2]


def test_v3_fan_exchange_operates_with_roof_closed():
    off = np.zeros(8)
    fan = off.copy()
    fan[6] = 1.0
    assert _derivative(fan)[2] < _derivative(off)[2]


def test_v3_background_infiltration_remains_when_roof_and_fan_are_off():
    off = np.zeros(8)
    assert _derivative(off, wind=5.0)[2] < _derivative(off, wind=0.0)[2]
    assert _derivative(off, wind=0.0)[2] < 0.0


def test_v3_zero_pad_parameters_make_sparse_pad_action_non_identifiable():
    off = np.zeros(8)
    pad = off.copy()
    pad[7] = 1.0
    np.testing.assert_allclose(_derivative(pad), _derivative(off), atol=1e-12)
