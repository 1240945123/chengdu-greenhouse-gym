from __future__ import annotations

import casadi as ca
import numpy as np

from experiments.reports.evaluate_chengdu_physics import default_parameter_vector
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres, vaporPres2rh


def _state() -> np.ndarray:
    x = np.zeros(28, dtype=float)
    x[0:2] = 650.0
    x[2:15] = 25.0
    x[15] = vaporDens2pres(25.0, rh2vaporDens(25.0, 70.0))
    x[16] = x[15]
    x[17:22] = 25.0
    x[23:26] = [9.5283e4, 2.5107e5, 5.5338e4]
    return x


def _params() -> np.ndarray:
    p = default_parameter_vector()
    p[208:216] = [8.0, 0.45, 4.0, 6.0, 0.6, 1.0, 0.6, 3.0]
    return p


def test_v2_partitions_solar_energy_into_canopy_and_floor_instead_of_air_only():
    from glassgym.models.ChengduPhysics.ode import ODE as legacy_ode
    from glassgym.models.ChengduPhysicsV2.ode import ODE as v2_ode

    x = _state()
    u = np.zeros(6)
    d = np.array([900.0, 25.0, x[15], 420.0, 1.0, 19.0, 22.0, 0.0, 1.0, 1.0])
    p = _params()
    legacy = ca.Function("legacy_partition", [], [legacy_ode(ca.DM(x), ca.DM(u), ca.DM(d), ca.DM(p))])()["o0"].full().ravel()
    v2 = ca.Function("v2_partition", [], [v2_ode(ca.DM(x), ca.DM(u), ca.DM(d), ca.DM(p))])()["o0"].full().ravel()

    assert v2[4] > 0.0
    assert v2[8] > 0.0
    assert v2[2] < legacy[2]


def test_v2_closed_roof_retains_wind_dependent_rain_safe_exchange():
    from glassgym.models.ChengduPhysicsV2.ode import ODE

    x = _state()
    x[2] = 35.0
    u = np.zeros(6)
    p = _params()
    calm = np.array([0.0, 20.0, x[15], 420.0, 0.0, 14.0, 22.0, 0.0, 0.0, 0.0])
    windy = calm.copy()
    windy[4] = 5.0
    dx_calm = ca.Function("v2_calm", [], [ODE(ca.DM(x), ca.DM(u), ca.DM(calm), ca.DM(p))])()["o0"].full().ravel()
    dx_windy = ca.Function("v2_windy", [], [ODE(ca.DM(x), ca.DM(u), ca.DM(windy), ca.DM(p))])()["o0"].full().ravel()

    assert dx_windy[2] < dx_calm[2]


def test_v2_roof_vent_vapor_exchange_is_bounded_by_air_exchange_rate():
    from glassgym.models.ChengduPhysicsV2.ode import ODE

    x = _state()
    x[15] = 3000.0
    p = _params()
    u = np.zeros(6)
    u[3] = 1.0
    d = np.array([0.0, 25.0, 1500.0, 420.0, 1.0, 19.0, 22.0, 0.0, 0.0, 0.0])
    dx = ca.Function("v2_vapor_bound", [], [ODE(ca.DM(x), ca.DM(u), ca.DM(d), ca.DM(p))])()["o0"].full().ravel()

    assert abs(dx[15]) < 100.0


def test_v2_remains_finite_and_inside_safety_envelope_for_72_hour_heat_wave():
    from glassgym.models.ChengduPhysicsV2.utils import define_model

    integrator = define_model(nx=28, nu=6, nd=10, n_params=216, dt=900.0)
    x = _state()
    p = _params()
    u = np.zeros(6)
    temperatures = []
    humidities = []
    for step in range(72 * 4):
        hour = (step / 4.0) % 24.0
        radiation = 1100.0 * max(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
        outdoor_temperature = 27.0 + 6.0 * max(0.0, np.sin(np.pi * (hour - 6.0) / 12.0))
        outdoor_vp = vaporDens2pres(outdoor_temperature, rh2vaporDens(outdoor_temperature, 75.0))
        d = np.array([radiation, outdoor_temperature, outdoor_vp, 420.0, 1.5, outdoor_temperature - 6.0, 24.0, 0.0, radiation > 0, radiation > 0])
        result = integrator(x0=ca.DM(x), u=ca.DM(u), p=ca.vertcat(ca.DM(d), ca.DM(p)))
        x = result["xf"].full().ravel()
        temperatures.append(float(x[2]))
        humidities.append(float(vaporPres2rh(x[2], x[15])))

    assert np.isfinite(x).all()
    assert max(temperatures) <= 60.0
    assert min(temperatures) >= -10.0
    assert min(humidities) >= 0.0
    assert max(humidities) <= 100.0
