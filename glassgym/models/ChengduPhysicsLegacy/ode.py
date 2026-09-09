import casadi as ca
import numpy as np

from glassgym.models.ChengduPhysics.ode import ODE as primary_ode, _multiplier


def ODE(x: np.ndarray, u: np.ndarray, d: np.ndarray, p: np.ndarray):
    """Pre-V2 model where roof ventilation also activates an 18 kW wet pad."""
    dxdt = primary_ode(x, u, d, p)
    t_air = ca.fmin(60.0, ca.fmax(-20.0, x[2]))
    t_out = d[1]
    u_vent = u[3] if u.shape[0] > 3 else 0.0
    heat_capacity_air = 1.2 * 1005.0 * (192.0 * 4.7)
    heat_capacity_scale = _multiplier(p, 208)
    fan_pad_cooling = 18_000.0 * u_vent * ca.fmax(0.0, t_air - t_out)
    dxdt[2] -= fan_pad_cooling / (heat_capacity_air * heat_capacity_scale)
    return dxdt
