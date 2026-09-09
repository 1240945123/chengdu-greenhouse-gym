import casadi as ca

from glassgym.models.ChengduPhysicsV4.ode import ODE


def define_model(nx: int, nu: int, nd: int, n_params: int, dt: float):
    if nu != 8:
        raise ValueError("ChengduPhysicsV4 requires eight control inputs")
    if n_params < 225:
        raise ValueError("ChengduPhysicsV4 requires at least 225 parameters")
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)
    return ca.integrator(
        "F",
        "cvodes",
        {"x": x, "u": u, "p": ca.vertcat(d, p), "ode": ODE(x, u, d, p)},
        0.0,
        dt,
        {"abstol": 1e-5, "reltol": 1e-5, "max_num_steps": 7e4, "max_step_size": 60.0},
    )

