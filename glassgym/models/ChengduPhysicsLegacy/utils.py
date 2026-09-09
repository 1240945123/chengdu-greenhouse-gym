import casadi as ca

from glassgym.models.ChengduPhysicsLegacy.ode import ODE


def define_model(nx: int, nu: int, nd: int, n_params: int, dt: float):
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)
    input_args_sym = ca.vertcat(d, p)
    return ca.integrator(
        "F",
        "cvodes",
        {"x": x, "u": u, "p": input_args_sym, "ode": ODE(x, u, d, p)},
        0.0,
        dt,
        {
            "abstol": 1e-5,
            "reltol": 1e-5,
            "max_num_steps": 7e4,
            "max_step_size": 60.0,
        },
    )
