import casadi as ca
import numpy as np

from glassgym.models.GreenLight.crop import crop_derivatives


def _sat(name, values, idx, default):
    return values[idx] if values.shape[0] > idx else default


def _multiplier(values, idx, default=1.0):
    if values.shape[0] <= idx:
        return default
    return ca.if_else(values[idx] > 0.0, values[idx], default)


def ODE(x: np.ndarray, u: np.ndarray, d: np.ndarray, p: np.ndarray):
    """Simplified 28-state compatible physics model for the NKY Chengdu Pidu greenhouse.

    The model keeps the GreenLight state layout so existing observations, rewards,
    PID, MPC, and surrogate pipelines can run while the physical parameters are
    calibrated from local greenhouse data.
    """
    dxdt = ca.SX.zeros(x.shape[0])

    i_glob = d[0]
    t_out = d[1]
    vp_out = d[2]
    co2_out = d[3]
    wind = d[4]

    co2_air = ca.fmin(2500.0, ca.fmax(0.0, x[0]))
    t_air = ca.fmin(60.0, ca.fmax(-20.0, x[2]))
    t_can = ca.fmin(60.0, ca.fmax(-20.0, x[4]))
    t_pipe = ca.fmin(80.0, ca.fmax(-20.0, x[9]))
    vp_air = ca.fmin(5000.0, ca.fmax(0.0, x[15]))

    u_boil = _sat("uBoil", u, 0, 0.0)
    u_co2 = _sat("uCO2", u, 1, 0.0)
    u_th_scr = _sat("uThScr", u, 2, 0.0)
    u_vent = _sat("uVent", u, 3, 0.0)
    u_lamp = _sat("uLamp", u, 4, 0.0)
    u_bl_scr = _sat("uBlScr", u, 5, 0.0)

    # Geometry currently reflects the extracted 192 m2 NKY target greenhouse.
    area = 192.0
    volume = 192.0 * 4.7
    rho_air = 1.2
    cp_air = 1005.0
    heat_capacity_air = rho_air * cp_air * volume

    heat_capacity_scale = _multiplier(p, 208)
    solar_gain_scale = _multiplier(p, 209)
    ua_scale = _multiplier(p, 210)
    vent_scale = _multiplier(p, 211)
    co2_vent_scale = _multiplier(p, 212)
    co2_photo_scale = _multiplier(p, 213)
    vapor_transpiration_scale = _multiplier(p, 214)
    vapor_vent_scale = _multiplier(p, 215)

    ua_base = 850.0 * ua_scale
    thermal_screen_saving = 0.45
    ua = ua_base * (1.0 - thermal_screen_saving * u_th_scr)

    vent_ach = (0.08 + (5.0 + 0.4 * wind) * u_vent) * vent_scale
    vent_rate = vent_ach / 3600.0

    solar_gain = solar_gain_scale * area * 0.62 * i_glob * (1.0 - 0.75 * u_bl_scr)
    heating_power = 50000.0 * u_boil
    lamp_power = 10000.0 * u_lamp
    envelope_loss = ua * (t_air - t_out)
    ventilation_loss = rho_air * cp_air * volume * vent_rate * (t_air - t_out)
    dxdt[2] = (solar_gain + heating_power + 0.75 * lamp_power - envelope_loss - ventilation_loss) / (heat_capacity_air * heat_capacity_scale)
    canopy_target = t_air + 0.006 * i_glob * (1.0 - 0.75 * u_bl_scr)
    dxdt[4] = (canopy_target - x[4]) / 600.0
    dxdt[9] = (45.0 * u_boil + t_air - t_pipe) / 900.0

    co2_dose = 0.25 * u_co2
    photo_assim = co2_photo_scale * 0.00003 * ca.fmax(0.0, i_glob) * ca.fmax(0.0, co2_air - 400.0)
    dxdt[0] = co2_vent_scale * vent_rate * (co2_out - co2_air) + co2_dose - photo_assim

    transpiration = vapor_transpiration_scale * (0.0015 * ca.fmax(0.0, i_glob) + 0.0005 * ca.fmax(0.0, t_air - 15.0))
    dehumidification = vapor_vent_scale * 0.35 * u_vent * ca.fmax(0.0, vp_air - vp_out)
    dxdt[15] = vapor_vent_scale * vent_rate * (vp_out - vp_air) + transpiration - dehumidification
    dxdt[16] = (vp_air - x[16]) / 1800.0

    dxdt[3] = (t_air - x[3]) / 1800.0
    dxdt[5] = (t_air - x[5]) / 1800.0
    dxdt[6] = (t_out - x[6]) / 1800.0
    dxdt[7] = (t_air - x[7]) / 1800.0
    dxdt[8] = (t_air - x[8]) / 3600.0
    for idx in range(10, 15):
        dxdt[idx] = (t_out - x[idx]) / 86400.0

    dxdt[21] = (t_can - x[21]) / 86400.0
    dxdt[22:28] = crop_derivatives(x, u, d, p)
    return dxdt
