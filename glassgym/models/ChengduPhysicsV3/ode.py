import casadi as ca
import numpy as np

from glassgym.models.GreenLight.crop import crop_derivatives


def _sat(values, idx, default):
    return values[idx] if values.shape[0] > idx else default


def _positive(values, idx, default=1.0):
    if values.shape[0] <= idx:
        return default
    return ca.if_else(values[idx] > 0.0, values[idx], default)


def _nonnegative(values, idx, default=0.0):
    if values.shape[0] <= idx:
        return default
    return ca.fmax(0.0, values[idx])


def ODE(x: np.ndarray, u: np.ndarray, d: np.ndarray, p: np.ndarray):
    """V2 thermal nodes with independent roof, fan, and pad control channels."""
    if u.shape[0] != 8:
        raise ValueError("ChengduPhysicsV3 requires eight control inputs")
    dxdt = ca.SX.zeros(x.shape[0])
    i_glob = ca.fmax(0.0, d[0])
    t_out = d[1]
    vp_out = d[2]
    co2_out = d[3]
    wind = ca.fmax(0.0, d[4])

    co2_air = ca.fmin(2500.0, ca.fmax(0.0, x[0]))
    t_air = ca.fmin(70.0, ca.fmax(-20.0, x[2]))
    t_can = ca.fmin(70.0, ca.fmax(-20.0, x[4]))
    t_cover = ca.fmin(70.0, ca.fmax(-20.0, x[5]))
    t_floor = ca.fmin(70.0, ca.fmax(-20.0, x[8]))
    t_pipe = ca.fmin(80.0, ca.fmax(-20.0, x[9]))
    vp_air = ca.fmin(8000.0, ca.fmax(0.0, x[15]))

    u_boil = u[0]
    u_co2 = u[1]
    u_th_scr = u[2]
    u_roof = u[3]
    u_lamp = u[4]
    u_bl_scr = u[5]
    u_fan = u[6]
    u_pad = u[7]

    area = 192.0
    volume = area * 4.7
    rho_air = 1.2
    cp_air = 1005.0
    heat_capacity_scale = _positive(p, 208)
    solar_gain_scale = _positive(p, 209)
    ua_scale = _positive(p, 210)
    roof_vent_scale = _positive(p, 211)
    co2_vent_scale = _positive(p, 212)
    co2_photo_scale = _positive(p, 213)
    vapor_transpiration_scale = _positive(p, 214)
    vapor_vent_scale = _positive(p, 215)
    background_ach = _nonnegative(p, 216, 0.2)
    fan_max_ach = _nonnegative(p, 217, 8.0)
    pad_cooling_power = _nonnegative(p, 218, 0.0)
    pad_vapor_source = _nonnegative(p, 219, 0.0)

    c_air = rho_air * cp_air * volume * heat_capacity_scale
    c_canopy = area * 2.0e4
    c_floor = area * 1.5e5
    c_cover = area * 1.2e4
    h_air_canopy = 300.0
    h_air_floor = 700.0
    h_air_cover = 900.0
    h_cover_out = 1300.0
    h_floor_soil = 250.0

    ua = 850.0 * ua_scale * (1.0 - 0.45 * u_th_scr)
    infiltration_ach = background_ach + 0.08 * wind
    roof_ach = (5.0 + 0.4 * wind) * u_roof * roof_vent_scale
    fan_ach = fan_max_ach * u_fan
    exchange_rate = (infiltration_ach + roof_ach + fan_ach) / 3600.0

    solar_total = solar_gain_scale * area * 0.62 * i_glob * (1.0 - 0.75 * u_bl_scr)
    solar_air = 0.08 * solar_total
    solar_canopy = 0.37 * solar_total
    solar_floor = 0.55 * solar_total
    heating_power = 50000.0 * u_boil
    lamp_power = 10000.0 * u_lamp
    ventilation_loss = rho_air * cp_air * volume * exchange_rate * (t_air - t_out)

    dxdt[2] = (
        solar_air
        + heating_power
        + 0.75 * lamp_power
        - pad_cooling_power * u_pad
        - ua * (t_air - t_out)
        - ventilation_loss
        - h_air_canopy * (t_air - t_can)
        - h_air_floor * (t_air - t_floor)
        - h_air_cover * (t_air - t_cover)
    ) / c_air
    dxdt[4] = (solar_canopy + h_air_canopy * (t_air - t_can)) / c_canopy
    dxdt[5] = (h_air_cover * (t_air - t_cover) - h_cover_out * (t_cover - t_out)) / c_cover
    dxdt[8] = (
        solar_floor + h_air_floor * (t_air - t_floor) - h_floor_soil * (t_floor - d[6])
    ) / c_floor
    dxdt[9] = (45.0 * u_boil + t_air - t_pipe) / 900.0

    co2_dose = 0.25 * u_co2
    photo_assim = co2_photo_scale * 0.00003 * i_glob * ca.fmax(0.0, co2_air - 400.0)
    dxdt[0] = co2_vent_scale * exchange_rate * (co2_out - co2_air) + co2_dose - photo_assim

    transpiration = vapor_transpiration_scale * (
        0.0015 * i_glob + 0.0005 * ca.fmax(0.0, t_air - 15.0)
    )
    saturation_pressure = 610.78 * ca.exp(17.2694 * t_air / (t_air + 237.3))
    condensation = 0.002 * ca.fmax(0.0, vp_air - saturation_pressure)
    dxdt[15] = (
        vapor_vent_scale * exchange_rate * (vp_out - vp_air)
        + transpiration
        + pad_vapor_source * u_pad
        - condensation
    )
    dxdt[16] = (vp_air - x[16]) / 1800.0

    dxdt[3] = (t_air - x[3]) / 1800.0
    dxdt[6] = (t_out - x[6]) / 1800.0
    dxdt[7] = (t_air - x[7]) / 1800.0
    for idx in range(10, 15):
        dxdt[idx] = (d[6] - x[idx]) / 86400.0
    for idx in range(17, 21):
        dxdt[idx] = (t_air - x[idx]) / 3600.0
    dxdt[21] = (t_can - x[21]) / 86400.0
    dxdt[22:28] = crop_derivatives(x, u[:6], d, p)
    return dxdt
