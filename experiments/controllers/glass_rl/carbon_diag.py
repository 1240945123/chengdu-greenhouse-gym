"""碳平衡通量诊断：定位亏碳的温度区间与时段。

逐时记录净光合(a200)、维持呼吸(a210-212)、温度，量化日碳平衡。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import casadi as ca

sys.path.insert(0, ".")
sys.path.insert(0, ".tmp")

from glass_env import GlassGreenhouseEnv
from glassgym.models.GreenLight.crop import crop_fluxes

FLUX_KEYS = [
    "photosynthesis", "leaf_maintenance", "stem_maintenance",
    "fruit_maintenance", "growth_respiration", "buffer_to_fruit",
]


def make_flux_function(nx=28, nu=10, nd=11, n_params=228):
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)
    # 与 GlassGreenhouse.ode 一致：x[0] 由温室维护为 ppm，作物需要 mg/m^3
    conv = 101325.0 * 0.04401 / (8.3144598 * (x[2] + 273.15))
    x_crop = ca.vertcat(conv * x[0], x[1:])
    flux = crop_fluxes(x_crop, u[:6], d, p)
    expr = ca.vertcat(*[flux[k] for k in FLUX_KEYS])
    return ca.Function("crop_flux", [x, u, d, p], [expr])


def phys_control(env):
    t = float(env.x[2]); h = env._hour
    a = np.zeros(9, dtype=int)
    if 6 <= h < 20:
        if t > 30: a = [1, 0, 0, 2, 0, 0, 3, 1, 1]
        elif t > 27: a = [1, 0, 0, 2, 0, 0, 2, 1, 1]
        elif t > 25: a = [1, 0, 0, 1, 0, 0, 1, 0, 0]
        elif t < 15: a = [0, 1, 1, 0, 0, 0, 0, 0, 0]
        else: a = [0, 0, 0, 1, 0, 0, 0, 0, 0]
    else:
        if t > 23: a = [0, 0, 0, 1, 0, 0, 1, 0, 0]
        elif t < 19: a = [0, 1, 1, 0, 0, 0, 0, 0, 0]
    return a


def run_days(day: int, n_days: int, fflux, start_crop=None):
    env = GlassGreenhouseEnv(episode_days=n_days, start_day_index=day)
    env.reset(seed=0)
    if start_crop:
        for i, v in start_crop.items():
            env.x[i] = v
    env._prev_fruit = float(env.x[25])
    rows = []
    for _ in range(n_days * 24):
        a = phys_control(env)
        u = env._action_to_u(a)
        d = env._weather_at(env._w_idx)
        val = np.asarray(fflux(env.x, u, d, env.params)).flatten()
        rows.append({
            "hour": float(env._hour), "t_air": float(env.x[2]),
            "t_can": float(env.x[4]), "co2": float(env.x[0]),
            **{k: v for k, v in zip(FLUX_KEYS, val)},
        })
        env.step(a)
    return pd.DataFrame(rows)


def main() -> None:
    fflux = make_flux_function()
    print("flux function 编译 OK")
    # 5月中(day49)与7月初(day95)各跑2天
    for day, label in [(49, "5月中·温和"), (75, "6月中·湿热"), (95, "7月初·高温")]:
        df = run_days(day, 2, fflux)
        # 净碳 = 光合 - 维持呼吸(leaf+stem+fruit) - 生长呼吸
        df["net"] = (df["photosynthesis"] - df["leaf_maintenance"]
                     - df["stem_maintenance"] - df["fruit_maintenance"]
                     - df["growth_respiration"]) * 3600.0  # per hour, mg CH2O
        day_net = df["net"].sum()
        daytime = df[df["hour"].between(6, 20)]
        nighttime = df[~df["hour"].between(6, 20)]
        print(f"\n=== {label} (day {day}) 2天碳平衡 ===")
        print(f"  2天净碳: {day_net:+.0f} mg/m2  | 白天(光合)净 {daytime['net'].sum():+.0f} / 夜间 {nighttime['net'].sum():+.0f}")
        print(f"  温度: 均 {df['t_air'].mean():.1f}°C | 昼均 {daytime['t_air'].mean():.1f} | 最高 {df['t_air'].max():.1f}")
        # 光合 vs 温度关系
        hot = df[df["t_air"] > 30]
        mid = df[df["t_air"].between(20, 30)]
        resp_cols = ["leaf_maintenance", "stem_maintenance", "fruit_maintenance"]
        print(f"  >30C 时段: {len(hot)}h (光合 {hot['photosynthesis'].sum()*3600:+.0f}, 呼吸 {hot[resp_cols].sum().sum()*3600:+.0f})")
        print(f"  20-30C 时段: {len(mid)}h (光合 {mid['photosynthesis'].sum()*3600:+.0f})")
        # 全天逐6h采样
        for h in [6, 12, 18, 0, 23]:
            r = df[(df.hour >= h) & (df.hour < h + 1)].iloc[0]
            print(f"    {h:02d}时: t={r.t_air:5.1f} 光合={r.photosynthesis:8.1f} 叶呼={r.leaf_maintenance:6.1f} 茎呼={r.stem_maintenance:6.1f} 果呼={r.fruit_maintenance:6.1f} (mg/m2/s)")


if __name__ == "__main__":
    main()
