"""完整生长季作物生长诊断（冠层过热 bug 修复后的验证脚本）。

逐时记录作物通量，对比幼苗起点(102天)与 GH2024 实测起点(57天)的生长轨迹，
验证「叶饿死」已修复、冠层温度贴近空气温度。

用法：
    $PY experiments/controllers/glass_rl/full_season_growth_diag.py

背景：完整生长季覆盖时作物叶量在 4 月/7 月饿死。根因是 GlassGreenhouse
物理模型冠层过热（h_air_canopy=300 低 19 倍 → 冠层 66°C → 光合归零）。
修复见 docs/notes/04-完整生长季_冠层过热bug修复.md。
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import casadi as ca

sys.path.insert(0, ".")

from experiments.controllers.glass_rl.glass_env import GlassGreenhouseEnv
from glassgym.models.GreenLight.crop import crop_fluxes
from glassgym.models.GreenLight.aux_states import update


def make_flux_fn(nx=28, nu=10, nd=11, n_params=228):
    x = ca.SX.sym("x", nx); u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd); p = ca.SX.sym("p", n_params)
    conv = 101325.0 * 0.04401 / (8.3144598 * (x[2] + 273.15))
    x_crop = ca.vertcat(conv * x[0], x[1:])
    flux = crop_fluxes(x_crop, u[:6], d, p)
    aux = update(x_crop, u[:6], d, p)
    keys = ["photosynthesis", "buffer_to_leaf", "buffer_to_stem", "buffer_to_fruit",
            "growth_respiration", "leaf_maintenance", "stem_maintenance",
            "fruit_maintenance", "leaf_pruning", "fruit_harvest"]
    extra = ca.vertcat(aux[199], aux[205], aux[204], aux[31], aux[200])
    return ca.Function("diag", [x, u, d, p], [ca.vertcat(*[flux[k] for k in keys], extra)]), keys


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


def run(env, fn, keys, n_days):
    rows = []
    for _ in range(n_days * 24):
        u = env._action_to_u(phys_control(env))
        d = env._weather_at(env._w_idx)
        val = np.asarray(fn(env.x, u, d, env.params)).flatten()
        rec = {k: v for k, v in zip(keys, val[:len(keys)])}
        rec.update({"cBuf": float(env.x[22]), "cLeaf": float(env.x[23]),
                    "cStem": float(env.x[24]), "cFruit": float(env.x[25]),
                    "tCanSum": float(env.x[26]), "LAI": float(val[len(keys) + 3]),
                    "net_photo": float(val[len(keys) + 4]), "t": float(env.x[2]),
                    "t_can": float(env.x[4])})
        rows.append(rec)
        env.step(phys_control(env))
    return pd.DataFrame(rows)


def summarize(df, label, targets=None):
    df = df.copy(); df["day"] = np.arange(len(df)) // 24
    daily = df.groupby("day").agg(
        photo=("net_photo", "sum"), leaf=("cLeaf", "last"), fruit=("cFruit", "last"),
        stem=("cStem", "last"), buf=("cBuf", "last"), ts=("tCanSum", "last"),
        lai=("LAI", "last"), t=("t", "mean"), t_can=("t_can", "mean"))
    daily["photo"] = daily["photo"] * 3600.0
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"起始: 叶{df.cLeaf.iloc[0]:.0f} 茎{df.cStem.iloc[0]:.0f} 果{df.cFruit.iloc[0]:.0f} "
          f"缓冲{df.cBuf.iloc[0]:.0f} 积温{df.tCanSum.iloc[0]:.0f} LAI{df.LAI.iloc[0]:.2f}")
    print(f"结束({len(daily)}天): 叶{df.cLeaf.iloc[-1]:.0f} 茎{df.cStem.iloc[-1]:.0f} "
          f"果{df.cFruit.iloc[-1]:.0f} 缓冲{df.cBuf.iloc[-1]:.0f} LAI{df.LAI.iloc[-1]:.2f}")
    print(f"叶: {df.cLeaf.iloc[0]:.0f} -> 峰值{df.cLeaf.max():.0f} -> {df.cLeaf.iloc[-1]:.0f}")
    print(f"果: {df.cFruit.iloc[0]:.0f} -> {df.cFruit.iloc[-1]:.0f}")
    print(f"冠层-空气温差: 均值{np.mean(df.t_can - df.t):+.1f}°C 峰值{np.max(df.t_can - df.t):+.1f}°C")
    if targets:
        print(f"目标(实测参考): 叶->{targets['leaf']}, 果->{targets['fruit']}")
    print(f"\n{'day':>4} {'叶':>9} {'茎':>9} {'果':>9} {'缓冲':>8} {'积温':>7} {'LAI':>5} {'光合mg/d':>9}")
    for i in range(0, len(daily), max(1, len(daily) // 10)):
        r = daily.iloc[i]
        print(f"{i:>4} {r.leaf:>9.0f} {r.stem:>9.0f} {r.fruit:>9.0f} {r.buf:>8.0f} "
              f"{r.ts:>7.0f} {r.lai:>5.2f} {r.photo:>9.0f}")
    return daily


if __name__ == "__main__":
    fn, keys = make_flux_fn()
    print("flux fn 编译 OK")

    # 1) 完整生长季：幼苗起点 4/1 起 102 天
    env = GlassGreenhouseEnv(episode_days=102, start_day_index=0,
                             crop_start="seedling", disable_supplements=True)
    env.reset(seed=0)
    summarize(run(env, fn, keys, 102), "完整生长季 幼苗起点 4/1 -> 7/12 (102天)")

    # 2) GH2024 实测起点 57 天对比
    env2 = GlassGreenhouseEnv(episode_days=57, start_day_index=0,
                              crop_start="seedling", disable_supplements=True)
    env2.reset(seed=0)
    for i, v in {22: 15000.0, 23: 14750.0, 24: 14250.0, 25: 500.0, 26: 0.0}.items():
        env2.x[i] = v
    env2._prev_fruit = float(env2.x[25])
    summarize(run(env2, fn, keys, 57), "GH2024实测起点(叶14750) 4/1起57天",
              targets={"leaf": 69500, "fruit": 229250})
