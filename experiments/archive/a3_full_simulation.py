"""A3 完整仿真：ChengduPhysics + ERA5 2025 模拟 GH2024 春茬作物轨迹。

跑 126 天（2025-01-15 ~ 05-20），记录逐时室内温度 + 作物干物质状态，
保存为 CSV 供 standing-crop 对齐使用。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from RL.utils import load_env_params, build_env_kwargs
from glassgym.environments.greenlight_env import GreenLightEnv
from processing.chengdu_sql_controls import split_mysql_values

SQL = r"E:/school/NKY/玻璃温室数据库信息/tomato_full_backup_20260724.sql"
OUT = Path("results/chengdu_agri_greenhouse_001/harvest_model_gh2024/simulated_crop_trajectory.csv")


def load_daily_temperature(sql_text: str) -> pd.DataFrame:
    acc = []
    for m in re.finditer(r"INSERT INTO `accumulate_temperature` VALUES \((.*?)\);", sql_text, re.S):
        v = split_mysql_values(m.group(1))
        acc.append((v[6], v[7], v[8], v[9]))
    frame = pd.DataFrame(acc, columns=["greenhouse_id", "date", "normal", "active"])
    for c in ("normal", "active"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["greenhouse_id"] == "1"].sort_values("date").set_index("date")


def main():
    start_day = 15
    season_length = 126  # 01-15 ~ 05-20

    env_kwargs = load_env_params("ChengduControllerBenchmark", "configs/envs/")
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    env_kwargs["season_length"] = season_length
    env_kwargs["normalize_actions"] = False
    env_kwargs["valid_reward_bounds"] = (-1e6, 1e6)  # 放宽，A3 只关心作物轨迹

    env = GreenLightEnv(**env_kwargs)

    # 应用 GH63 标定参数（关键：vent_scale=6, solar_gain=0.6, ua=2.5, heat_capacity=4）
    import json
    calib = json.load(open(
        "results/chengdu_agri_greenhouse_001/physics_calibration/chengdu_physics_calibrated_params.json",
        encoding="utf-8",
    ))
    calibrated = env.base_p.copy()
    for index, value in calib["multipliers"].items():
        calibrated[int(index)] = 1.0 + 1.0 * (float(value) - 1.0)
    env.base_p = calibrated
    env.parameter_provider.base_p = calibrated.copy()
    env.p = calibrated.copy()
    print(f"已应用标定参数: {calib['multipliers']}")

    env.reset(seed=2026, options={"scenario": {"location": "Chengdu", "growth_year": 2025, "start_day": start_day}})

    controlled_idx = list(env.action_scheme.controlled_idx)
    n_action = len(controlled_idx)
    vent_idx = controlled_idx.index(3)  # uVent
    blscr_idx = controlled_idx.index(5)  # uBlScr 遮阳幕
    print(f"controlled_idx={controlled_idx}, action={n_action}, uVent idx={vent_idx}, uBlScr idx={blscr_idx}")

    t0 = pd.Timestamp("2025-01-01") + pd.to_timedelta(start_day, unit="D")
    records = []
    for step in range(env.N):
        # 固定基础通风 0.2 + 高温遮阳（26°C 以上开遮阳幕，避免强辐射下温度发散）
        t_air = float(env.x[2])
        vent = 0.2
        blscr = 1.0 if t_air > 26.0 else 0.0
        u = np.zeros(n_action, dtype=np.float32)
        u[vent_idx] = vent
        u[blscr_idx] = blscr
        _, _, terminated, truncated, _ = env.step(u)
        records.append({
            "timestamp": t0 + pd.to_timedelta(step * 900, unit="s"),
            "t_air_c": float(env.x[2]),
            "t_can_c": float(env.x[4]),
            "rh_percent": float(env.x[16]) if env.nx > 16 else np.nan,
            "c_buffer_mg_m2": float(env.x[22]),
            "c_leaf_mg_m2": float(env.x[23]),
            "c_stem_mg_m2": float(env.x[24]),
            "c_fruit_mg_m2": float(env.x[25]),
        })
        if terminated or truncated:
            break

    traj = pd.DataFrame(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"轨迹已保存: {OUT} ({len(traj)} 步)")

    # 温度对比（完整窗口）
    daily = traj.set_index("timestamp")["t_air_c"].resample("1D").mean()
    observed = load_daily_temperature(open(SQL, encoding="utf-8-sig").read())
    overlap = pd.DataFrame({"sim_t_air": daily, "obs_normal": observed["normal"]}).dropna()
    diff = overlap["sim_t_air"] - overlap["obs_normal"]
    print(f"\n完整窗口温度对比: 重叠 {len(overlap)} 天, sim mean={overlap['sim_t_air'].mean():.1f}°C, obs mean={overlap['obs_normal'].mean():.1f}°C, bias={diff.mean():+.2f}°C, MAE={diff.abs().mean():.2f}°C, corr={overlap['sim_t_air'].corr(overlap['obs_normal']):.3f}")

    # 作物轨迹摘要
    last = traj.iloc[-1]
    print(f"\n作物状态(season 末): 果实={last['c_fruit_mg_m2']*1e-6:.3f} kg/m2, 叶={last['c_leaf_mg_m2']*1e-6:.3f}, 茎={last['c_stem_mg_m2']*1e-6:.3f}, 缓冲={last['c_buffer_mg_m2']*1e-6:.3f}")
    peak_fruit = traj["c_fruit_mg_m2"].max() * 1e-6
    print(f"果实峰值: {peak_fruit:.3f} kg/m2")


if __name__ == "__main__":
    main()
