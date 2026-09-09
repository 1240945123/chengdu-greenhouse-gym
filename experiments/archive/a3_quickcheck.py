"""A3 快速验证：ChengduPhysics 模拟 2025 室内温度 vs GH2024 日积温。

用 ChengduPhysics 后端 + ERA5 2025 天气跑一个 2025 春茬 season，
提取模拟室内温度（env.x[2]），算日平均，与 GH2024 的 accumulate_temperature.normal
（日平均温度）对比，判断「跨棚迁移模拟」是否合理。
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
    # GH2024 定植 2025-01-19 ~ 成熟 04-23，拉秧 ~05-20。模拟 01-15 ~ 05-20。
    start_day = 15  # 2025-01-15（0-indexed 从 1/1 起算，留少量预热）
    season_length = 30  # 先跑 30 天做快速量级验证（01-15 ~ 02-14）

    env_kwargs = load_env_params("ChengduControllerBenchmark", "configs/envs/")
    env_kwargs, default_scenarios = build_env_kwargs(env_kwargs)
    env_kwargs["season_length"] = season_length
    env_kwargs["normalize_actions"] = False  # 直接传物理控制值

    env = GreenLightEnv(**env_kwargs)
    env.reset(
        seed=2026,
        options={"scenario": {"location": "Chengdu", "growth_year": 2025, "start_day": start_day}},
    )

    # 固定控制：轻度通风 uVent=0.2，其余关闭
    # controlled_inputs = [uThScr, uVent, uLamp, uBlScr]，动作维度 = 4
    controlled_idx = list(env.action_scheme.controlled_idx)
    n_action = len(controlled_idx)
    vent_idx = controlled_idx.index(3)  # 3 = uVent 的物理索引
    print(f"controlled_idx={controlled_idx}, action 维度={n_action}, uVent action index={vent_idx}")

    records = []
    for step in range(env.N):
        u = np.zeros(n_action, dtype=np.float32)
        u[vent_idx] = 0.2
        obs, reward, terminated, truncated, info = env.step(u)
        records.append({
            "step": step,
            "t_air": float(env.x[2]),
            "t_can": float(env.x[4]),
            "fruit_mg_m2": float(env.x[25]),
            "leaf_mg_m2": float(env.x[23]),
            "stem_mg_m2": float(env.x[24]),
        })
        if terminated or truncated:
            break

    traj = pd.DataFrame(records)
    # 时间轴：start_day 天 + step * 900s
    traj["timestamp"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(start_day, unit="D") + pd.to_timedelta(traj["step"] * 900, unit="s")
    traj["date"] = traj["timestamp"].dt.date

    # 日平均室内温度
    daily = traj.groupby("date")["t_air"].mean()

    # GH2024 日积温
    observed = load_daily_temperature(open(SQL, encoding="utf-8-sig").read())
    overlap = pd.DataFrame({
        "sim_t_air": daily,
        "obs_normal": observed["normal"],
    }).dropna()

    print(f"\n=== 模拟室内温度 vs GH2024 日积温(normal) ===\n")
    print(f"重叠天数: {len(overlap)}")
    print(f"模拟室内温度: mean={overlap['sim_t_air'].mean():.1f}°C, range={overlap['sim_t_air'].min():.1f}~{overlap['sim_t_air'].max():.1f}")
    print(f"观测日积温normal: mean={overlap['obs_normal'].mean():.1f}°C, range={overlap['obs_normal'].min():.1f}~{overlap['obs_normal'].max():.1f}")
    diff = overlap["sim_t_air"] - overlap["obs_normal"]
    print(f"偏差(sim - obs): mean={diff.mean():.2f}°C, MAE={diff.abs().mean():.2f}°C, RMSE={np.sqrt((diff**2).mean()):.2f}°C")
    print(f"相关系数: {overlap['sim_t_air'].corr(overlap['obs_normal']):.3f}")

    # 按旬对比
    print("\n=== 按旬对比（sim vs obs normal）===")
    overlap["decade"] = pd.to_datetime(overlap.index).to_period("10D")
    for decade, g in overlap.groupby("decade"):
        print(f"  {decade}: sim={g['sim_t_air'].mean():.1f}°C, obs={g['obs_normal'].mean():.1f}°C, diff={g['sim_t_air'].mean()-g['obs_normal'].mean():+.1f}")

    # 果实状态
    print(f"\n=== 作物状态（season 末）===")
    print(f"  果实干重 x[25]: {traj['fruit_mg_m2'].iloc[-1]:.0f} mg/m2 = {traj['fruit_mg_m2'].iloc[-1]*1e-6:.3f} kg/m2")
    print(f"  叶 x[23]: {traj['leaf_mg_m2'].iloc[-1]*1e-6:.3f} kg/m2, 茎 x[24]: {traj['stem_mg_m2'].iloc[-1]*1e-6:.3f} kg/m2")


if __name__ == "__main__":
    main()
