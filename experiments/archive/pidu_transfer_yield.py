"""方案 b：郫都 GH63 2026 春茬迁移产量估计（v2，自己处理气候缺口）。

用新都源域验证的作物模型（默认 GreenLight 参数 + 干物质比例 0.081）
+ 郫都 2026 实测室内气候/控制，模拟郫都春茬（04-01~07-20）果实生长。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from experiments.crop.run_chengdu_target_crop_validation import (
    _load_environment_kwargs,
    prepare_observed_control_targets,
    simulate_forced_crop_trajectory,
)
from processing.chengdu_crop_initialization import build_target_crop_initial_state

OBS = "results/chengdu_agri_greenhouse_001/harvest_model/processed/standing_crop_samples.csv"
CONTROLS = "data/processed/chengdu_agri/greenhouse_001/controls/controls_1h.csv"
CLIMATE = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
ENV_CONFIG = "configs/envs/ChengduSingleGreenhouseEnv.yml"
OUT = Path("results/chengdu_agri_greenhouse_001/harvest_model_gh2024/pidu_transfer_yield_trajectory.csv")

TARGET_ID = 63
TARGET_CODE = "GH202602061448266452514"
BASELINE_DATE = "2026-03-30"
SIM_START = "2026-04-01 00:00:00"
SIM_END = "2026-07-20 00:00:00"
DMF = 0.081


def prepare_climate(path: str, start: str, end: str) -> pd.DataFrame:
    """提取郫都室内气候并填充缺口：canopy 用 air 代理，CO2 用 e3036 列 + 400 兜底。"""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()

    # canopy_temperature 缺口（06-02~06-09）用 air + 中位温差 代理
    delta = (df["canopy_temperature"] - df["air_temperature"]).median()
    df["canopy_temperature"] = df["canopy_temperature"].fillna(
        df["air_temperature"] + delta
    )

    # CO2：优先 e3036（缺失 0.4%），其余用 400 ppm（郫都无 CO2 施肥）
    co2 = df["co2_concentration__e3036__s1529"].copy()
    df["co2_used"] = co2.fillna(400.0)

    # RH + air 时间插值（逐时等间隔，线性插值即可）
    for col in ("relative_humidity", "air_temperature"):
        df[col] = df[col].interpolate(limit_direction="both").bfill().ffill()

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "air_temperature": df["air_temperature"],
        "canopy_temperature": df["canopy_temperature"],
        "relative_humidity": df["relative_humidity"],
        "co2_concentration": df["co2_used"],
    })
    return out.reset_index(drop=True)


def main():
    observations = pd.read_csv(OBS)
    env_kwargs = _load_environment_kwargs(ENV_CONFIG)

    # 1) 郫都气候（自己填充缺口）+ 控制
    climate = prepare_climate(CLIMATE, SIM_START, SIM_END)
    controls = prepare_observed_control_targets(
        pd.read_csv(CONTROLS), start=SIM_START, end=SIM_END, dt_seconds=3600
    )
    print(f"气候行数={len(climate)}, 控制行数={len(controls)}")

    # 2) 郫都作物初始状态
    initial_state, init_audit = build_target_crop_initial_state(
        observations,
        target_greenhouse_id=TARGET_ID,
        target_greenhouse_code=TARGET_CODE,
        baseline_date=BASELINE_DATE,
    )
    dmf_estimate = init_audit.get("dry_matter_fraction_transfer", {}).get("ripe_fruit", {}).get("estimate", DMF)
    print(f"初始作物状态: {initial_state}")
    print(f"干物质比例: {dmf_estimate}")

    # 3) weather = ERA5 2026（110 天覆盖 04-01~07-20）
    weather = env_kwargs["weather_repository"].load(
        location="Chengdu", growth_year=2026, start_day=0,
        season_length=110, pred_horizon=0, dt=3600, nd=10,
    )

    # 4) 驱动作物模型（默认参数 = 新都验证）
    trajectory = simulate_forced_crop_trajectory(
        climate=climate, controls=controls, weather=weather,
        initial_crop_state=initial_state, fruit_dry_matter_fraction=float(dmf_estimate),
        substep_seconds=300, include_flux_diagnostics=False,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n郫都迁移产量轨迹已保存: {OUT} ({len(trajectory)} 行)")

    # 5) 关键产出
    last = trajectory.iloc[-1]
    peak = trajectory["standing_fresh_kg_m2"].max()
    print(f"\n=== 郫都迁移产量估计 ===")
    print(f"果实鲜重(season 末): {last['standing_fresh_kg_m2']:.3f} kg/m2")
    print(f"果实鲜重峰值: {peak:.3f} kg/m2")
    print(f"果实干重峰值: {trajectory['standing_dry_kg_m2'].max():.3f} kg/m2")

    # 6) 与郫都 9 条观测对比
    obs = observations.loc[
        observations["greenhouse_id"].astype(str).eq(str(TARGET_ID))
        & observations["greenhouse_code"].astype(str).eq(str(TARGET_CODE))
    ].copy()
    obs["observation_date"] = pd.to_datetime(obs["observation_date"]).dt.tz_localize(None)
    by_date = obs.groupby(obs["observation_date"].dt.normalize())["ripe_fruit_fresh_kg_m2"].mean()
    by_date.index = pd.to_datetime(by_date.index).tz_localize(None)

    traj = trajectory.copy()
    traj["timestamp"] = pd.to_datetime(traj["timestamp"]).dt.tz_localize(None)
    traj_x = (traj["timestamp"] - traj["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    print(f"\n=== 模拟 vs 郫都 9 条观测 ===")
    for d, ov in by_date.items():
        sim = np.interp((d - traj["timestamp"].iloc[0]).total_seconds(), traj_x,
                        traj["standing_fresh_kg_m2"].to_numpy())
        print(f"  {d.date()}: 观测成熟果鲜重={ov:.3f} kg/m2, 模拟总果实鲜重={sim:.3f} kg/m2, 比值={ov/sim:.2f}")


if __name__ == "__main__":
    main()
