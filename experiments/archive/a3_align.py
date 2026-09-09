"""A3 对齐：模拟作物轨迹 vs GH2024 standing-crop 观测。

读 simulated_crop_trajectory.csv（果实/叶/茎干重逐时），转鲜重，
在 GH2024 的 19 个观测日期插值，与成熟果鲜重观测对比。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

TRAJ = "results/chengdu_agri_greenhouse_001/harvest_model_gh2024/simulated_crop_trajectory.csv"
OBS = "results/chengdu_agri_greenhouse_001/harvest_model_gh2024/processed/standing_crop_samples.csv"
DMF = 0.081  # 果实干物质比例（来自 source_precalibration）


def main():
    traj = pd.read_csv(TRAJ)
    traj["timestamp"] = pd.to_datetime(traj["timestamp"])
    traj["fruit_dry_kg_m2"] = traj["c_fruit_mg_m2"] * 1e-6
    traj["fruit_fresh_kg_m2"] = traj["fruit_dry_kg_m2"] / DMF
    traj["leaf_dry_kg_m2"] = traj["c_leaf_mg_m2"] * 1e-6
    traj["stem_dry_kg_m2"] = traj["c_stem_mg_m2"] * 1e-6

    obs = pd.read_csv(OBS)
    obs["observation_date"] = pd.to_datetime(obs["observation_date"]).dt.tz_localize(None)
    obs = obs[obs["ripe_fruit_fresh_g_per_plant"].notna()].copy()
    obs["ripe_fruit_fresh_kg_m2"] = (
        obs["ripe_fruit_fresh_g_per_plant"] * obs["plant_density_plants_m2"] / 1000.0
    )
    obs["ripe_fruit_dry_kg_m2"] = (
        obs["ripe_fruit_dry_g_per_plant"] * obs["plant_density_plants_m2"] / 1000.0
    )

    # 按日期聚合
    by_date = obs.groupby(obs["observation_date"].dt.normalize()).agg(
        fresh_mean=("ripe_fruit_fresh_kg_m2", "mean"),
        fresh_count=("ripe_fruit_fresh_kg_m2", "count"),
        dry_mean=("ripe_fruit_dry_kg_m2", "mean"),
    ).reset_index()

    # 插值模拟值
    traj_x = (traj["timestamp"] - traj["timestamp"].iloc[0]).dt.total_seconds().to_numpy()
    by_date["sim_fruit_fresh_kg_m2"] = np.interp(
        (by_date["observation_date"] - traj["timestamp"].iloc[0]).dt.total_seconds().to_numpy(),
        traj_x,
        traj["fruit_fresh_kg_m2"].to_numpy(),
    )
    by_date["sim_fruit_dry_kg_m2"] = np.interp(
        (by_date["observation_date"] - traj["timestamp"].iloc[0]).dt.total_seconds().to_numpy(),
        traj_x,
        traj["fruit_dry_kg_m2"].to_numpy(),
    )

    print("=== 模拟总果实鲜重 vs 观测成熟果鲜重 ===")
    print(f"{'日期':<12}{'观测鲜重':>10}{'模拟鲜重':>10}{'比值':>8}")
    for _, r in by_date.iterrows():
        print(f"{str(r['observation_date'].date()):<12}{r['fresh_mean']:>10.3f}{r['sim_fruit_fresh_kg_m2']:>10.3f}{r['fresh_mean']/max(r['sim_fruit_fresh_kg_m2'],1e-9):>8.2f}")

    # 只对比有观测的日期
    err = by_date["sim_fruit_fresh_kg_m2"] - by_date["fresh_mean"]
    denom = by_date["fresh_mean"].sum()
    print(f"\n观测日期数: {len(by_date)}, 观测样本数: {by_date['fresh_count'].sum()}")
    print(f"MAE(鲜重) = {err.abs().mean():.3f} kg/m2")
    print(f"RMSE = {np.sqrt((err**2).mean()):.3f} kg/m2")
    print(f"bias(模拟-观测) = {err.mean():+.3f} kg/m2")
    print(f"WMAPE = {err.abs().sum()/denom:.3f}")

    # 干重对比
    dry_err = by_date["sim_fruit_dry_kg_m2"] - by_date["dry_mean"]
    print(f"\n干重对比: MAE={dry_err.abs().mean():.3f} kg/m2, bias={dry_err.mean():+.3f} kg/m2")

    # 模拟轨迹摘要
    print(f"\n=== 模拟轨迹摘要 ===")
    print(f"果实鲜重峰值: {traj['fruit_fresh_kg_m2'].max():.3f} kg/m2 (观测成熟果峰值 {by_date['fresh_mean'].max():.3f})")
    print(f"果实干重峰值: {traj['fruit_dry_kg_m2'].max():.3f} kg/m2")


if __name__ == "__main__":
    main()
