"""补全郫都 GH63 缺失数据（迁移估计口径，非实测）。

从已生成的郫都迁移产量轨迹（pidu_transfer_yield_trajectory.csv）派生两类缺失数据：

1. 收获事件（采摘日期+重量）：用 HarvestCohortModel + WUR 成熟度先验（maturity≈596.5 deg-day，
   基温 10°C，5 天采摘一次，DMF 0.082）把轨迹的果实干物质流「老化成批」→ 逐次采摘事件。
2. 完整作物观测（生物量/果实定期测量）：按周采样轨迹，输出叶/茎/果实干物质 + 果实鲜重，
   补齐真实观测里缺的 dry matter 列（lai/plant_dry/leaf_dry/stem_dry/ripe_fruit_dry 全 NaN）。

所有产出统一标记 evidence_class=transfer_simulated、target_eligible=False（迁移估计，非实测），
避免论文误当实测产量验证使用。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from glassgym.models.harvest import HarvestCohortModel, HarvestCohortParameters

TRAJ = Path(
    "results/chengdu_agri_greenhouse_001/harvest_model_gh2024/"
    "pidu_transfer_yield_trajectory.csv"
)
OUT_DIR = Path("results/chengdu_agri_greenhouse_001/harvest_model_gh2024")

# 郫都 GH63 元信息（与 standing_crop_samples.csv 观测一致）
GREENHOUSE_ID = 63
GREENHOUSE_CODE = "GH202602061448266452514"
PLANTING_CODE = "P202603251720122975035"
CULTIVAR = "塞尼瑞"
AREA_M2 = 192.0
PLANT_DENSITY = 2.44791667  # plants/m2（实测）

# WUR 外部先验（data/processed/external/wur_agc2/wur_process_priors.json 的 metrics 估计值）
MATURITY_THERMAL_TIME_DEG_DAY = 596.5205024101232
BASE_TEMPERATURE_C = 10.0
DRY_MATTER_FRACTION = 0.082  # 与 pidu_transfer_yield 迁移所用 DMF 一致
PICK_INTERVAL_DAYS = 5
PICK_HOUR = 10


def load_trajectory() -> pd.DataFrame:
    df = pd.read_csv(TRAJ)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def derive_harvest_events(traj: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """用 HarvestCohortModel 把净果实干物质流推导为逐次采摘事件。"""
    params = HarvestCohortParameters(
        base_temperature_c=BASE_TEMPERATURE_C,
        maturity_thermal_time_deg_day=MATURITY_THERMAL_TIME_DEG_DAY,
        dry_matter_fraction=DRY_MATTER_FRACTION,
        minimum_pick_fresh_kg_m2=0.05,  # 单次采摘低于 0.05 kg/m2 视为无果，跳过
    )
    model = HarvestCohortModel(params, initial_fruit_dry_matter_kg_m2=0.0,
                               initial_fruit_maturity_fraction=0.0)

    first_day = traj["timestamp"].iloc[0].normalize()
    events: list[dict] = []
    for row in traj.itertuples(index=False):
        ts = row.timestamp
        net_change = float(row.net_fruit_dry_matter_change_kg_m2)
        air_temp = float(row.air_temperature_c)
        elapsed_days = (ts.normalize() - first_day).days
        pick = elapsed_days % PICK_INTERVAL_DAYS == 0 and ts.hour == PICK_HOUR
        result = model.step(
            net_fruit_dry_matter_change_kg_m2=net_change,
            air_temperature_c=air_temp,
            dt_hours=1.0,
            pick=pick,
        )
        if result.harvested_fresh_kg_m2 > 0.0:
            events.append(
                {
                    "harvest_datetime": ts,
                    "harvested_fresh_kg_m2": result.harvested_fresh_kg_m2,
                    "harvested_dry_kg_m2": result.harvested_dry_matter_kg_m2,
                    "cumulative_fresh_kg_m2": result.cumulative_harvested_fresh_kg_m2,
                    "standing_dry_kg_m2_after": result.standing_dry_matter_kg_m2,
                }
            )

    ev = pd.DataFrame(events)
    ev["harvest_event_id"] = [f"PIDU_TX_{i:03d}" for i in range(len(ev))]
    ev["greenhouse_id"] = GREENHOUSE_ID
    ev["greenhouse_code"] = GREENHOUSE_CODE
    ev["planting_code"] = PLANTING_CODE
    ev["cultivar"] = CULTIVAR
    ev["harvested_area_m2"] = AREA_M2
    ev["harvested_fresh_kg"] = ev["harvested_fresh_kg_m2"] * AREA_M2
    ev["evidence_class"] = "transfer_simulated"
    ev["target_eligible"] = False
    ev["method"] = "harvest_cohort_model + wur_maturity_prior"

    # 质量守恒：累积净果实生长 = 已采 + 存留（模型口径，非轨迹原值）
    seasonal_yield = float(ev["harvested_fresh_kg_m2"].sum())
    final_standing_fresh = (
        model.standing_dry_matter_kg_m2 / DRY_MATTER_FRACTION
    )
    total_fruit_fresh = seasonal_yield + final_standing_fresh
    # 轨迹 standing crop（含已成熟未采 + 未成熟），应与 total 同量级
    trajectory_standing = float(traj["standing_fresh_kg_m2"].iloc[-1])
    summary = {
        "experiment": "郫都 GH63 2026 春茬 · 迁移模拟收获事件（非实测）",
        "evidence_class": "transfer_simulated",
        "target_eligible": False,
        "method": "HarvestCohortModel 老化成批 + WUR 成熟度先验 + 5 天采摘",
        "parameters": {
            "maturity_thermal_time_deg_day": MATURITY_THERMAL_TIME_DEG_DAY,
            "base_temperature_c": BASE_TEMPERATURE_C,
            "dry_matter_fraction": DRY_MATTER_FRACTION,
            "pick_interval_days": PICK_INTERVAL_DAYS,
            "greenhouse_area_m2": AREA_M2,
            "plant_density_plants_m2": PLANT_DENSITY,
        },
        "harvest_event_count": int(len(ev)),
        "first_harvest_datetime": (
            str(ev["harvest_datetime"].iloc[0]) if len(ev) else None
        ),
        "last_harvest_datetime": (
            str(ev["harvest_datetime"].iloc[-1]) if len(ev) else None
        ),
        "seasonal_yield_kg_m2": seasonal_yield,
        "seasonal_yield_kg_total": seasonal_yield * AREA_M2,
        "final_standing_fresh_kg_m2": final_standing_fresh,
        "total_fruit_fresh_kg_m2": total_fruit_fresh,
        "trajectory_standing_crop_kg_m2": trajectory_standing,
        "mass_balance_error_kg_m2": total_fruit_fresh - trajectory_standing,
        "peak_batch_fresh_kg_m2": (
            float(ev["harvested_fresh_kg_m2"].max()) if len(ev) else 0.0
        ),
        "mean_batch_fresh_kg_m2": (
            float(ev["harvested_fresh_kg_m2"].mean()) if len(ev) else 0.0
        ),
        "limitations": [
            "transfer_simulated_not_measured",
            "crop_parameters_transferred_from_xindu_not_recalibrated_on_pidu",
            "maturity_and_pick_prior_from_wur_high_tech_glasshouse",
            "first_harvest_later_than_observed_ripe_fruit_because_wur_maturity_prior_not_chengdu_calibrated",
            "season_end_is_sensor_stream_end_2026_07_20",
        ],
    }
    return ev, summary


def derive_crop_observations(traj: pd.DataFrame) -> pd.DataFrame:
    """按周采样轨迹，补齐真实观测缺失的生物量（干物质）与果实列。"""
    weekly = traj.resample("7D", on="timestamp").last().dropna(
        subset=["standing_fresh_kg_m2"]
    )
    rows = []
    for ts, r in weekly.iterrows():
        fruit_fresh_kg_m2 = float(r["standing_fresh_kg_m2"])
        fruit_dry_kg_m2 = float(r["standing_dry_kg_m2"])
        leaf_dry_kg_m2 = float(r["c_leaf_mg_m2"]) * 1e-6
        stem_dry_kg_m2 = float(r["c_stem_mg_m2"]) * 1e-6
        rows.append(
            {
                "observation_date": ts,
                "greenhouse_id": GREENHOUSE_ID,
                "greenhouse_code": GREENHOUSE_CODE,
                "planting_code": PLANTING_CODE,
                "cultivar": CULTIVAR,
                "plant_density_plants_m2": PLANT_DENSITY,
                "fruit_fresh_kg_m2": fruit_fresh_kg_m2,
                "fruit_dry_kg_m2": fruit_dry_kg_m2,
                "leaf_dry_kg_m2": leaf_dry_kg_m2,
                "stem_dry_kg_m2": stem_dry_kg_m2,
                "fruit_fresh_g_per_plant": fruit_fresh_kg_m2 * 1000 / PLANT_DENSITY,
                "fruit_dry_g_per_plant": fruit_dry_kg_m2 * 1000 / PLANT_DENSITY,
                "leaf_dry_g_per_plant": leaf_dry_kg_m2 * 1000 / PLANT_DENSITY,
                "stem_dry_g_per_plant": stem_dry_kg_m2 * 1000 / PLANT_DENSITY,
                "evidence_class": "transfer_simulated",
                "target_eligible": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    traj = load_trajectory()
    events, summary = derive_harvest_events(traj)
    observations = derive_crop_observations(traj)

    events_path = OUT_DIR / "pidu_transfer_harvest_events.csv"
    summary_path = OUT_DIR / "pidu_transfer_harvest_summary.json"
    obs_path = OUT_DIR / "pidu_transfer_crop_observations.csv"

    events.to_csv(events_path, index=False, encoding="utf-8-sig")
    observations.to_csv(obs_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"收获事件: {len(events)} 条 -> {events_path}")
    print(f"作物观测: {len(observations)} 条(周) -> {obs_path}")
    print(f"季节采收量(迁移模拟): {summary['seasonal_yield_kg_m2']:.3f} kg/m2 "
          f"({summary['seasonal_yield_kg_total']:.1f} kg / {AREA_M2:.0f} m2)")
    print(f"最终存留(未成熟): {summary['final_standing_fresh_kg_m2']:.3f} kg/m2")
    print(f"总果实鲜重(采+存): {summary['total_fruit_fresh_kg_m2']:.3f} kg/m2")
    print(f"轨迹 standing crop: {summary['trajectory_standing_crop_kg_m2']:.3f} kg/m2")
    print(f"质量守恒误差: {summary['mass_balance_error_kg_m2']:.4f} kg/m2")
    print(f"首采: {summary['first_harvest_datetime']}, "
          f"末采: {summary['last_harvest_datetime']}")


if __name__ == "__main__":
    main()
