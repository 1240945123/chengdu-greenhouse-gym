"""作物模型参数 OAT 敏感性分析（郫都迁移产量，season 末果实鲜重）。

复用 pidu_transfer_yield 的郫都 2026 气候/控制/初始状态，
对 GreenLight 作物关键参数 ±20% 扰动，评估 season 末/峰值果实鲜重变化。
"""
import sys, json, time
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
from glassgym.configs.default_params import init_default_params

OBS = "results/chengdu_agri_greenhouse_001/harvest_model/processed/standing_crop_samples.csv"
CONTROLS = "data/processed/chengdu_agri/greenhouse_001/controls/controls_1h.csv"
CLIMATE = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
ENV_CONFIG = "configs/envs/ChengduSingleGreenhouseEnv.yml"

TARGET_ID = 63
TARGET_CODE = "GH202602061448266452514"
BASELINE_DATE = "2026-03-30"
SIM_START = "2026-04-01 00:00:00"
SIM_END = "2026-07-20 00:00:00"
DMF = 0.081

# 作物关键参数（索引 -> 名称）
CROP_PARAMS = {
    154: "fruit_growth_rate",       # buffer→fruit, 0.328
    155: "leaf_growth_rate",        # buffer→leaf, 0.095
    156: "stem_growth_rate",        # buffer→stem, 0.074
    163: "dev_stage_threshold",     # tCanSum 发育阈值, 1035
    164: "dev_stage_threshold_2",   # 1250
    146: "fruit_growth_respiration",# 0.27
}


def prepare_climate(path: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    delta = (df["canopy_temperature"] - df["air_temperature"]).median()
    df["canopy_temperature"] = df["canopy_temperature"].fillna(df["air_temperature"] + delta)
    co2 = df["co2_concentration__e3036__s1529"].copy()
    df["co2_used"] = co2.fillna(400.0)
    for col in ("relative_humidity", "air_temperature"):
        df[col] = df[col].interpolate(limit_direction="both").bfill().ffill()
    return pd.DataFrame({
        "timestamp": df["timestamp"],
        "air_temperature": df["air_temperature"],
        "canopy_temperature": df["canopy_temperature"],
        "relative_humidity": df["relative_humidity"],
        "co2_concentration": df["co2_used"],
    }).reset_index(drop=True)


def main():
    observations = pd.read_csv(OBS)
    env_kwargs = _load_environment_kwargs(ENV_CONFIG)
    climate = prepare_climate(CLIMATE, SIM_START, SIM_END)
    controls = prepare_observed_control_targets(
        pd.read_csv(CONTROLS), start=SIM_START, end=SIM_END, dt_seconds=3600
    )
    initial_state, _ = build_target_crop_initial_state(
        observations, target_greenhouse_id=TARGET_ID,
        target_greenhouse_code=TARGET_CODE, baseline_date=BASELINE_DATE,
    )
    weather = env_kwargs["weather_repository"].load(
        location="Chengdu", growth_year=2026, start_day=0,
        season_length=110, pred_horizon=0, dt=3600, nd=10,
    )
    print(f"气候={len(climate)} 行, 控制={len(controls)} 行")

    base_params = np.asarray(init_default_params(216), dtype=float)

    def run(params):
        traj = simulate_forced_crop_trajectory(
            climate=climate, controls=controls, weather=weather,
            initial_crop_state=initial_state, fruit_dry_matter_fraction=DMF,
            parameters=params, substep_seconds=300, include_flux_diagnostics=False,
        )
        return float(traj["standing_fresh_kg_m2"].iloc[-1]), float(traj["standing_fresh_kg_m2"].max())

    t0 = time.time()
    base_end, base_peak = run(base_params)
    print(f"基线: season末={base_end:.3f} kg/m2, 峰值={base_peak:.3f} kg/m2  ({time.time()-t0:.1f}s)")

    results = []
    for idx in sorted(CROP_PARAMS):
        name = CROP_PARAMS[idx]
        base_val = float(base_params[idx])
        row = {"param": name, "index": idx, "base_val": base_val}
        for tag, frac in [("low", -0.20), ("high", 0.20)]:
            p = base_params.copy()
            p[idx] = base_val * (1.0 + frac)
            end, peak = run(p)
            row[f"end_yield_{tag}"] = round(end, 4)
            row[f"peak_yield_{tag}"] = round(peak, 4)
        results.append(row)
        print(f"  [{idx}] {name} (base={base_val}): "
              f"end {row['end_yield_low']}/{row['end_yield_high']}, "
              f"peak {row['peak_yield_low']}/{row['peak_yield_high']}  ({time.time()-t0:.1f}s 累计)")

    df = pd.DataFrame(results)
    df["end_yield_base"] = base_end
    df["peak_yield_base"] = base_peak
    df["end_sensitivity_pct"] = ((df["end_yield_high"] - df["end_yield_low"]) / df["end_yield_base"]) * 100
    df["end_span"] = (df["end_yield_high"] - df["end_yield_low"]).abs()
    df = df.sort_values("end_span", ascending=False)

    out_dir = "results/chengdu_agri_greenhouse_001/sensitivity_analysis"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/oat_crop_params.csv", index=False)
    summary = {
        "base_end_yield_kg_m2": base_end,
        "base_peak_yield_kg_m2": base_peak,
        "perturbation": "+-20%",
        "results": df.to_dict(orient="records"),
    }
    with open(f"{out_dir}/oat_crop_params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 作物参数敏感性排序（按 season 末产量变化降序）===")
    print(df[["param", "base_val", "end_yield_low", "end_yield_base", "end_yield_high", "end_span"]].to_string(index=False))


if __name__ == "__main__":
    main()
