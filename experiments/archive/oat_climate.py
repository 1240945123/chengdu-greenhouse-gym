"""气候模型 8 参数 OAT 敏感性分析（单步预测，v5_cultivation_window/test.csv）。

对 ChengduPhysics 的 8 个 multiplier 参数在标定值 ±20% 扰动，
用单步预测评估温度/RH MAE 变化，输出敏感性排序。
"""
import sys, json, time
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from experiments.reports.evaluate_chengdu_physics import (
    evaluate_one_step_predictions,
    default_parameter_vector,
)

CAL = {208: 4.0, 209: 0.6, 210: 2.5, 211: 6.0, 212: 0.6, 213: 1.0, 214: 1.0, 215: 1.6}
PARAM_NAMES = {
    208: "heat_capacity_scale",
    209: "solar_gain_scale",
    210: "ua_scale",
    211: "vent_scale",
    212: "co2_vent_scale",
    213: "co2_photo_scale",
    214: "vapor_transpiration_scale",
    215: "vapor_vent_scale",
}


def build_params(mults: dict[int, float]) -> np.ndarray:
    params = default_parameter_vector(216)
    for k, v in mults.items():
        params[k] = float(v)
    return params


def main():
    traj = pd.read_csv(
        "data/processed/chengdu_agri/greenhouse_001/trajectories/v5_cultivation_window/test.csv"
    )
    print(f"验证集行数: {len(traj)}")

    base_params = build_params(CAL)
    base, _ = evaluate_one_step_predictions(traj, parameter_vector=base_params)
    print(f"基线: 温度MAE={base['air_temperature_mae']:.4f}, RH MAE={base['relative_humidity_mae']:.4f}")

    results = []
    for idx in sorted(PARAM_NAMES):
        name = PARAM_NAMES[idx]
        base_val = CAL[idx]
        t0 = time.time()
        row = {"param": name, "index": idx, "base_val": base_val}
        for tag, frac in [("low", -0.20), ("high", 0.20)]:
            mults = dict(CAL)
            mults[idx] = base_val * (1.0 + frac)
            params = build_params(mults)
            m, _ = evaluate_one_step_predictions(traj, parameter_vector=params)
            row[f"temp_mae_{tag}"] = round(m["air_temperature_mae"], 4)
            row[f"rh_mae_{tag}"] = round(m["relative_humidity_mae"], 4)
        results.append(row)
        print(f"  [{idx}] {name}: base={base_val} -> "
              f"T {row['temp_mae_low']} / {row['temp_mae_high']}, "
              f"RH {row['rh_mae_low']} / {row['rh_mae_high']}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(results)
    df["temp_mae_base"] = base["air_temperature_mae"]
    df["rh_mae_base"] = base["relative_humidity_mae"]
    # 敏感性 = MAE 随 ±20% 扰动的变化幅度（相对基线 %）
    df["temp_sensitivity_pct"] = (
        (df["temp_mae_high"] + df["temp_mae_low"]) / 2 - df["temp_mae_base"]
    ) / df["temp_mae_base"] * 100
    df["rh_sensitivity_pct"] = (
        (df["rh_mae_high"] + df["rh_mae_low"]) / 2 - df["rh_mae_base"]
    ) / df["rh_mae_base"] * 100
    df["temp_mae_span"] = (df["temp_mae_high"] - df["temp_mae_low"]).abs()
    df = df.sort_values("temp_mae_span", ascending=False)

    out_dir = "results/chengdu_agri_greenhouse_001/sensitivity_analysis"
    import os
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/oat_climate_params.csv", index=False)
    summary = {
        "base_temperature_mae": base["air_temperature_mae"],
        "base_relative_humidity_mae": base["relative_humidity_mae"],
        "num_samples": int(base["num_samples"]),
        "num_parameters": len(results),
        "perturbation": "+-20%",
        "results": df.to_dict(orient="records"),
    }
    with open(f"{out_dir}/oat_climate_params.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 敏感性排序（按温度 MAE 变化幅度降序）===")
    print(df[["param", "base_val", "temp_mae_low", "temp_mae_base", "temp_mae_high", "temp_mae_span"]].to_string(index=False))


if __name__ == "__main__":
    main()
