"""GlassGreenhouse 参数标定（坐标下降）。

优化参数子集使单步预测误差最小（温度 MAE + 0.5*湿度 MAE）。
每轮对每个参数在当前值附近 ±幅度 试探 3-5 个候选，取最优。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from experiments.reports.evaluate_chengdu_physics import evaluate_one_step_predictions
from glassgym.configs.default_params import init_default_params

VAL_CSV = "data/processed/chengdu_agri/greenhouse_001/trajectories/v5_cultivation_window/val.csv"
OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/glass_calibrated_params.json")

# 待优化参数: (索引, 名称, 初始值, 下界, 上界, 是否乘性)
PARAMS = [
    (208, "heat_capacity_scale", 4.0, 1.0, 8.0),
    (209, "solar_gain_scale", 0.6, 0.2, 1.2),
    (210, "ua_scale", 2.5, 0.5, 5.0),
    (211, "vent_scale", 6.0, 1.0, 10.0),
    (217, "fan_max_ach", 8.0, 2.0, 16.0),
    (218, "pad_cooling_power", 20000.0, 0.0, 60000.0),
    (225, "side_scr_saving", 0.35, 0.0, 0.6),
    (227, "glass_transmissivity", 0.85, 0.6, 0.95),
]
N_ROUNDS = 6
N_SAMPLE = 250


def base_params() -> np.ndarray:
    p = np.asarray(init_default_params(228), dtype=float)
    cal = json.load(open("results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json"))
    for k, v in cal["multipliers"].items():
        p[int(k)] = float(v)
    p[219] = 0.0005  # pad vapor source (fixed)
    p[226] = 0.2     # curtain fan fraction (fixed)
    for idx, _n, init_v, _lo, _hi in PARAMS:
        p[idx] = init_v
    return p


def score_params(p: np.ndarray, val: pd.DataFrame) -> float:
    m, _ = evaluate_one_step_predictions(
        val.head(N_SAMPLE), parameter_vector=p, model_backend="GlassGreenhouse"
    )
    return float(m["air_temperature_mae"] + 0.5 * m["relative_humidity_mae"])


def main():
    val = pd.read_csv(VAL_CSV)
    p = base_params()
    best_score = score_params(p, val)
    print(f"初始 score (T MAE + 0.5 RH MAE): {best_score:.4f}")

    for rnd in range(1, N_ROUNDS + 1):
        improved = False
        for idx, name, _init, lo, hi in PARAMS:
            current = float(p[idx])
            # 试探幅度：逐轮缩小
            span = 0.5 * (0.7 ** (rnd - 1)) * max(current, 1.0)
            candidates = {
                "low": max(lo, current - span),
                "mid": current,
                "high": min(hi, current + span),
            }
            best_val, best_c = current, best_score
            for tag, cand in candidates.items():
                trial = p.copy()
                trial[idx] = cand
                s = score_params(trial, val)
                if s < best_score - 1e-6:
                    best_score, best_val = s, cand
            if best_val != current:
                p[idx] = best_val
                improved = True
                print(f"  轮{rnd} [{idx}]{name}: {current:.4f} -> {best_val:.4f} (score {best_score:.4f})", flush=True)
        if not improved:
            print("无改进，提前收敛", flush=True)
            break

    # 保存
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_backend": "GlassGreenhouse",
        "num_params": 228,
        "score": best_score,
        "multipliers": {str(idx): float(p[idx]) for idx, _n, _i, _l, _h in PARAMS},
        "fixed": {"pad_vapor_source": 0.0005, "curtain_fan_fraction": 0.2},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n标定完成: score {best_score:.4f} -> {OUT}")
    # 最终详细误差
    m, _ = evaluate_one_step_predictions(val.head(N_SAMPLE), parameter_vector=p, model_backend="GlassGreenhouse")
    print(f"最终: 温度 MAE {m['air_temperature_mae']:.3f}°C, 湿度 MAE {m['relative_humidity_mae']:.3f}%")


if __name__ == "__main__":
    main()
