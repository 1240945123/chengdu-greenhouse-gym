from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.controllers.v5_hybrid_environment import build_v5_hybrid_environment


def _distance_outside(value: float, low: float, high: float) -> float:
    return max(low - value, 0.0) + max(value - high, 0.0)


def run_policy_scenario(
    *,
    scenario: str,
    target_controls: np.ndarray,
    growth_year: int,
    start_day: int,
    episode_days: int,
    max_steps: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    target = np.asarray(target_controls, dtype=float)
    if target.shape != (4,) or not np.isfinite(target).all() or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("V5 preflight target controls must be four values inside [0, 1]")
    env = build_v5_hybrid_environment(
        growth_year=growth_year,
        start_day=start_day,
        episode_days=episode_days,
        dt_seconds=900,
    )
    env.reset(seed=0)
    limit = env.N if max_steps is None else min(int(max_steps), env.N)
    rows = []
    terminated = False
    truncated = False
    for timestep in range(limit):
        hour = float(env.hour_of_day)
        current = env.u[env.action_scheme.controlled_idx]
        delta = env.delta_u_max[env.action_scheme.controlled_idx]
        action = np.clip((target - current) / delta, -1.0, 1.0).astype(np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        if info.get("failure", False):
            rows.append(
                {
                    "timestep": timestep,
                    "reward": float(reward),
                    "numerical_failure": True,
                }
            )
            break
        indoor = np.asarray(info["indoor_climate"], dtype=float)
        temperature = float(indoor[1])
        rh = float(indoor[2])
        is_day = 6.0 <= hour < 20.0
        temp_low, temp_high = (20.0, 28.0) if is_day else (16.0, 24.0)
        rows.append(
            {
                "timestep": timestep,
                "hour_of_day": hour,
                "reward": float(reward),
                "air_temperature": temperature,
                "relative_humidity": rh,
                "temperature_band_error": _distance_outside(temperature, temp_low, temp_high),
                "relative_humidity_band_error": _distance_outside(rh, 60.0, 85.0),
                "joint_comfort": bool(temp_low <= temperature <= temp_high and 60.0 <= rh <= 85.0),
                "uRoofVent": float(env.u[3]),
                "uFan": float(env.u[6]),
                "safety_intervened": bool(info.get("safety_intervened", False)),
                "residual_fallback": bool(info.get("residual_fallback", False)),
                "c_buffer_mg_m2": float(env.x[22]),
                "c_leaf_mg_m2": float(env.x[23]),
                "c_stem_mg_m2": float(env.x[24]),
                "c_fruit_mg_m2": float(env.x[25]),
                "numerical_failure": False,
            }
        )
        if terminated or truncated:
            break

    trajectory = pd.DataFrame(rows)
    numeric = trajectory.select_dtypes(include=[np.number])
    carbon_columns = ["c_buffer_mg_m2", "c_leaf_mg_m2", "c_stem_mg_m2", "c_fruit_mg_m2"]
    complete_carbon = all(column in trajectory for column in carbon_columns)
    crop_min = float(trajectory[carbon_columns].min().min()) if complete_carbon else float("nan")
    crop_nonnegative_tolerance = 1e-6
    metrics = {
        "scenario": str(scenario),
        "steps": int(len(trajectory)),
        "completed_episode": bool(terminated and not truncated),
        "numerical_failure": bool(trajectory.get("numerical_failure", pd.Series([True])).any()),
        "all_values_finite": bool(not numeric.empty and np.isfinite(numeric.to_numpy()).all()),
        "total_reward": float(trajectory["reward"].sum()),
        "temperature_band_mae_c": float(trajectory.get("temperature_band_error", pd.Series(dtype=float)).mean()),
        "relative_humidity_band_mae_percent": float(
            trajectory.get("relative_humidity_band_error", pd.Series(dtype=float)).mean()
        ),
        "joint_comfort_fraction": float(trajectory.get("joint_comfort", pd.Series([False])).mean()),
        "safety_intervention_fraction": float(
            trajectory.get("safety_intervened", pd.Series([True])).mean()
        ),
        "residual_fallback_count": int(
            trajectory.get("residual_fallback", pd.Series([True])).sum()
        ),
        "mean_roof_vent": float(trajectory.get("uRoofVent", pd.Series(dtype=float)).mean()),
        "mean_fan": float(trajectory.get("uFan", pd.Series(dtype=float)).mean()),
        "crop_carbon_min_mg_m2": crop_min,
        "crop_carbon_nonnegative_tolerance_mg_m2": crop_nonnegative_tolerance,
        "crop_carbon_nonnegative": bool(
            complete_carbon and crop_min >= -crop_nonnegative_tolerance
        ),
        "fruit_state_change_mg_m2": float(
            trajectory["c_fruit_mg_m2"].iloc[-1] - trajectory["c_fruit_mg_m2"].iloc[0]
        ) if complete_carbon and len(trajectory) else float("nan"),
    }
    return metrics, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-day V5 hybrid environment policy preflight.")
    parser.add_argument("--growth_year", type=int, default=2025)
    parser.add_argument("--start_day", type=int, default=226)
    parser.add_argument("--episode_days", type=int, default=1)
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/v5_hybrid_preflight",
    )
    args = parser.parse_args()
    scenarios = {
        "low": np.array([0.0, 0.0, 0.0, 0.0]),
        "mid": np.array([0.5, 0.5, 0.5, 0.5]),
        "high": np.array([1.0, 1.0, 1.0, 1.0]),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    for name, target in scenarios.items():
        metrics, trajectory = run_policy_scenario(
            scenario=name,
            target_controls=target,
            growth_year=args.growth_year,
            start_day=args.start_day,
            episode_days=args.episode_days,
        )
        trajectory.to_csv(output / f"trajectory_{name}.csv", index=False)
        metrics_rows.append(metrics)
        print(f"preflight {name}: steps={metrics['steps']} reward={metrics['total_reward']:.3f}", flush=True)
    summary = pd.DataFrame(metrics_rows)
    summary.to_csv(output / "summary.csv", index=False)
    promote = bool(
        summary["completed_episode"].all()
        and (~summary["numerical_failure"]).all()
        and summary["all_values_finite"].all()
        and summary["crop_carbon_nonnegative"].all()
        and summary["residual_fallback_count"].eq(0).all()
    )
    decision = {
        "schema_version": "chengdu-v5-hybrid-preflight-v1",
        "growth_year": args.growth_year,
        "start_day": args.start_day,
        "episode_days": args.episode_days,
        "scenarios": list(scenarios),
        "promote_environment_to_controller_smoke": promote,
        "yield_claim_allowed": False,
        "interpretation": "environment_mechanics_preflight_not_controller_ranking",
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
