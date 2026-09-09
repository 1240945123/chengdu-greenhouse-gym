from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.controllers.benchmark_metrics import summarize_episode
from experiments.controllers.evaluation_failures import append_environment_failure_row
from experiments.controllers.execution import ControllerExecutorV2
from experiments.controllers.benchmark_protocol import (
    BenchmarkConfig,
    DEFAULT_CONFIG_PATH,
    build_environment,
    full_control_target_to_action,
)
from glassgym.components.mpc import LightweightMPCController
from glassgym.components.pid import PIDController
from glassgym.components.rule_based import RuleBasedController
from glassgym.core.types import ControllerStepContextV2, StepContext
from glassgym.environments.utils import satVp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_CONFIG_DIR = PROJECT_ROOT / "configs" / "agents"
CONTROLLER_CLASSES = {
    "baseline": RuleBasedController,
    "pid": PIDController,
    "mpc": LightweightMPCController,
}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _agent_params(name: str) -> dict[str, Any]:
    filename = "rule_based.yml" if name == "baseline" else f"{name}.yml"
    raw = yaml.safe_load((AGENT_CONFIG_DIR / filename).read_text(encoding="utf-8"))
    return deepcopy(raw.get("ChengduControllerBenchmark", raw["GreenLightEnv"]))


def _benchmark_search(config_path: str | Path, algorithm: str) -> list[dict[str, Any]]:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(raw["classical_search"][algorithm])


def _step_context(env) -> StepContext:
    weather_history = env.controller_weather_history()
    return StepContext(
        t=env.timestep,
        dt=env.dt,
        Np=env.Np,
        x_prev=env.x_prev,
        x=env.x,
        u=env.u,
        p=env.p,
        d=weather_history,
        hour_of_day=env.hour_of_day,
        day_of_year=env.day_of_year,
        forecast=env.issue_forecast(),
    )


def _mpc_step_context(env) -> ControllerStepContextV2:
    return ControllerStepContextV2(
        t=env.timestep,
        dt=env.dt,
        indoor_temperature=float(env.x[2]),
        indoor_relative_humidity=float(100.0 * env.x[15] / satVp(env.x[2])),
        u=env.u,
        forecast=env.issue_forecast(),
        hour_of_day=env.hour_of_day,
        day_of_year=env.day_of_year,
    )


def build_classical_controller(algorithm: str, params: dict[str, Any], env):
    if algorithm not in CONTROLLER_CLASSES:
        raise ValueError(f"Unsupported classical algorithm: {algorithm}")
    controller = CONTROLLER_CLASSES[algorithm](**params)
    if hasattr(controller, "reset"):
        controller.reset()
    return controller


def resolve_classical_params(
    algorithm: str,
    params: dict[str, Any],
    env,
) -> dict[str, Any]:
    resolved = deepcopy(params)
    if algorithm != "mpc":
        return resolved
    resolved["candidate_levels"] = {
        "uThScr": [0.0, 1.0],
        "uVent": [0.0, 0.5, 1.0],
        "uLamp": [0.0, 1.0],
        "uBlScr": [0.0, 1.0],
    }
    resolved["include_conservative"] = False
    resolved["fallback_control"] = [0.0, 0.0, 0.5, 0.1, 0.0, 0.0]
    resolved["objective_weights"].update({
        "co2_error": 0.0,
        "heat_cost": 0.0,
        "co2_cost": 0.0,
        "fruit_growth": 0.0,
    })
    reward = env.reward_fn
    resolved["max_action_change"] = float(
        env.delta_u_max[env.action_scheme.controlled_idx][0]
    )
    resolved["comfort_objective"] = {
        "temp_day_low": reward.temp_day_low,
        "temp_day_high": reward.temp_day_high,
        "temp_night_low": reward.temp_night_low,
        "temp_night_high": reward.temp_night_high,
        "rh_low": reward.rh_low,
        "rh_high": reward.rh_high,
        "day_start": reward.day_start,
        "day_end": reward.day_end,
        "temperature_weight": reward.temperature_weight,
        "humidity_weight": reward.humidity_weight,
        "lamp_weight": reward.lamp_weight,
        "effort_weight": reward.effort_weight,
        "action_change_weight": reward.action_change_weight,
    }
    return resolved


def evaluate_classical_controller(
    config: BenchmarkConfig,
    *,
    algorithm: str,
    params: dict[str, Any] | None,
    seed: int,
    start_day: int,
    growth_year: int | None = None,
    episode_days: int | None = None,
    max_steps: int | None = None,
    return_resolved_params: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    days = int(config.episode_days if episode_days is None else episode_days)
    env = build_environment(
        config,
        start_day,
        episode_days=days,
        growth_year=growth_year,
    )
    base_params = _agent_params(algorithm) if params is None else params
    controller_params = resolve_classical_params(algorithm, base_params, env)
    controller = build_classical_controller(algorithm, controller_params, env)
    forecast_manifest = getattr(env.forecast_provider, "artifact_manifest", None) or {}
    fallback_controller = RuleBasedController(**_agent_params("baseline"))
    if hasattr(fallback_controller, "reset"):
        fallback_controller.reset()
    executor = ControllerExecutorV2(
        timeout_seconds=config.controller_timeout_seconds,
        expected_shape=(env.nu,),
    )
    rows: list[dict[str, Any]] = []
    env.reset(seed=seed)
    limit = int(max_steps if max_steps is not None else env.N)
    try:
        for timestep in range(limit):
            started = perf_counter()
            fallback_ctx = _step_context(env)
            ctx = _mpc_step_context(env) if algorithm == "mpc" else fallback_ctx
            execution = executor.execute(
                lambda: controller.predict(ctx),
                fallback=lambda: fallback_controller.predict(fallback_ctx),
                final_fallback=np.asarray(env.u, dtype=np.float32),
            )
            target = execution.action
            action = full_control_target_to_action(
                env.u,
                target,
                delta=env.delta_u_max[env.action_scheme.controlled_idx],
            )
            _obs, _reward, terminated, truncated, info = env.step(action)
            wall_time = perf_counter() - started
            if info.get("failure", False):
                append_environment_failure_row(rows, info, timestep=timestep)
                break
            controls = np.asarray(info["controls"], dtype=float)
            proposed_controls = np.asarray(
                info.get("proposed_controls", controls), dtype=float
            )
            indoor = np.asarray(info["indoor_climate"], dtype=float)
            row = {
                "algorithm": algorithm,
                "seed": int(seed),
                "location": str(env.location),
                "growth_year": int(env.growth_year),
                "start_day": int(start_day),
                "timestep": timestep,
                "reward": float(info["reward"]),
                "air_temperature": float(indoor[1]),
                "relative_humidity": float(indoor[2]),
                "wall_time_seconds": float(wall_time),
                "controller_inference_seconds": float(execution.elapsed_seconds),
                "controller_fallback_used": bool(execution.fallback_used),
                "controller_failure_kind": execution.failure_kind,
                "controller_failure_message": execution.failure_message,
                "controller_fallback_failure_kind": execution.fallback_failure_kind,
                "controller_fallback_duration_steps": int(
                    execution.fallback_duration_steps
                ),
                "forecast_source_model_version": ctx.forecast.source_model_version,
                "forecast_error_model_id": ctx.forecast.error_model_id,
                "controller_model_fingerprint": (
                    controller.controller_model.config.fingerprint
                    if hasattr(controller, "controller_model")
                    else "not_applicable"
                ),
                "forecast_artifact_fingerprint": getattr(
                    env.forecast_provider, "artifact_fingerprint", None
                ),
                "forecast_fit_role": forecast_manifest.get("fit_role", "not_applicable"),
                "forecast_select_role": forecast_manifest.get("select_role", "not_applicable"),
                "forecast_model_label": forecast_manifest.get("model_label", "not_applicable"),
                "forecast_metric_variable_indices": json.dumps(
                    forecast_manifest.get("metric_variable_indices", [])
                ),
                "forecast_metric_error_scales": json.dumps(
                    forecast_manifest.get("metric_error_scales", [])
                ),
                "forecast_source_sha256": json.dumps(
                    forecast_manifest.get("source_sha256", {}), sort_keys=True
                ),
                "forecast_leap_day_policy": forecast_manifest.get(
                    "leap_day_policy", "not_applicable"
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "environment_failure": False,
                "environment_failure_kind": "",
                "environment_failure_phase": "",
                "environment_failure_exception": "",
                "environment_failure_message": "",
                "environment_failure_timestep": -1,
                **{name: float(controls[index]) for index, name in enumerate(
                    ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")
                )},
                **{f"proposed_{name}": float(proposed_controls[index]) for index, name in enumerate(
                    ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")
                )},
                "safety_intervened": bool(info.get("safety_intervened", False)),
                "safety_interventions": "|".join(info.get("safety_interventions", ())),
                "safety_fallback_used": bool(info.get("safety_fallback_used", False)),
                "safety_fallback_duration_steps": int(
                    info.get("safety_fallback_duration_steps", 0)
                ),
                **{
                    key: float(value)
                    for key, value in info.items()
                    if key.endswith(("_penalty", "_target", "_low", "_high"))
                },
                "crop_model": str(info["crop_model"]),
                **{
                    key: float(info[key])
                    for key in (
                        "fruit_harvest_rate_mg_m2_s",
                        "fruit_allocation_rate_mg_m2_s",
                        "harvested_dry_matter_mg_m2",
                        "allocated_fruit_dry_matter_mg_m2",
                        "dry_matter_fraction",
                        "c_buffer_mg_m2",
                        "c_leaf_mg_m2",
                        "c_stem_mg_m2",
                        "c_fruit_mg_m2",
                        "c_fruit_previous_mg_m2",
                    )
                },
            }
            if row["uBoil"] != 0.0 or row["uCO2"] != 0.0:
                raise RuntimeError("Disabled heating or CO2 control became nonzero")
            rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()
    trajectory = pd.DataFrame(rows)
    if return_resolved_params:
        return trajectory, controller_params
    return trajectory


def select_best_trial(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("At least one trial is required")
    return min(
        trials,
        key=lambda trial: (
            -float(trial["cumulative_reward"]),
            float(trial["total_action_variation"]),
            str(trial["candidate_id"]),
        ),
    )


def assert_validation_only(trials: list[dict[str, Any]], config: BenchmarkConfig) -> None:
    invalid = [
        trial for trial in trials
        if int(trial["start_day"]) != config.validation_start_day
    ]
    if invalid:
        raise ValueError(
            f"Classical tuning must use validation start day {config.validation_start_day} only"
        )


def tune_classical_controller(
    config: BenchmarkConfig,
    *,
    algorithm: str,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    episode_days: int | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if algorithm not in {"pid", "mpc"}:
        raise ValueError("Only PID and MPC have validation searches")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    base = _agent_params(algorithm)
    trial_records: list[dict[str, Any]] = []
    resolved_params: dict[str, dict[str, Any]] = {}
    for candidate in _benchmark_search(config_path, algorithm):
        candidate_id = str(candidate["candidate_id"])
        params = _deep_merge(base, candidate.get("overrides", {}))
        trajectory, effective_params = evaluate_classical_controller(
            config,
            algorithm=algorithm,
            params=params,
            seed=0,
            start_day=config.validation_start_day,
            episode_days=episode_days,
            max_steps=max_steps,
            return_resolved_params=True,
        )
        metrics = summarize_episode(trajectory, dt_seconds=900)
        trial_records.append({
            "candidate_id": candidate_id,
            "start_day": config.validation_start_day,
            **metrics,
        })
        resolved_params[candidate_id] = effective_params
    assert_validation_only(trial_records, config)
    selected = select_best_trial(trial_records)
    payload = {
        "algorithm": algorithm,
        "selected_candidate_id": selected["candidate_id"],
        "selected_metrics": selected,
        "selected_params": resolved_params[selected["candidate_id"]],
        "forecast_provenance": {
            key: trajectory[key].iloc[0]
            for key in (
                "forecast_source_model_version",
                "forecast_error_model_id",
                "forecast_artifact_fingerprint",
                "forecast_fit_role",
                "forecast_select_role",
                "forecast_model_label",
                "forecast_metric_variable_indices",
                "forecast_metric_error_scales",
                "forecast_source_sha256",
                "forecast_leap_day_policy",
            )
        },
    }
    pd.DataFrame(trial_records).to_csv(output / "validation_trials.csv", index=False)
    (output / "selected_params.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload
