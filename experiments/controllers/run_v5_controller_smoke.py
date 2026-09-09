from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from experiments.controllers.v5_hybrid_environment import build_v5_hybrid_environment
from experiments.controllers.v5_smoke_controllers import (
    V5FixedBaseline,
    V5HybridMPC,
    V5PIDController,
)
from glassgym.components.weather import RandomScenarioWeatherSampler


RL_CLASSES = {"ppo": PPO, "sac": SAC}
SMOKE_ALGORITHMS = ("baseline", "pid", "mpc", "ppo", "sac")
SMOKE_ROLES = ("validation", "test")


def _checkpoint_hyperparameters_compatible(
    saved: dict | None,
    requested: dict | None,
) -> bool:
    def canonicalize(values: dict | None) -> dict | None:
        if values is None:
            return None
        normalized = deepcopy(values)
        policy = normalized.get("policy_kwargs")
        if isinstance(policy, dict) and policy.get("use_sde") is False:
            policy.pop("use_sde")
        return normalized

    return canonicalize(saved) == canonicalize(requested)


class TrainingProgressCallback(BaseCallback):
    def __init__(
        self,
        *,
        output_dir: str | Path,
        interval_timesteps: int,
        unit_id: str,
        target_checkpoint: int,
    ) -> None:
        super().__init__(verbose=0)
        self.output_dir = Path(output_dir)
        self.interval_timesteps = int(interval_timesteps)
        self.unit_id = str(unit_id)
        self.target_checkpoint = int(target_checkpoint)
        self.last_written = 0
        if self.interval_timesteps <= 0:
            raise ValueError("progress interval must be positive")

    def _on_step(self) -> bool:
        completed = int(self.model.num_timesteps)
        if completed - self.last_written < self.interval_timesteps:
            return True
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "chengdu-v5-training-progress-v1",
            "unit_id": self.unit_id,
            "completed_timesteps": completed,
            "target_checkpoint": self.target_checkpoint,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.output_dir / "training_progress.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        self.last_written = completed
        return True


class PIDResidualActionWrapper(gym.ActionWrapper):
    def __init__(self, env, *, residual_scale: float = 0.25) -> None:
        super().__init__(env)
        self.residual_scale = float(residual_scale)
        if not np.isfinite(self.residual_scale) or self.residual_scale < 0.0:
            raise ValueError("Residual PID scale must be finite and nonnegative")
        self.pid_controller = V5PIDController()

    def reset(self, **kwargs):
        self.pid_controller.reset()
        return self.env.reset(**kwargs)

    def action(self, action):
        residual = np.asarray(action, dtype=np.float32)
        if residual.shape != self.action_space.shape or not np.isfinite(residual).all():
            raise ValueError("Residual action must match the common finite action space")
        base = self.pid_controller.predict_action(self.env.unwrapped)
        return np.clip(
            base + self.residual_scale * residual,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)


def _build_rl_model(
    *,
    algorithm: str,
    env,
    seed: int,
    total_timesteps: int,
    hyperparameters: dict | None = None,
):
    if algorithm == "ppo":
        defaults = {
            "learning_rate": 3e-4,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 2,
            "gamma": 0.99,
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        }
    else:
        defaults = {
            "learning_rate": 7e-4,
            "buffer_size": max(512, int(total_timesteps)),
            "learning_starts": min(128, max(16, int(total_timesteps) // 4)),
            "batch_size": 32,
            "tau": 0.01,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "qf": [64, 64]}},
        }
    parameters = deepcopy(defaults if hyperparameters is None else hyperparameters)
    return RL_CLASSES[algorithm](
        "MlpPolicy",
        env,
        **parameters,
        seed=int(seed),
        device="cpu",
        verbose=0,
    )


def render_markdown_table(frame: pd.DataFrame, *, float_digits: int = 4) -> str:
    def render(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{int(float_digits)}f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def _distance_outside(value: float, low: float, high: float) -> float:
    return max(low - value, 0.0) + max(value - high, 0.0)


def classify_safety_intervention(
    proposed: np.ndarray,
    executed: np.ndarray,
    *,
    active_indices: tuple[int, ...] = (3, 6),
    material_tolerance: float = 1e-6,
    reasons: tuple[str, ...] = (),
) -> dict:
    proposed_values = np.asarray(proposed, dtype=float)
    executed_values = np.asarray(executed, dtype=float)
    if proposed_values.shape != executed_values.shape or proposed_values.ndim != 1:
        raise ValueError("Safety proposal and execution must be matching vectors")
    if not np.isfinite(proposed_values).all() or not np.isfinite(executed_values).all():
        raise ValueError("Safety proposal and execution must be finite")
    if material_tolerance <= 0.0 or not np.isfinite(material_tolerance):
        raise ValueError("Material safety tolerance must be finite and positive")
    active = np.asarray(active_indices, dtype=int)
    if np.any(active < 0) or np.any(active >= len(proposed_values)):
        raise ValueError("Active safety indices are outside the control vector")
    difference = np.abs(proposed_values - executed_values)
    active_difference = difference[active]
    return {
        "strict_safety_intervened": bool(
            not np.allclose(proposed_values, executed_values, rtol=0.0, atol=1e-12)
        ),
        "material_safety_intervened": bool(np.any(difference > material_tolerance)),
        "active_safety_intervened": bool(
            np.any(active_difference > material_tolerance)
        ),
        "material_projection_max": float(difference.max(initial=0.0)),
        "active_projection_max": float(active_difference.max(initial=0.0)),
        "safety_intervention_reasons": ";".join(str(reason) for reason in reasons),
    }


def summarize_smoke_trajectory(
    algorithm: str,
    trajectory: pd.DataFrame,
) -> dict:
    numeric = trajectory.select_dtypes(include=[np.number])
    carbon_columns = ["c_buffer_mg_m2", "c_leaf_mg_m2", "c_stem_mg_m2", "c_fruit_mg_m2"]
    complete_carbon = all(column in trajectory for column in carbon_columns)
    crop_min = float(trajectory[carbon_columns].min().min()) if complete_carbon else float("nan")
    executed = trajectory[
        ["executed_uRoofVent", "executed_uBlScr", "executed_uFan", "executed_uPadPump"]
    ].to_numpy(dtype=float)
    changes = np.abs(np.diff(executed, axis=0)) if len(executed) > 1 else np.empty((0, 2))
    material = trajectory.get("material_safety_intervened", trajectory["safety_intervened"])
    active = trajectory.get("active_safety_intervened", material)
    material_projection = trajectory.get(
        "material_projection_max", pd.Series(np.zeros(len(trajectory)))
    )
    active_projection = trajectory.get(
        "active_projection_max", pd.Series(np.zeros(len(trajectory)))
    )
    return {
        "algorithm": str(algorithm),
        "steps": int(len(trajectory)),
        "completed_episode": bool(
            len(trajectory)
            and trajectory["terminated"].iloc[-1]
            and not trajectory["truncated"].any()
        ),
        "numerical_failure": bool(trajectory["numerical_failure"].any()),
        "all_values_finite": bool(not numeric.empty and np.isfinite(numeric.to_numpy()).all()),
        "cumulative_reward": float(trajectory["reward"].sum()),
        "temperature_band_mae_c": float(trajectory["temperature_band_error"].mean()),
        "relative_humidity_band_mae_percent": float(
            trajectory["relative_humidity_band_error"].mean()
        ),
        "joint_comfort_fraction": float(trajectory["joint_comfort"].mean()),
        "safety_intervention_fraction": float(trajectory["safety_intervened"].mean()),
        "material_safety_intervention_fraction": float(material.mean()),
        "active_safety_intervention_fraction": float(active.mean()),
        "mean_material_projection_max": float(material_projection.mean()),
        "mean_active_projection_max": float(active_projection.mean()),
        "residual_fallback_count": int(trajectory["residual_fallback"].sum()),
        "mean_executed_actuator_effort": float(np.mean(np.abs(executed))),
        "total_executed_action_variation": float(changes.sum()) if len(changes) else 0.0,
        "mean_controller_inference_ms": float(
            trajectory["controller_inference_seconds"].mean() * 1000.0
        ),
        "crop_carbon_min_mg_m2": crop_min,
        "crop_carbon_nonnegative_tolerance_mg_m2": 1e-6,
        "crop_carbon_nonnegative": bool(complete_carbon and crop_min >= -1e-6),
        "fruit_state_change_mg_m2": float(
            trajectory["c_fruit_mg_m2"].iloc[-1]
            - trajectory["c_fruit_mg_m2"].iloc[0]
        ) if complete_carbon and len(trajectory) else float("nan"),
    }


def _record_step(
    *,
    env,
    timestep: int,
    hour: float,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    inference_seconds: float,
) -> dict:
    indoor = np.asarray(info["indoor_climate"], dtype=float)
    temperature = float(indoor[1])
    rh = float(indoor[2])
    is_day = 6.0 <= hour < 20.0
    temp_low, temp_high = (20.0, 28.0) if is_day else (16.0, 24.0)
    proposed = np.asarray(info.get("proposed_controls", info["controls"]), dtype=float)
    executed = np.asarray(info["controls"], dtype=float)
    safety = classify_safety_intervention(
        proposed,
        executed,
        reasons=tuple(info.get("safety_interventions", ())),
    )
    return {
        "timestep": int(timestep),
        "hour_of_day": float(hour),
        "reward": float(reward),
        "air_temperature": temperature,
        "relative_humidity": rh,
        "temperature_band_error": _distance_outside(temperature, temp_low, temp_high),
        "relative_humidity_band_error": _distance_outside(rh, 60.0, 85.0),
        "joint_comfort": bool(temp_low <= temperature <= temp_high and 60.0 <= rh <= 85.0),
        "proposed_uRoofVent": float(proposed[3]),
        "proposed_uBlScr": float(proposed[5]),
        "proposed_uFan": float(proposed[6]),
        "proposed_uPadPump": float(proposed[7]),
        "executed_uRoofVent": float(executed[3]),
        "executed_uBlScr": float(executed[5]),
        "executed_uFan": float(executed[6]),
        "executed_uPadPump": float(executed[7]),
        "safety_intervened": bool(info.get("safety_intervened", False)),
        **safety,
        "residual_fallback": bool(info.get("residual_fallback", False)),
        "controller_inference_seconds": float(inference_seconds),
        "c_buffer_mg_m2": float(env.x[22]),
        "c_leaf_mg_m2": float(env.x[23]),
        "c_stem_mg_m2": float(env.x[24]),
        "c_fruit_mg_m2": float(env.x[25]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "numerical_failure": bool(info.get("failure", False)),
    }


def evaluate_classical_policy(
    *,
    algorithm: str,
    policy,
    growth_year: int,
    start_day: int,
    episode_days: int,
    max_steps: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    env = build_v5_hybrid_environment(
        growth_year=growth_year,
        start_day=start_day,
        episode_days=episode_days,
        dt_seconds=900,
    )
    env.reset(seed=0)
    if hasattr(policy, "reset"):
        policy.reset()
    limit = env.N if max_steps is None else min(int(max_steps), env.N)
    rows = []
    for timestep in range(limit):
        hour = float(env.hour_of_day)
        started = perf_counter()
        action = np.asarray(policy.predict_action(env), dtype=np.float32)
        inference_seconds = perf_counter() - started
        if action.shape != env.action_space.shape or not np.isfinite(action).all():
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        if info.get("failure", False):
            break
        rows.append(
            _record_step(
                env=env,
                timestep=timestep,
                hour=hour,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                inference_seconds=inference_seconds,
            )
        )
        if terminated or truncated:
            break
    trajectory = pd.DataFrame(rows)
    return summarize_smoke_trajectory(algorithm, trajectory), trajectory


def _flat_environment_factory(
    *,
    growth_year: int,
    start_day: int,
    episode_days: int,
    training_scenarios: list[tuple[int, int]] | None = None,
    residual_pid_scale: float | None = None,
    reward_fn: str | None = None,
    reward_kwargs: dict | None = None,
):
    def factory():
        env = build_v5_hybrid_environment(
            growth_year=growth_year,
            start_day=start_day,
            episode_days=episode_days,
            dt_seconds=900,
            reward_fn=reward_fn,
            reward_kwargs=reward_kwargs,
        )
        if training_scenarios:
            env.weather_scenario_sampler = RandomScenarioWeatherSampler(
                scenarios=[
                    {
                        "location": "Chengdu",
                        "growth_year": int(year),
                        "start_day": int(day),
                    }
                    for year, day in training_scenarios
                ]
            )
        controlled = (
            PIDResidualActionWrapper(env, residual_scale=residual_pid_scale)
            if residual_pid_scale is not None
            else env
        )
        return gym.wrappers.FlattenObservation(controlled)

    return factory


def train_rl_smoke(
    *,
    algorithm: str,
    output_dir: str | Path,
    total_timesteps: int,
    seed: int,
    growth_year: int,
    start_day: int,
    episode_days: int,
    training_scenarios: list[tuple[int, int]] | None = None,
    reward_fn: str | None = None,
    reward_kwargs: dict | None = None,
) -> dict:
    if algorithm not in RL_CLASSES:
        raise ValueError(f"Unsupported V5 smoke RL algorithm: {algorithm}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = list(training_scenarios or [(int(growth_year), int(start_day))])
    vector = DummyVecEnv(
        [
            _flat_environment_factory(
                growth_year=growth_year,
                start_day=start_day,
                episode_days=episode_days,
                training_scenarios=scenarios,
                reward_fn=reward_fn,
                reward_kwargs=reward_kwargs,
            )
        ]
    )
    normalized = VecNormalize(
        vector,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )
    model = _build_rl_model(
        algorithm=algorithm,
        env=normalized,
        seed=seed,
        total_timesteps=total_timesteps,
    )
    started = perf_counter()
    model.learn(total_timesteps=int(total_timesteps))
    elapsed = perf_counter() - started
    model.save(output / "model")
    normalized.save(output / "vecnormalize.pkl")
    raw_env = normalized.venv.envs[0].unwrapped
    metadata = {
        "schema_version": "chengdu-v5-rl-smoke-training-v1",
        "algorithm": algorithm,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "growth_year": int(growth_year),
        "start_day": int(start_day),
        "episode_days": int(episode_days),
        "training_scenarios": [
            {"growth_year": int(year), "start_day": int(day)}
            for year, day in scenarios
        ],
        "elapsed_seconds": float(elapsed),
        "environment_manifest": raw_env.v5_hybrid_manifest,
    }
    (output / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    normalized.close()
    return metadata


def train_rl_checkpoints(
    *,
    algorithm: str,
    output_dir: str | Path,
    checkpoints: list[int] | tuple[int, ...],
    seed: int,
    growth_year: int,
    start_day: int,
    episode_days: int,
    training_scenarios: list[tuple[int, int]] | None = None,
    residual_pid_scale: float | None = None,
    hyperparameters: dict | None = None,
    profile_label: str = "smoke",
    progress_interval_timesteps: int = 1024,
    resume: bool = True,
    reward_fn: str | None = None,
    reward_kwargs: dict | None = None,
) -> list[dict]:
    if algorithm not in RL_CLASSES:
        raise ValueError(f"Unsupported V5 RL algorithm: {algorithm}")
    targets = sorted({int(value) for value in checkpoints})
    if not targets or targets[0] <= 0:
        raise ValueError("RL checkpoints must contain positive cumulative steps")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios = list(training_scenarios or [(int(growth_year), int(start_day))])
    scenario_records = [
        {"growth_year": int(year), "start_day": int(day)} for year, day in scenarios
    ]

    def load_matching(target: int) -> dict | None:
        checkpoint_dir = output / f"checkpoint_{target:08d}"
        metadata_path = checkpoint_dir / "training_metadata.json"
        required = [checkpoint_dir / "model.zip", checkpoint_dir / "vecnormalize.pkl"]
        if algorithm == "sac":
            required.append(checkpoint_dir / "replay_buffer.pkl")
        if not resume or not metadata_path.exists() or not all(path.exists() for path in required):
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatible = bool(
            metadata.get("schema_version") == "chengdu-v5-rl-checkpoint-v1"
            and metadata.get("algorithm") == algorithm
            and int(metadata.get("seed", -1)) == int(seed)
            and int(metadata.get("total_timesteps", -1)) == target
            and int(metadata.get("episode_days", -1)) == int(episode_days)
            and metadata.get("training_scenarios") == scenario_records
            and metadata.get("residual_pid_scale") == residual_pid_scale
            and _checkpoint_hyperparameters_compatible(
                metadata.get("hyperparameters"), hyperparameters
            )
            and metadata.get("profile_label") == str(profile_label)
        )
        return metadata if compatible else None

    existing = {target: load_matching(target) for target in targets}
    if all(existing.values()):
        return [{**existing[target], "reused": True} for target in targets]

    vector = DummyVecEnv(
        [
            _flat_environment_factory(
                growth_year=growth_year,
                start_day=start_day,
                episode_days=episode_days,
                training_scenarios=scenarios,
                residual_pid_scale=residual_pid_scale,
                reward_fn=reward_fn,
                reward_kwargs=reward_kwargs,
            )
        ]
    )
    completed_targets = [target for target in targets if existing[target] is not None]
    start_target = max(completed_targets, default=0)
    if start_target:
        checkpoint_dir = output / f"checkpoint_{start_target:08d}"
        normalized = VecNormalize.load(checkpoint_dir / "vecnormalize.pkl", vector)
        normalized.training = True
        normalized.norm_reward = True
        model = RL_CLASSES[algorithm].load(
            checkpoint_dir / "model.zip", env=normalized, device="cpu"
        )
        if algorithm == "sac":
            model.load_replay_buffer(checkpoint_dir / "replay_buffer.pkl")
    else:
        normalized = VecNormalize(vector, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = _build_rl_model(
            algorithm=algorithm,
            env=normalized,
            seed=seed,
            total_timesteps=max(targets),
            hyperparameters=hyperparameters,
        )

    records: list[dict] = []
    for target in targets:
        if target <= start_target and existing[target] is not None:
            records.append({**existing[target], "reused": True})
            continue
        increment = target - int(model.num_timesteps)
        if increment <= 0:
            raise ValueError("Checkpoint sequence is inconsistent with model timesteps")
        started = perf_counter()
        progress = TrainingProgressCallback(
            output_dir=output,
            interval_timesteps=progress_interval_timesteps,
            unit_id=f"{profile_label}:{algorithm}/seed_{seed}",
            target_checkpoint=target,
        )
        model.learn(
            total_timesteps=increment,
            reset_num_timesteps=False,
            callback=progress,
        )
        elapsed = perf_counter() - started
        checkpoint_dir = output / f"checkpoint_{target:08d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_dir / "model")
        normalized.save(checkpoint_dir / "vecnormalize.pkl")
        if algorithm == "sac":
            model.save_replay_buffer(checkpoint_dir / "replay_buffer.pkl")
        raw_env = normalized.venv.envs[0].unwrapped
        metadata = {
            "schema_version": "chengdu-v5-rl-checkpoint-v1",
            "algorithm": algorithm,
            "seed": int(seed),
            "total_timesteps": int(target),
            "actual_model_timesteps": int(model.num_timesteps),
            "growth_year": int(growth_year),
            "start_day": int(start_day),
            "episode_days": int(episode_days),
            "training_scenarios": scenario_records,
            "residual_pid_scale": residual_pid_scale,
            "hyperparameters": deepcopy(hyperparameters),
            "profile_label": str(profile_label),
            "elapsed_seconds_this_chunk": float(elapsed),
            "environment_manifest": raw_env.v5_hybrid_manifest,
        }
        (checkpoint_dir / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        records.append({**metadata, "reused": False})
    normalized.close()
    return records


def evaluate_saved_rl_smoke(
    *,
    algorithm: str,
    model_dir: str | Path,
    growth_year: int,
    start_day: int,
    episode_days: int,
    max_steps: int | None = None,
    residual_pid_scale: float | None = None,
) -> tuple[dict, pd.DataFrame]:
    if algorithm not in RL_CLASSES:
        raise ValueError(f"Unsupported V5 smoke RL algorithm: {algorithm}")
    model_dir = Path(model_dir)
    statistics_env = DummyVecEnv(
        [
            _flat_environment_factory(
                growth_year=growth_year,
                start_day=start_day,
                episode_days=episode_days,
                residual_pid_scale=residual_pid_scale,
            )
        ]
    )
    normalizer = VecNormalize.load(model_dir / "vecnormalize.pkl", statistics_env)
    normalizer.training = False
    normalizer.norm_reward = False
    model = RL_CLASSES[algorithm].load(model_dir / "model.zip", device="cpu")

    wrapped = _flat_environment_factory(
        growth_year=growth_year,
        start_day=start_day,
        episode_days=episode_days,
        residual_pid_scale=residual_pid_scale,
    )()
    observation, _reset_info = wrapped.reset(seed=0)
    env = wrapped.unwrapped
    limit = env.N if max_steps is None else min(int(max_steps), env.N)
    rows = []
    for timestep in range(limit):
        hour = float(env.hour_of_day)
        normalized_observation = normalizer.normalize_obs(
            np.asarray(observation, dtype=np.float32)[None, :]
        )
        started = perf_counter()
        action, _state = model.predict(normalized_observation, deterministic=True)
        inference_seconds = perf_counter() - started
        action = np.asarray(action, dtype=np.float32).reshape(-1, 4)[0]
        observation, reward, terminated, truncated, info = wrapped.step(action)
        if info.get("failure", False):
            break
        rows.append(
            _record_step(
                env=env,
                timestep=timestep,
                hour=hour,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                inference_seconds=inference_seconds,
            )
        )
        if terminated or truncated:
            break
    wrapped.close()
    normalizer.close()
    trajectory = pd.DataFrame(rows)
    return summarize_smoke_trajectory(algorithm, trajectory), trajectory


def build_smoke_decision(comparison: pd.DataFrame) -> dict:
    expected = {
        (role, algorithm)
        for role in SMOKE_ROLES
        for algorithm in SMOKE_ALGORITHMS
    }
    observed = set(
        zip(
            comparison.get("role", pd.Series(dtype=str)).astype(str),
            comparison.get("algorithm", pd.Series(dtype=str)).astype(str),
        )
    )
    complete = observed == expected and len(comparison) == len(expected)
    gates = bool(
        complete
        and comparison["completed_episode"].astype(bool).all()
        and (~comparison["numerical_failure"].astype(bool)).all()
        and comparison["all_values_finite"].astype(bool).all()
        and comparison["residual_fallback_count"].astype(int).eq(0).all()
        and comparison["crop_carbon_nonnegative"].astype(bool).all()
    )
    validation_reward_improvement: dict[str, float] = {}
    if complete and "cumulative_reward" in comparison.columns:
        validation = comparison.loc[comparison["role"].eq("validation")].set_index(
            "algorithm"
        )
        baseline_reward = float(validation.loc["baseline", "cumulative_reward"])
        validation_reward_improvement = {
            algorithm: float(validation.loc[algorithm, "cumulative_reward"] - baseline_reward)
            for algorithm in SMOKE_ALGORITHMS
            if algorithm != "baseline"
        }
    all_candidates_beat_baseline = bool(
        len(validation_reward_improvement) == len(SMOKE_ALGORITHMS) - 1
        and all(value > 0.0 for value in validation_reward_improvement.values())
    )
    test_reward_improvement: dict[str, float] = {}
    if complete and "cumulative_reward" in comparison.columns:
        test = comparison.loc[comparison["role"].eq("test")].set_index("algorithm")
        test_baseline_reward = float(test.loc["baseline", "cumulative_reward"])
        test_reward_improvement = {
            algorithm: float(test.loc[algorithm, "cumulative_reward"] - test_baseline_reward)
            for algorithm in SMOKE_ALGORITHMS
            if algorithm != "baseline"
        }
    baseline_is_worst_on_test = bool(
        len(test_reward_improvement) == len(SMOKE_ALGORITHMS) - 1
        and all(value > 0.0 for value in test_reward_improvement.values())
    )
    return {
        "schema_version": "chengdu-v5-controller-smoke-decision-v1",
        "required_algorithms": list(SMOKE_ALGORITHMS),
        "required_roles": list(SMOKE_ROLES),
        "complete_algorithm_role_matrix": bool(complete),
        "validation_reward_improvement": validation_reward_improvement,
        "all_candidates_beat_validation_baseline": all_candidates_beat_baseline,
        "test_reward_improvement": test_reward_improvement,
        "baseline_is_worst_on_test": baseline_is_worst_on_test,
        "promote_to_longer_training": bool(gates and all_candidates_beat_baseline),
        "yield_claim_allowed": False,
        "statistical_superiority_claim_allowed": False,
        "interpretation": "single_seed_single_day_smoke_benchmark",
    }


def _training_complete(
    path: Path,
    algorithm: str,
    total_timesteps: int,
    training_scenarios: list[tuple[int, int]],
) -> bool:
    metadata_path = path / "training_metadata.json"
    if not all(
        candidate.exists()
        for candidate in (metadata_path, path / "model.zip", path / "vecnormalize.pkl")
    ):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_scenarios = [
        {"growth_year": int(year), "start_day": int(day)}
        for year, day in training_scenarios
    ]
    return bool(
        metadata.get("algorithm") == algorithm
        and int(metadata.get("total_timesteps", -1)) == int(total_timesteps)
        and metadata.get("training_scenarios") == expected_scenarios
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V5 five-controller smoke benchmark.")
    parser.add_argument("--rl_timesteps", type=int, default=2048)
    parser.add_argument(
        "--output_dir",
        default="results/chengdu_agri_greenhouse_001/controller_benchmark/v5_hybrid_smoke",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    training_root = output / "training"
    trajectory_root = output / "trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)

    training_scenarios = [(2023, 59), (2023, 226), (2024, 60), (2024, 227)]
    for algorithm in ("ppo", "sac"):
        model_dir = training_root / algorithm / "seed_0"
        if not _training_complete(
            model_dir, algorithm, args.rl_timesteps, training_scenarios
        ):
            print(f"training {algorithm}: {args.rl_timesteps} timesteps", flush=True)
            train_rl_smoke(
                algorithm=algorithm,
                output_dir=model_dir,
                total_timesteps=args.rl_timesteps,
                seed=0,
                growth_year=2023,
                start_day=59,
                episode_days=1,
                training_scenarios=training_scenarios,
            )
        else:
            print(f"training {algorithm}: reusing complete artifact", flush=True)

    roles = {
        "validation": {"growth_year": 2025, "start_day": 59},
        "test": {"growth_year": 2025, "start_day": 226},
    }
    policy_factories = {
        "baseline": V5FixedBaseline,
        "pid": V5PIDController,
        "mpc": lambda: V5HybridMPC(horizon_steps=2),
    }
    metrics_rows = []
    for role, weather in roles.items():
        for algorithm, factory in policy_factories.items():
            print(f"evaluating {role}/{algorithm}", flush=True)
            metrics, trajectory = evaluate_classical_policy(
                algorithm=algorithm,
                policy=factory(),
                episode_days=1,
                **weather,
            )
            metrics["role"] = role
            metrics.update(weather)
            metrics_rows.append(metrics)
            trajectory.to_csv(trajectory_root / f"{role}_{algorithm}.csv", index=False)
        for algorithm in ("ppo", "sac"):
            print(f"evaluating {role}/{algorithm}", flush=True)
            metrics, trajectory = evaluate_saved_rl_smoke(
                algorithm=algorithm,
                model_dir=training_root / algorithm / "seed_0",
                episode_days=1,
                **weather,
            )
            metrics["role"] = role
            metrics.update(weather)
            metrics_rows.append(metrics)
            trajectory.to_csv(trajectory_root / f"{role}_{algorithm}.csv", index=False)

    comparison = pd.DataFrame(metrics_rows).sort_values(["role", "algorithm"])
    comparison.to_csv(output / "comparison.csv", index=False)
    decision = build_smoke_decision(comparison)
    decision.update(
        {
            "rl_training_year": 2023,
            "rl_training_start_day": 59,
            "rl_training_scenarios": [
                {"growth_year": year, "start_day": day}
                for year, day in training_scenarios
            ],
            "rl_timesteps": int(args.rl_timesteps),
            "rl_seed": 0,
        }
    )
    (output / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = [
        "role",
        "algorithm",
        "cumulative_reward",
        "temperature_band_mae_c",
        "relative_humidity_band_mae_percent",
        "joint_comfort_fraction",
        "safety_intervention_fraction",
        "mean_controller_inference_ms",
    ]
    report = "\n".join(
        [
            "# V5 controller smoke benchmark",
            "",
            render_markdown_table(comparison[columns], float_digits=4),
            "",
            f"Promote to longer training: `{decision['promote_to_longer_training']}`",
            "",
            "This is a single-seed, single-day smoke benchmark. It validates execution",
            "mechanics only and does not support superiority or yield claims.",
        ]
    )
    (output / "result_analysis.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
