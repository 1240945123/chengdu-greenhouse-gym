from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from experiments.controllers.benchmark_protocol import (
    BenchmarkConfig,
    build_environment,
    full_control_target_to_action,
)
from experiments.controllers.evaluation_failures import append_environment_failure_row
from experiments.controllers.execution import (
    ControllerExecutorV2,
    ControllerExecutionResultV2,
    execute_controller_v2,
)
from glassgym.components.rule_based import RuleBasedController
from glassgym.components.weather import RandomScenarioWeatherSampler
from glassgym.core.types import StepContext


ALGORITHM_CLASSES = {"ppo": PPO, "sac": SAC}
DEFAULT_CHECKPOINT_INTERVAL_TIMESTEPS = 131_072


def _raw_step_context(env) -> StepContext:
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


def execute_rl_action_v2(
    *,
    model,
    observations: np.ndarray,
    raw_env,
    fallback_controller: RuleBasedController,
    timeout_seconds: float,
    executor: ControllerExecutorV2 | None = None,
) -> ControllerExecutionResultV2:
    ctx = _raw_step_context(raw_env)
    delta = raw_env.delta_u_max[raw_env.action_scheme.controlled_idx]

    def fallback_action() -> np.ndarray:
        target = fallback_controller.predict(ctx)
        return full_control_target_to_action(raw_env.u, target, delta=delta)[None, :]

    final = full_control_target_to_action(raw_env.u, raw_env.u, delta=delta)[None, :]
    if executor is None:
        return execute_controller_v2(
            lambda: model.predict(observations, deterministic=True)[0],
            fallback=fallback_action,
            final_fallback=final,
            timeout_seconds=timeout_seconds,
            expected_shape=(1, raw_env.action_space.shape[0]),
        )
    return executor.execute(
        lambda: model.predict(observations, deterministic=True)[0],
        fallback=fallback_action,
        final_fallback=final,
    )


class SaveBestVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str | Path):
        super().__init__(verbose=0)
        self.save_path = Path(save_path)

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            raise RuntimeError("Best-model callback requires a VecNormalize environment")
        vec_normalize.save(self.save_path)
        return True


class ProtectedEvalCallback(EvalCallback):
    """SB3 validation callback with the benchmark controller execution contract."""

    def __init__(
        self,
        *args,
        fallback_controller: RuleBasedController,
        timeout_seconds: float,
        absolute_eval_interval_timesteps: int | None = None,
        next_eval_timestep: int | None = None,
        evaluation_timesteps: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.fallback_controller = fallback_controller
        self.timeout_seconds = float(timeout_seconds)
        self.controller_calls = 0
        self.controller_fallback_steps = 0
        self.controller_timeout_steps = 0
        self._executor: ControllerExecutorV2 | None = None
        self.absolute_eval_interval_timesteps = absolute_eval_interval_timesteps
        self.next_eval_timestep = next_eval_timestep
        self.evaluation_timesteps = list(evaluation_timesteps or [])

    def _new_controller_executor(self, raw_env) -> ControllerExecutorV2:
        return ControllerExecutorV2(
            timeout_seconds=self.timeout_seconds,
            expected_shape=(1, raw_env.action_space.shape[0]),
        )

    def _on_step(self) -> bool:
        absolute_schedule = self.absolute_eval_interval_timesteps is not None
        if absolute_schedule:
            if self.next_eval_timestep is None:
                raise RuntimeError("Absolute validation schedule has no next timestep")
            if self.num_timesteps < self.next_eval_timestep:
                return True
        elif self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        raw_env = self.eval_env.venv.envs[0].unwrapped
        self._executor = self._new_controller_executor(raw_env)
        had_instance_predict = "predict" in self.model.__dict__
        original_instance_predict = self.model.__dict__.get("predict")
        original_predict = self.model.predict

        class PredictProxy:
            def predict(proxy_self, observations, deterministic=True):
                return original_predict(observations, deterministic=deterministic)

        def protected_predict(
            observations,
            state=None,
            episode_start=None,
            deterministic=False,
        ):
            result = execute_rl_action_v2(
                model=PredictProxy(),
                observations=observations,
                raw_env=raw_env,
                fallback_controller=self.fallback_controller,
                timeout_seconds=self.timeout_seconds,
                executor=self._executor,
            )
            self.controller_calls += 1
            self.controller_fallback_steps += int(result.fallback_used)
            self.controller_timeout_steps += int(
                bool(result.failure_kind)
                and result.failure_kind.startswith("timeout")
            )
            return result.action, state

        self.model.predict = protected_predict
        try:
            original_eval_freq = self.eval_freq
            if absolute_schedule:
                self.eval_freq = 1
            result = super()._on_step()
            if absolute_schedule:
                self.evaluation_timesteps.append(int(self.num_timesteps))
                interval = int(self.absolute_eval_interval_timesteps)
                while self.next_eval_timestep <= self.num_timesteps:
                    self.next_eval_timestep += interval
            return result
        finally:
            if absolute_schedule:
                self.eval_freq = original_eval_freq
            if had_instance_predict:
                self.model.predict = original_instance_predict
            else:
                del self.model.predict


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def _training_contract_sha256(
    *,
    algorithm: str,
    seed: int,
    total_timesteps: int,
    episode_days: int,
    n_envs: int,
    hyperparameters: dict[str, Any],
) -> str:
    payload = {
        "algorithm": algorithm,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "episode_days": int(episode_days),
        "n_envs": int(n_envs),
        "hyperparameters": hyperparameters,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_model_save(model, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.zip")
    model.save(temporary)
    temporary.replace(path)


def _atomic_vecnormalize_save(environment: VecNormalize, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.pkl")
    environment.save(temporary)
    temporary.replace(path)


def _atomic_replay_buffer_save(model, path: Path) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.pkl")
    model.save_replay_buffer(temporary)
    temporary.replace(path)


def _load_resume_state(
    output: Path,
    *,
    contract_sha256: str,
    algorithm: str,
) -> dict[str, Any] | None:
    state_path = output / "resume_state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("training_contract_sha256") != contract_sha256:
        raise ValueError("Resume checkpoint training contract does not match this run")
    required = [output / "resume_model.zip", output / "resume_vecnormalize.pkl"]
    if algorithm == "sac":
        required.append(output / "resume_replay_buffer.pkl")
    missing = [path.name for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"Resume checkpoint is incomplete: {', '.join(missing)}")
    return state


def _save_resume_checkpoint(
    *,
    output: Path,
    model,
    training_env: VecNormalize,
    algorithm: str,
    contract_sha256: str,
    target_timesteps: int,
    elapsed_seconds: float,
    callback: ProtectedEvalCallback,
    started_at: str,
) -> dict[str, Any]:
    _atomic_model_save(model, output / "resume_model.zip")
    _atomic_vecnormalize_save(training_env, output / "resume_vecnormalize.pkl")
    if algorithm == "sac":
        _atomic_replay_buffer_save(model, output / "resume_replay_buffer.pkl")
    state = {
        "schema_version": 1,
        "status": "partial",
        "algorithm": algorithm,
        "training_contract_sha256": contract_sha256,
        "completed_timesteps": int(model.num_timesteps),
        "target_timesteps": int(target_timesteps),
        "elapsed_seconds": float(elapsed_seconds),
        "started_at": started_at,
        "best_mean_reward": float(callback.best_mean_reward),
        "last_mean_reward": float(callback.last_mean_reward),
        "next_eval_timestep": int(callback.next_eval_timestep),
        "validation_evaluation_timesteps": callback.evaluation_timesteps,
        "validation_controller_calls": int(callback.controller_calls),
        "validation_controller_fallback_steps": int(
            callback.controller_fallback_steps
        ),
        "validation_controller_timeout_steps": int(
            callback.controller_timeout_steps
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "resume_state.json", state)
    return state


def _make_training_env(
    config: BenchmarkConfig,
    seed: int,
    rank: int,
    episode_days: int,
):
    def factory():
        first_year, first_start = config.training_scenarios[0]
        env = build_environment(
            config,
            first_start,
            episode_days=episode_days,
            growth_year=first_year,
        )
        env.weather_scenario_sampler = RandomScenarioWeatherSampler(
            scenarios=[
                {
                    "location": config.location,
                    "growth_year": growth_year,
                    "start_day": start_day,
                }
                for growth_year, start_day in config.training_scenarios
            ],
        )
        env.reset(seed=seed + rank)
        return gym.wrappers.FlattenObservation(env)

    return factory


def _make_eval_env(
    config: BenchmarkConfig,
    seed: int,
    episode_days: int,
    start_day: int,
    growth_year: int,
):
    def factory():
        env = build_environment(
            config,
            start_day,
            episode_days=episode_days,
            growth_year=growth_year,
        )
        env.reset(seed=seed)
        return gym.wrappers.FlattenObservation(env)

    return factory


def train_rl_agent(
    config: BenchmarkConfig,
    *,
    algorithm: str,
    seed: int,
    run_dir: str | Path,
    total_timesteps: int | None = None,
    episode_days: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    checkpoint_interval_timesteps: int = DEFAULT_CHECKPOINT_INTERVAL_TIMESTEPS,
    stop_after_checkpoint: bool = False,
) -> dict[str, Any]:
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHM_CLASSES:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    steps = int(config.rl_total_timesteps if total_timesteps is None else total_timesteps)
    days = int(config.episode_days if episode_days is None else episode_days)
    params = deepcopy(
        config.algorithms[algorithm] if hyperparameters is None else hyperparameters
    )
    n_envs = 1 if episode_days is not None else config.rl_n_envs
    checkpoint_interval = int(checkpoint_interval_timesteps)
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval_timesteps must be positive")
    contract_sha256 = _training_contract_sha256(
        algorithm=algorithm,
        seed=seed,
        total_timesteps=steps,
        episode_days=days,
        n_envs=n_envs,
        hyperparameters=params,
    )
    resume_state = _load_resume_state(
        output, contract_sha256=contract_sha256, algorithm=algorithm
    )
    started = (
        datetime.fromisoformat(str(resume_state["started_at"]))
        if resume_state is not None
        else datetime.now(timezone.utc)
    )
    resumed_from_timesteps = int(
        resume_state.get("completed_timesteps", 0) if resume_state else 0
    )
    prior_elapsed = float(resume_state.get("elapsed_seconds", 0.0) if resume_state else 0.0)
    metadata: dict[str, Any] = {
        "status": "running",
        "algorithm": algorithm,
        "seed": int(seed),
        "profile": config.profile,
        "total_timesteps": steps,
        "episode_days": days,
        "n_envs": n_envs,
        "hyperparameters": params,
        "started_at": started.isoformat(),
        "python": platform.python_version(),
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "training_contract_sha256": contract_sha256,
        "checkpoint_interval_timesteps": checkpoint_interval,
        "resumed_from_timesteps": resumed_from_timesteps,
    }
    _write_json(output / "training_metadata.json", metadata)

    training_env = None
    evaluation_env = None
    try:
        base_training_env = DummyVecEnv([
            _make_training_env(config, seed, rank, days) for rank in range(n_envs)
        ])
        if resume_state is None:
            training_env = VecNormalize(
                base_training_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                gamma=float(params.get("gamma", 0.99)),
            )
        else:
            training_env = VecNormalize.load(
                output / "resume_vecnormalize.pkl", base_training_env
            )
            training_env.training = True
            training_env.norm_reward = True
        evaluation_env = DummyVecEnv([
            _make_eval_env(
                config,
                seed,
                days,
                config.validation_start_day,
                config.validation_growth_year,
            )
        ])
        evaluation_env = VecNormalize(
            evaluation_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            gamma=float(params.get("gamma", 0.99)),
        )
        evaluation_env.training = False
        eval_raw_env = evaluation_env.venv.envs[0].unwrapped
        forecast_manifest = (
            getattr(eval_raw_env.forecast_provider, "artifact_manifest", None) or {}
        )
        metadata.update({
            "forecast_source_model_version": eval_raw_env.forecast_provider.source_model_version,
            "forecast_error_model_id": eval_raw_env.forecast_provider.error_model_id,
            "forecast_artifact_fingerprint": getattr(
                eval_raw_env.forecast_provider, "artifact_fingerprint", None
            ),
            "forecast_fit_role": forecast_manifest.get("fit_role", "not_applicable"),
            "forecast_select_role": forecast_manifest.get("select_role", "not_applicable"),
            "forecast_model_label": forecast_manifest.get("model_label", "not_applicable"),
            "forecast_metric_variable_indices": forecast_manifest.get(
                "metric_variable_indices", []
            ),
            "forecast_metric_error_scales": forecast_manifest.get(
                "metric_error_scales", []
            ),
            "forecast_source_sha256": forecast_manifest.get("source_sha256", {}),
            "forecast_leap_day_policy": forecast_manifest.get(
                "leap_day_policy", "not_applicable"
            ),
        })
        _write_json(output / "training_metadata.json", metadata)
        if resume_state is None:
            model = ALGORITHM_CLASSES[algorithm](
                "MlpPolicy",
                training_env,
                seed=seed,
                device="cpu",
                verbose=0,
                **params,
            )
        else:
            model = ALGORITHM_CLASSES[algorithm].load(
                output / "resume_model.zip", env=training_env, device="cpu"
            )
            if int(model.num_timesteps) != resumed_from_timesteps:
                raise ValueError("Resume model timestep does not match resume state")
            if algorithm == "sac":
                model.load_replay_buffer(output / "resume_replay_buffer.pkl")
        eval_interval_timesteps = max(1, steps // 2)
        next_eval_timestep = int(
            resume_state.get("next_eval_timestep")
            if resume_state is not None
            else eval_interval_timesteps
        )
        from experiments.controllers.tune_benchmark_classical import _agent_params

        callback = ProtectedEvalCallback(
            evaluation_env,
            best_model_save_path=str(output),
            log_path=str(output / "evaluation"),
            eval_freq=1,
            n_eval_episodes=1,
            deterministic=True,
            warn=False,
            callback_on_new_best=SaveBestVecNormalizeCallback(
                output / "best_vecnormalize.pkl"
            ),
            fallback_controller=RuleBasedController(**_agent_params("baseline")),
            timeout_seconds=config.controller_timeout_seconds,
            absolute_eval_interval_timesteps=eval_interval_timesteps,
            next_eval_timestep=next_eval_timestep,
            evaluation_timesteps=(
                resume_state.get("validation_evaluation_timesteps", [])
                if resume_state is not None
                else []
            ),
        )
        if resume_state is not None:
            callback.best_mean_reward = float(resume_state["best_mean_reward"])
            callback.last_mean_reward = float(resume_state["last_mean_reward"])
            callback.controller_calls = int(resume_state["validation_controller_calls"])
            callback.controller_fallback_steps = int(
                resume_state["validation_controller_fallback_steps"]
            )
            callback.controller_timeout_steps = int(
                resume_state["validation_controller_timeout_steps"]
            )

        invocation_clock = perf_counter()
        chunks_this_invocation = 0
        while int(model.num_timesteps) < steps:
            remaining = steps - int(model.num_timesteps)
            chunk_timesteps = min(checkpoint_interval, remaining)
            model.learn(
                total_timesteps=chunk_timesteps,
                callback=callback,
                progress_bar=False,
                reset_num_timesteps=(int(model.num_timesteps) == 0),
            )
            elapsed = prior_elapsed + (perf_counter() - invocation_clock)
            state = _save_resume_checkpoint(
                output=output,
                model=model,
                training_env=training_env,
                algorithm=algorithm,
                contract_sha256=contract_sha256,
                target_timesteps=steps,
                elapsed_seconds=elapsed,
                callback=callback,
                started_at=started.isoformat(),
            )
            metadata.update(
                status="partial",
                completed_timesteps=int(model.num_timesteps),
                elapsed_seconds=float(elapsed),
                validation_evaluation_timesteps=callback.evaluation_timesteps,
                updated_at=state["updated_at"],
            )
            _write_json(output / "training_metadata.json", metadata)
            chunks_this_invocation += 1
            if stop_after_checkpoint and chunks_this_invocation >= 1:
                return metadata

        elapsed = prior_elapsed + (perf_counter() - invocation_clock)
        model.save(output / "final_model")
        training_env.save(output / "vecnormalize.pkl")
        best_path = output / "best_model.zip"
        if not best_path.exists():
            model.save(output / "best_model")
            training_env.save(output / "best_vecnormalize.pkl")

        metadata.update(
            status="complete",
            completed_timesteps=int(model.num_timesteps),
            elapsed_seconds=float(elapsed),
            best_validation_reward=float(callback.best_mean_reward),
            validation_evaluation_timesteps=callback.evaluation_timesteps,
            validation_controller_calls=int(callback.controller_calls),
            validation_controller_fallback_steps=int(
                callback.controller_fallback_steps
            ),
            validation_controller_timeout_steps=int(callback.controller_timeout_steps),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(output / "training_metadata.json", metadata)
        return metadata
    except Exception as exc:
        metadata.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        _write_json(output / "training_metadata.json", metadata)
        raise
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        if training_env is not None:
            training_env.close()


def evaluate_saved_agent(
    config: BenchmarkConfig,
    *,
    algorithm: str,
    run_dir: str | Path,
    seed: int,
    episode_days: int | None = None,
    max_steps: int | None = None,
    growth_year: int | None = None,
    start_day: int | None = None,
) -> pd.DataFrame:
    algorithm = algorithm.lower()
    if algorithm not in ALGORITHM_CLASSES:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")
    path = Path(run_dir)
    model_path = path / "best_model.zip"
    normalization_path = path / "best_vecnormalize.pkl"
    if not model_path.exists() or not normalization_path.exists():
        raise FileNotFoundError(f"Missing saved model or normalization in {path}")

    days = int(config.episode_days if episode_days is None else episode_days)
    evaluation_year = int(config.growth_year if growth_year is None else growth_year)
    evaluation_start_day = int(config.test_start_day if start_day is None else start_day)
    base_vec = DummyVecEnv([
        _make_eval_env(config, seed, days, evaluation_start_day, evaluation_year)
    ])
    vec_env = VecNormalize.load(normalization_path, base_vec)
    vec_env.training = False
    vec_env.norm_reward = False
    model = ALGORITHM_CLASSES[algorithm].load(model_path, env=vec_env, device="cpu")
    if model.action_space != vec_env.action_space:
        vec_env.close()
        raise ValueError("Saved model action space does not match benchmark environment")

    rows: list[dict[str, Any]] = []
    observations = vec_env.reset()
    raw_env = base_vec.envs[0].unwrapped
    forecast_manifest = getattr(raw_env.forecast_provider, "artifact_manifest", None) or {}
    from experiments.controllers.tune_benchmark_classical import _agent_params

    fallback_controller = RuleBasedController(**_agent_params("baseline"))
    executor = ControllerExecutorV2(
        timeout_seconds=config.controller_timeout_seconds,
        expected_shape=(1, raw_env.action_space.shape[0]),
    )
    limit = int(max_steps if max_steps is not None else raw_env.N)
    try:
        for timestep in range(limit):
            started = perf_counter()
            forecast_identity = raw_env.issue_forecast()
            execution = execute_rl_action_v2(
                model=model,
                observations=observations,
                raw_env=raw_env,
                fallback_controller=fallback_controller,
                timeout_seconds=config.controller_timeout_seconds,
                executor=executor,
            )
            action = execution.action
            observations, _normalized_reward, dones, infos = vec_env.step(action)
            wall_time = perf_counter() - started
            info = infos[0]
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
                "location": str(raw_env.location),
                "growth_year": int(raw_env.growth_year),
                "start_day": int(raw_env.start_day),
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
                "forecast_source_model_version": forecast_identity.source_model_version,
                "forecast_error_model_id": forecast_identity.error_model_id,
                "forecast_artifact_fingerprint": getattr(
                    raw_env.forecast_provider, "artifact_fingerprint", None
                ),
                "forecast_fit_role": forecast_manifest.get("fit_role", "not_applicable"),
                "forecast_select_role": forecast_manifest.get("select_role", "not_applicable"),
                "controller_model_fingerprint": "not_applicable",
                "forecast_model_label": forecast_manifest.get(
                    "model_label", "not_applicable"
                ),
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
                "terminated": bool(dones[0] and not info.get("TimeLimit.truncated", False)),
                "truncated": bool(info.get("TimeLimit.truncated", False)),
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
            if dones[0]:
                break
    finally:
        vec_env.close()
    return pd.DataFrame(rows)
