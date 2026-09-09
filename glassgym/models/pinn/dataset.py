from __future__ import annotations

import numpy as np

from glassgym.components.pid import PIDController
from glassgym.components.rule_based import RuleBasedController
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_model_hyperparams


def build_step_context(env: GreenLightEnv) -> StepContext:
    return StepContext(
        t=env.timestep,
        dt=env.dt,
        Np=env.Np,
        x_prev=env.x_prev,
        x=env.x,
        u=env.u,
        p=env.p,
        d=env.weather_data,
        hour_of_day=env.hour_of_day,
        day_of_year=env.day_of_year,
    )


def _controller_for_source(source: str):
    if source == "baseline":
        return RuleBasedController(**load_model_hyperparams("rule_based", "GreenLightEnv"))
    if source == "pid":
        return PIDController(**load_model_hyperparams("pid", "GreenLightEnv"))
    return None


def _action_for_source(env: GreenLightEnv, source: str, rng: np.random.Generator, controller):
    if source == "random":
        return rng.uniform(0.0, 1.0, size=env.nu).astype(np.float32)
    ctx = build_step_context(env)
    return controller.predict(ctx).astype(np.float32)


def generate_mixed_transitions(
    env_kwargs: dict,
    source_counts: dict[str, int],
    max_steps_per_episode: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    env_kwargs = env_kwargs.copy()
    env_kwargs["normalize_actions"] = False
    env = GreenLightEnv(**env_kwargs)

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    previous_states: list[np.ndarray] = []
    sources: list[str] = []

    for source, episodes in source_counts.items():
        controller = _controller_for_source(source)
        for episode in range(int(episodes)):
            env.reset(seed=seed + len(sources) + episode)
            if hasattr(controller, "reset"):
                controller.reset()
            for _ in range(max_steps_per_episode):
                x_t = env.x.copy()
                d_t = env.weather_data[env.timestep].copy()
                u_t = _action_for_source(env, source, rng, controller)
                _obs, _reward, terminated, truncated, _info = env.step(u_t)
                x_next = env.x.copy()

                inputs.append(np.concatenate([x_t, u_t, d_t]).astype(np.float32))
                targets.append(x_next.astype(np.float32))
                previous_states.append(x_t.astype(np.float32))
                sources.append(source)

                if terminated or truncated:
                    break

    env.close()
    return {
        "inputs": np.asarray(inputs, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "previous_states": np.asarray(previous_states, dtype=np.float32),
        "sources": np.asarray(sources),
    }
