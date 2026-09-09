"""
Evaluate the lightweight MPC controller on GreenLightEnv.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from glassgym.components.mpc import LightweightMPCController
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


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


def evaluate_episode(env: GreenLightEnv, controller: LightweightMPCController) -> list[dict]:
    steps: list[dict] = []
    controller.reset()
    while True:
        ctx = build_step_context(env)
        u = controller.predict(ctx)
        _obs, reward, terminated, truncated, info = env.step(u.astype(np.float32))
        steps.append({
            "timestep": env.timestep,
            "reward": float(reward),
            **{k: v for k, v in info.items() if k != "controls"},
        })
        if terminated or truncated:
            break
    return steps


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the lightweight MPC controller on the GreenLight environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env_id", default="GreenLightEnv")
    parser.add_argument("--env_config", default="configs/envs/")
    parser.add_argument("--mpc_config", default="configs/agents/")
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--n_sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon_steps", type=int, default=None)
    parser.add_argument("--parameter_provider", type=str, default=None)
    parser.add_argument("--parameter_provider_kwargs", type=str, default=None)
    parser.add_argument("--reward_kwargs", type=str, default=None)
    parser.add_argument("--save_dir", default="results/amsterdam_reference/mpc/")
    args = parser.parse_args()

    env_kwargs = load_env_params(args.env_id, args.env_config)
    env_kwargs, default_scenarios = build_env_kwargs(env_kwargs)
    env_kwargs["normalize_actions"] = False

    scenarios = default_scenarios
    if args.scenarios is not None:
        scenarios = json.loads(args.scenarios)

    if args.parameter_provider is not None:
        env_kwargs["parameter_provider"] = args.parameter_provider
    if args.parameter_provider_kwargs is not None:
        env_kwargs["parameter_provider_kwargs"] = json.loads(args.parameter_provider_kwargs)
    if args.reward_kwargs is not None:
        overrides = json.loads(args.reward_kwargs)
        env_kwargs.setdefault("reward_kwargs", {}).update(overrides)

    mpc_params = load_model_hyperparams("mpc", args.env_id)
    if args.horizon_steps is not None:
        mpc_params["horizon_steps"] = args.horizon_steps

    env = GreenLightEnv(**env_kwargs)
    controller = LightweightMPCController(**mpc_params)
    controller.bind_env(env)

    os.makedirs(args.save_dir, exist_ok=True)
    all_results: list[dict] = []

    for scenario in scenarios:
        desc = f"{scenario['location']}/{scenario['growth_year']}d{scenario['start_day']}"
        for sim in tqdm(range(args.n_sims), desc=desc):
            env.reset(seed=args.seed + sim, options={"scenario": scenario})
            episode_steps = evaluate_episode(env, controller)
            for step in episode_steps:
                step.update(sim=sim, **scenario)
            all_results.extend(episode_steps)

    df = pd.DataFrame(all_results)
    save_path = os.path.join(args.save_dir, f"mpc_{args.env_id}.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df)} rows to {save_path}")


if __name__ == "__main__":
    main()


