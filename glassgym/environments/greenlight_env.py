from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, SupportsFloat
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from importlib import import_module

import numpy as np
import casadi as ca
import gymnasium as gym
from gymnasium import spaces, Space

from glassgym.components.actions import NamedControlActionScheme
from glassgym.components.climate_postprocessor import ClimateStatePostprocessor
from glassgym.components.forecast import (
    HistoricalForecastEmulatorV2,
    PersistenceForecastProviderV2,
)
from glassgym.components.observations import BaseObservations, OBSERVATION_MODULES
from glassgym.components.parameters import PARAMETER_PROVIDERS, BaseParameterProvider
from glassgym.configs.default_params import init_default_params
from glassgym.configs.greenlight_parameters import TOMATO_PARAMETER_REGISTRY
from glassgym.components.rewards import BaseReward, REWARDS_MODULES
from glassgym.components.safety import SafetyConfigV2, SafetyProjectorV2
from glassgym.components.weather import BaseWeatherSampler, WeatherRepository, WEATHER_SAMPLERS
from glassgym.core.types import ActionSchemaV2, RewardContext, StepContext, WeatherScenario
from glassgym.environments.utils import init_state, load_weather_data, vaporPres2rh
from glassgym.models.GreenLight.crop import crop_fluxes

_DEFAULT_WEATHER_DIR = Path(__file__).resolve().parent.parent / "data" / "weather"

ObservationSpec = Union[
    str,                              # registry key
    type[BaseObservations],            # class
    BaseObservations,                  # pre-built instance
    Callable[["GreenLightEnv"], BaseObservations],  # factory
]

class GreenLightEnv(gym.Env):
    def __init__(
        self,
        num_params: int,                # number of model parameters
        nx: int,                        # number of states
        nu: int,                        # number of control inputs
        nd: int,                        # number of disturbances
        dt: float,                      # [s] time step for the underlying GreenLight solver
        u_min: List[float],             # min bound for the control inputs
        u_max: List[float],             # max for the control inputs
        delta_u_max: float,             # max change for the control inputs
        season_length: int,             # length of the growing season [days]
        pred_horizon: int,              # lookahead horizon for weather predictions/realizations [days]
        observation_modules: Iterable[ObservationSpec] | None,
        constraints: Dict[str, Any],            # constraints for the environment
        reward_fn: Union[str, type[BaseReward], BaseReward, Callable[[], BaseReward]],
        weather_scenario_sampler: Union[str, BaseWeatherSampler, type[BaseWeatherSampler], Callable[[], BaseWeatherSampler]],
        weather_scenario_sampler_kwargs: Dict[str, Any],
        weather_repository: WeatherRepository | None = None,
        weather_data_dir: str | Path | None = None,
        reward_kwargs: Dict[str, Any] = {},     # reward function arguments
        controlled_inputs: List[str] = ["uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr"],
        normalize_actions: bool = True,
        parameter_provider: Union[str, type[BaseParameterProvider], BaseParameterProvider, Callable[[], BaseParameterProvider]] = "fixed",
        parameter_provider_kwargs: Dict[str, Any] = {},
        model_backend: str = "GreenLight",
        valid_reward_bounds: Tuple[float, float] = (-100.0, 100.0),
        failure_penalty_epsilon: float = 1e-6,
        safety_config: Mapping[str, Any] | None = None,
        forecast_provider: Mapping[str, Any] | PersistenceForecastProviderV2 | HistoricalForecastEmulatorV2 | None = None,
        state_postprocessor: ClimateStatePostprocessor | None = None,
        ) -> None:
        super(GreenLightEnv, self).__init__()

        if weather_repository is None:
            base_dir = Path(weather_data_dir) if weather_data_dir is not None else _DEFAULT_WEATHER_DIR
            weather_repository = WeatherRepository(
                weather_data_dir=base_dir,
                load_weather_data_fn=load_weather_data,
            )

        # arguments that are kept the same over various simulations
        self.c = 86400  # Class constant (seconds in a day)
        self.num_params = num_params
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        self.delta_u_max = self.u_max * delta_u_max
        self.dt = dt
        self.season_length = season_length
        self.N = int(self.season_length * self.c/self.dt)
        self.pred_horizon = pred_horizon
        self.Np = int(self.pred_horizon * self.c/self.dt)
        self.valid_reward_bounds = tuple(float(value) for value in valid_reward_bounds)
        self.failure_penalty_epsilon = float(failure_penalty_epsilon)
        if (
            len(self.valid_reward_bounds) != 2
            or not np.all(np.isfinite(self.valid_reward_bounds))
            or self.valid_reward_bounds[0] > self.valid_reward_bounds[1]
        ):
            raise ValueError("valid_reward_bounds must be two ordered finite values")
        if (
            not np.isfinite(self.failure_penalty_epsilon)
            or self.failure_penalty_epsilon <= 0.0
        ):
            raise ValueError("failure_penalty_epsilon must be finite and positive")
        longest_worst_return = self.valid_reward_bounds[0] * max(self.N, 1)
        strict_failure_return = np.nextafter(longest_worst_return, -np.inf)
        if not np.isfinite(longest_worst_return) or not np.isfinite(strict_failure_return):
            raise ValueError(
                "valid_reward_bounds must permit a finite failure return for "
                "the configured episode length"
            )


        # initialise the observation and action spaces
        self.observation_modules = self._init_observations_modules(observation_modules)

        # Preferred: structured observation space
        self.observation_space = self._build_observation_space()
        self.action_scheme = NamedControlActionScheme(
            nu=self.nu,
            controlled_inputs=controlled_inputs,
            low=self.u_min,
            high=self.u_max,
            normalize_actions=normalize_actions,
        )
        self.weather_scenario_sampler = self._init_weather_sampler(weather_scenario_sampler, weather_scenario_sampler_kwargs)
        self.weather_repository = weather_repository
        self.action_space = self.action_scheme.action_space
        self.safety_projector = self._init_safety_projector(safety_config)
        self.forecast_provider = self._init_forecast_provider(forecast_provider)
        self.state_postprocessor = state_postprocessor
        self.safety_fallback_duration_steps = 0
        self._last_safety_info: Dict[str, Any] = {}

        # initialise the model
        self.model_backend = model_backend
        self.F = self._define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
        )

        # The lowerbound constraints
        self.constraints_low = np.array([
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
        ])

        # The upperbound constraints
        self.constraints_high = np.array([
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
        ])

        # Parameter initialization
        self.base_p = np.asarray(init_default_params(self.num_params), dtype=np.float64)
        self.parameter_registry = TOMATO_PARAMETER_REGISTRY

        self.parameter_provider = self._init_parameter_provider(
            parameter_provider=parameter_provider,
            parameter_provider_kwargs=parameter_provider_kwargs,
        )

        # keep a valid initial vector before first reset
        self.p = self.base_p.copy()

        # initialise the reward function
        self.reward_fn = self._init_reward(reward_fn, reward_kwargs)

    @staticmethod
    def _init_forecast_provider(
        config: Mapping[str, Any] | PersistenceForecastProviderV2 | None,
    ) -> PersistenceForecastProviderV2:
        if config is None:
            return PersistenceForecastProviderV2()
        if isinstance(config, PersistenceForecastProviderV2):
            return config
        values = dict(config)
        schema_version = values.pop("schema_version", "forecast-provider-v2")
        kind = values.pop("kind", "persistence")
        if kind == "historical_emulator":
            if schema_version != "forecast-provider-v2":
                raise ValueError("Unsupported forecast provider configuration")
            artifact_path = Path(values.pop("artifact_manifest"))
            expected_fingerprint = values.pop("expected_artifact_fingerprint", None)
            expected_sources = values.pop("expected_source_sha256", None)
            if values:
                raise ValueError(f"Unexpected historical forecast options: {sorted(values)}")
            return HistoricalForecastEmulatorV2.from_artifact(
                artifact_path,
                expected_artifact_fingerprint=expected_fingerprint,
                expected_source_sha256=expected_sources,
            )
        if schema_version != "forecast-provider-v2" or kind != "persistence":
            raise ValueError("Unsupported forecast provider configuration")
        return PersistenceForecastProviderV2(**values)

    def issue_forecast(self):
        steps_per_day = max(1, int(round(self.c / float(self.dt))))
        return self.forecast_provider.issue(
            self.controller_weather_history(),
            issue_timestep=self.timestep,
            horizon_steps=self.Np,
            issue_day_index=int(np.floor(self.day_of_year)) % 365,
            issue_step_of_day=int(round(self.hour_of_day * 3600.0 / self.dt)) % steps_per_day,
        )

    def controller_weather_history(self) -> np.ndarray:
        """Return immutable disturbances containing no post-issue information."""
        history = np.asarray(
            self.weather_data[: self.timestep + 1], dtype=np.float64
        ).copy()
        if history.shape[1] > 7:
            energy = np.maximum(history[:, 0], 0.0) * float(self.dt) / 1_000_000.0
            steps_per_day = max(1, int(round(self.c / float(self.dt))))
            for start in range(0, len(history), steps_per_day):
                stop = min(start + steps_per_day, len(history))
                history[start:stop, 7] = np.cumsum(energy[start:stop])
        history.setflags(write=False)
        return history

    def _define_model(self, nx: int, nu: int, nd: int, n_params: int, dt: float):
        module = import_module(f"glassgym.models.{self.model_backend}.utils")
        return module.define_model(nx=nx, nu=nu, nd=nd, n_params=n_params, dt=dt)

    def _init_safety_projector(
        self,
        config: Mapping[str, Any] | None,
    ) -> SafetyProjectorV2 | None:
        if config is None:
            return None
        values = dict(config)
        schema_version = values.pop("schema_version", "safety-projection-v2")
        if schema_version != "safety-projection-v2":
            raise ValueError(f"Unsupported safety schema version: {schema_version}")
        values["schema_version"] = schema_version
        return SafetyProjectorV2(
            schema=ActionSchemaV2.for_count(self.nu),
            config=SafetyConfigV2(**values),
        )

    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        """
        # The current transition advances from timestep to timestep + 1.
        if self.timestep + 1 >= self.N:
            return True
        return False

    def _build_observation_space(self) -> Space:
        return spaces.Dict(OrderedDict(
            (module.key, module.space)
            for module in self.observation_modules
        ))

    def _init_observations_modules(
        self,
        observation_modules: Iterable[ObservationSpec],
    ) -> list[BaseObservations]:
        modules: list[BaseObservations] = []

        for spec in observation_modules:
            if isinstance(spec, str):
                modules.append(OBSERVATION_MODULES[spec](self))
            elif isinstance(spec, type) and issubclass(spec, BaseObservations):
                modules.append(spec(self))
            elif isinstance(spec, BaseObservations):
                modules.append(spec)
            elif callable(spec):
                modules.append(spec(self))
            else:
                raise TypeError(f"Unsupported observation spec: {spec!r}")

        return modules

    def _init_reward(self, reward, reward_kwargs):
        reward_kwargs = reward_kwargs or {}

        if isinstance(reward, str):
            reward_cls = REWARDS_MODULES[reward]
            return reward_cls(p=self.p, dt=self.dt, **reward_kwargs)
        elif isinstance(reward, type) and issubclass(reward, BaseReward):
            return reward(p=self.p, dt=self.dt, **reward_kwargs)
        elif isinstance(reward, BaseReward):
            return reward
        elif callable(reward):
            return reward(self, **reward_kwargs)
        else:
            raise TypeError(f"Unsupported reward spec: {reward!r}")

    def _init_weather_sampler(self, weather_sampler, weather_sampler_kwargs):
        kwargs = weather_sampler_kwargs or {}

        if isinstance(weather_sampler, str):
            cls = WEATHER_SAMPLERS[weather_sampler]
            return cls(**kwargs)
        elif isinstance(weather_sampler, BaseWeatherSampler):
            return weather_sampler
        elif isinstance(weather_sampler, type) and issubclass(weather_sampler, BaseWeatherSampler):
            return weather_sampler(**kwargs)
        elif callable(weather_sampler):
            return weather_sampler(**kwargs)
        else:
            raise TypeError(f"Unsupported weather_sampler spec: {weather_sampler!r}")

    def _init_parameter_provider(
        self,
        parameter_provider="fixed",
        parameter_provider_kwargs=None,
    ):
        kwargs = dict(parameter_provider_kwargs or {})

        if isinstance(parameter_provider, str):
            provider_cls = PARAMETER_PROVIDERS[parameter_provider]
            return provider_cls(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        if isinstance(parameter_provider, type) and issubclass(parameter_provider, BaseParameterProvider):
            return parameter_provider(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        if isinstance(parameter_provider, BaseParameterProvider):
            return parameter_provider

        if callable(parameter_provider):
            return parameter_provider(
                env=self,
                base_p=self.base_p,
                registry=self.parameter_registry,
                **kwargs,
            )

        raise TypeError(f"Unsupported parameter_provider spec: {parameter_provider!r}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        self.x_prev = np.copy(self.x)
        u_prev = np.copy(self.u)

        # Convert the controller proposal, then apply the common execution layer.
        proposed_u = self.action_scheme.to_full_control_input(action)
        disturbance = self.weather_data[self.timestep]
        if self.safety_projector is None:
            self.u = proposed_u
            self._last_safety_info = {}
        else:
            projection = self.safety_projector.project(
                proposed_u,
                previous=u_prev,
                disturbance=disturbance,
                hour_of_day=self.hour_of_day,
                indoor_relative_humidity=float(vaporPres2rh(self.x[2], self.x[15])),
                indoor_temperature=float(self.x[2]),
            )
            self.u = projection.executed.astype(np.float32)
            self.action_scheme.commit_full_control_input(self.u)
            if projection.fallback_used:
                self.safety_fallback_duration_steps += 1
            else:
                self.safety_fallback_duration_steps = 0
            self._last_safety_info = {
                "proposed_controls": projection.proposed.copy(),
                "executed_controls": projection.executed.copy(),
                "safety_interventions": projection.interventions,
                "safety_intervened": not np.allclose(
                    projection.proposed,
                    projection.executed,
                    rtol=0.0,
                    atol=1e-12,
                ),
                "safety_fallback_used": projection.fallback_used,
                "invalid_proposal_indices": projection.invalid_proposal_indices,
                "safety_fallback_duration_steps": self.safety_fallback_duration_steps,
            }
        crop_info: Dict[str, Any] = {}

        flux_start = crop_fluxes(
            ca.DM(self.x_prev), ca.DM(self.u), ca.DM(disturbance), ca.DM(self.p)
        )
        p_dyn = ca.vertcat(ca.DM(disturbance), self.p)
        try:
            res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
        except (RuntimeError, ArithmeticError) as exc:
            return self._numerical_failure(exc, "integration")

        self.x = res["xf"].full().flatten()
        if not np.all(np.isfinite(self.x)):
            return self._numerical_failure(
                FloatingPointError("solver returned a non-finite state"),
                "integration",
            )
        postprocess_info: Dict[str, Any] = {}
        if self.state_postprocessor is not None:
            corrected, postprocess_info = self.state_postprocessor.correct(
                previous_state=self.x_prev.copy(),
                raw_state=self.x.copy(),
                controls=self.u.copy(),
                disturbance=np.asarray(disturbance, dtype=np.float64).copy(),
                dt_seconds=float(self.dt),
                hour_of_day=float(self.hour_of_day),
                day_of_year=float(self.day_of_year),
            )
            corrected = np.asarray(corrected, dtype=np.float64)
            if corrected.shape != self.x.shape:
                raise ValueError("State postprocessor changed the state vector shape")
            if not np.all(np.isfinite(corrected)):
                raise ValueError("State postprocessor returned a non-finite state")
            self.x = corrected
        flux_end = crop_fluxes(
            ca.DM(self.x), ca.DM(self.u), ca.DM(disturbance), ca.DM(self.p)
        )
        harvest_rate = max(0.0, 0.5 * (
            float(flux_start["fruit_harvest"]) + float(flux_end["fruit_harvest"])
        ))
        fruit_allocation_rate = max(0.0, 0.5 * (
            float(flux_start["buffer_to_fruit"]) + float(flux_end["buffer_to_fruit"])
        ))
        dmfm = float(self.p[206])
        crop_info = {
            "crop_model": "Vanthoor2011_GreenLight",
            "fruit_harvest_rate_mg_m2_s": harvest_rate,
            "fruit_allocation_rate_mg_m2_s": fruit_allocation_rate,
            "harvested_dry_matter_mg_m2": harvest_rate * float(self.dt),
            "allocated_fruit_dry_matter_mg_m2": fruit_allocation_rate * float(self.dt),
            "dry_matter_fraction": dmfm,
            "c_buffer_mg_m2": float(self.x[22]),
            "c_leaf_mg_m2": float(self.x[23]),
            "c_stem_mg_m2": float(self.x[24]),
            "c_fruit_mg_m2": float(self.x[25]),
            "c_fruit_previous_mg_m2": float(self.x_prev[25]),
        }

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day +=  (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()
        if self._terminalState():
            self.terminated = True

        # create reward context such that reward function does not rely on env
        ctx = RewardContext(
            t=self.timestep,
            dt=self.dt,
            Np=self.Np,
            x_prev=self.x_prev,
            x=self.x,
            u=self.u,
            p=self.p,
            d=self.controller_weather_history(),
            obs=self.obs,
            day_of_year=self.day_of_year,
            hour_of_day=self.hour_of_day,
            constraints_low=self.constraints_low,
            constraints_high=self.constraints_high,
            u_prev=u_prev,
        )

        reward, reward_info = self.reward_fn.compute_reward(ctx)
        reward = float(reward)
        reward_low, reward_high = self.valid_reward_bounds
        if not np.isfinite(reward) or not reward_low <= reward <= reward_high:
            raise ValueError(
                "reward must be finite and within valid_reward_bounds "
                f"[{reward_low}, {reward_high}], received {reward}"
            )
        transition_info = {
            **reward_info,
            **crop_info,
            **self._last_safety_info,
            **postprocess_info,
        }
        if "IndoorClimateObservations" in self.obs:
            transition_info["indoor_climate"] = self.obs[
                "IndoorClimateObservations"
            ].copy()
        info = self._get_info(transition_info)
        self.timestep += 1

        return (
                self.obs,
                reward, 
                self.terminated, 
                self.truncated,
                info
                )

    def _failure_observation(self) -> Dict[str, np.ndarray]:
        observation: Dict[str, np.ndarray] = {}
        for module in self.observation_modules:
            space = module.space
            sentinel = np.clip(
                np.zeros(space.shape, dtype=np.float64),
                space.low,
                space.high,
            ).astype(space.dtype)
            if not space.contains(sentinel):
                raise ValueError(
                    f"Cannot construct failure observation for {module.key}"
                )
            observation[module.key] = sentinel
        return observation

    def _numerical_failure(
        self,
        exc: BaseException,
        phase: str,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.x = self.x_prev.copy()
        self.terminated = False
        self.truncated = True
        self.obs = self._failure_observation()

        remaining_steps = max(self.N - self.timestep, 1)
        reward_lower_bound = self.valid_reward_bounds[0]
        worst_valid_continuation = reward_lower_bound * remaining_steps
        epsilon_candidate = worst_valid_continuation - self.failure_penalty_epsilon
        failure_reward = (
            epsilon_candidate
            if np.isfinite(epsilon_candidate)
            and epsilon_candidate < worst_valid_continuation
            else float(np.nextafter(worst_valid_continuation, -np.inf))
        )
        info = self._get_info(
            {
                "reward": failure_reward,
                "failure": True,
                "failure_kind": "numerical_integration",
                "failure_phase": phase,
                "failure_exception": type(exc).__name__,
                "failure_message": str(exc),
                "failure_timestep": self.timestep,
                "failure_remaining_steps": remaining_steps,
                "failure_reward_lower_bound": reward_lower_bound,
                "failure_penalty_epsilon": self.failure_penalty_epsilon,
                "scenario": {
                    "location": self.location,
                    "growth_year": self.growth_year,
                    "start_day": self.start_day,
                },
                **self._last_safety_info,
            }
        )
        self.timestep += 1
        return self.obs, failure_reward, False, True, info

    def _get_obs(self):
        ctx = StepContext(
            t=self.timestep,
            dt=self.dt,
            Np=self.Np,
            x_prev=self.x_prev,
            x=self.x,
            u=self.u,
            p=self.p,
            d=self.controller_weather_history(),
            hour_of_day=self.hour_of_day,
            day_of_year=self.day_of_year,
            forecast=self.issue_forecast(),
        )
        obs = {
            module.key: np.asarray(module.compute_obs(ctx), dtype=np.float32)
            for module in self.observation_modules
        }
        return obs

    def get_obs_names(self):
        """
        """
        obs_names = []
        for module in self.observation_modules:
            obs_names.extend(module.obs_names)
        return obs_names

    def _get_info(self, reward_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            **reward_info,
            "controls": self.u,
        }

    def _resolve_weather_scenario(self, options: Dict[str, Any] | None) -> WeatherScenario:
        if options is not None and "scenario" in options:
            return WeatherScenario(
                location=options["scenario"]["location"],
                growth_year=int(options["scenario"]["growth_year"]),
                start_day=int(options["scenario"]["start_day"]),
            )
        return self.weather_scenario_sampler.sample(self.np_random, options)

    def reset(
        self, seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed, options=options)

        scenario = self._resolve_weather_scenario(options)
        weather_data = self.weather_repository.load(
            location=scenario.location,
            growth_year=scenario.growth_year,
            start_day=scenario.start_day,
            season_length=self.season_length,
            pred_horizon=self.pred_horizon,
            dt=self.dt,
            nd=self.nd,
        )

        self.weather_data = weather_data
        self.location = scenario.location
        self.growth_year = scenario.growth_year
        self.start_day = scenario.start_day

        self.day_of_year = self.start_day
        self.hour_of_day = 0

        self.p = self.parameter_provider.sample(
            rng=self.np_random,
            scenario=scenario,
            options=options,
        )

        self.u  = self.action_scheme.reset_full_control_input()
        self.x = init_state(self.weather_data[0])
        self._apply_initial_crop_state(options)
        self.x_prev = np.copy(self.x)
        self.timestep = 0
        self.obs = self._get_obs()

        self.terminated = False
        self.truncated = False
        self.safety_fallback_duration_steps = 0
        self._last_safety_info = {}
        return self.obs, {
            "scenario":
                {"location": self.location, "growth_year": self.growth_year, "start_day": self.start_day}
            }

    def _apply_initial_crop_state(self, options: Optional[Dict[str, Any]]) -> None:
        if not options or "initial_crop_state_mg_m2" not in options:
            return
        override = options["initial_crop_state_mg_m2"]
        if not isinstance(override, Mapping):
            raise ValueError("initial_crop_state_mg_m2 must be a mapping")
        state_indices = {
            "cBuf": 22,
            "cLeaf": 23,
            "cStem": 24,
            "cFruit": 25,
            "tCanSum": 26,
        }
        unknown = sorted(set(override) - set(state_indices))
        if unknown:
            raise ValueError(f"unknown crop state: {', '.join(unknown)}")
        for name, raw_value in override.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"initial crop state {name} must be finite and non-negative"
                ) from exc
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"initial crop state {name} must be finite and non-negative"
                )
            self.x[state_indices[name]] = value
