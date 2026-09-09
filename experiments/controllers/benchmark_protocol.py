from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from glassgym.components.weather import WeatherRepository
from glassgym.environments.greenlight_env import GreenLightEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmarks" / "chengdu_controllers.yml"
ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "envs" / "ChengduControllerBenchmark.yml"
CONTROLLED_INDICES = np.array([2, 3, 4, 5], dtype=int)


@dataclass(frozen=True)
class BenchmarkConfig:
    profile: str
    dataset_id: str
    environment_id: str
    location: str
    growth_year: int
    validation_growth_year: int
    training_growth_years: tuple[int, ...]
    training_scenarios: tuple[tuple[int, int], ...]
    available_weather_days: int
    episode_days: int
    train_start_days: tuple[int, ...]
    validation_start_day: int
    test_start_day: int
    seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    rl_total_timesteps: int
    rl_n_envs: int
    calibration_path: Path
    output_dir: Path
    algorithms: dict[str, dict[str, Any]]
    controller_timeout_seconds: float


def load_benchmark_config(
    profile: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> BenchmarkConfig:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        selected = raw["profiles"][profile]
    except KeyError as exc:
        valid = ", ".join(sorted(raw.get("profiles", {})))
        raise ValueError(f"Unknown benchmark profile '{profile}'. Valid profiles: {valid}") from exc

    calibration_path = Path(raw["calibration_path"])
    if not calibration_path.is_absolute():
        calibration_path = PROJECT_ROOT / calibration_path
    episode_days = int(raw["episode_days"])
    training_scenarios: list[tuple[int, int]] = []
    for season in raw["training_seasons"]:
        year = int(season["growth_year"])
        start = int(season["start_day"])
        end = int(season["end_day_exclusive"])
        if end - start != 120:
            raise ValueError(f"Training season {season['name']} must contain 120 days")
        training_scenarios.extend(
            (year, day) for day in range(start, end - episode_days)
        )
    config = BenchmarkConfig(
        profile=profile,
        dataset_id=str(raw["dataset_id"]),
        environment_id=str(raw["environment_id"]),
        location=str(raw["location"]),
        growth_year=int(raw["growth_year"]),
        validation_growth_year=int(raw["validation_growth_year"]),
        training_growth_years=tuple(sorted({year for year, _ in training_scenarios})),
        training_scenarios=tuple(training_scenarios),
        available_weather_days=int(raw["available_weather_days"]),
        episode_days=episode_days,
        train_start_days=tuple(sorted({day for _, day in training_scenarios})),
        validation_start_day=int(raw["validation_start_day"]),
        test_start_day=int(raw["test_start_day"]),
        seeds=tuple(int(value) for value in selected["seeds"]),
        evaluation_seeds=tuple(int(value) for value in selected["evaluation_seeds"]),
        rl_total_timesteps=int(selected["rl_total_timesteps"]),
        rl_n_envs=int(selected["rl_n_envs"]),
        calibration_path=calibration_path,
        output_dir=Path("results") / str(raw["dataset_id"]) / "controller_benchmark" / profile,
        algorithms=dict(raw["algorithms"]),
        controller_timeout_seconds=float(raw.get("controller_timeout_seconds", 30.0)),
    )
    if not np.isfinite(config.controller_timeout_seconds) or config.controller_timeout_seconds <= 0:
        raise ValueError("controller_timeout_seconds must be finite and positive")
    validate_windows(config, config.available_weather_days)
    return config


def validate_windows(config: BenchmarkConfig, available_days: int) -> None:
    if not config.training_scenarios:
        raise ValueError("At least one training scenario is required")
    if min(day for _, day in config.training_scenarios) < 0:
        raise ValueError("Training scenario start days must be non-negative")
    if not config.training_growth_years:
        raise ValueError("At least one training weather year is required")
    if any(
        start_day + config.episode_days + 0.5 > available_days
        for _, start_day in config.training_scenarios
    ):
        raise ValueError("A training episode exceeds available weather")
    if max(config.training_growth_years) >= config.validation_growth_year:
        raise ValueError("Training years must precede the validation year")
    if config.validation_growth_year > config.growth_year:
        raise ValueError("Validation year must not follow the test year")
    if (
        config.validation_growth_year == config.growth_year
        and config.validation_start_day >= config.test_start_day
    ):
        raise ValueError("Validation must precede test")
    if (
        config.validation_growth_year == config.growth_year
        and config.validation_start_day + config.episode_days > config.test_start_day
    ):
        raise ValueError("The validation episode overlaps test weather")
    required_days = config.test_start_day + config.episode_days + 0.5
    if required_days > float(available_days):
        raise ValueError(
            f"Test episode exceeds available weather: needs {required_days:g} days, "
            f"has {available_days}"
        )


def full_control_target_to_action(
    current_full: np.ndarray,
    target_full: np.ndarray,
    delta: float | np.ndarray,
) -> np.ndarray:
    current = np.asarray(current_full, dtype=np.float32)
    target = np.asarray(target_full, dtype=np.float32)
    if current.shape != (6,) or target.shape != (6,):
        raise ValueError("current_full and target_full must each contain six controls")
    delta_values = np.asarray(delta, dtype=np.float32)
    if delta_values.ndim == 0:
        delta_values = np.full(4, float(delta_values), dtype=np.float32)
    if delta_values.shape != (4,) or np.any(delta_values <= 0):
        raise ValueError("delta must be positive and scalar or four-dimensional")
    action = (target[CONTROLLED_INDICES] - current[CONTROLLED_INDICES]) / delta_values
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _load_environment_kwargs() -> dict[str, Any]:
    raw = yaml.safe_load(ENV_CONFIG_PATH.read_text(encoding="utf-8"))
    kwargs = dict(raw["ChengduControllerBenchmark"])
    repository_kwargs = kwargs.pop("weather_repository_kwargs")
    kwargs.pop("eval_scenarios", None)
    weather_dir = Path(repository_kwargs["weather_data_dir"])
    if not weather_dir.is_absolute():
        weather_dir = PROJECT_ROOT / weather_dir
    repository_kwargs["weather_data_dir"] = weather_dir
    kwargs["weather_repository"] = WeatherRepository.from_config(repository_kwargs)
    return kwargs


def _apply_calibration(
    env: GreenLightEnv,
    calibration_path: Path,
    *,
    strength: float = 1.0,
) -> None:
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing calibrated parameter file: {calibration_path}")
    if not np.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("calibration strength must be finite and non-negative")
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibrated = env.base_p.copy()
    for index, value in payload["multipliers"].items():
        calibrated[int(index)] = 1.0 + float(strength) * (float(value) - 1.0)
    env.base_p = calibrated
    env.parameter_provider.base_p = calibrated.copy()
    env.p = calibrated.copy()


def build_environment(
    config: BenchmarkConfig,
    start_day: int,
    *,
    episode_days: int | None = None,
    growth_year: int | None = None,
    dt_seconds: int | None = None,
    calibration_strength: float = 1.0,
) -> GreenLightEnv:
    kwargs = _load_environment_kwargs()
    kwargs["season_length"] = int(config.episode_days if episode_days is None else episode_days)
    if dt_seconds is not None:
        if int(dt_seconds) <= 0:
            raise ValueError("dt_seconds must be positive")
        kwargs["dt"] = int(dt_seconds)
    kwargs["weather_scenario_sampler"] = "fixed"
    kwargs["weather_scenario_sampler_kwargs"] = {
        "location": config.location,
        "growth_year": int(config.growth_year if growth_year is None else growth_year),
        "start_day": int(start_day),
    }
    env = GreenLightEnv(**kwargs)
    _apply_calibration(
        env,
        config.calibration_path,
        strength=float(calibration_strength),
    )
    return env
