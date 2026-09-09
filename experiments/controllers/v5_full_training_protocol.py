from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from experiments.controllers.benchmark_protocol import PROJECT_ROOT


DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "benchmarks" / "chengdu_v5_full_training.yml"
)


@dataclass(frozen=True)
class V5FullTrainingProtocol:
    schema_version: str
    dataset_id: str
    episode_days: int
    training_scenarios: tuple[tuple[int, int], ...]
    validation_scenarios: tuple[tuple[int, int], ...]
    temporal_holdout: tuple[int, int, int]
    seeds: tuple[int, ...]
    checkpoints: tuple[int, ...]
    minimum_full_checkpoint: int
    plateau_relative_improvement: float
    residual_ppo_scale: float
    active_safety_limit: float
    algorithms: dict[str, dict[str, Any]]
    source_path: Path
    source_sha256: str

    def resolved_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "episode_days": self.episode_days,
            "training_scenarios": [
                {"growth_year": year, "start_day": day}
                for year, day in self.training_scenarios
            ],
            "validation_scenarios": [
                {"growth_year": year, "start_day": day}
                for year, day in self.validation_scenarios
            ],
            "temporal_holdout": {
                "growth_year": self.temporal_holdout[0],
                "start_day": self.temporal_holdout[1],
                "episode_days": self.temporal_holdout[2],
            },
            "seeds": list(self.seeds),
            "checkpoints": list(self.checkpoints),
            "minimum_full_checkpoint": self.minimum_full_checkpoint,
            "plateau_relative_improvement": self.plateau_relative_improvement,
            "residual_ppo_scale": self.residual_ppo_scale,
            "active_safety_limit": self.active_safety_limit,
            "algorithms": self.algorithms,
            "config_path": str(self.source_path),
            "config_sha256": self.source_sha256,
            "data_roles": {
                "training": "2023-2024 spring and autumn rolling windows",
                "validation": "2025 spring fixed non-overlapping windows",
                "temporal_holdout": "2025 autumn complete 120-day season",
            },
        }


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved


def _windows_overlap(start_a: int, length_a: int, start_b: int, length_b: int) -> bool:
    return start_a < start_b + length_b and start_b < start_a + length_a


def load_v5_full_training_protocol(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> V5FullTrainingProtocol:
    source = _resolve_path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    episode_days = int(raw["episode_days"])
    if episode_days <= 0:
        raise ValueError("episode_days must be positive")

    training: list[tuple[int, int]] = []
    for season in raw["training_seasons"]:
        year = int(season["growth_year"])
        start = int(season["start_day"])
        end = int(season["end_day_exclusive"])
        if end - start < episode_days:
            raise ValueError(f"training season {season['name']} is shorter than one episode")
        training.extend((year, day) for day in range(start, end - episode_days + 1))

    validation_year = int(raw["validation_growth_year"])
    validation = tuple(
        (validation_year, int(day)) for day in raw["validation_start_days"]
    )
    ordered_validation_days = [day for _year, day in validation]
    for index, start in enumerate(ordered_validation_days):
        for other in ordered_validation_days[index + 1 :]:
            if _windows_overlap(start, episode_days, other, episode_days):
                raise ValueError("validation windows must be non-overlapping")

    holdout_raw = raw["temporal_holdout"]
    holdout = (
        int(holdout_raw["growth_year"]),
        int(holdout_raw["start_day"]),
        int(holdout_raw["episode_days"]),
    )
    if any(
        year == holdout[0]
        and _windows_overlap(day, episode_days, holdout[1], holdout[2])
        for year, day in validation
    ):
        raise ValueError("validation windows overlap the temporal holdout")
    if any(year >= validation_year for year, _day in training):
        raise ValueError("training weather must precede validation weather")

    seeds = tuple(int(value) for value in raw["seeds"])
    checkpoints = tuple(sorted({int(value) for value in raw["checkpoints"]}))
    minimum = int(raw["minimum_full_checkpoint"])
    threshold = float(raw["plateau_relative_improvement"])
    residual_scale = float(raw["residual_ppo_scale"])
    active_safety_limit = float(raw.get("active_safety_limit", 0.05))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be non-empty and unique")
    if not checkpoints or checkpoints[0] <= 0 or minimum not in checkpoints:
        raise ValueError("checkpoints must be positive and include the minimum full checkpoint")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("plateau_relative_improvement must be between zero and one")
    if not np.isfinite(residual_scale) or not 0.0 <= residual_scale <= 1.0:
        raise ValueError("residual_ppo_scale must be between zero and one")
    if not np.isfinite(active_safety_limit) or not 0.0 <= active_safety_limit <= 1.0:
        raise ValueError("active_safety_limit must be between zero and one")

    return V5FullTrainingProtocol(
        schema_version=str(raw["schema_version"]),
        dataset_id=str(raw["dataset_id"]),
        episode_days=episode_days,
        training_scenarios=tuple(training),
        validation_scenarios=validation,
        temporal_holdout=holdout,
        seeds=seeds,
        checkpoints=checkpoints,
        minimum_full_checkpoint=minimum,
        plateau_relative_improvement=threshold,
        residual_ppo_scale=residual_scale,
        active_safety_limit=active_safety_limit,
        algorithms={key: dict(value) for key, value in raw["algorithms"].items()},
        source_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def should_continue_training(
    checkpoint_scores: Mapping[int, float],
    *,
    relative_improvement_threshold: float = 0.01,
) -> bool:
    finite = sorted(
        (int(step), float(score))
        for step, score in checkpoint_scores.items()
        if np.isfinite(float(score))
    )
    if len(finite) < 2:
        return True
    previous = finite[-2][1]
    current = finite[-1][1]
    relative_improvement = (current - previous) / max(abs(previous), 1e-12)
    return bool(relative_improvement > float(relative_improvement_threshold))


def write_resolved_protocol(
    protocol: V5FullTrainingProtocol,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(protocol.resolved_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
