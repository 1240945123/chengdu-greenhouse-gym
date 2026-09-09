from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_dataset_config(dataset_id: str, config_dir: str | Path = "configs/datasets") -> dict[str, Any]:
    path = Path(config_dir) / f"{dataset_id}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get("dataset_id") != dataset_id:
        raise ValueError(f"Dataset config {path} has dataset_id={config.get('dataset_id')!r}")
    return config
