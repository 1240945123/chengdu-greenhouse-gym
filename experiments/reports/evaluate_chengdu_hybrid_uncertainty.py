from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.reports.chengdu_residual_correction import load_residual_model


def evaluate_interval_metrics(
    rollouts: pd.DataFrame,
    model: dict,
    *,
    horizons: list[int],
    quantile_key: str = "q90",
) -> dict[str, float]:
    if "uncertainty" not in model:
        raise ValueError("Residual model does not contain calibrated uncertainty")
    metrics: dict[str, float] = {"num_rollout_rows": float(len(rollouts))}
    for horizon in horizons:
        selected = rollouts.loc[rollouts["horizon"].astype(int).eq(int(horizon))]
        if selected.empty:
            raise ValueError(f"No rollout rows for horizon {horizon}")
        for target in ("air_temperature", "relative_humidity"):
            radius = float(model["uncertainty"][target][quantile_key])
            if radius < 0.0 or not np.isfinite(radius):
                raise ValueError(f"Invalid uncertainty radius for {target}: {radius}")
            error = np.abs(
                selected[f"pred_{target}"].to_numpy(dtype=float)
                - selected[f"true_{target}"].to_numpy(dtype=float)
            )
            prefix = f"horizon_{horizon}_{target}"
            metrics[f"{prefix}_coverage"] = float(np.mean(error <= radius))
            metrics[f"{prefix}_mean_width"] = float(2.0 * radius)
            metrics[f"{prefix}_radius"] = radius
    metrics["quantile_key"] = quantile_key
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate calibrated intervals for Chengdu hybrid rollouts.")
    parser.add_argument("--rollouts_csv", required=True)
    parser.add_argument("--model_json", required=True)
    parser.add_argument("--horizons", default="1,6,24,72")
    parser.add_argument("--quantile_key", default="q90")
    parser.add_argument(
        "--output",
        default="results/chengdu_agri_greenhouse_001/physics_v4_greybox_residual/uncertainty_metrics.json",
    )
    args = parser.parse_args()
    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    metrics = evaluate_interval_metrics(
        pd.read_csv(args.rollouts_csv),
        load_residual_model(args.model_json),
        horizons=horizons,
        quantile_key=args.quantile_key,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

