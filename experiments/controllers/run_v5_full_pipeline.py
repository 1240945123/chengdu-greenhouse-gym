from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import pandas as pd

from experiments.controllers.evaluate_v5_full_training import (
    run_full_season_evaluation,
    run_validation_selection,
)
from experiments.controllers.run_v5_full_training import (
    DEFAULT_OUTPUT_ROOT,
    run_full_training,
)
from experiments.controllers.v5_full_training_protocol import (
    DEFAULT_CONFIG_PATH,
    load_v5_full_training_protocol,
    should_continue_training,
)


def pipeline_stage_names() -> tuple[str, ...]:
    return (
        "minimum_training",
        "minimum_validation",
        "extended_training",
        "final_validation",
        "temporal_holdout_evaluation",
        "six_season_evaluation",
    )


def _write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _extension_algorithms(
    summary_path: Path,
    *,
    threshold: float,
    minimum_checkpoint: int,
) -> tuple[str, ...]:
    summary = pd.read_csv(summary_path)
    selected: list[str] = []
    for algorithm, group in summary.loc[summary["feasible"].astype(bool)].groupby(
        "algorithm", sort=True
    ):
        best_by_checkpoint = group.groupby("checkpoint")[
            "mean_cumulative_reward"
        ].max()
        scores = {
            int(step): float(score)
            for step, score in best_by_checkpoint.items()
            if int(step) <= int(minimum_checkpoint)
        }
        if should_continue_training(
            scores, relative_improvement_threshold=float(threshold)
        ):
            selected.append(str(algorithm))
    return tuple(selected)


def run_pipeline(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    max_workers: int = 6,
) -> None:
    protocol = load_v5_full_training_protocol(config_path)
    output = Path(output_root)
    status_path = output / "pipeline_status.json"
    status: dict[str, object] = {
        "schema_version": "chengdu-v5-full-pipeline-v1",
        "status": "running",
        "current_stage": None,
        "completed_stages": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    def begin(stage: str) -> None:
        status["current_stage"] = stage
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_status(status_path, status)
        print(f"pipeline stage started: {stage}", flush=True)

    def complete(stage: str) -> None:
        completed = list(status["completed_stages"])
        if stage not in completed:
            completed.append(stage)
        status["completed_stages"] = completed
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_status(status_path, status)
        print(f"pipeline stage complete: {stage}", flush=True)

    try:
        begin("minimum_training")
        run_full_training(
            config_path=config_path,
            output_root=output,
            max_checkpoint=protocol.minimum_full_checkpoint,
            max_workers=max_workers,
        )
        complete("minimum_training")

        begin("minimum_validation")
        minimum_selection_checkpoints = tuple(
            value
            for value in protocol.checkpoints
            if 20_480 < value <= protocol.minimum_full_checkpoint
        )
        run_validation_selection(
            config_path=config_path,
            output_root=output,
            checkpoints=minimum_selection_checkpoints,
            max_workers=max_workers,
        )
        complete("minimum_validation")

        begin("extended_training")
        extension = _extension_algorithms(
            output / "validation" / "candidate_summary.csv",
            threshold=protocol.plateau_relative_improvement,
            minimum_checkpoint=protocol.minimum_full_checkpoint,
        )
        status["extended_algorithms"] = list(extension)
        if extension:
            run_full_training(
                config_path=config_path,
                output_root=output,
                max_checkpoint=max(protocol.checkpoints),
                max_workers=min(max_workers, len(extension) * len(protocol.seeds)),
                algorithms=extension,
            )
        complete("extended_training")

        begin("final_validation")
        run_validation_selection(
            config_path=config_path,
            output_root=output,
            max_workers=max_workers,
        )
        complete("final_validation")

        begin("temporal_holdout_evaluation")
        run_full_season_evaluation(
            output_root=output,
            season_ids=("2025_autumn",),
            max_workers=max_workers,
        )
        complete("temporal_holdout_evaluation")

        begin("six_season_evaluation")
        run_full_season_evaluation(
            output_root=output,
            max_workers=max_workers,
        )
        complete("six_season_evaluation")

        status["status"] = "complete"
        status["current_stage"] = None
        status["completed_at"] = datetime.now(timezone.utc).isoformat()
        status["updated_at"] = status["completed_at"]
        _write_status(status_path, status)
    except BaseException as exc:
        status["status"] = "failed"
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        status["traceback"] = traceback.format_exc()
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_status(status_path, status)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_pipeline(
        config_path=args.config,
        output_root=args.output_root,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
