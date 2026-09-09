from __future__ import annotations

from typing import Any


def append_environment_failure_row(
    rows: list[dict[str, Any]],
    info: dict[str, Any],
    *,
    timestep: int,
) -> None:
    if not rows:
        raise RuntimeError(
            "environment failed before the evaluator recorded a valid transition"
        )
    row = rows[-1].copy()
    row.update(
        {
            "timestep": int(timestep),
            "reward": float(info["reward"]),
            "terminated": False,
            "truncated": True,
            "environment_failure": True,
            "environment_failure_kind": str(info.get("failure_kind", "unknown")),
            "environment_failure_phase": str(info.get("failure_phase", "unknown")),
            "environment_failure_exception": str(
                info.get("failure_exception", "unknown")
            ),
            "environment_failure_message": str(info.get("failure_message", "")),
            "environment_failure_timestep": int(
                info.get("failure_timestep", timestep)
            ),
        }
    )
    rows.append(row)
