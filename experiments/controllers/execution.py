from __future__ import annotations

from dataclasses import dataclass, replace
from queue import Empty, Queue
from threading import Thread
from time import perf_counter
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ControllerExecutionResultV2:
    action: np.ndarray
    fallback_used: bool
    failure_kind: str | None
    failure_message: str | None
    fallback_failure_kind: str | None
    elapsed_seconds: float
    fallback_duration_steps: int


def _call_with_timeout(
    function: Callable[[], object], timeout_seconds: float
) -> tuple[object | None, str | None, str | None]:
    outcomes: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcomes.put((True, function()))
        except BaseException as exc:
            outcomes.put((False, exc))

    worker = Thread(target=invoke, daemon=True)
    worker.start()
    try:
        succeeded, payload = outcomes.get(timeout=timeout_seconds)
    except Empty:
        return None, "timeout", f"controller exceeded {timeout_seconds:g} seconds"
    if not succeeded:
        return None, "exception", f"{type(payload).__name__}: {payload}"
    return payload, None, None


def _validated_action(
    value: object, expected_shape: tuple[int, ...]
) -> tuple[np.ndarray | None, str | None]:
    action = np.asarray(value, dtype=np.float32)
    if action.shape != expected_shape:
        return None, "shape"
    if not np.all(np.isfinite(action)):
        return None, "nonfinite"
    return action, None


def execute_controller_v2(
    primary: Callable[[], object],
    *,
    fallback: Callable[[], object],
    final_fallback: np.ndarray,
    timeout_seconds: float,
    expected_shape: tuple[int, ...],
) -> ControllerExecutionResultV2:
    """Execute one proposal under a wall-clock deadline and finite-value contract."""
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    final_action, final_error = _validated_action(final_fallback, expected_shape)
    if final_error is not None:
        raise ValueError(f"final_fallback violates controller contract: {final_error}")

    started = perf_counter()
    validated, failure_kind, failure_message = _call_with_timeout(
        lambda: _validated_action(primary(), expected_shape),
        float(timeout_seconds),
    )
    if failure_kind is None:
        action, validation_error = validated
        if validation_error is None:
            return ControllerExecutionResultV2(
                action=action,
                fallback_used=False,
                failure_kind=None,
                failure_message=None,
                fallback_failure_kind=None,
                elapsed_seconds=perf_counter() - started,
                fallback_duration_steps=0,
            )
        failure_kind = validation_error
        failure_message = f"primary proposal violates controller contract: {validation_error}"

    fallback_validated, fallback_call_error, _ = _call_with_timeout(
        lambda: _validated_action(fallback(), expected_shape),
        float(timeout_seconds),
    )
    fallback_action = None
    fallback_error = fallback_call_error
    if fallback_call_error is None:
        fallback_action, fallback_error = fallback_validated

    return ControllerExecutionResultV2(
        action=fallback_action if fallback_action is not None else final_action,
        fallback_used=True,
        failure_kind=failure_kind,
        failure_message=failure_message,
        fallback_failure_kind=fallback_error,
        elapsed_seconds=perf_counter() - started,
        fallback_duration_steps=1,
    )


class ControllerExecutorV2:
    """Stateful execution boundary that permanently isolates a timed-out controller."""

    def __init__(self, *, timeout_seconds: float, expected_shape: tuple[int, ...]) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.expected_shape = expected_shape
        self._timeout_latched = False
        self._fallback_duration_steps = 0

    def execute(
        self,
        primary: Callable[[], object],
        *,
        fallback: Callable[[], object],
        final_fallback: np.ndarray,
    ) -> ControllerExecutionResultV2:
        if self._timeout_latched:
            fallback_result = execute_controller_v2(
                fallback,
                fallback=fallback,
                final_fallback=final_fallback,
                timeout_seconds=self.timeout_seconds,
                expected_shape=self.expected_shape,
            )
            self._fallback_duration_steps += 1
            return replace(
                fallback_result,
                fallback_used=True,
                failure_kind="timeout_latched",
                failure_message="primary controller disabled after an earlier timeout",
                fallback_duration_steps=self._fallback_duration_steps,
            )

        result = execute_controller_v2(
            primary,
            fallback=fallback,
            final_fallback=final_fallback,
            timeout_seconds=self.timeout_seconds,
            expected_shape=self.expected_shape,
        )
        if result.fallback_used:
            self._fallback_duration_steps += 1
            result = replace(result, fallback_duration_steps=self._fallback_duration_steps)
        else:
            self._fallback_duration_steps = 0
        if result.failure_kind == "timeout":
            self._timeout_latched = True
        return result
