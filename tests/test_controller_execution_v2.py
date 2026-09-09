from __future__ import annotations

from time import perf_counter, sleep

import numpy as np

from experiments.controllers.execution import ControllerExecutorV2, execute_controller_v2
from experiments.controllers.benchmark_protocol import build_environment, load_benchmark_config
from experiments.controllers.train_benchmark_rl import execute_rl_action_v2
from experiments.controllers.train_benchmark_rl import ProtectedEvalCallback
from glassgym.components.rule_based import RuleBasedController


def test_controller_timeout_uses_rule_fallback_without_waiting_for_primary():
    fallback = np.array([0.0, 0.0, 0.5, 0.1, 0.0, 0.0], dtype=np.float32)

    def slow_controller():
        sleep(0.5)
        return np.ones(6, dtype=np.float32)

    started = perf_counter()
    result = execute_controller_v2(
        slow_controller,
        fallback=lambda: fallback,
        final_fallback=np.zeros(6, dtype=np.float32),
        timeout_seconds=0.02,
        expected_shape=(6,),
    )

    assert perf_counter() - started < 0.25
    np.testing.assert_array_equal(result.action, fallback)
    assert result.fallback_used is True
    assert result.failure_kind == "timeout"


def test_timeout_latches_fallback_and_does_not_reinvoke_primary():
    calls = 0

    def slow_controller():
        nonlocal calls
        calls += 1
        sleep(0.2)
        return np.ones(6)

    executor = ControllerExecutorV2(timeout_seconds=0.01, expected_shape=(6,))
    first = executor.execute(
        slow_controller,
        fallback=lambda: np.zeros(6),
        final_fallback=np.zeros(6),
    )
    second = executor.execute(
        slow_controller,
        fallback=lambda: np.zeros(6),
        final_fallback=np.zeros(6),
    )

    assert calls == 1
    assert first.fallback_duration_steps == 1
    assert second.failure_kind == "timeout_latched"
    assert second.fallback_duration_steps == 2


def test_controller_exception_and_nonfinite_output_use_rule_fallback():
    fallback = np.arange(6, dtype=np.float32) / 10.0

    def raises():
        raise RuntimeError("controller failed")

    exception_result = execute_controller_v2(
        raises,
        fallback=lambda: fallback,
        final_fallback=np.zeros(6, dtype=np.float32),
        timeout_seconds=0.1,
        expected_shape=(6,),
    )
    nonfinite_result = execute_controller_v2(
        lambda: np.full(6, np.nan, dtype=np.float32),
        fallback=lambda: fallback,
        final_fallback=np.zeros(6, dtype=np.float32),
        timeout_seconds=0.1,
        expected_shape=(6,),
    )

    np.testing.assert_array_equal(exception_result.action, fallback)
    np.testing.assert_array_equal(nonfinite_result.action, fallback)
    assert exception_result.failure_kind == "exception"
    assert nonfinite_result.failure_kind == "nonfinite"


def test_conversion_exception_and_slow_conversion_remain_inside_boundary():
    class Unconvertible:
        def __array__(self, dtype=None):
            raise TypeError("cannot convert")

    class SlowArray:
        def __array__(self, dtype=None):
            sleep(0.2)
            return np.ones(6, dtype=dtype)

    fallback = np.zeros(6, dtype=np.float32)
    invalid = execute_controller_v2(
        lambda: Unconvertible(),
        fallback=lambda: fallback,
        final_fallback=fallback,
        timeout_seconds=0.05,
        expected_shape=(6,),
    )
    started = perf_counter()
    slow = execute_controller_v2(
        lambda: SlowArray(),
        fallback=lambda: fallback,
        final_fallback=fallback,
        timeout_seconds=0.02,
        expected_shape=(6,),
    )

    assert invalid.failure_kind == "exception"
    assert slow.failure_kind == "timeout"
    assert perf_counter() - started < 0.15


def test_invalid_fallback_uses_finite_final_fallback():
    final = np.array([0.0, 0.0, 0.4, 0.0, 0.0, 0.0], dtype=np.float32)
    result = execute_controller_v2(
        lambda: np.full(6, np.nan),
        fallback=lambda: np.ones(5),
        final_fallback=final,
        timeout_seconds=0.1,
        expected_shape=(6,),
    )

    np.testing.assert_array_equal(result.action, final)
    assert result.fallback_used is True
    assert result.fallback_failure_kind == "shape"


def test_vectorized_rl_nonfinite_action_uses_rule_baseline():
    class NonfiniteModel:
        def predict(self, observations, deterministic=True):
            return np.full((1, 4), np.nan, dtype=np.float32), None

    config = load_benchmark_config("smoke")
    env = build_environment(config, config.test_start_day, episode_days=1)
    env.reset(seed=0)
    fallback = RuleBasedController(
        lamps_on=0,
        lamps_off=18,
        lamps_day_start=-1,
        lamps_day_stop=366,
        lamps_off_sun=400,
        lamp_rad_sum_limit=10,
        temp_setpoint_day=19.5,
        temp_setpoint_night=16.5,
        heat_correction=0,
        heat_deadzone=5,
        co2_day=800,
        vent_heat_Pband=4,
        rh_max=85,
        mech_dehumid_Pband=2,
        vent_rh_Pband=5,
        t_vent_off=1,
        vent_cold_Pband=-1,
        thScrSpDay=5,
        thScrSpNight=10,
        thScrPband=-1,
        thScrDeadZone=4,
        thScrRh=-2,
        thScrRhPband=2,
        lampExtraHeat=2,
        blScrExtraRh=100,
        rhMax=85,
        tHeatBand=-1,
        co2Band=-100,
        useBlScr=1,
    )
    try:
        result = execute_rl_action_v2(
            model=NonfiniteModel(),
            observations=np.zeros((1, 1), dtype=np.float32),
            raw_env=env,
            fallback_controller=fallback,
            timeout_seconds=0.1,
        )
    finally:
        env.close()

    assert result.action.shape == (1, 4)
    assert np.isfinite(result.action).all()
    assert result.fallback_used is True
    assert result.failure_kind == "nonfinite"


def test_validation_events_receive_fresh_timeout_latches():
    class ActionSpace:
        shape = (4,)

    class Env:
        action_space = ActionSpace()

    callback = object.__new__(ProtectedEvalCallback)
    callback.timeout_seconds = 0.1

    first = callback._new_controller_executor(Env())
    second = callback._new_controller_executor(Env())

    assert first is not second
