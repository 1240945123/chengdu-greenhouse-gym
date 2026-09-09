import numpy as np
import pytest

from glassgym.components.rewards import ChengduClimateReward
from glassgym.core.types import RewardContext


def make_context(
    *,
    temp: float = 24.0,
    rh: float = 72.5,
    co2: float = 500.0,
    hour: float = 12.0,
    u: np.ndarray | None = None,
    previous_u: np.ndarray | None = None,
) -> RewardContext:
    controls = np.zeros(8, dtype=np.float64) if u is None else np.asarray(u, dtype=np.float64)
    old_controls = controls if previous_u is None else np.asarray(previous_u, dtype=np.float64)
    return RewardContext(
        t=0,
        dt=900,
        Np=0,
        x_prev=np.zeros(28, dtype=np.float64),
        x=np.zeros(28, dtype=np.float64),
        u=controls,
        p=np.zeros(208, dtype=np.float64),
        d=np.zeros((1, 10), dtype=np.float64),
        hour_of_day=hour,
        day_of_year=100.0,
        obs={
            "IndoorClimateObservations": np.array([co2, temp, rh], dtype=np.float64),
            "ControlObservations": old_controls,
        },
        constraints_low=np.array([300.0, 15.0, 50.0]),
        constraints_high=np.array([1600.0, 34.0, 85.0]),
        u_prev=old_controls,
    )


def make_reward() -> ChengduClimateReward:
    return ChengduClimateReward(
        dt=900,
        p=np.zeros(208),
        temp_day_low=20.0,
        temp_day_high=28.0,
        temp_night_low=16.0,
        temp_night_high=24.0,
        rh_low=60.0,
        rh_high=85.0,
        day_start=6.0,
        day_end=20.0,
        temperature_weight=1.0,
        humidity_weight=1.0,
        lamp_weight=0.1,
        effort_weight=0.2,
        action_change_weight=0.3,
    )


def test_reward_is_zero_inside_comfort_with_idle_controls():
    reward, info = make_reward().compute_reward(make_context())

    assert reward == pytest.approx(0.0)
    assert info["temperature_penalty"] == pytest.approx(0.0)
    assert info["humidity_penalty"] == pytest.approx(0.0)
    assert info["temperature_target"] == pytest.approx(24.0)
    assert info["humidity_target"] == pytest.approx(72.5)
    assert info["temperature_low"] == pytest.approx(20.0)
    assert info["temperature_high"] == pytest.approx(28.0)
    assert info["humidity_low"] == pytest.approx(60.0)
    assert info["humidity_high"] == pytest.approx(85.0)


def test_reward_reports_scaled_climate_and_allowed_control_penalties():
    controls = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 1.0, 0.0])
    previous = np.zeros(8)
    reward, info = make_reward().compute_reward(
        make_context(temp=30.0, rh=90.0, u=controls, previous_u=previous)
    )

    assert info["temperature_penalty"] == pytest.approx(0.2)
    assert info["humidity_penalty"] == pytest.approx(0.125)
    assert info["lamp_penalty"] == pytest.approx(0.1)
    assert info["effort_penalty"] == pytest.approx(0.125)
    assert info["action_change_penalty"] == pytest.approx(0.1875)
    assert reward == pytest.approx(-0.7375)


def test_reward_excludes_heating_and_co2_actions_and_state():
    reward_a, _ = make_reward().compute_reward(
        make_context(temp=30.0, rh=90.0, co2=300.0, u=np.zeros(8))
    )
    reward_b, _ = make_reward().compute_reward(
        make_context(
            temp=30.0,
            rh=90.0,
            co2=1600.0,
            u=np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )

    assert reward_a == pytest.approx(reward_b)


def test_reward_uses_night_temperature_band():
    reward, info = make_reward().compute_reward(make_context(temp=26.0, hour=2.0))

    assert info["temperature_target"] == pytest.approx(20.0)
    assert info["temperature_penalty"] == pytest.approx(0.2)
    assert reward < 0.0
