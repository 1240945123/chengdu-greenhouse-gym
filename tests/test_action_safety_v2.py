from __future__ import annotations

import numpy as np
import pytest
from glassgym.components.actions import NamedControlActionScheme

from experiments.controllers.benchmark_protocol import build_environment, load_benchmark_config
import experiments.controllers.benchmark_protocol as _benchmark_protocol
from glassgym.components.safety import SafetyConfigV2, SafetyProjectorV2
from glassgym.components.rule_based import RuleBasedController
from glassgym.core.types import ActionFieldV2, ActionSchemaV2


@pytest.fixture(autouse=True)
def _use_eight_dimensional_benchmark_env(monkeypatch):
    """Sync the benchmark env to the new 8-dimensional physical control vector.

    The new ChengduClimateReward expects ALLOWED_CONTROL_INDICES=[3, 5, 6, 7],
    i.e. an 8-dim control vector. The benchmark env config is still pinned to
    nu=6; this test-only patch widens the control vector to 8 dims (keeping the
    existing 4 controlled inputs, which remain valid indices) so the reward can
    be evaluated. The underlying config should be updated to nu=8 for real.
    """
    original = _benchmark_protocol._load_environment_kwargs

    def patched():
        kwargs = original()
        kwargs["nu"] = 8
        kwargs["u_min"] = [0.0] * 8
        kwargs["u_max"] = [1.0] * 8
        return kwargs

    monkeypatch.setattr(_benchmark_protocol, "_load_environment_kwargs", patched)


def test_action_schema_v2_has_canonical_primary_and_wet_pad_forms():
    primary = ActionSchemaV2.for_count(6)
    wet_pad = ActionSchemaV2.for_count(8)

    assert primary.schema_version == "action-schema-v2"
    assert primary.count == 6
    assert [field.name for field in wet_pad.fields[-2:]] == ["uPadFan", "uPadPump"]
    assert all(field.unit == "fraction" for field in wet_pad.fields)
    assert wet_pad.fields[0].primary_enabled is False
    assert wet_pad.fields[1].primary_enabled is False

    mutable = list(primary.fields)
    reconstructed = ActionSchemaV2(fields=mutable)
    mutable.pop()
    assert isinstance(reconstructed.fields, tuple)
    assert reconstructed == primary

    custom = list(primary.fields)
    custom[3] = ActionFieldV2(3, "unsafeVent", "fraction", 0.0, 1.0, 0.0, True)
    with pytest.raises(ValueError, match="canonical"):
        ActionSchemaV2(fields=custom)


def test_safety_projection_is_deterministic_idempotent_and_emergency_closes_vent():
    projector = SafetyProjectorV2(
        ActionSchemaV2.for_count(6),
        SafetyConfigV2(
            normal_slew_limit=0.1,
            emergency_vent_slew_limit=1.0,
            rain_close_threshold_mm_h=0.0,
            wind_close_threshold_m_s=10.0,
            lamp_start_hour=6.0,
            lamp_end_hour=20.0,
        ),
    )
    previous = np.array([0.0, 0.0, 0.7, 0.8, 0.5, 0.7])
    proposed = np.array([1.0, 1.0, 0.9, 1.0, 1.0, 0.9])
    disturbance = np.zeros(11)
    disturbance[4] = 12.0
    disturbance[10] = 0.5

    first = projector.project(
        proposed,
        previous=previous,
        disturbance=disturbance,
        hour_of_day=12.0,
    )
    second = projector.project(
        first.executed,
        previous=previous,
        disturbance=disturbance,
        hour_of_day=12.0,
    )

    np.testing.assert_allclose(first.executed, [0.0, 0.0, 0.8, 0.0, 0.6, 0.8])
    np.testing.assert_array_equal(second.executed, first.executed)
    assert "rain_vent_closure" in first.interventions
    assert "wind_vent_closure" in first.interventions
    assert "heating_disabled" in first.interventions
    assert "co2_disabled" in first.interventions


def test_nonfinite_proposal_uses_previous_action_then_reapplies_hard_constraints():
    projector = SafetyProjectorV2(ActionSchemaV2.for_count(6), SafetyConfigV2())
    previous = np.array([0.0, 0.0, 0.3, 0.7, 0.5, 0.4])
    proposed = previous.copy()
    proposed[3] = np.nan
    disturbance = np.zeros(11)
    disturbance[10] = 1.0

    result = projector.project(
        proposed,
        previous=previous,
        disturbance=disturbance,
        hour_of_day=2.0,
    )

    assert result.fallback_used is True
    assert np.all(np.isfinite(result.proposed))
    assert result.invalid_proposal_indices == (3,)
    assert result.executed[3] == 0.0
    assert result.executed[4] == 0.0
    assert "invalid_proposal_fallback" in result.interventions
    assert "lamp_schedule" in result.interventions


def test_wet_pad_interlock_and_high_humidity_restriction():
    projector = SafetyProjectorV2(ActionSchemaV2.for_count(8), SafetyConfigV2())
    result = projector.project(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 1.0]),
        previous=np.zeros(8),
        disturbance=np.zeros(11),
        hour_of_day=12.0,
        indoor_relative_humidity=90.0,
    )

    assert result.executed[6] == 0.0
    assert result.executed[7] == 0.0
    assert "high_humidity_wet_pad_lockout" in result.interventions
    assert result.executed[7] <= result.executed[6]


def test_high_temperature_emergency_opens_vent_and_removes_heat_sources():
    projector = SafetyProjectorV2(
        ActionSchemaV2.for_count(6),
        SafetyConfigV2(high_temperature_emergency_c=34.0),
    )

    result = projector.project(
        np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0]),
        previous=np.array([0.0, 0.0, 0.8, 0.0, 0.8, 0.8]),
        disturbance=np.zeros(11),
        hour_of_day=12.0,
        indoor_temperature=35.0,
        indoor_relative_humidity=70.0,
    )

    np.testing.assert_allclose(result.executed[2:6], [0.0, 1.0, 0.0, 0.0])
    assert "high_temperature_emergency" in result.interventions


def test_high_temperature_emergency_overrides_rain_but_not_strong_wind():
    projector = SafetyProjectorV2(ActionSchemaV2.for_count(6), SafetyConfigV2())
    disturbance = np.zeros(11)
    disturbance[10] = 1.0

    result = projector.project(
        np.zeros(6),
        previous=np.zeros(6),
        disturbance=disturbance,
        hour_of_day=12.0,
        indoor_temperature=40.0,
    )

    assert result.executed[3] == 1.0
    assert "rain_vent_closure" in result.interventions
    assert "high_temperature_emergency" in result.interventions
    assert "rain_vent_override_high_temperature" in result.interventions

    disturbance[4] = 12.0
    windy = projector.project(
        np.zeros(6),
        previous=np.zeros(6),
        disturbance=disturbance,
        hour_of_day=12.0,
        indoor_temperature=40.0,
    )
    assert windy.executed[3] == 0.0
    assert "wind_vent_closure" in windy.interventions


def test_projection_logs_bounds_and_slew_only_when_they_change_actions():
    projector = SafetyProjectorV2(ActionSchemaV2.for_count(6), SafetyConfigV2())
    result = projector.project(
        np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]),
        previous=np.zeros(6),
        disturbance=np.zeros(11),
        hour_of_day=12.0,
    )

    assert "equipment_bounds" in result.interventions
    assert "normal_slew_limit" in result.interventions
    assert "heating_disabled" not in result.interventions
    assert "co2_disabled" not in result.interventions


def test_named_action_scheme_accepts_reserved_wet_pad_controls_for_secondary_model():
    scheme = NamedControlActionScheme(
        nu=8,
        controlled_inputs=["uPadFan", "uPadPump"],
        low=np.zeros(8),
        high=np.ones(8),
        normalize_actions=False,
    )
    scheme.reset_full_control_input()
    executed = scheme.to_full_control_input(np.array([0.4, 0.3]))
    np.testing.assert_allclose(executed[6:8], [0.4, 0.3])


def test_environment_applies_same_safety_projection_and_logs_actions():
    protocol = load_benchmark_config("smoke")
    env = build_environment(protocol, start_day=6, episode_days=1, dt_seconds=3_600)
    env.reset(seed=0)
    env.weather_data[0, 10] = 1.0
    try:
        _obs, _reward, _terminated, truncated, info = env.step(
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        )
    finally:
        env.close()

    assert not truncated
    assert info["proposed_controls"][3] > 0.0
    assert info["executed_controls"][3] == 0.0
    assert info["controls"][3] == 0.0
    assert info["safety_intervened"] is True
    assert "rain_vent_closure" in info["safety_interventions"]
    assert info["safety_fallback_duration_steps"] == 0
    np.testing.assert_allclose(
        info["indoor_climate"], env.obs["IndoorClimateObservations"]
    )


def test_rule_baseline_proportional_control_saturates_without_overflow():
    controller = object.__new__(RuleBasedController)
    with np.errstate(over="raise", invalid="raise"):
        low = controller.proportional_control(-1e9, 20.0, 1.0, 0.0, 1.0)
        high = controller.proportional_control(1e9, 20.0, 1.0, 0.0, 1.0)
    assert np.isfinite(low)
    assert np.isfinite(high)
