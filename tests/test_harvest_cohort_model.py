from __future__ import annotations

import math

import pandas as pd

from glassgym.models.harvest.cohort_model import (
    HarvestCohortModel,
    HarvestCohortParameters,
    is_scheduled_pick,
    scheduled_pick_between,
)


def _parameters(**overrides: float) -> HarvestCohortParameters:
    values = {
        "base_temperature_c": 10.0,
        "maturity_thermal_time_deg_day": 20.0,
        "dry_matter_fraction": 0.08,
        "minimum_pick_fresh_kg_m2": 0.0,
    }
    values.update(overrides)
    return HarvestCohortParameters(**values)


def test_allocation_is_conserved_between_standing_and_harvested_mass():
    model = HarvestCohortModel(_parameters())

    for _ in range(5):
        model.step(
            net_fruit_dry_matter_change_kg_m2=0.04,
            air_temperature_c=22.0,
            dt_hours=24.0,
            pick=False,
        )
    result = model.step(
        net_fruit_dry_matter_change_kg_m2=0.0,
        air_temperature_c=22.0,
        dt_hours=24.0,
        pick=True,
    )

    assert math.isclose(
        model.cumulative_net_fruit_growth_dry_matter_kg_m2,
        model.standing_dry_matter_kg_m2
        + model.cumulative_harvested_dry_matter_kg_m2,
        abs_tol=1e-12,
    )
    assert result.harvested_fresh_kg_m2 >= 0.0


def test_temperature_controls_maturity_through_thermal_time():
    cold = HarvestCohortModel(_parameters())
    warm = HarvestCohortModel(_parameters())
    cold.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)
    warm.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)

    cold.step(0.0, air_temperature_c=10.0, dt_hours=48.0, pick=False)
    warm.step(0.0, air_temperature_c=20.0, dt_hours=48.0, pick=False)

    assert cold.mature_dry_matter_kg_m2 == 0.0
    assert warm.mature_dry_matter_kg_m2 == 0.08


def test_thermal_time_is_capped_at_tomato_upper_threshold():
    hot = HarvestCohortModel(
        _parameters(
            maturity_thermal_time_deg_day=21.0,
            upper_temperature_c=30.0,
        )
    )
    cutoff = HarvestCohortModel(
        _parameters(
            maturity_thermal_time_deg_day=21.0,
            upper_temperature_c=30.0,
        )
    )
    hot.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)
    cutoff.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)

    hot.step(0.0, air_temperature_c=40.0, dt_hours=24.0, pick=False)
    cutoff.step(0.0, air_temperature_c=30.0, dt_hours=24.0, pick=False)

    assert hot.mature_dry_matter_kg_m2 == 0.0
    assert hot.mature_dry_matter_kg_m2 == cutoff.mature_dry_matter_kg_m2


def test_pick_removes_only_mature_fruit_and_converts_to_fresh_mass():
    model = HarvestCohortModel(_parameters())
    model.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)
    model.step(0.0, air_temperature_c=20.0, dt_hours=48.0, pick=False)

    result = model.step(0.04, air_temperature_c=10.0, dt_hours=1.0, pick=True)

    assert math.isclose(result.harvested_dry_matter_kg_m2, 0.08)
    assert math.isclose(result.harvested_fresh_kg_m2, 1.0)
    assert math.isclose(model.standing_dry_matter_kg_m2, 0.04)


def test_minimum_pick_mass_defers_small_batches():
    model = HarvestCohortModel(
        _parameters(minimum_pick_fresh_kg_m2=1.1)
    )
    model.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)
    model.step(0.0, air_temperature_c=20.0, dt_hours=48.0, pick=False)

    result = model.step(0.0, air_temperature_c=20.0, dt_hours=1.0, pick=True)

    assert result.harvested_fresh_kg_m2 == 0.0
    assert math.isclose(model.mature_dry_matter_kg_m2, 0.08)


def test_thermal_age_is_invariant_to_time_step_for_existing_fruit():
    daily = HarvestCohortModel(_parameters())
    hourly = HarvestCohortModel(_parameters())
    daily.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)
    hourly.step(0.08, air_temperature_c=10.0, dt_hours=0.0, pick=False)

    daily.step(0.0, air_temperature_c=20.0, dt_hours=48.0, pick=False)
    for _ in range(48):
        hourly.step(0.0, air_temperature_c=20.0, dt_hours=1.0, pick=False)

    assert math.isclose(
        daily.mature_dry_matter_kg_m2,
        hourly.mature_dry_matter_kg_m2,
        abs_tol=1e-12,
    )


def test_continuous_allocation_is_invariant_to_external_time_step():
    long_step = HarvestCohortModel(_parameters())
    short_steps = HarvestCohortModel(_parameters())

    long_step.step(0.08, air_temperature_c=20.0, dt_hours=48.0, pick=False)
    short_steps.step(0.04, air_temperature_c=20.0, dt_hours=24.0, pick=False)
    short_steps.step(0.04, air_temperature_c=20.0, dt_hours=24.0, pick=False)

    assert math.isclose(
        long_step.mature_dry_matter_kg_m2,
        short_steps.mature_dry_matter_kg_m2,
        abs_tol=1e-12,
    )
    assert math.isclose(
        long_step.standing_dry_matter_kg_m2,
        short_steps.standing_dry_matter_kg_m2,
        abs_tol=1e-12,
    )


def test_allocation_is_invariant_between_hourly_and_greenlight_quarter_hour_steps():
    parameters = _parameters(maturity_thermal_time_deg_day=0.2)
    hourly = HarvestCohortModel(parameters)
    quarter_hourly = HarvestCohortModel(parameters)

    hourly.step(0.04, air_temperature_c=20.0, dt_hours=1.0, pick=False)
    for _ in range(4):
        quarter_hourly.step(0.01, air_temperature_c=20.0, dt_hours=0.25, pick=False)

    assert math.isclose(
        hourly.mature_dry_matter_kg_m2,
        quarter_hourly.mature_dry_matter_kg_m2,
        abs_tol=1e-12,
    )


def test_pick_schedule_uses_local_weekday_and_hour():
    schedule = {0, 3}

    assert is_scheduled_pick(
        pd.Timestamp("2026-04-06 08:00:00"),
        weekdays=schedule,
        local_hour=8,
    )
    assert not is_scheduled_pick(
        pd.Timestamp("2026-04-07 08:00:00"),
        weekdays=schedule,
        local_hour=8,
    )
    assert not is_scheduled_pick(
        pd.Timestamp("2026-04-06 09:00:00"),
        weekdays=schedule,
        local_hour=8,
    )
    assert not is_scheduled_pick(
        pd.Timestamp("2026-04-06 08:30:00"),
        weekdays=schedule,
        local_hour=8,
    )


def test_pick_schedule_detects_event_crossed_by_step_in_chengdu_time():
    schedule = {0}

    event = scheduled_pick_between(
        pd.Timestamp("2026-04-05 23:00:00", tz="UTC"),
        pd.Timestamp("2026-04-06 02:00:00", tz="UTC"),
        weekdays=schedule,
        local_hour=8,
        timezone="Asia/Shanghai",
    )

    assert event == pd.Timestamp("2026-04-06 08:00:00", tz="Asia/Shanghai")


def test_pick_schedule_rejects_fractional_hours_and_weekdays():
    try:
        is_scheduled_pick(
            pd.Timestamp("2026-04-06 08:00:00"),
            weekdays={0},
            local_hour=8.5,
        )
    except ValueError as exc:
        assert "integer" in str(exc)
    else:
        raise AssertionError("fractional local_hour must be rejected")


def test_invalid_parameters_and_inputs_are_rejected():
    try:
        _parameters(dry_matter_fraction=0.0)
    except ValueError as exc:
        assert "dry_matter_fraction" in str(exc)
    else:
        raise AssertionError("zero dry-matter fraction must be rejected")

    model = HarvestCohortModel(_parameters())
    try:
        model.step(-0.1, air_temperature_c=20.0, dt_hours=1.0, pick=False)
    except ValueError as exc:
        assert "exceeds standing" in str(exc)
    else:
        raise AssertionError("fruit loss cannot exceed standing fruit mass")


def test_negative_net_fruit_growth_reduces_cohorts_without_breaking_balance():
    model = HarvestCohortModel(_parameters())
    model.step(0.10, air_temperature_c=10.0, dt_hours=0.0, pick=False)

    model.step(-0.02, air_temperature_c=20.0, dt_hours=1.0, pick=False)

    assert math.isclose(model.standing_dry_matter_kg_m2, 0.08)
    assert math.isclose(
        model.cumulative_net_fruit_growth_dry_matter_kg_m2,
        model.standing_dry_matter_kg_m2
        + model.cumulative_harvested_dry_matter_kg_m2,
        abs_tol=1e-12,
    )
