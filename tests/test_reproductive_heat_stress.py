from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.crop.simulate_chengdu_greenlight_harvest_priors import (
    simulate_cohort_scenario,
)
from glassgym.models.harvest.reproductive_stress import (
    ReproductiveHeatStressParameters,
    trailing_heat_retention,
)


def test_heat_retention_is_bounded_continuous_and_monotonic():
    params = ReproductiveHeatStressParameters(
        optimal_mean_temperature_c=25.0,
        severe_mean_temperature_c=29.0,
        floor_retention=0.5,
        trailing_window_hours=1,
    )

    result = trailing_heat_retention(
        np.array([24.0, 25.0, 27.0, 29.0, 31.0]), params
    )

    assert result.tolist() == pytest.approx([1.0, 1.0, 0.75, 0.5, 0.5])
    assert np.all(np.diff(result) <= 0.0)


def test_heat_retention_uses_available_trailing_temperature_window():
    params = ReproductiveHeatStressParameters(
        floor_retention=0.5,
        trailing_window_hours=2,
    )

    result = trailing_heat_retention(np.array([20.0, 30.0, 30.0]), params)

    assert result.tolist() == pytest.approx([1.0, 1.0, 0.5])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"optimal_mean_temperature_c": 29.0, "severe_mean_temperature_c": 25.0},
        {"floor_retention": -0.1},
        {"floor_retention": 1.1},
        {"trailing_window_hours": 0},
    ],
)
def test_heat_stress_parameters_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ReproductiveHeatStressParameters(**kwargs)


def _toy_trajectory(temperature_c: float = 30.0) -> pd.DataFrame:
    timestamps = pd.date_range("2023-03-01", periods=48, freq="h")
    fruit = np.arange(len(timestamps) + 1, dtype=float) * 10_000.0
    return pd.DataFrame(
        {
            "season_id": "2023_spring",
            "timestamp": timestamps,
            "c_fruit_previous_mg_m2": fruit[:-1],
            "c_fruit_mg_m2": fruit[1:],
            "harvested_dry_matter_mg_m2": 0.0,
            "allocated_fruit_dry_matter_mg_m2": 10_000.0,
            "air_temperature": temperature_c,
            "uBoil": 0.0,
            "uCO2": 0.0,
        }
    )


def _sample(**updates: object) -> pd.Series:
    values: dict[str, object] = {
        "scenario_id": "SIM_STRESS",
        "season_id": "2023_spring",
        "climate_scenario": "calibrated",
        "climate_calibration_strength": 1.0,
        "base_temperature_c": 10.0,
        "maturity_thermal_time_deg_day": 10_000.0,
        "dry_matter_fraction": 0.08,
        "minimum_pick_fresh_kg_m2": 0.0,
        "pick_interval_days": 4,
        "pick_hour": 8,
        "initial_fruit_maturity_fraction": 0.0,
    }
    values.update(updates)
    return pd.Series(values)


def test_missing_heat_stress_configuration_reproduces_no_stress_balance():
    _drivers, _events, summary = simulate_cohort_scenario(
        _toy_trajectory(), _sample()
    )

    assert summary["reproductive_heat_stress_loss_dry_matter_kg_m2"] == 0.0
    assert summary["dry_matter_balance_error_kg_m2"] == pytest.approx(0.0, abs=1e-12)


def test_heat_stress_rejects_allocation_and_preserves_mass_balance():
    drivers, _events, summary = simulate_cohort_scenario(
        _toy_trajectory(),
        _sample(
            heat_stress_scenario="central",
            heat_stress_optimal_mean_temperature_c=25.0,
            heat_stress_severe_mean_temperature_c=29.0,
            heat_stress_floor_retention=0.5,
            heat_stress_trailing_window_hours=24,
        ),
    )

    assert drivers["reproductive_heat_retention"].eq(0.5).all()
    assert summary["gross_fruit_allocation_dry_matter_kg_m2"] == pytest.approx(0.48)
    assert summary["reproductive_heat_stress_loss_dry_matter_kg_m2"] == pytest.approx(0.24)
    assert summary["retained_fruit_allocation_dry_matter_kg_m2"] == pytest.approx(0.24)
    assert summary["dry_matter_balance_error_kg_m2"] == pytest.approx(0.0, abs=1e-12)

