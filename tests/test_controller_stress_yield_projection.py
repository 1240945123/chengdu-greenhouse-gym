from __future__ import annotations

import pandas as pd
import pytest

from experiments.crop.project_controller_stress_yield import (
    aggregate_stress_projection,
    expand_heat_stress_samples,
)
from experiments.crop.audit_controller_stress_yield import build_stress_audit


def test_heat_stress_expansion_pairs_three_scenarios_per_cohort_draw():
    samples = pd.DataFrame(
        {
            "scenario_id": ["SIM_A", "SIM_B"],
            "season_id": ["2023_spring", "2023_spring"],
            "cohort_draw": [0, 1],
            "maturity_thermal_time_deg_day": [700.0, 800.0],
            "dry_matter_fraction": [0.08, 0.085],
        }
    )
    stress_scenarios = {
        "sensitive": 0.25,
        "central": 0.50,
        "tolerant": 0.75,
    }

    expanded = expand_heat_stress_samples(
        samples,
        stress_scenarios=stress_scenarios,
        optimal_mean_temperature_c=25.0,
        severe_mean_temperature_c=29.0,
        trailing_window_hours=240,
    )

    assert len(expanded) == 6
    assert expanded.groupby("cohort_draw")["heat_stress_scenario"].nunique().eq(3).all()
    assert expanded.groupby("cohort_draw")["maturity_thermal_time_deg_day"].nunique().eq(1).all()
    assert expanded["scenario_id"].str.contains("__heat_").all()


def test_stress_projection_aggregation_reports_loss_and_adjusted_yield():
    scenarios = pd.DataFrame(
        {
            "algorithm": ["pid"] * 4,
            "seed": [0] * 4,
            "season_id": ["2023_spring"] * 4,
            "heat_stress_scenario": ["central"] * 4,
            "partial_yield_kg_m2": [2.0, 2.2, 2.4, 2.6],
            "reproductive_heat_stress_loss_dry_matter_kg_m2": [0.1, 0.2, 0.3, 0.4],
            "mean_reproductive_heat_retention": [0.8, 0.7, 0.6, 0.5],
            "dry_matter_balance_error_kg_m2": [0.0] * 4,
            "first_harvest_day": [70.0, 71.0, 72.0, 73.0],
        }
    )

    episodes, seasons, algorithms = aggregate_stress_projection(scenarios)

    assert episodes.iloc[0]["projected_harvest_median_kg_m2"] == pytest.approx(2.3)
    assert seasons.iloc[0]["projected_harvest_kg_m2"] == pytest.approx(2.3)
    assert algorithms.iloc[0]["projected_harvest_mean_kg_m2"] == pytest.approx(2.3)
    assert algorithms.iloc[0]["mean_reproductive_heat_retention"] == pytest.approx(0.65)


def test_stress_audit_compares_central_scenario_with_v2_and_checks_pairs():
    scenarios = pd.DataFrame(
        {
            "algorithm": ["pid"] * 3,
            "seed": [0] * 3,
            "season_id": ["2023_spring"] * 3,
            "cohort_draw": [0] * 3,
            "heat_stress_scenario": ["sensitive", "central", "tolerant"],
            "partial_yield_kg_m2": [1.8, 2.2, 2.6],
            "reproductive_heat_stress_loss_dry_matter_kg_m2": [0.3, 0.2, 0.1],
            "mean_reproductive_heat_retention": [0.6, 0.7, 0.8],
            "dry_matter_balance_error_kg_m2": [0.0] * 3,
        }
    )
    seasons = pd.DataFrame(
        {
            "algorithm": ["pid"] * 3,
            "season_id": ["2023_spring"] * 3,
            "heat_stress_scenario": ["sensitive", "central", "tolerant"],
            "projected_harvest_kg_m2": [1.8, 2.2, 2.6],
            "projected_first_harvest_day": [70.0] * 3,
            "reproductive_heat_stress_loss_mean_kg_m2": [0.3, 0.2, 0.1],
            "mean_reproductive_heat_retention": [0.6, 0.7, 0.8],
        }
    )
    v2 = pd.DataFrame(
        {
            "algorithm": ["pid"],
            "season_id": ["2023_spring"],
            "projected_harvest_kg_m2": [2.5],
            "projected_first_harvest_day": [70.0],
        }
    )
    priors = {
        "spring_partial_yield_envelope": {"low_kg_m2": 2.0, "high_kg_m2": 2.8},
        "maturity_thermal_time": {
            "target_first_harvest_day": 70,
            "onset_days": [65, 70, 75],
        },
    }

    audit, by_algorithm, comparison = build_stress_audit(
        priors,
        scenarios,
        seasons,
        v2,
        expected_scenario_count=3,
        expected_controller_episode_count=1,
        expected_stress_scenarios=3,
    )

    assert audit["complete"] is True
    assert audit["paired_stress_scenarios_complete"] is True
    assert audit["target_harvest_validated"] is False
    assert by_algorithm.iloc[0]["central_spring_envelope_coverage_fraction"] == 1.0
    assert comparison.iloc[0]["yield_change_from_v2_fraction"] == pytest.approx(-0.12)


def test_stress_audit_reports_draw_harvest_reversal_without_failing_physics_gate():
    scenarios = pd.DataFrame(
        {
            "algorithm": ["pid"] * 6,
            "seed": [0] * 6,
            "season_id": ["2023_autumn"] * 6,
            "cohort_draw": [0, 0, 0, 1, 1, 1],
            "heat_stress_scenario": [
                "sensitive", "central", "tolerant",
                "sensitive", "central", "tolerant",
            ],
            "partial_yield_kg_m2": [1.0, 0.9, 1.2, 1.0, 1.2, 1.4],
            "reproductive_heat_stress_loss_dry_matter_kg_m2": [
                0.3, 0.2, 0.1, 0.3, 0.2, 0.1,
            ],
            "mean_reproductive_heat_retention": [
                0.6, 0.7, 0.8, 0.6, 0.7, 0.8,
            ],
            "dry_matter_balance_error_kg_m2": [0.0] * 6,
        }
    )
    seasons = pd.DataFrame(
        {
            "algorithm": ["pid"] * 3,
            "season_id": ["2023_autumn"] * 3,
            "heat_stress_scenario": ["sensitive", "central", "tolerant"],
            "projected_harvest_kg_m2": [1.0, 1.05, 1.3],
            "projected_first_harvest_day": [70.0] * 3,
            "reproductive_heat_stress_loss_mean_kg_m2": [0.3, 0.2, 0.1],
            "mean_reproductive_heat_retention": [0.6, 0.7, 0.8],
        }
    )
    v2 = pd.DataFrame(
        {
            "algorithm": ["pid"],
            "season_id": ["2023_autumn"],
            "projected_harvest_kg_m2": [1.2],
            "projected_first_harvest_day": [70.0],
        }
    )
    priors = {
        "spring_partial_yield_envelope": {"low_kg_m2": 2.0, "high_kg_m2": 2.8},
        "maturity_thermal_time": {
            "target_first_harvest_day": 70,
            "onset_days": [65, 70, 75],
        },
    }

    audit, _by_algorithm, _comparison = build_stress_audit(
        priors,
        scenarios,
        seasons,
        v2,
        expected_scenario_count=6,
        expected_controller_episode_count=1,
        expected_stress_scenarios=3,
    )

    assert audit["complete"] is True
    assert audit["draw_harvest_monotonicity_violation_count"] == 1
    assert audit["season_aggregate_yield_monotonic_with_heat_tolerance"] is True
    assert audit["reproductive_retention_monotonic_with_heat_tolerance"] is True
