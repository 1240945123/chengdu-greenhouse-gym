import pandas as pd
import pytest

from experiments.crop.report_chengdu_greenlight_harvest_priors import (
    build_paired_climate_sensitivity,
    build_plausibility_report,
)


def test_plausibility_report_uses_scenario_level_intervals_and_marks_comparison_limits():
    summaries = pd.DataFrame(
        {
            "scenario_id": ["SIM_a", "SIM_b"],
            "season_id": ["2023_spring", "2023_spring"],
            "climate_scenario": ["calibrated_low", "calibrated_high"],
            "event_count": [2, 2],
            "first_harvest_datetime": ["2023-03-11 08:00", "2023-03-13 08:00"],
            "peak_harvest_datetime": ["2023-03-15 08:00", "2023-03-17 08:00"],
            "median_inter_pick_days": [4.0, 6.0],
            "partial_yield_kg_m2": [4.0, 6.0],
            "dry_matter_balance_error_kg_m2": [1e-12, -1e-12],
            "evidence_class": ["simulated_prior", "simulated_prior"],
            "target_eligible": [False, False],
        }
    )
    events = pd.DataFrame(
        {
            "scenario_id": ["SIM_a", "SIM_a", "SIM_b", "SIM_b"],
            "fresh_kg_m2": [1.0, 3.0, 2.0, 4.0],
            "evidence_class": ["simulated_prior"] * 4,
            "target_eligible": [False] * 4,
        }
    )
    trajectories = pd.DataFrame(
        {
            "season_id": ["2023_spring"] * 4,
            "climate_scenario": ["calibrated_low"] * 2 + ["calibrated_high"] * 2,
            "timestamp": pd.to_datetime(
                ["2023-03-01 01:00", "2023-03-01 02:00"] * 2
            ),
            "air_temperature": [20.0, 30.0, 22.0, 24.0],
            "relative_humidity": [70.0, 90.0, 65.0, 75.0],
            "uBoil": [0.0] * 4,
            "uCO2": [0.0] * 4,
            "uVent": [0.0, 1.0, 0.5, 0.5],
            "uLamp": [0.0, 0.0, 1.0, 1.0],
        }
    )
    external = {
        "metrics": {
            "first_harvest_days_from_crop_start": {"estimate": 65.0},
            "mean_batch_kg_m2": {"estimate": 0.61},
            "median_interpick_days": {"estimate": 5.0},
            "seasonal_yield_kg_m2": {"estimate": 14.1},
        }
    }
    standing = pd.DataFrame(
        {"ripe_fruit_fresh_kg_m2": [0.0, 0.2], "target_eligible": [False, False]}
    )

    report, season_table, climate_table = build_plausibility_report(
        summaries, events, trajectories, external, standing
    )

    season = season_table.iloc[0]
    assert season["scenario_count"] == 2
    assert season["partial_yield_median_kg_m2"] == pytest.approx(5.0)
    assert season["partial_yield_ci80_width_kg_m2"] == pytest.approx(1.6)
    assert season["mean_batch_median_kg_m2"] == pytest.approx(2.5)
    assert len(climate_table) == 2
    assert report["comparison_scope"] == "plausibility_only_not_target_accuracy"
    assert report["wur_comparison"]["yield_comparison"] == "partial_120_day_vs_full_external_season_not_directly_comparable"
    assert report["target_validated"] is False
    assert report["quality_gates"]["all_target_ineligible"] is True


def test_paired_climate_sensitivity_compares_same_cohort_draw():
    samples = pd.DataFrame(
        {
            "scenario_id": ["SIM_low", "SIM_mid", "SIM_high"],
            "season_id": ["2023_spring"] * 3,
            "cohort_draw": [0, 0, 0],
            "climate_scenario": ["calibrated_low", "calibrated", "calibrated_high"],
        }
    )
    summaries = pd.DataFrame(
        {
            "scenario_id": ["SIM_low", "SIM_mid", "SIM_high"],
            "partial_yield_kg_m2": [4.0, 5.0, 6.0],
            "event_count": [8, 9, 10],
            "first_harvest_datetime": [
                "2023-05-03 08:00",
                "2023-05-02 08:00",
                "2023-05-01 08:00",
            ],
        }
    )

    paired = build_paired_climate_sensitivity(summaries, samples)

    assert len(paired) == 1
    assert paired.iloc[0]["yield_high_minus_low_kg_m2"] == pytest.approx(2.0)
    assert paired.iloc[0]["event_count_high_minus_low"] == 2
    assert paired.iloc[0]["first_harvest_high_minus_low_days"] == pytest.approx(-2.0)
