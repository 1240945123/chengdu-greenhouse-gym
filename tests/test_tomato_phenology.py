from __future__ import annotations

import pandas as pd
import pytest
import math

from common.tomato_phenology import (
    derive_initial_thermal_sum_scenarios,
    evaluate_temperature_driven_trusses,
)


def test_temperature_driven_trusses_integrates_hourly_rate_and_scores_replicates():
    climate = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01", periods=49, freq="1h", tz="Asia/Shanghai"
            ),
            "air_temperature": 20.0,
        }
    )
    observations = pd.DataFrame(
        {
            "observation_date": ["2026-04-02", "2026-04-02", "2026-04-03"],
            "fruit_truss_count": [0.25, 0.35, 0.60],
        }
    )

    metrics, aligned = evaluate_temperature_driven_trusses(
        climate,
        observations,
        baseline_timestamp="2026-04-01",
    )

    daily_rate = -0.2903 + 0.1454 * math.log(20.0)
    assert aligned["predicted_fruit_truss_count"].tolist() == pytest.approx(
        [daily_rate, 2.0 * daily_rate]
    )
    assert aligned["observed_mean_fruit_truss_count"].tolist() == pytest.approx(
        [0.30, 0.60]
    )
    assert metrics["observation_date_count"] == 2
    assert metrics["replicate_count"] == 3
    assert metrics["model"] == "de_koning_1994_temperature_truss_rate"


def test_temperature_driven_trusses_rejects_observations_outside_climate():
    climate = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01", periods=25, freq="1h", tz="Asia/Shanghai"
            ),
            "air_temperature": 20.0,
        }
    )
    observations = pd.DataFrame(
        {"observation_date": ["2026-04-03"], "fruit_truss_count": [1.0]}
    )

    with pytest.raises(ValueError, match="outside the climate interval"):
        evaluate_temperature_driven_trusses(
            climate,
            observations,
            baseline_timestamp="2026-04-01",
        )


def test_initial_thermal_sum_scenarios_use_only_baseline_trusses_and_early_temperature():
    baseline = pd.DataFrame(
        {
            "observation_date": ["2026-03-30"] * 3,
            "flower_truss_count": [1.5, 2.0, 2.5],
        }
    )
    climate = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01", periods=7 * 24 + 1, freq="1h", tz="Asia/Shanghai"
            ),
            "air_temperature": [18.0, 20.0, 22.0] * 56 + [20.0],
        }
    )

    result = derive_initial_thermal_sum_scenarios(
        baseline,
        climate,
        baseline_timestamp="2026-03-30",
        temperature_window_days=7,
    )

    scenarios = result["scenarios"]
    assert [row["name"] for row in scenarios] == ["low", "median", "high"]
    values = [row["initial_t_can_sum_deg_day"] for row in scenarios]
    assert 0.0 < values[0] <= values[1] <= values[2]
    assert result["baseline_replicate_count"] == 3
    assert result["temperature_sample_count"] == 169
    assert result["held_out_mass_used"] is False
    assert result["uncertainty_type"] == "structural_sensitivity_not_predictive_interval"
    assert result["temperature_proxy_scope"] == "first_postbaseline_week_not_prebaseline_measurement"


def test_initial_thermal_sum_scenarios_reject_nonpositive_truss_rate():
    baseline = pd.DataFrame(
        {"observation_date": ["2026-03-30"], "flower_truss_count": [2.0]}
    )
    climate = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-04-01", periods=25, freq="1h", tz="Asia/Shanghai"
            ),
            "air_temperature": 5.0,
        }
    )

    with pytest.raises(ValueError, match="positive truss appearance rate"):
        derive_initial_thermal_sum_scenarios(
            baseline,
            climate,
            baseline_timestamp="2026-03-30",
            temperature_window_days=1,
        )
