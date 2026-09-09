from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.crop.project_controller_cohort_yield import (
    load_regional_projection_priors,
    prepare_hourly_harvest_trajectory,
)
from experiments.crop.simulate_chengdu_greenlight_harvest_priors import (
    sample_harvest_scenarios,
)


def test_controller_trajectory_is_aggregated_to_mass_conserving_hourly_drivers():
    frame = pd.DataFrame(
        {
            "timestep": np.arange(8),
            "air_temperature": np.arange(8, dtype=float) + 20.0,
            "c_fruit_previous_mg_m2": np.arange(8, dtype=float) * 10.0,
            "c_fruit_mg_m2": (np.arange(8, dtype=float) + 1.0) * 10.0,
            "harvested_dry_matter_mg_m2": np.zeros(8),
            "allocated_fruit_dry_matter_mg_m2": np.ones(8) * 10.0,
            "uBoil": np.zeros(8),
            "uCO2": np.zeros(8),
        }
    )

    hourly = prepare_hourly_harvest_trajectory(
        frame,
        season_id="2024_spring",
        growth_year=2024,
        start_day=60,
    )

    assert len(hourly) == 2
    assert hourly["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert hourly["allocated_fruit_dry_matter_mg_m2"].tolist() == [40.0, 40.0]
    assert hourly["c_fruit_previous_mg_m2"].tolist() == [0.0, 40.0]
    assert hourly["c_fruit_mg_m2"].tolist() == [40.0, 80.0]
    assert hourly["air_temperature"].tolist() == [21.5, 25.5]
    assert hourly["season_id"].eq("2024_spring").all()


def test_regional_projection_prior_overrides_only_maturity_and_status(tmp_path):
    path = tmp_path / "regional_priors.json"
    path.write_text(
        json.dumps(
            {
                "evidence_class": "external_transfer_prior",
                "target_eligible": False,
                "calibration_status": "regional_external_transfer_not_target_validated",
                "source_ids": ["chengdu_timing", "chengdu_yield"],
                "maturity_thermal_time": {
                    "low_deg_day": 720.0,
                    "high_deg_day": 980.0,
                    "season_intervals_deg_day": {
                        "spring": {
                            "low_deg_day": 500.0,
                            "central_deg_day": 600.0,
                            "high_deg_day": 700.0,
                        },
                        "autumn": {
                            "low_deg_day": 900.0,
                            "central_deg_day": 1000.0,
                            "high_deg_day": 1100.0,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    priors = load_regional_projection_priors(path)

    assert priors["maturity_thermal_time_deg_day"] == (720.0, 980.0)
    assert priors["maturity_thermal_time_deg_day_by_season"] == {
        "autumn": (900.0, 1100.0),
        "spring": (500.0, 700.0),
    }
    assert priors["pick_interval_days"] == (5, 5)
    assert priors["source_dois"]
    assert priors["source_path"] == path.as_posix()
    assert (
        priors["transfer_calibration_status"]
        == "regional_external_transfer_not_target_validated"
    )


def test_scenario_sampling_uses_matching_season_maturity_interval():
    priors = {
        "maturity_thermal_time_deg_day": (100.0, 2000.0),
        "maturity_thermal_time_deg_day_by_season": {
            "spring": (500.0, 600.0),
            "autumn": (900.0, 1000.0),
        },
        "pick_interval_days": (5, 5),
        "transfer_calibration_status": "season_adaptive_test",
    }

    samples = sample_harvest_scenarios(
        ["spring", "autumn"],
        draws_per_climate=8,
        seed=12,
        process_priors=priors,
    )

    spring = samples.loc[samples["season_id"].eq("spring")]
    autumn = samples.loc[samples["season_id"].eq("autumn")]
    assert spring["maturity_thermal_time_deg_day"].between(500.0, 600.0).all()
    assert autumn["maturity_thermal_time_deg_day"].between(900.0, 1000.0).all()
    assert samples["maturity_prior_scope"].eq("season_specific").all()


def test_regional_projection_prior_rejects_target_eligible_payload(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "evidence_class": "external_transfer_prior",
                "target_eligible": True,
                "maturity_thermal_time": {
                    "low_deg_day": 720.0,
                    "high_deg_day": 980.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target-ineligible"):
        load_regional_projection_priors(path)
