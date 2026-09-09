from __future__ import annotations

import pandas as pd

from experiments.reports.audit_controller_and_yield_results import (
    audit_results,
    build_calibration_summary,
)


def _comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "algorithm": ["baseline", "pid", "mpc", "ppo", "sac"],
            "episodes": [1, 1, 1, 5, 5],
            "reward_mean": [-426.0, -421.0, -420.0, -254.0, -236.0],
            "temperature_mae_mean": [12.4, 12.2, 12.3, 7.2, 6.8],
            "humidity_mae_mean": [18.8, 19.8, 19.7, 20.9, 21.1],
            "joint_comfort_fraction_mean": [0.005, 0.0, 0.0, 0.003, 0.002],
            "safety_intervention_fraction_mean": [0.91, 0.86, 0.61, 0.67, 0.68],
            "fresh_yield_kg_m2_mean": [0.0004, 0.0002, 0.0001, 0.0004, 0.0004],
            "simulated_fresh_fruit_production_kg_m2_mean": [-0.05] * 5,
        }
    )


def test_short_single_scenario_benchmark_is_not_yield_evaluable():
    episodes = pd.DataFrame(
        {
            "algorithm": ["baseline", "pid", "mpc", "ppo", "sac"],
            "seed": [0, 0, 0, 0, 0],
            "growth_year": [2025] * 5,
            "start_day": [226] * 5,
            "episode_steps": [384] * 5,
            "episode_complete": [True] * 5,
        }
    )
    manifest = {
        "runs": {
            name: {"0": {"status": "complete"}}
            for name in ("baseline", "pid", "mpc", "ppo", "sac")
        }
    }
    readiness = {
        "target_validated": False,
        "harvest_data": {
            "status": "missing_target_harvest_events",
            "target_harvest_event_count": 0,
        },
        "blocking_reasons": ["missing_target_harvest_events"],
    }
    plausibility = {
        "season_count": 6,
        "scenario_count": 72,
        "quality_gates": {
            "maximum_abs_mass_balance_error_kg_m2": 1e-12,
            "all_target_ineligible": True,
        },
    }

    audit = audit_results(
        comparison=_comparison(),
        episodes=episodes,
        manifest=manifest,
        readiness=readiness,
        plausibility=plausibility,
        episode_days=4,
    )

    assert audit["controller_experiment"]["training_complete"] is True
    assert audit["controller_experiment"]["climate_comparison_evaluable"] is True
    assert audit["controller_experiment"]["yield_comparison_evaluable"] is False
    assert audit["controller_experiment"]["unique_test_scenarios"] == 1
    assert audit["crop_model"]["target_validated"] is False
    assert audit["crop_model"]["calibration_level"] == "external_transfer_prior"
    assert audit["overall_verdict"] == "control_results_preliminary_yield_not_target_validated"
    finding_codes = {finding["code"] for finding in audit["findings"]}
    assert "SHORT_HORIZON_YIELD" in finding_codes
    assert "AMBIGUOUS_NEGATIVE_NET_PRODUCTION" in finding_codes
    assert "LOW_JOINT_COMFORT" in finding_codes
    assert "HIGH_SAFETY_INTERVENTION" in finding_codes
    assert "MISSING_TARGET_HARVEST" in finding_codes


def test_calibration_summary_separates_external_prior_from_target_validation():
    process_prior = {
        "metrics": {
            "maturity_thermal_time_deg_day": {
                "estimate": 596.5,
                "ci95_low": 583.8,
                "ci95_high": 612.7,
            },
            "fruit_dry_matter_fraction": {
                "estimate": 0.091,
                "ci95_low": 0.0894,
                "ci95_high": 0.0929,
            },
            "median_interpick_days": {
                "estimate": 5.0,
                "ci95_low": 5.0,
                "ci95_high": 5.0,
            },
        }
    }
    readiness = {
        "target_validated": False,
        "harvest_data": {"target_harvest_event_count": 0},
        "standing_crop_validation": {"validation_wmape": 0.289},
    }

    summary = build_calibration_summary(process_prior, readiness)

    assert summary["metric"].tolist() == [
        "maturity_thermal_time_deg_day",
        "fruit_dry_matter_fraction",
        "median_interpick_days",
        "pidu_standing_crop_validation_wmape",
    ]
    assert summary.iloc[0]["evidence_scope"] == "external_observed_wur"
    assert summary.iloc[-1]["evidence_scope"] == "target_standing_crop_not_harvest"
    assert not summary["target_harvest_validated"].any()
