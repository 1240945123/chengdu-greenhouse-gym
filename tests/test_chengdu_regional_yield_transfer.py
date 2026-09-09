from __future__ import annotations

import pandas as pd
import pytest

from experiments.crop.calibrate_chengdu_regional_yield_transfer import (
    build_regional_partial_yield_envelope,
    derive_maturity_thermal_time_prior,
    validate_regional_source_registry,
)
from experiments.crop.audit_regional_controller_yield import (
    build_regional_reasonableness_audit,
    compare_regional_audit_versions,
    compare_yield_transfer_versions,
    dataframe_to_markdown_table,
    run_audit,
)


def test_regional_registry_rejects_target_eligible_evidence():
    registry = pd.DataFrame(
        {
            "source_id": ["regional-study"],
            "evidence_class": ["external_observed"],
            "target_eligible": [True],
            "transfer_role": ["regional_yield_constraint"],
        }
    )

    with pytest.raises(ValueError, match="target_eligible=false"):
        validate_regional_source_registry(registry)


def test_maturity_prior_uses_regional_onset_days_and_degree_days():
    hourly = {}
    for season_id, temperature in (("spring", 20.0), ("autumn", 22.0)):
        hourly[season_id] = pd.DataFrame(
            {
                "elapsed_hours": range(1, 75 * 24 + 1),
                "air_temperature_c": temperature,
            }
        )

    prior = derive_maturity_thermal_time_prior(
        hourly,
        base_temperature_c=10.0,
        onset_days=(65, 70, 75),
    )

    assert prior["central_deg_day"] == pytest.approx(770.0)
    assert prior["low_deg_day"] == pytest.approx(650.0)
    assert prior["high_deg_day"] == pytest.approx(900.0)
    assert prior["season_count"] == 2
    assert prior["target_first_harvest_day"] == 70
    assert prior["season_intervals_deg_day"] == {
        "autumn": {
            "low_deg_day": pytest.approx(780.0),
            "central_deg_day": pytest.approx(840.0),
            "high_deg_day": pytest.approx(900.0),
        },
        "spring": {
            "low_deg_day": pytest.approx(650.0),
            "central_deg_day": pytest.approx(700.0),
            "high_deg_day": pytest.approx(750.0),
        },
    }


def test_partial_yield_envelope_time_warps_complete_wur_curves():
    observations = pd.DataFrame(
        {
            "team": ["a", "a", "b", "b"],
            "harvest_date": [
                "2020-01-10",
                "2020-03-10",
                "2020-01-10",
                "2020-03-10",
            ],
            "batch_fresh_kg_m2": [2.0, 8.0, 4.0, 6.0],
            "target_eligible": [False, False, False, False],
        }
    )

    envelope = build_regional_partial_yield_envelope(
        observations,
        regional_total_yields_kg_m2=[10.0, 12.0],
        wur_crop_start="2020-01-01",
        wur_crop_duration_days=100,
        regional_crop_duration_days=200,
        evaluation_day=40,
    )

    assert envelope["mapped_wur_crop_day"] == 20
    assert envelope["scenario_count"] == 4
    assert envelope["low_kg_m2"] == pytest.approx(2.0)
    assert envelope["median_kg_m2"] == pytest.approx(3.2)
    assert envelope["high_kg_m2"] == pytest.approx(4.8)
    assert envelope["target_eligible"] is False


def test_partial_yield_envelope_rejects_incomplete_team_curve():
    observations = pd.DataFrame(
        {
            "team": ["a", "a", "b"],
            "harvest_date": ["2020-01-10", "2020-03-10", "2020-01-10"],
            "batch_fresh_kg_m2": [2.0, 8.0, 4.0],
            "target_eligible": [False, False, False],
        }
    )

    with pytest.raises(ValueError, match="complete harvest curve"):
        build_regional_partial_yield_envelope(
            observations,
            regional_total_yields_kg_m2=[10.0],
            wur_crop_start="2020-01-01",
            wur_crop_duration_days=100,
            regional_crop_duration_days=200,
            evaluation_day=40,
        )


def test_regional_audit_reports_spring_coverage_and_onset_error():
    priors = {
        "spring_partial_yield_envelope": {"low_kg_m2": 2.0, "high_kg_m2": 3.0},
        "maturity_thermal_time": {"target_first_harvest_day": 70},
        "target_eligible": False,
    }
    seasons = pd.DataFrame(
        {
            "algorithm": ["baseline", "baseline", "pid", "pid"],
            "season_id": ["2023_spring", "2023_autumn"] * 2,
            "projected_harvest_kg_m2": [2.5, 0.5, 3.5, 0.4],
            "projected_first_harvest_day": [72.0, 68.0, 75.0, 65.0],
        }
    )
    scenarios = pd.DataFrame(
        {
            "algorithm": ["baseline", "baseline", "pid", "pid"],
            "seed": [0, 0, 0, 0],
            "season_id": ["2023_spring", "2023_autumn"] * 2,
            "dry_matter_balance_error_kg_m2": [1e-12, 0.0, -1e-12, 0.0],
        }
    )

    audit, by_algorithm, by_season = build_regional_reasonableness_audit(
        priors,
        scenarios,
        seasons,
        expected_scenario_count=4,
        expected_episode_count=4,
    )

    baseline = by_algorithm.set_index("algorithm").loc["baseline"]
    pid = by_algorithm.set_index("algorithm").loc["pid"]
    assert baseline["spring_envelope_coverage_fraction"] == 1.0
    assert pid["spring_envelope_coverage_fraction"] == 0.0
    assert baseline["first_harvest_mae_days"] == 2.0
    assert pid["first_harvest_mae_days"] == 5.0
    assert baseline["first_harvest_window_coverage_fraction"] == 1.0
    assert pid["first_harvest_window_coverage_fraction"] == 1.0
    assert by_season.loc[0, "spring_envelope_status"] == "inside"
    assert by_season["first_harvest_window_status"].tolist() == [
        "inside",
        "inside",
        "inside",
        "inside",
    ]
    assert audit["complete"] is True
    assert audit["target_harvest_validated"] is False
    assert audit["first_harvest_window_days"] == {"low": 65.0, "high": 75.0}


def test_markdown_table_renderer_has_no_optional_dependency():
    frame = pd.DataFrame({"algorithm": ["pid"], "yield": [2.34567]})

    rendered = dataframe_to_markdown_table(frame, float_digits=3)

    assert rendered.splitlines() == [
        "| algorithm | yield |",
        "| --- | --- |",
        "| pid | 2.346 |",
    ]


def test_transfer_sensitivity_preserves_algorithm_alignment():
    original = pd.DataFrame(
        {
            "algorithm": ["pid", "baseline"],
            "projected_harvest_mean_kg_m2": [2.0, 3.0],
        }
    )
    regional = pd.DataFrame(
        {
            "algorithm": ["baseline", "pid"],
            "projected_yield_mean_kg_m2": [2.1, 1.5],
        }
    )

    sensitivity = compare_yield_transfer_versions(original, regional)

    baseline = sensitivity.set_index("algorithm").loc["baseline"]
    pid = sensitivity.set_index("algorithm").loc["pid"]
    assert baseline["absolute_change_kg_m2"] == pytest.approx(-0.9)
    assert baseline["relative_change_fraction"] == pytest.approx(-0.3)
    assert pid["absolute_change_kg_m2"] == pytest.approx(-0.5)


def test_regional_version_comparison_reports_timing_and_yield_changes():
    v1 = pd.DataFrame(
        {
            "algorithm": ["baseline"],
            "projected_yield_mean_kg_m2": [2.0],
            "first_harvest_mae_days": [6.0],
            "spring_envelope_coverage_fraction": [0.5],
        }
    )
    v2 = pd.DataFrame(
        {
            "algorithm": ["baseline"],
            "projected_yield_mean_kg_m2": [2.4],
            "first_harvest_mae_days": [3.0],
            "spring_envelope_coverage_fraction": [1.0],
        }
    )

    result = compare_regional_audit_versions(v1, v2).iloc[0]

    assert result["yield_change_fraction"] == pytest.approx(0.2)
    assert result["first_harvest_mae_improvement_days"] == pytest.approx(3.0)
    assert result["spring_envelope_coverage_change"] == pytest.approx(0.5)


def test_regional_audit_uses_artifact_prefix_without_overwriting_v1(tmp_path):
    prior = tmp_path / "priors.json"
    prior.write_text(
        __import__("json").dumps(
            {
                "spring_partial_yield_envelope": {
                    "low_kg_m2": 2.0,
                    "high_kg_m2": 3.0,
                },
                "maturity_thermal_time": {
                    "target_first_harvest_day": 70,
                    "onset_days": [65, 70, 75],
                },
                "limitations": ["external_transfer"],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "algorithm": ["baseline"],
            "seed": [0],
            "season_id": ["2023_spring"],
            "dry_matter_balance_error_kg_m2": [0.0],
        }
    ).to_csv(tmp_path / "regional_v2_cohort_yield_scenarios.csv", index=False)
    pd.DataFrame(
        {
            "algorithm": ["baseline"],
            "season_id": ["2023_spring"],
            "projected_harvest_kg_m2": [2.5],
            "projected_first_harvest_day": [70.0],
        }
    ).to_csv(tmp_path / "regional_v2_cohort_yield_by_season.csv", index=False)

    output = run_audit(tmp_path, prior, artifact_prefix="regional_v2_")

    assert output.name == "regional_v2_yield_reasonableness.json"
    assert (tmp_path / "regional_v2_yield_reasonableness_by_algorithm.csv").exists()
    assert not (tmp_path / "regional_yield_reasonableness.json").exists()
