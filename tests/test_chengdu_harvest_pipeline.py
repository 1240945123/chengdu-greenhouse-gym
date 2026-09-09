from __future__ import annotations

import sys

import pandas as pd
import pytest
import yaml

import experiments.crop.run_chengdu_harvest_pipeline as harvest_pipeline
from experiments.crop.run_chengdu_harvest_pipeline import (
    build_harvest_readiness_report,
    calibrate_and_validate_harvest_seasons,
    combine_crop_observation_sources,
    evaluate_target_acceptance,
    parse_acceptance_thresholds,
    resolve_target_plant_density,
    validate_harvest_pipeline_inputs,
)
from experiments.crop.calibrate_harvest_model import simulate_harvest_series
from glassgym.models.harvest import HarvestCohortParameters


def _confirmed_protocols_for_drivers(drivers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season_id, season in drivers.groupby("season_id", sort=False):
        start = pd.Timestamp(season["timestamp"].min())
        start -= pd.Timedelta(hours=float(season["dt_hours"].iloc[0]))
        if start.tzinfo is None:
            start = start.tz_localize("Asia/Shanghai")
        rows.append(
            {
                "protocol_id": f"PROTOCOL-{season_id}",
                "season_id": str(season_id),
                "greenhouse_id": int(season["greenhouse_id"].iloc[0]),
                "greenhouse_code": str(season["greenhouse_code"].iloc[0]),
                "planting_code": str(season["planting_code"].iloc[0]),
                "effective_start": start,
                "effective_end": pd.NaT,
                "protocol_status": "confirmed",
                "pick_weekdays": (0, 2, 4),
                "pick_local_hour": 8,
                "confirmation_source": "site_manager_signed_schedule",
            }
        )
    return pd.DataFrame(rows)


def test_target_density_prefers_count_divided_by_area_over_inconsistent_registered_value():
    density, audit = resolve_target_plant_density(
        {
            "greenhouse_area_m2": 192.0,
            "registered_plant_count": 470,
            "registered_plant_density_plants_m2": 1.0,
        }
    )

    assert density == pytest.approx(470 / 192)
    assert audit["selected_source"] == "registered_plant_count_divided_by_greenhouse_area"
    assert audit["registered_density_relative_difference"] > 0.5
    assert audit["registered_density_consistent"] is False


def test_production_pipeline_requires_season_manifest_with_harvest_events():
    with pytest.raises(ValueError, match="season-manifest"):
        validate_harvest_pipeline_inputs(
            harvest_events_path="events.csv",
            season_manifest_path=None,
            drivers_path=None,
        )
    with pytest.raises(ValueError, match="harvest-events"):
        validate_harvest_pipeline_inputs(
            harvest_events_path=None,
            season_manifest_path="seasons.csv",
            drivers_path=None,
        )
    with pytest.raises(ValueError, match="harvest-events"):
        validate_harvest_pipeline_inputs(
            harvest_events_path=None,
            season_manifest_path=None,
            drivers_path="drivers.csv",
        )

    with pytest.raises(ValueError, match="harvest-protocol"):
        validate_harvest_pipeline_inputs(
            harvest_events_path="events.csv",
            season_manifest_path="seasons.csv",
            drivers_path="drivers.csv",
        )

    assert validate_harvest_pipeline_inputs(
        harvest_events_path="events.csv",
        season_manifest_path="seasons.csv",
        harvest_protocol_path="protocol.csv",
        drivers_path="drivers.csv",
    ) == ("events.csv", "seasons.csv", "protocol.csv")

    assert validate_harvest_pipeline_inputs(
        harvest_events_path="events.csv",
        season_manifest_path="seasons.csv",
        drivers_path=None,
    ) == ("events.csv", "seasons.csv", None)

    with pytest.raises(ValueError, match="uncertainty-driver-scenarios.*drivers"):
        validate_harvest_pipeline_inputs(
            harvest_events_path="events.csv",
            season_manifest_path="seasons.csv",
            harvest_protocol_path="protocol.csv",
            drivers_path=None,
            uncertainty_driver_scenarios_path="scenarios.csv",
        )


def test_production_pipeline_resolves_one_harvest_workbook_and_rejects_mixed_inputs():
    resolved = validate_harvest_pipeline_inputs(
        harvest_events_path=None,
        season_manifest_path=None,
        harvest_workbook_path="field.xlsx",
        drivers_path="drivers.csv",
    )

    assert resolved == ("field.xlsx", "field.xlsx", "field.xlsx")
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_harvest_pipeline_inputs(
            harvest_events_path="events.csv",
            season_manifest_path="seasons.csv",
            harvest_workbook_path="field.xlsx",
            drivers_path=None,
        )


def test_production_cli_accepts_explicit_harvest_protocol(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_chengdu_harvest_pipeline",
            "--sql",
            "tomato.sql",
            "--harvest-events",
            "events.csv",
            "--season-manifest",
            "seasons.csv",
            "--harvest-protocol",
            "protocol.csv",
            "--drivers",
            "drivers.csv",
            "--uncertainty-driver-scenarios",
            "scenarios.csv",
        ],
    )

    args = harvest_pipeline._parse_args()

    assert args.harvest_protocol == "protocol.csv"
    assert args.uncertainty_driver_scenarios == "scenarios.csv"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_harvest_pipeline_inputs(
            harvest_events_path=None,
            season_manifest_path=None,
            harvest_protocol_path="protocol.csv",
            harvest_workbook_path="field.xlsx",
            drivers_path=None,
        )
    with pytest.raises(ValueError, match="XLSX"):
        validate_harvest_pipeline_inputs(
            harvest_events_path=None,
            season_manifest_path=None,
            harvest_workbook_path="field.csv",
            drivers_path=None,
        )


def test_readiness_report_does_not_mislabel_source_samples_as_target_harvest():
    crop = pd.DataFrame(
        {
            "greenhouse_code": ["GH-XINDU"] * 2,
            "target_eligible": ["False", "0"],
            "observation_type": ["standing_crop_sample"] * 2,
            "observation_date": pd.to_datetime(["2025-04-01", "2025-04-18"]),
        }
    )
    harvest = pd.DataFrame(
        columns=[
            "greenhouse_id", "greenhouse_code", "target_eligible",
            "season_id", "timestamp", "fresh_kg_m2",
        ]
    )

    report = build_harvest_readiness_report(
        crop,
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["source_crop_observation_count"] == 2
    assert report["target_crop_observation_count"] == 0
    assert report["harvest_data"]["status"] == "missing_target_harvest_events"
    assert not report["target_validated"]
    assert report["blocking_reasons"]
    assert "driver_validation_evidence_missing" in report["blocking_reasons"]
    assert "conditional_uncertainty_evidence_missing" in report["blocking_reasons"]


def test_readiness_counts_target_workbook_samples_without_treating_them_as_harvest():
    crop = pd.DataFrame(
        {
            "target_eligible": [True] * 9,
            "observation_type": ["standing_crop_sample"] * 9,
        }
    )
    harvest = pd.DataFrame(
        columns=[
            "greenhouse_id", "greenhouse_code", "target_eligible",
            "season_id", "timestamp", "fresh_kg_m2",
        ]
    )

    report = build_harvest_readiness_report(
        crop,
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["target_crop_observation_count"] == 9
    assert report["harvest_data"]["target_harvest_event_count"] == 0
    assert not report["target_validated"]


def test_readiness_includes_failed_preliminary_standing_crop_validation_without_promoting_harvest():
    crop = pd.DataFrame(
        {
            "target_eligible": [True] * 9,
            "observation_type": ["standing_crop_sample"] * 9,
        }
    )
    harvest = pd.DataFrame(
        columns=[
            "greenhouse_id", "greenhouse_code", "target_eligible",
            "season_id", "timestamp", "fresh_kg_m2",
        ]
    )
    standing = {
        "target_greenhouse_id": 63,
        "target_greenhouse_code": "GH-PIDU",
        "status": "preliminary_target_standing_crop_validation",
        "standing_crop_precalibration": {
            "validation_metrics": {"wmape": 0.289},
            "validation_date": "2026-05-19T00:00:00+08:00",
        },
        "temperature_driven_phenology_validation": {"wmape": 0.049},
        "fixed_multi_organ_calibration": {
            "validation_metrics": {"organ_normalized_rmse": 0.248}
        },
        "initial_thermal_sum_sensitivity": {
            "validation_envelope_contains_observed": False,
            "best_case_absolute_relative_error": 0.45,
            "uncertainty_type": "structural_sensitivity_not_predictive_interval",
        },
        "late_biomass_flux_diagnostics": {
            "model_fraction_of_observed_change": 0.581,
            "required_to_modeled_gross_ratio": 1.429,
            "mass_balance_residual_absolute_sum_kg_m2": 9e-15,
        },
        "assimilation_capacity_sensitivity": {
            "validation_fresh_envelope_contains_observed": False,
            "best_case_validation_fresh_absolute_relative_error": 0.299,
            "calibration_organ_nrmse_range": [0.02, 0.298],
        },
        "scored_observation_dates": 2,
        "target_harvest_validated": False,
    }

    report = build_harvest_readiness_report(
        crop,
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        standing_crop_validation=standing,
    )

    assert report["standing_crop_validation"]["validation_wmape"] == pytest.approx(0.289)
    assert report["standing_crop_validation"]["phenology_wmape"] == pytest.approx(0.049)
    assert report["standing_crop_validation"]["organ_validation_nrmse"] == pytest.approx(0.248)
    assert report["standing_crop_validation"]["thermal_sum_envelope_contains_observed"] is False
    assert report["standing_crop_validation"]["thermal_sum_best_case_relative_error"] == pytest.approx(0.45)
    assert report["standing_crop_validation"]["late_biomass_fraction_of_observed"] == pytest.approx(0.581)
    assert report["standing_crop_validation"]["required_to_modeled_gross_ratio"] == pytest.approx(1.429)
    assert report["standing_crop_validation"]["capacity_envelope_contains_observed"] is False
    assert report["standing_crop_validation"]["capacity_best_case_relative_error"] == pytest.approx(0.299)
    assert report["standing_crop_validation"]["evidence_sufficient"] is False
    assert "standing_crop_validation_insufficient" in report["blocking_reasons"]
    assert report["target_validated"] is False


def test_combine_crop_sources_preserves_target_rows_and_source_audit():
    columns = [
        "observation_id", "observation_type", "observation_date", "greenhouse_id",
        "greenhouse_code", "planting_id", "planting_code", "crop_name", "cultivar",
        "plant_stage", "sample_plant_id", "source_site", "target_eligible",
        "quality_flags", "plant_density_plants_m2", "lai_m2_m2", "plant_height_cm",
        "stem_diameter_mm", "stem_node_count", "flower_truss_count", "fruit_truss_count",
        "plant_fresh_g_per_plant", "plant_fresh_kg_m2", "plant_dry_g_per_plant",
        "plant_dry_kg_m2", "leaf_fresh_g_per_plant", "leaf_fresh_kg_m2",
        "leaf_dry_g_per_plant", "leaf_dry_kg_m2", "petiole_fresh_g_per_plant",
        "petiole_fresh_kg_m2", "petiole_dry_g_per_plant", "petiole_dry_kg_m2",
        "stem_fresh_g_per_plant", "stem_fresh_kg_m2", "stem_dry_g_per_plant",
        "stem_dry_kg_m2", "ripe_fruit_fresh_g_per_plant", "ripe_fruit_fresh_kg_m2",
        "ripe_fruit_dry_g_per_plant", "ripe_fruit_dry_kg_m2", "root_fresh_g_per_plant",
        "root_fresh_kg_m2", "root_dry_g_per_plant", "root_dry_kg_m2", "source_table",
        "source_line",
    ]
    base = {column: None for column in columns}
    base.update(
        observation_type="standing_crop_sample", plant_density_plants_m2=1.0,
        crop_name="tomato", cultivar="c", quality_flags="", source_line=1,
    )
    source = pd.DataFrame([
        dict(base, observation_id=1, observation_date=pd.Timestamp("2025-01-01"),
             greenhouse_id=1, greenhouse_code="X", planting_id=1, planting_code="PX",
             source_site="xindu", target_eligible=False, source_table="sql")
    ])
    source.attrs["duplicate_rows_removed"] = 2
    target = pd.DataFrame([
        dict(base, observation_id=2, observation_date=pd.Timestamp("2026-04-27"),
             greenhouse_id=63, greenhouse_code="P", planting_id=59, planting_code="PP",
             source_site="pidu", target_eligible=True, source_table="xlsx")
    ])

    combined = combine_crop_observation_sources(source, target)

    assert len(combined) == 2
    assert combined["target_eligible"].sum() == 1
    assert combined.attrs["duplicate_rows_removed"] == 2
    assert combined.attrs["source_row_counts"] == {"sql": 1, "target_workbook": 1}


def test_acceptance_requires_all_timing_yield_and_uncertainty_gates():
    good = {
        "first_harvest_date_error_days": 2.0,
        "harvest_timing_wasserstein_days": 3.0,
        "median_harvest_date_error_days": 2.0,
        "p90_harvest_date_error_days": 4.0,
        "batch_wmape": 0.15,
        "cumulative_wmape": 0.08,
        "final_yield_bias_percent": -4.0,
        "cumulative_r2": 0.90,
        "batch_interval_80_coverage": 0.78,
        "batch_interval_95_coverage": 0.92,
        "batch_interval_80_normalized_mean_width": 0.80,
        "batch_interval_95_normalized_mean_width": 1.20,
        "cumulative_interval_80_coverage": 0.78,
        "cumulative_interval_95_coverage": 0.92,
        "cumulative_interval_80_normalized_mean_width": 0.20,
        "cumulative_interval_95_normalized_mean_width": 0.30,
    }

    accepted = evaluate_target_acceptance(good)

    assert accepted["accepted"]
    assert all(accepted["gates"].values())

    bad_batch = dict(good, batch_wmape=0.25)
    rejected = evaluate_target_acceptance(bad_batch)
    assert not rejected["accepted"]
    assert not rejected["gates"]["batch_wmape"]

    bad_batch_interval = dict(good, batch_interval_80_coverage=0.60)
    rejected_interval = evaluate_target_acceptance(bad_batch_interval)
    assert not rejected_interval["accepted"]
    assert not rejected_interval["gates"]["batch_interval_80_coverage"]

    bad_timing = dict(good, harvest_timing_wasserstein_days=8.0)
    rejected_timing = evaluate_target_acceptance(bad_timing)
    assert not rejected_timing["accepted"]
    assert not rejected_timing["gates"]["harvest_timing_wasserstein_days"]


def test_acceptance_config_maps_all_directional_keys_to_canonical_metrics():
    config = yaml.safe_load(
        open("configs/crops/chengdu_tomato_harvest.yml", encoding="utf-8")
    )

    thresholds = parse_acceptance_thresholds(
        config["validation"]["acceptance"]
    )

    assert len(thresholds) == 16
    assert thresholds["first_harvest_date_error_days"] == 5.0
    assert thresholds["harvest_timing_wasserstein_days"] == 7.0
    assert thresholds["cumulative_r2"] == 0.80
    assert thresholds["batch_interval_95_coverage"] == 0.85


def test_acceptance_config_rejects_missing_unknown_and_invalid_values():
    config = yaml.safe_load(
        open("configs/crops/chengdu_tomato_harvest.yml", encoding="utf-8")
    )["validation"]["acceptance"]
    missing = dict(config)
    missing.pop("batch_wmape_max")
    with pytest.raises(ValueError, match="missing"):
        parse_acceptance_thresholds(missing)

    unknown = dict(config, undocumented_gate_max=1.0)
    with pytest.raises(ValueError, match="unknown"):
        parse_acceptance_thresholds(unknown)

    invalid_coverage = dict(config, batch_interval_80_coverage_min=1.1)
    with pytest.raises(ValueError, match="coverage"):
        parse_acceptance_thresholds(invalid_coverage)

    nonfinite = dict(config, cumulative_wmape_max=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        parse_acceptance_thresholds(nonfinite)


def test_readiness_needs_held_out_target_metrics_even_with_two_seasons():
    crop = pd.DataFrame(
        {
            "greenhouse_code": ["GH-PIDU"],
            "target_eligible": [True],
            "observation_type": ["standing_crop_sample"],
            "observation_date": pd.to_datetime(["2026-04-01"]),
        }
    )
    harvest = pd.DataFrame(
        {
            "greenhouse_code": ["GH-PIDU"] * 16,
            "greenhouse_id": [63] * 16,
            "target_eligible": [True] * 16,
            "season_id": ["s1"] * 8 + ["s2"] * 8,
                "timestamp": list(pd.date_range("2026-05-01", periods=8, freq="5D"))
                + list(pd.date_range("2027-05-01", periods=8, freq="5D")),
                "fresh_kg_m2": [1.0] * 16,
                "season_complete_evidence": [True] * 16,
            }
        )

    report = build_harvest_readiness_report(
        crop,
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    assert report["harvest_data"]["can_validate_target_independently"]
    assert not report["target_validated"]
    assert "held_out_target_metrics_missing" in report["blocking_reasons"]


def test_readiness_requires_structured_held_out_validation_evidence():
    harvest = pd.DataFrame(
        {
            "greenhouse_id": [63] * 16,
            "greenhouse_code": ["GH-PIDU"] * 16,
            "target_eligible": [True] * 16,
            "season_id": ["s1"] * 8 + ["s2"] * 8,
                "timestamp": list(pd.date_range("2026-05-01", periods=8, freq="5D"))
                + list(pd.date_range("2027-05-01", periods=8, freq="5D")),
                "fresh_kg_m2": 1.0,
                "season_complete_evidence": True,
            }
        )
    metrics = {
        "first_harvest_date_error_days": 2.0,
        "harvest_timing_wasserstein_days": 3.0,
        "median_harvest_date_error_days": 2.0,
        "p90_harvest_date_error_days": 4.0,
        "batch_wmape": 0.15,
        "cumulative_wmape": 0.08,
        "final_yield_bias_percent": -4.0,
        "cumulative_r2": 0.90,
        "batch_interval_80_coverage": 0.78,
        "batch_interval_95_coverage": 0.92,
        "batch_interval_80_normalized_mean_width": 0.80,
        "batch_interval_95_normalized_mean_width": 1.20,
        "cumulative_interval_80_coverage": 0.78,
        "cumulative_interval_95_coverage": 0.92,
        "cumulative_interval_80_normalized_mean_width": 0.20,
        "cumulative_interval_95_normalized_mean_width": 0.30,
    }
    config = yaml.safe_load(
        open("configs/crops/chengdu_tomato_harvest.yml", encoding="utf-8")
    )
    configured_thresholds = parse_acceptance_thresholds(
        config["validation"]["acceptance"]
    )
    evidence = {
        "target_greenhouse_id": 63,
        "target_greenhouse_code": "GH-PIDU",
        "split": {"train_seasons": ["s1"], "validation_seasons": [], "test_seasons": ["s2"]},
        "metrics_scope": "held_out_target_test",
        "driver_validation_passed": True,
        "test_metrics": metrics,
        "acceptance_thresholds": configured_thresholds,
        "uncertainty_assessment": {
            "method": "joint_parameter_refit_and_residual_moving_block_bootstrap",
            "scope": "conditional_on_supplied_greenlight_drivers_and_model_structure",
            "interval_levels": [0.80, 0.95],
            "output_targets": ["batch_fresh_kg_m2", "cumulative_fresh_kg_m2"],
            "block_boundary_policy": "never_cross_training_season_boundaries",
            "forecast_cumulative_reset_policy": "reset_for_each_forecast_season",
            "included_sources": [
                "training_batch_residual_temporal_dependence",
                "harvest_parameter_estimation_uncertainty",
            ],
            "excluded_sources": [
                "greenhouse_climate_model_parameter_uncertainty",
                "outdoor_weather_scenario_uncertainty",
                "xindu_to_pidu_site_transfer_uncertainty",
                "structural_uncertainty_of_dry_matter_age_cohorts",
            ],
        },
    }

    accepted = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=evidence,
    )
    rejected = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=dict(evidence, driver_validation_passed=False),
    )

    assert accepted["target_validated"]
    scenario_assessment = {
        **evidence["uncertainty_assessment"],
        "method": (
            "joint_driver_scenario_parameter_refit_and_residual_"
            "moving_block_bootstrap"
        ),
        "scope": (
            "conditional_on_supplied_greenlight_driver_ensemble_and_"
            "model_structure"
        ),
        "included_sources": [
            "training_batch_residual_temporal_dependence",
            "harvest_parameter_estimation_uncertainty",
            "greenhouse_climate_model_parameter_uncertainty",
            "outdoor_weather_scenario_uncertainty",
        ],
        "excluded_sources": [
            "xindu_to_pidu_site_transfer_uncertainty",
            "structural_uncertainty_of_dry_matter_age_cohorts",
        ],
        "driver_scenarios": {"scenario_count": 2},
    }
    scenario_report = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=dict(
            evidence,
            uncertainty_assessment=scenario_assessment,
        ),
    )
    assert scenario_report["target_validated"]
    assert not rejected["target_validated"]
    assert "driver_validation_evidence_missing" in rejected["blocking_reasons"]
    strict_thresholds = dict(configured_thresholds, batch_wmape=0.10)
    strict_report = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=dict(evidence, acceptance_thresholds=strict_thresholds),
    )
    assert not strict_report["target_validated"]
    assert strict_report["held_out_acceptance"]["thresholds"]["batch_wmape"] == 0.10
    assert not strict_report["held_out_acceptance"]["gates"]["batch_wmape"]
    missing_uncertainty = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result={key: value for key, value in evidence.items() if key != "uncertainty_assessment"},
    )
    assert not missing_uncertainty["target_validated"]
    assert "conditional_uncertainty_evidence_missing" in missing_uncertainty["blocking_reasons"]
    unsafe_uncertainty = dict(evidence)
    unsafe_uncertainty["uncertainty_assessment"] = dict(
        evidence["uncertainty_assessment"],
        block_boundary_policy="may_cross_training_season_boundaries",
    )
    unsafe_report = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=unsafe_uncertainty,
    )
    assert not unsafe_report["target_validated"]
    assert "conditional_uncertainty_evidence_missing" in unsafe_report["blocking_reasons"]
    omitted_exclusion = dict(evidence)
    omitted_exclusion["uncertainty_assessment"] = dict(
        evidence["uncertainty_assessment"],
        excluded_sources=["greenhouse_climate_model_parameter_uncertainty"],
    )
    omitted_report = build_harvest_readiness_report(
        pd.DataFrame(columns=["target_eligible"]),
        harvest,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        validation_result=omitted_exclusion,
    )
    assert not omitted_report["target_validated"]
    assert "conditional_uncertainty_evidence_missing" in omitted_report["blocking_reasons"]


def test_driver_scenario_uncertainty_assessment_moves_declared_sources_to_included():
    assessment = harvest_pipeline._conditional_uncertainty_assessment(
        {"accepted_draws": 20},
        {
            "scenario_count": 3,
            "scenario_ids": ["a", "b", "c"],
            "uncertainty_sources": [
                "greenhouse_climate_model_parameter_uncertainty"
            ],
        },
    )

    assert assessment["method"] == (
        "joint_driver_scenario_parameter_refit_and_residual_"
        "moving_block_bootstrap"
    )
    assert "greenhouse_climate_model_parameter_uncertainty" in assessment[
        "included_sources"
    ]
    assert "greenhouse_climate_model_parameter_uncertainty" not in assessment[
        "excluded_sources"
    ]
    assert "outdoor_weather_scenario_uncertainty" in assessment["excluded_sources"]
    assert assessment["driver_scenarios"]["scenario_count"] == 3


def test_harvest_config_separates_source_estimate_from_target_validation():
    config = yaml.safe_load(
        open("configs/crops/chengdu_tomato_harvest.yml", encoding="utf-8")
    )

    assert config["model"]["architecture"] == "greenlight_plus_dry_matter_age_cohorts"
    assert config["source_precalibration"]["greenhouse_code"] == "GH2024"
    assert config["target_site"]["greenhouse_code"] == "GH202602061448266452514"
    assert config["target_site"]["planting_code"] == "P202603251720122975035"
    parameter_uncertainty = config["uncertainty"]["parameter_estimation"]
    assert parameter_uncertainty["method"] == (
        "season_residual_moving_block_parameter_refit_bootstrap"
    )
    assert parameter_uncertainty["bootstrap_samples"] >= 200
    assert "harvest_parameter_estimation_uncertainty" not in config["uncertainty"][
        "not_yet_propagated"
    ]
    scenario_config = config["uncertainty"]["driver_scenario_ensemble"]
    assert scenario_config["status"] == "optional_not_currently_supplied"
    assert scenario_config["builder_module"] == (
        "experiments.crop.build_chengdu_harvest_driver_scenarios"
    )
    assert scenario_config["minimum_scenarios"] == 2
    assert scenario_config["point_driver_implicitly_included"] is False
    assert set(scenario_config["allowed_sources"]) == {
        "greenhouse_climate_model_parameter_uncertainty",
        "outdoor_weather_scenario_uncertainty",
    }
    acceptance = config["validation"]["acceptance"]
    assert acceptance["harvest_timing_wasserstein_days_max"] == 7.0
    assert acceptance["median_harvest_date_error_days_abs_max"] == 7.0
    assert acceptance["p90_harvest_date_error_days_abs_max"] == 10.0
    assert config["target_site"]["cultivar"] == "塞尼瑞"
    assert "declared_phenology_dates_are_not_observations" in config["target_site"]["quality_flags"]
    assert not config["status"]["target_harvest_calibrated"]
    assert not config["status"]["target_harvest_validated"]
    assert config["validation"]["minimum_complete_target_seasons"] >= 2


def test_multiseason_pipeline_fits_train_and_scores_only_held_out_test():
    true_parameters = HarvestCohortParameters(
        base_temperature_c=10.0,
        maturity_thermal_time_deg_day=100.0,
        dry_matter_fraction=0.08,
    )
    driver_parts = []
    event_parts = []
    for year, season_id in zip((2024, 2025, 2026), ("s1", "s2", "s3"), strict=True):
        timestamps = pd.date_range(f"{year}-03-01 08:00", periods=100, freq="D")
        drivers = pd.DataFrame(
            {
                "season_id": season_id,
                "timestamp": timestamps,
                "net_fruit_dry_matter_change_kg_m2": 0.01,
                "air_temperature_c": 20.0,
                "dt_hours": 24.0,
                "pick": [
                    timestamp.weekday() in {0, 2, 4}
                    for timestamp in timestamps
                ],
                "pick_source": "management_protocol",
                "driver_model_status": "accepted_target_crop_model",
                "season_complete": True,
                "fruit_change_source": "greenlight_fruit_state_balance",
                "air_temperature_source": "greenlight_simulated_indoor",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": f"P-{season_id}",
                "initial_fruit_dry_matter_kg_m2": 0.0,
                "initial_fruit_maturity_fraction": 0.0,
            }
        )
        predicted = simulate_harvest_series(drivers, true_parameters)
        events = predicted.loc[predicted["fresh_kg_m2"] > 0.0, ["season_id", "timestamp", "fresh_kg_m2"]]
        events = events.assign(
            greenhouse_id=63,
            greenhouse_code="GH-PIDU",
            planting_code=f"P-{season_id}",
            target_eligible=True,
            season_complete_evidence=True,
        )
        driver_parts.append(drivers)
        event_parts.append(events)

    configured_thresholds = parse_acceptance_thresholds(
        yaml.safe_load(
            open("configs/crops/chengdu_tomato_harvest.yml", encoding="utf-8")
        )["validation"]["acceptance"]
    )
    combined_drivers = pd.concat(driver_parts, ignore_index=True)
    result = calibrate_and_validate_harvest_seasons(
        combined_drivers,
        pd.concat(event_parts, ignore_index=True),
        protocols=_confirmed_protocols_for_drivers(combined_drivers),
        base_temperature_c=10.0,
        maturity_bounds_deg_day=(70.0, 130.0),
        dry_matter_fraction_bounds=(0.06, 0.10),
        maturity_grid_size=13,
        parameter_bootstrap_samples=4,
        bootstrap_samples=200,
        bootstrap_seed=3,
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        acceptance_thresholds=configured_thresholds,
    )

    assert result["split"] == {
        "train_seasons": ["s1"],
        "validation_seasons": ["s2"],
        "test_seasons": ["s3"],
    }
    assert result["metrics_scope"] == "held_out_target_test"
    assert result["batch_semantics"] == "greenhouse_calendar_day_total"
    assert result["acceptance_thresholds"] == configured_thresholds
    assert result["test_metrics"]["batch_wmape"] < 1e-4
    assert result["test_metrics"]["batch_interval_80_coverage"] == 1.0
    assert result["test_metrics"]["batch_interval_95_coverage"] == 1.0
    assert "batch_p10_kg_m2" in result["test_predictions"].columns
    assert "cumulative_p90_kg_m2" in result["test_predictions"].columns
    assert result["acceptance"]["accepted"]
    assert len(result["parameter_bootstrap_draws"]) == 4
    assert result["parameter_bootstrap_audit"]["accepted_draws"] == 4
    assert result["uncertainty_method"] == (
        "joint_parameter_refit_and_residual_moving_block_bootstrap"
    )
    assert result["uncertainty_assessment"]["scope"] == (
        "conditional_on_supplied_greenlight_drivers_and_model_structure"
    )
    assert "outdoor_weather_scenario_uncertainty" in result["uncertainty_assessment"]["excluded_sources"]
    assert "harvest_parameter_estimation_uncertainty" in result["uncertainty_assessment"]["included_sources"]
    assert "harvest_parameter_estimation_uncertainty" not in result["uncertainty_assessment"]["excluded_sources"]
    assert result["uncertainty_assessment"]["block_boundary_policy"] == (
        "never_cross_training_season_boundaries"
    )
    assert result["uncertainty_assessment"]["forecast_cumulative_reset_policy"] == (
        "reset_for_each_forecast_season"
    )
    assert result["uncertainty_assessment"]["output_targets"] == [
        "batch_fresh_kg_m2",
        "cumulative_fresh_kg_m2",
    ]


def test_multiseason_pipeline_rejects_pick_labels_derived_from_harvest_events():
    drivers = pd.DataFrame(
        {
            "season_id": ["s1", "s2"],
            "timestamp": pd.to_datetime(["2025-03-01", "2026-03-01"]),
            "pick_source": ["observed_harvest_events"] * 2,
            "season_complete": [True] * 2,
            "fruit_change_source": ["greenlight_fruit_state_balance"] * 2,
            "air_temperature_source": ["greenlight_simulated_indoor"] * 2,
        }
    )
    events = pd.DataFrame(
        {
            "season_id": ["s1", "s2"],
            "timestamp": pd.to_datetime(["2025-05-01", "2026-05-01"]),
            "fresh_kg_m2": [1.0, 1.0],
        }
    )

    try:
        calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=pd.DataFrame(),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
        )
    except ValueError as exc:
        assert "pick_source" in str(exc) and "management_protocol" in str(exc)
    else:
        raise AssertionError("observed harvest dates must not define model pick labels")


def test_multiseason_pipeline_rejects_unaccepted_crop_model_drivers():
    drivers = pd.DataFrame(
        {
            "season_id": ["s1", "s2"],
            "timestamp": pd.to_datetime(["2025-03-01", "2026-03-01"]),
            "pick_source": ["management_protocol"] * 2,
            "driver_model_status": ["diagnostic_not_final"] * 2,
            "fruit_change_source": ["greenlight_fruit_state_balance"] * 2,
            "air_temperature_source": ["observed_indoor"] * 2,
        }
    )
    events = pd.DataFrame(
        {
            "season_id": ["s1", "s2"],
            "timestamp": pd.to_datetime(["2025-05-01", "2026-05-01"]),
            "fresh_kg_m2": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="driver_model_status"):
        calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=pd.DataFrame(),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
        )


def test_multiseason_pipeline_rejects_incomplete_or_short_driver_seasons():
    dates = list(pd.date_range("2025-03-01", periods=10, freq="D")) + list(
        pd.date_range("2026-03-01", periods=10, freq="D")
    )
    drivers = pd.DataFrame(
        {
            "season_id": ["s1"] * 10 + ["s2"] * 10,
            "timestamp": dates,
            "pick_source": "management_protocol",
            "driver_model_status": "accepted_target_crop_model",
            "season_complete": False,
            "fruit_change_source": "greenlight_fruit_state_balance",
            "air_temperature_source": "greenlight_simulated_indoor",
        }
    )
    events = pd.DataFrame(
        {
            "season_id": ["s1"] * 8 + ["s2"] * 8,
            "timestamp": list(pd.date_range("2025-03-01", periods=8, freq="D"))
            + list(pd.date_range("2026-03-01", periods=8, freq="D")),
            "fresh_kg_m2": [1.0] * 16,
        }
    )

    try:
        calibrate_and_validate_harvest_seasons(
            drivers,
            events,
            protocols=pd.DataFrame(),
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
        )
    except ValueError as exc:
        assert "complete" in str(exc) or "90" in str(exc)
    else:
        raise AssertionError("partial driver seasons must not enter target validation")


def test_pipeline_rejects_string_false_sparse_identity_mismatch_and_overlap():
    base = pd.DataFrame(
        {
            "season_id": ["s1"] * 91 + ["s2"] * 91,
            "timestamp": list(pd.date_range("2025-01-01", periods=91, freq="D"))
            + list(pd.date_range("2026-01-01", periods=91, freq="D")),
            "net_fruit_dry_matter_change_kg_m2": 0.01,
            "initial_fruit_dry_matter_kg_m2": 0.0,
            "initial_fruit_maturity_fraction": 0.0,
            "air_temperature_c": 20.0,
            "dt_hours": 24.0,
            "pick": False,
            "pick_source": "management_protocol",
            "driver_model_status": "accepted_target_crop_model",
            "season_complete": True,
            "fruit_change_source": "greenlight_fruit_state_balance",
            "air_temperature_source": "greenlight_simulated_indoor",
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "planting_code": ["P1"] * 91 + ["P2"] * 91,
        }
    )
    events = pd.DataFrame(
        {
            "season_id": ["s1"] * 8 + ["s2"] * 8,
            "timestamp": list(pd.date_range("2025-02-01", periods=8, freq="5D"))
            + list(pd.date_range("2026-02-01", periods=8, freq="5D")),
            "fresh_kg_m2": 1.0,
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "planting_code": ["P1"] * 8 + ["P2"] * 8,
            "target_eligible": True,
        }
    )
    kwargs = dict(
        protocols=_confirmed_protocols_for_drivers(base),
        base_temperature_c=10.0,
        maturity_bounds_deg_day=(70.0, 130.0),
        dry_matter_fraction_bounds=(0.06, 0.10),
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
    )

    for changed, expected in (
        (base.assign(season_complete="False"), "complete"),
        (base.assign(greenhouse_code="OTHER"), "identity"),
    ):
        try:
            calibrate_and_validate_harvest_seasons(changed, events, **kwargs)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid drivers must fail: {expected}")

    sparse = pd.concat([base.iloc[[0, 90]], base.iloc[[91, 181]]], ignore_index=True)
    try:
        calibrate_and_validate_harvest_seasons(sparse, events, **kwargs)
    except ValueError as exc:
        assert "interval" in str(exc) or "coverage" in str(exc)
    else:
        raise AssertionError("two-point seasons must not count as complete")

    missing_step = base.drop(index=[10, 101]).reset_index(drop=True)
    missing_step.loc[missing_step["timestamp"].isin([pd.Timestamp("2025-01-12"), pd.Timestamp("2026-01-12")]), "dt_hours"] = 48.0
    try:
        calibrate_and_validate_harvest_seasons(
            missing_step,
            events,
            maximum_driver_interval_hours=48.0,
            **kwargs,
        )
    except ValueError as exc:
        assert "fixed nominal" in str(exc) or "coverage" in str(exc)
    else:
        raise AssertionError("missing samples must not be hidden by changing row dt_hours")

    overlap = base.copy()
    overlap.loc[overlap["season_id"] == "s2", "timestamp"] -= pd.Timedelta(days=330)
    overlap_events = events.copy()
    overlap_events.loc[overlap_events["season_id"] == "s2", "timestamp"] -= pd.Timedelta(days=330)
    try:
        calibrate_and_validate_harvest_seasons(overlap, overlap_events, **kwargs)
    except ValueError as exc:
        assert "overlap" in str(exc) or "chronological" in str(exc)
    else:
        raise AssertionError("overlapping seasons must not enter validation")
