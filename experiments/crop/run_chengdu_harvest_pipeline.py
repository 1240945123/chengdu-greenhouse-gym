from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from experiments.crop.calibrate_harvest_model import (
    assess_target_harvest_data,
    bootstrap_harvest_parameter_estimates,
    estimate_fruit_dry_matter_fraction,
    joint_harvest_prediction_intervals,
    simulate_harvest_series,
    validate_uncertainty_driver_scenarios,
)
from experiments.crop.harvest_package_validation import (
    assess_harvest_calibration_package,
    require_independent_validation_package,
)
from common.harvest_evaluation import evaluate_harvest_predictions
from glassgym.models.harvest import HarvestCohortParameters
from processing.chengdu_crop_observations import (
    CROP_OBSERVATION_COLUMNS,
    extract_crop_observations,
    validate_crop_observations,
)
from processing.chengdu_crop_workbook import extract_target_crop_workbook
from processing.chengdu_harvest_events import load_harvest_events
from processing.chengdu_harvest_protocols import load_harvest_protocols
from processing.chengdu_harvest_seasons import (
    bind_harvest_events_to_seasons,
    load_harvest_seasons,
)
from processing.chengdu_measured_harvest import build_measured_harvest_artifacts


DEFAULT_ACCEPTANCE_THRESHOLDS = {
    "first_harvest_date_error_days": 5.0,
    "harvest_timing_wasserstein_days": 7.0,
    "median_harvest_date_error_days": 7.0,
    "p90_harvest_date_error_days": 10.0,
    "batch_wmape": 0.20,
    "cumulative_wmape": 0.15,
    "final_yield_bias_percent": 10.0,
    "cumulative_r2": 0.80,
    "batch_interval_80_coverage": 0.70,
    "batch_interval_95_coverage": 0.85,
    "batch_interval_80_normalized_mean_width": 1.50,
    "batch_interval_95_normalized_mean_width": 2.00,
    "cumulative_interval_80_coverage": 0.70,
    "cumulative_interval_95_coverage": 0.85,
    "cumulative_interval_80_normalized_mean_width": 0.50,
    "cumulative_interval_95_normalized_mean_width": 0.80,
}

ACCEPTANCE_CONFIG_KEY_MAP = {
    "first_harvest_date_error_days_max": "first_harvest_date_error_days",
    "harvest_timing_wasserstein_days_max": "harvest_timing_wasserstein_days",
    "median_harvest_date_error_days_abs_max": "median_harvest_date_error_days",
    "p90_harvest_date_error_days_abs_max": "p90_harvest_date_error_days",
    "batch_wmape_max": "batch_wmape",
    "cumulative_wmape_max": "cumulative_wmape",
    "final_yield_bias_percent_abs_max": "final_yield_bias_percent",
    "cumulative_r2_min": "cumulative_r2",
    "batch_interval_80_coverage_min": "batch_interval_80_coverage",
    "batch_interval_95_coverage_min": "batch_interval_95_coverage",
    "batch_interval_80_normalized_mean_width_max": "batch_interval_80_normalized_mean_width",
    "batch_interval_95_normalized_mean_width_max": "batch_interval_95_normalized_mean_width",
    "cumulative_interval_80_coverage_min": "cumulative_interval_80_coverage",
    "cumulative_interval_95_coverage_min": "cumulative_interval_95_coverage",
    "cumulative_interval_80_normalized_mean_width_max": "cumulative_interval_80_normalized_mean_width",
    "cumulative_interval_95_normalized_mean_width_max": "cumulative_interval_95_normalized_mean_width",
}

CONDITIONAL_UNCERTAINTY_SCOPE = (
    "conditional_on_supplied_greenlight_drivers_and_model_structure"
)
DRIVER_ENSEMBLE_UNCERTAINTY_SCOPE = (
    "conditional_on_supplied_greenlight_driver_ensemble_and_model_structure"
)
FIXED_DRIVER_UNCERTAINTY_METHOD = (
    "joint_parameter_refit_and_residual_moving_block_bootstrap"
)
DRIVER_ENSEMBLE_UNCERTAINTY_METHOD = (
    "joint_driver_scenario_parameter_refit_and_residual_moving_block_bootstrap"
)
REQUIRED_UNCERTAINTY_EXCLUSIONS = {
    "greenhouse_climate_model_parameter_uncertainty",
    "outdoor_weather_scenario_uncertainty",
    "xindu_to_pidu_site_transfer_uncertainty",
    "structural_uncertainty_of_dry_matter_age_cohorts",
}
REQUIRED_UNCERTAINTY_INCLUSIONS = {
    "training_batch_residual_temporal_dependence",
    "harvest_parameter_estimation_uncertainty",
}
PROPAGATABLE_DRIVER_UNCERTAINTY_SOURCES = {
    "greenhouse_climate_model_parameter_uncertainty",
    "outdoor_weather_scenario_uncertainty",
}
ALL_DECLARED_UNCERTAINTY_SOURCES = (
    REQUIRED_UNCERTAINTY_INCLUSIONS | REQUIRED_UNCERTAINTY_EXCLUSIONS
)


def parse_acceptance_thresholds(config: dict[str, object]) -> dict[str, float]:
    """Validate directional YAML keys and return canonical metric thresholds."""
    provided = set(config)
    expected = set(ACCEPTANCE_CONFIG_KEY_MAP)
    if missing := sorted(expected - provided):
        raise ValueError(f"acceptance configuration missing keys: {', '.join(missing)}")
    if unknown := sorted(provided - expected):
        raise ValueError(f"acceptance configuration has unknown keys: {', '.join(unknown)}")
    thresholds: dict[str, float] = {}
    for config_key, metric_name in ACCEPTANCE_CONFIG_KEY_MAP.items():
        try:
            value = float(config[config_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"acceptance threshold {config_key} must be numeric and finite"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"acceptance threshold {config_key} must be numeric and finite"
            )
        if "coverage" in metric_name and not 0.0 <= value <= 1.0:
            raise ValueError(f"acceptance coverage threshold {config_key} must be in [0, 1]")
        if metric_name == "cumulative_r2" and not -1.0 <= value <= 1.0:
            raise ValueError("acceptance cumulative R2 threshold must be in [-1, 1]")
        if metric_name != "cumulative_r2" and value < 0.0:
            raise ValueError(f"acceptance threshold {config_key} must be non-negative")
        thresholds[metric_name] = value
    return thresholds


def _conditional_uncertainty_assessment(
    parameter_bootstrap_audit: dict[str, object] | None = None,
    driver_scenario_audit: dict[str, object] | None = None,
) -> dict[str, object]:
    driver_sources = (
        set(driver_scenario_audit.get("uncertainty_sources", []))
        if driver_scenario_audit is not None
        else set()
    )
    if not driver_sources.issubset(PROPAGATABLE_DRIVER_UNCERTAINTY_SOURCES):
        raise ValueError("driver scenario uncertainty sources are not recognized")
    included_sources = REQUIRED_UNCERTAINTY_INCLUSIONS | driver_sources
    assessment = {
        "method": (
            DRIVER_ENSEMBLE_UNCERTAINTY_METHOD
            if driver_scenario_audit is not None
            else FIXED_DRIVER_UNCERTAINTY_METHOD
        ),
        "scope": (
            DRIVER_ENSEMBLE_UNCERTAINTY_SCOPE
            if driver_scenario_audit is not None
            else CONDITIONAL_UNCERTAINTY_SCOPE
        ),
        "interval_levels": [0.80, 0.95],
        "output_targets": ["batch_fresh_kg_m2", "cumulative_fresh_kg_m2"],
        "block_boundary_policy": "never_cross_training_season_boundaries",
        "forecast_cumulative_reset_policy": "reset_for_each_forecast_season",
        "included_sources": sorted(included_sources),
        "excluded_sources": sorted(
            ALL_DECLARED_UNCERTAINTY_SOURCES - included_sources
        ),
    }
    if parameter_bootstrap_audit is not None:
        assessment["parameter_bootstrap"] = parameter_bootstrap_audit
    if driver_scenario_audit is not None:
        assessment["driver_scenarios"] = driver_scenario_audit
    return assessment


def _strict_boolean_series(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = values.map(
        lambda value: mapping.get(value.lower().strip() if isinstance(value, str) else value)
    )
    if parsed.isna().any():
        raise ValueError(f"{name} must contain strict boolean values")
    return parsed.astype(bool)


def validate_harvest_pipeline_inputs(
    *,
    harvest_events_path: str | Path | None,
    season_manifest_path: str | Path | None,
    harvest_workbook_path: str | Path | None = None,
    drivers_path: str | Path | None,
    harvest_protocol_path: str | Path | None = None,
    uncertainty_driver_scenarios_path: str | Path | None = None,
) -> tuple[str | Path | None, str | Path | None, str | Path | None]:
    if harvest_workbook_path is not None:
        if (
            harvest_events_path is not None
            or season_manifest_path is not None
            or harvest_protocol_path is not None
        ):
            raise ValueError(
                "--harvest-workbook is mutually exclusive with "
                "--harvest-events, --season-manifest, and --harvest-protocol"
            )
        if Path(harvest_workbook_path).suffix.lower() not in {".xlsx", ".xlsm"}:
            raise ValueError("--harvest-workbook must be an XLSX or XLSM file")
        harvest_events_path = harvest_workbook_path
        season_manifest_path = harvest_workbook_path
        harvest_protocol_path = harvest_workbook_path
    if harvest_events_path is not None and season_manifest_path is None:
        raise ValueError("--harvest-events requires --season-manifest")
    if season_manifest_path is not None and harvest_events_path is None:
        raise ValueError("--season-manifest requires --harvest-events")
    if harvest_protocol_path is not None and (
        harvest_events_path is None or season_manifest_path is None
    ):
        raise ValueError(
            "--harvest-protocol requires --harvest-events and --season-manifest"
        )
    if drivers_path is not None and harvest_events_path is None:
        raise ValueError("--drivers requires --harvest-events and --season-manifest")
    if drivers_path is not None and harvest_protocol_path is None:
        raise ValueError("--drivers requires --harvest-protocol")
    if uncertainty_driver_scenarios_path is not None and drivers_path is None:
        raise ValueError("--uncertainty-driver-scenarios requires --drivers")
    return harvest_events_path, season_manifest_path, harvest_protocol_path


def resolve_target_plant_density(
    target_site: dict[str, object], *, consistency_tolerance: float = 0.10
) -> tuple[float, dict[str, object]]:
    registered = float(target_site["registered_plant_density_plants_m2"])
    plant_count = float(target_site["registered_plant_count"])
    greenhouse_area = float(target_site["greenhouse_area_m2"])
    values = np.asarray([registered, plant_count, greenhouse_area], dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("target plant density inputs must be finite and positive")
    count_area_density = plant_count / greenhouse_area
    relative_difference = abs(registered - count_area_density) / count_area_density
    selected = count_area_density
    audit = {
        "registered_density_plants_m2": registered,
        "registered_plant_count": int(plant_count),
        "greenhouse_area_m2": greenhouse_area,
        "count_area_density_plants_m2": count_area_density,
        "selected_density_plants_m2": selected,
        "selected_source": "registered_plant_count_divided_by_greenhouse_area",
        "registered_density_relative_difference": relative_difference,
        "registered_density_consistent": bool(
            relative_difference <= float(consistency_tolerance)
        ),
        "consistency_tolerance": float(consistency_tolerance),
    }
    return selected, audit


def evaluate_target_acceptance(
    metrics: dict[str, float],
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, object]:
    limits = dict(DEFAULT_ACCEPTANCE_THRESHOLDS)
    if thresholds:
        limits.update(thresholds)
    missing = sorted(set(limits) - set(metrics))
    if missing:
        return {
            "accepted": False,
            "gates": {name: False for name in limits},
            "missing_metrics": missing,
            "thresholds": limits,
        }
    gates = {
        "first_harvest_date_error_days": abs(
            float(metrics["first_harvest_date_error_days"])
        ) <= limits["first_harvest_date_error_days"],
        "harvest_timing_wasserstein_days": (
            float(metrics["harvest_timing_wasserstein_days"])
            <= limits["harvest_timing_wasserstein_days"]
        ),
        "median_harvest_date_error_days": abs(
            float(metrics["median_harvest_date_error_days"])
        ) <= limits["median_harvest_date_error_days"],
        "p90_harvest_date_error_days": abs(
            float(metrics["p90_harvest_date_error_days"])
        ) <= limits["p90_harvest_date_error_days"],
        "batch_wmape": float(metrics["batch_wmape"]) <= limits["batch_wmape"],
        "cumulative_wmape": (
            float(metrics["cumulative_wmape"]) <= limits["cumulative_wmape"]
        ),
        "final_yield_bias_percent": abs(
            float(metrics["final_yield_bias_percent"])
        ) <= limits["final_yield_bias_percent"],
        "cumulative_r2": float(metrics["cumulative_r2"]) >= limits["cumulative_r2"],
        "batch_interval_80_coverage": (
            float(metrics["batch_interval_80_coverage"])
            >= limits["batch_interval_80_coverage"]
        ),
        "batch_interval_95_coverage": (
            float(metrics["batch_interval_95_coverage"])
            >= limits["batch_interval_95_coverage"]
        ),
        "batch_interval_80_normalized_mean_width": (
            float(metrics["batch_interval_80_normalized_mean_width"])
            <= limits["batch_interval_80_normalized_mean_width"]
        ),
        "batch_interval_95_normalized_mean_width": (
            float(metrics["batch_interval_95_normalized_mean_width"])
            <= limits["batch_interval_95_normalized_mean_width"]
        ),
        "cumulative_interval_80_coverage": (
            float(metrics["cumulative_interval_80_coverage"])
            >= limits["cumulative_interval_80_coverage"]
        ),
        "cumulative_interval_95_coverage": (
            float(metrics["cumulative_interval_95_coverage"])
            >= limits["cumulative_interval_95_coverage"]
        ),
        "cumulative_interval_80_normalized_mean_width": (
            float(metrics["cumulative_interval_80_normalized_mean_width"])
            <= limits["cumulative_interval_80_normalized_mean_width"]
        ),
        "cumulative_interval_95_normalized_mean_width": (
            float(metrics["cumulative_interval_95_normalized_mean_width"])
            <= limits["cumulative_interval_95_normalized_mean_width"]
        ),
    }
    return {
        "accepted": all(gates.values()),
        "gates": gates,
        "missing_metrics": [],
        "thresholds": limits,
    }


def build_harvest_readiness_report(
    crop_observations: pd.DataFrame,
    harvest_events: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    validation_result: dict[str, Any] | None = None,
    standing_crop_validation: dict[str, Any] | None = None,
) -> dict[str, object]:
    harvest_status = assess_target_harvest_data(
        harvest_events,
        target_greenhouse_id=target_greenhouse_id,
        target_greenhouse_code=target_greenhouse_code,
    )
    crop_target_flags = _strict_boolean_series(
        crop_observations.get(
            "target_eligible",
            pd.Series(False, index=crop_observations.index),
        ),
        "crop target_eligible",
    )
    target_crop_count = int(crop_target_flags.sum())
    source_crop_count = int(len(crop_observations) - target_crop_count)
    standing_summary: dict[str, object] | None = None
    if standing_crop_validation is not None:
        identity_matches = bool(
            standing_crop_validation.get("target_greenhouse_id")
            == int(target_greenhouse_id)
            and str(standing_crop_validation.get("target_greenhouse_code"))
            == str(target_greenhouse_code)
        )
        precalibration = standing_crop_validation.get(
            "standing_crop_precalibration", {}
        )
        validation_metrics = (
            precalibration.get("validation_metrics", {})
            if isinstance(precalibration, dict)
            else {}
        )
        validation_wmape = validation_metrics.get("wmape")
        phenology = standing_crop_validation.get(
            "temperature_driven_phenology_validation", {}
        )
        phenology_wmape = phenology.get("wmape") if isinstance(phenology, dict) else None
        organ_calibration = standing_crop_validation.get(
            "fixed_multi_organ_calibration", {}
        )
        organ_validation = (
            organ_calibration.get("validation_metrics", {})
            if isinstance(organ_calibration, dict)
            else {}
        )
        organ_validation_nrmse = (
            organ_validation.get("organ_normalized_rmse")
            if isinstance(organ_validation, dict)
            else None
        )
        thermal_sum_sensitivity = standing_crop_validation.get(
            "initial_thermal_sum_sensitivity", {}
        )
        thermal_sum_contains = (
            thermal_sum_sensitivity.get("validation_envelope_contains_observed")
            if isinstance(thermal_sum_sensitivity, dict)
            else None
        )
        thermal_sum_best_error = (
            thermal_sum_sensitivity.get("best_case_absolute_relative_error")
            if isinstance(thermal_sum_sensitivity, dict)
            else None
        )
        biomass_flux = standing_crop_validation.get(
            "late_biomass_flux_diagnostics", {}
        )
        late_biomass_fraction = (
            biomass_flux.get("model_fraction_of_observed_change")
            if isinstance(biomass_flux, dict)
            else None
        )
        required_gross_ratio = (
            biomass_flux.get("required_to_modeled_gross_ratio")
            if isinstance(biomass_flux, dict)
            else None
        )
        capacity_sensitivity = standing_crop_validation.get(
            "assimilation_capacity_sensitivity", {}
        )
        capacity_contains = (
            capacity_sensitivity.get("validation_fresh_envelope_contains_observed")
            if isinstance(capacity_sensitivity, dict)
            else None
        )
        capacity_best_error = (
            capacity_sensitivity.get(
                "best_case_validation_fresh_absolute_relative_error"
            )
            if isinstance(capacity_sensitivity, dict)
            else None
        )
        held_out_dates = int(validation_metrics.get("observation_date_count", 0))
        evidence_sufficient = bool(
            identity_matches
            and held_out_dates >= 2
            and validation_wmape is not None
            and float(validation_wmape) <= 0.25
        )
        standing_summary = {
            "identity_matches": identity_matches,
            "status": standing_crop_validation.get("status"),
            "scored_observation_dates": int(
                standing_crop_validation.get("scored_observation_dates", 0)
            ),
            "held_out_validation_dates": held_out_dates,
            "validation_wmape": float(validation_wmape)
            if validation_wmape is not None
            else None,
            "phenology_wmape": float(phenology_wmape)
            if phenology_wmape is not None
            else None,
            "organ_validation_nrmse": float(organ_validation_nrmse)
            if organ_validation_nrmse is not None
            else None,
            "thermal_sum_envelope_contains_observed": (
                bool(thermal_sum_contains)
                if thermal_sum_contains is not None
                else None
            ),
            "thermal_sum_best_case_relative_error": (
                float(thermal_sum_best_error)
                if thermal_sum_best_error is not None
                else None
            ),
            "late_biomass_fraction_of_observed": (
                float(late_biomass_fraction)
                if late_biomass_fraction is not None
                else None
            ),
            "required_to_modeled_gross_ratio": (
                float(required_gross_ratio)
                if required_gross_ratio is not None
                else None
            ),
            "capacity_envelope_contains_observed": (
                bool(capacity_contains) if capacity_contains is not None else None
            ),
            "capacity_best_case_relative_error": (
                float(capacity_best_error)
                if capacity_best_error is not None
                else None
            ),
            "wmape_acceptance_max": 0.25,
            "minimum_held_out_validation_dates": 2,
            "evidence_sufficient": evidence_sufficient,
            "metrics_scope": "target_standing_fruit_not_harvest_yield",
        }
    metrics = validation_result.get("test_metrics") if validation_result else None
    metrics_scope = validation_result.get("metrics_scope") if validation_result else None
    result_thresholds = (
        validation_result.get("acceptance_thresholds")
        if validation_result
        else None
    )
    acceptance = (
        evaluate_target_acceptance(
            metrics,
            thresholds=result_thresholds if isinstance(result_thresholds, dict) else None,
        )
        if isinstance(metrics, dict)
        else None
    )
    evidence_identity_matches = bool(
        validation_result
        and validation_result.get("target_greenhouse_id") == int(target_greenhouse_id)
        and str(validation_result.get("target_greenhouse_code"))
        == str(target_greenhouse_code)
    )
    driver_validation_passed = bool(
        validation_result and validation_result.get("driver_validation_passed") is True
    )
    uncertainty_assessment = (
        validation_result.get("uncertainty_assessment") if validation_result else None
    )
    included_uncertainty_sources = set(
        uncertainty_assessment.get("included_sources", [])
        if isinstance(uncertainty_assessment, dict)
        else []
    )
    excluded_uncertainty_sources = set(
        uncertainty_assessment.get("excluded_sources", [])
        if isinstance(uncertainty_assessment, dict)
        else []
    )
    uncertainty_method = (
        uncertainty_assessment.get("method")
        if isinstance(uncertainty_assessment, dict)
        else None
    )
    uncertainty_scope = (
        uncertainty_assessment.get("scope")
        if isinstance(uncertainty_assessment, dict)
        else None
    )
    method_scope_valid = (
        uncertainty_method == FIXED_DRIVER_UNCERTAINTY_METHOD
        and uncertainty_scope == CONDITIONAL_UNCERTAINTY_SCOPE
    ) or (
        uncertainty_method == DRIVER_ENSEMBLE_UNCERTAINTY_METHOD
        and uncertainty_scope == DRIVER_ENSEMBLE_UNCERTAINTY_SCOPE
        and isinstance(uncertainty_assessment.get("driver_scenarios"), dict)
        and int(
            uncertainty_assessment.get("driver_scenarios", {}).get(
                "scenario_count", 0
            )
        )
        >= 2
    )
    source_accounting_complete = bool(
        not (included_uncertainty_sources & excluded_uncertainty_sources)
        and included_uncertainty_sources | excluded_uncertainty_sources
        == ALL_DECLARED_UNCERTAINTY_SOURCES
        and REQUIRED_UNCERTAINTY_INCLUSIONS.issubset(
            included_uncertainty_sources
        )
    )
    uncertainty_evidence_complete = bool(
        isinstance(uncertainty_assessment, dict)
        and method_scope_valid
        and uncertainty_assessment.get("interval_levels") == [0.80, 0.95]
        and uncertainty_assessment.get("output_targets")
        == ["batch_fresh_kg_m2", "cumulative_fresh_kg_m2"]
        and uncertainty_assessment.get("block_boundary_policy")
        == "never_cross_training_season_boundaries"
        and uncertainty_assessment.get("forecast_cumulative_reset_policy")
        == "reset_for_each_forecast_season"
        and source_accounting_complete
    )
    test_seasons = (
        [str(value) for value in validation_result.get("split", {}).get("test_seasons", [])]
        if validation_result else []
    )
    latest_target_season: str | None = None
    if not harvest_events.empty:
        event_times = pd.to_datetime(harvest_events["timestamp"], errors="coerce")
        target_mask = (
            pd.to_numeric(harvest_events["greenhouse_id"], errors="coerce").eq(
                int(target_greenhouse_id)
            )
            & harvest_events["greenhouse_code"].astype(str).eq(str(target_greenhouse_code))
            & _strict_boolean_series(harvest_events["target_eligible"], "target_eligible")
        )
        target_times = pd.DataFrame(
            {
                "season_id": harvest_events.loc[target_mask, "season_id"].astype(str),
                "timestamp": event_times.loc[target_mask],
            }
        )
        if not target_times.empty:
            latest_target_season = str(
                target_times.groupby("season_id")["timestamp"].max().idxmax()
            )
    test_season_matches = bool(
        latest_target_season is not None and test_seasons == [latest_target_season]
    )
    blocking_reasons: list[str] = []
    if not harvest_status["can_calibrate_target"]:
        blocking_reasons.append(str(harvest_status["status"]))
    if not harvest_status["can_validate_target_independently"]:
        blocking_reasons.append("independent_target_season_missing")
    if metrics is None:
        blocking_reasons.append("held_out_target_metrics_missing")
    elif metrics_scope != "held_out_target_test":
        blocking_reasons.append("metrics_are_not_held_out_target_test")
    elif acceptance is not None and not acceptance["accepted"]:
        blocking_reasons.append("held_out_acceptance_gates_failed")
    if validation_result is not None and not evidence_identity_matches:
        blocking_reasons.append("validation_target_identity_mismatch")
    if not driver_validation_passed:
        blocking_reasons.append("driver_validation_evidence_missing")
    if validation_result is not None and not test_season_matches:
        blocking_reasons.append("held_out_test_is_not_latest_target_season")
    if not uncertainty_evidence_complete:
        blocking_reasons.append("conditional_uncertainty_evidence_missing")
    standing_evidence_required = target_crop_count > 0
    if standing_evidence_required and standing_summary is None:
        blocking_reasons.append("standing_crop_validation_missing")
    elif standing_summary is not None and not standing_summary["evidence_sufficient"]:
        blocking_reasons.append("standing_crop_validation_insufficient")
    target_validated = bool(
        harvest_status["can_validate_target_independently"]
        and evidence_identity_matches
        and driver_validation_passed
        and test_season_matches
        and uncertainty_evidence_complete
        and metrics_scope == "held_out_target_test"
        and acceptance is not None
        and acceptance["accepted"]
        and (
            not standing_evidence_required
            or (
                standing_summary is not None
                and standing_summary["evidence_sufficient"] is True
            )
        )
    )
    return {
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "source_crop_observation_count": source_crop_count,
        "target_crop_observation_count": target_crop_count,
        "harvest_data": harvest_status,
        "held_out_acceptance": acceptance,
        "metrics_scope": metrics_scope,
        "held_out_test_seasons": test_seasons,
        "uncertainty_assessment": uncertainty_assessment,
        "standing_crop_validation": standing_summary,
        "target_validated": target_validated,
        "blocking_reasons": blocking_reasons,
    }


def combine_crop_observation_sources(
    sql_observations: pd.DataFrame,
    target_workbook_observations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine canonical source and target samples without changing semantics."""
    target = (
        target_workbook_observations
        if target_workbook_observations is not None
        else pd.DataFrame(columns=CROP_OBSERVATION_COLUMNS)
    )
    combined = pd.concat(
        [sql_observations[CROP_OBSERVATION_COLUMNS], target[CROP_OBSERVATION_COLUMNS]],
        ignore_index=True,
    )
    if combined["observation_id"].duplicated().any():
        raise ValueError("crop observation IDs must be unique across SQL and workbook sources")
    combined = combined.sort_values(
        ["observation_date", "observation_id"], kind="stable"
    ).reset_index(drop=True)
    validate_crop_observations(combined)
    combined.attrs["duplicate_rows_removed"] = int(
        sql_observations.attrs.get("duplicate_rows_removed", 0)
    )
    combined.attrs["source_row_counts"] = {
        "sql": int(len(sql_observations)),
        "target_workbook": int(len(target)),
    }
    return combined


def calibrate_and_validate_harvest_seasons(
    drivers: pd.DataFrame,
    harvest_events: pd.DataFrame,
    *,
    protocols: pd.DataFrame,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    base_temperature_c: float,
    maturity_bounds_deg_day: tuple[float, float],
    dry_matter_fraction_bounds: tuple[float, float],
    maturity_grid_size: int = 61,
    parameter_bootstrap_samples: int = 200,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
    acceptance_thresholds: dict[str, float] | None = None,
    minimum_driver_span_days: float = 90.0,
    minimum_harvest_events_per_season: int = 8,
    minimum_harvest_span_days: float = 30.0,
    maximum_driver_interval_hours: float = 24.0,
    minimum_driver_coverage_fraction: float = 0.95,
    uncertainty_driver_scenarios: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Fit complete training seasons and evaluate one untouched target season."""
    if "season_id" not in drivers.columns or "season_id" not in harvest_events.columns:
        raise ValueError("drivers and harvest events must include season_id")
    if "pick_source" not in drivers.columns or not drivers["pick_source"].eq(
        "management_protocol"
    ).all():
        raise ValueError(
            "pick_source must be management_protocol and must not be derived from harvest events"
        )
    if "driver_model_status" not in drivers.columns or not drivers[
        "driver_model_status"
    ].eq("accepted_target_crop_model").all():
        raise ValueError(
            "driver_model_status must be accepted_target_crop_model for calibration"
        )
    if "fruit_change_source" not in drivers.columns or not drivers[
        "fruit_change_source"
    ].eq("greenlight_fruit_state_balance").all():
        raise ValueError(
            "fruit_change_source must be greenlight_fruit_state_balance"
        )
    allowed_temperature_sources = {
        "observed_indoor",
        "greenlight_simulated_indoor",
    }
    if "air_temperature_source" not in drivers.columns or not set(
        drivers["air_temperature_source"].astype(str)
    ).issubset(allowed_temperature_sources):
        raise ValueError(
            "air_temperature_source must be observed_indoor or "
            "greenlight_simulated_indoor"
        )
    if "season_complete" not in drivers.columns:
        raise ValueError("season_complete must contain strict boolean values")
    complete = _strict_boolean_series(drivers["season_complete"], "season_complete")
    if not complete.all():
        raise ValueError("all calibration and validation driver seasons must be complete")
    driver_frame = drivers.copy()
    driver_frame["season_complete"] = complete
    driver_frame["pick"] = _strict_boolean_series(driver_frame["pick"], "pick")
    identity_columns = {"greenhouse_id", "greenhouse_code", "planting_code"}
    missing_identity = sorted(identity_columns - set(driver_frame.columns))
    if missing_identity:
        raise ValueError(f"driver identity columns missing: {', '.join(missing_identity)}")
    driver_ids = pd.to_numeric(driver_frame["greenhouse_id"], errors="coerce")
    identity_matches = (
        driver_ids.eq(int(target_greenhouse_id))
        & driver_frame["greenhouse_code"].astype(str).eq(str(target_greenhouse_code))
    )
    if driver_ids.isna().any() or not identity_matches.all():
        raise ValueError("driver greenhouse identity does not match the target greenhouse")
    required_numeric = {
        "net_fruit_dry_matter_change_kg_m2",
        "air_temperature_c",
        "dt_hours",
        "initial_fruit_dry_matter_kg_m2",
        "initial_fruit_maturity_fraction",
    }
    missing_numeric = sorted(required_numeric - set(driver_frame.columns))
    if missing_numeric:
        raise ValueError(f"driver columns missing: {', '.join(missing_numeric)}")
    for column in required_numeric:
        driver_frame[column] = pd.to_numeric(driver_frame[column], errors="coerce")
    if not np.isfinite(driver_frame[list(required_numeric)].to_numpy(dtype=float)).all():
        raise ValueError("driver numeric values must be finite")
    if (driver_frame["dt_hours"] <= 0.0).any():
        raise ValueError("driver dt_hours must be positive")
    driver_frame["timestamp"] = pd.to_datetime(driver_frame["timestamp"], errors="coerce")
    if driver_frame["timestamp"].isna().any():
        raise ValueError("driver timestamps must be valid")
    if driver_frame.duplicated(["season_id", "timestamp"]).any():
        raise ValueError("driver timestamps must be unique within each season")
    order = (
        driver_frame.groupby("season_id")["timestamp"]
        .min()
        .sort_values()
        .index.astype(str)
        .tolist()
    )
    event_seasons = set(harvest_events["season_id"].astype(str))
    usable = [season for season in order if season in event_seasons]
    if len(usable) < 2:
        raise ValueError("at least two complete seasons are required for held-out validation")
    event_frame = harvest_events.copy()
    event_frame["timestamp"] = pd.to_datetime(event_frame["timestamp"], errors="coerce")
    if event_frame["timestamp"].isna().any():
        raise ValueError("harvest event timestamps must be valid")
    event_identity = {"greenhouse_id", "greenhouse_code", "planting_code", "target_eligible"}
    missing_event_identity = sorted(event_identity - set(event_frame.columns))
    if missing_event_identity:
        raise ValueError(
            f"harvest event identity columns missing: {', '.join(missing_event_identity)}"
        )
    eligible = _strict_boolean_series(event_frame["target_eligible"], "target_eligible")
    event_ids = pd.to_numeric(event_frame["greenhouse_id"], errors="coerce")
    if (
        event_ids.isna().any()
        or not event_ids.eq(int(target_greenhouse_id)).all()
        or not event_frame["greenhouse_code"].astype(str).eq(str(target_greenhouse_code)).all()
        or not eligible.all()
    ):
        raise ValueError("harvest event identity does not match the target greenhouse")
    for season_id in usable:
        season_drivers = driver_frame[
            driver_frame["season_id"].astype(str) == season_id
        ]
        season_events = event_frame[
            event_frame["season_id"].astype(str) == season_id
        ]
        season_drivers = season_drivers.sort_values("timestamp", kind="stable")
        driver_span = float(
            (
                season_drivers["timestamp"].max()
                - season_drivers["timestamp"].min()
            ).total_seconds()
            / 86400.0
        )
        event_span = float(
            (season_events["timestamp"].max() - season_events["timestamp"].min()).total_seconds()
            / 86400.0
        )
        gaps_hours = (
            season_drivers["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
        )
        nominal_interval_hours = float(season_drivers["dt_hours"].iloc[0])
        if not np.allclose(
            season_drivers["dt_hours"].to_numpy(dtype=float),
            nominal_interval_hours,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"season {season_id} must use one fixed nominal driver interval")
        if gaps_hours.empty or (gaps_hours > float(maximum_driver_interval_hours) + 1e-8).any():
            raise ValueError(
                f"season {season_id} driver interval exceeds "
                f"{maximum_driver_interval_hours:g} hours"
            )
        row_dt = season_drivers["dt_hours"].iloc[1:].to_numpy(dtype=float)
        if not np.allclose(gaps_hours.to_numpy(dtype=float), row_dt, rtol=1e-6, atol=1e-6):
            raise ValueError(f"season {season_id} dt_hours does not match timestamp intervals")
        expected_rows = int(np.floor(driver_span * 24.0 / nominal_interval_hours)) + 1
        coverage_fraction = len(season_drivers) / max(expected_rows, 1)
        if coverage_fraction < float(minimum_driver_coverage_fraction):
            raise ValueError(f"season {season_id} driver coverage is too sparse")
        initial_mass = season_drivers["initial_fruit_dry_matter_kg_m2"]
        initial_age = season_drivers["initial_fruit_maturity_fraction"]
        if (initial_mass < 0.0).any() or not initial_mass.iloc[1:].eq(0.0).all():
            raise ValueError("initial fruit inventory must be non-negative and occur once per season")
        if not initial_age.between(0.0, 1.0).all() or not initial_age.iloc[1:].eq(0.0).all():
            raise ValueError("initial fruit maturity fraction must be in [0, 1] and occur once")
        driver_plantings = set(season_drivers["planting_code"].astype(str))
        event_plantings = set(season_events["planting_code"].astype(str))
        if len(driver_plantings) != 1 or driver_plantings != event_plantings:
            raise ValueError(f"season {season_id} driver and harvest planting identity mismatch")
        if driver_span < float(minimum_driver_span_days):
            raise ValueError(
                f"season {season_id} driver coverage must span at least "
                f"{minimum_driver_span_days:g} days"
            )
        if len(season_events) < int(minimum_harvest_events_per_season):
            raise ValueError(
                f"season {season_id} requires at least "
                f"{minimum_harvest_events_per_season} harvest events"
            )
        if event_span < float(minimum_harvest_span_days):
            raise ValueError(
                f"season {season_id} harvest events must span at least "
                f"{minimum_harvest_span_days:g} days"
            )
        if (
            season_drivers["timestamp"].min() > season_events["timestamp"].min()
            or season_drivers["timestamp"].max() < season_events["timestamp"].max()
        ):
            raise ValueError(
                f"season {season_id} drivers must cover every harvest event"
            )
    if len(usable) == 2:
        train_seasons = [usable[0]]
        validation_seasons: list[str] = []
        test_seasons = [usable[1]]
    else:
        train_seasons = usable[:-2]
        validation_seasons = [usable[-2]]
        test_seasons = [usable[-1]]
    season_bounds = [
        (
            season,
            driver_frame.loc[
                driver_frame["season_id"].astype(str).eq(season), "timestamp"
            ].min(),
            driver_frame.loc[
                driver_frame["season_id"].astype(str).eq(season), "timestamp"
            ].max(),
        )
        for season in usable
    ]
    if any(previous[2] >= following[1] for previous, following in zip(season_bounds, season_bounds[1:])):
        raise ValueError("driver seasons overlap and are not strictly chronological")

    package_assessment = assess_harvest_calibration_package(
        driver_frame,
        event_frame,
        target_greenhouse_id=target_greenhouse_id,
        target_greenhouse_code=target_greenhouse_code,
        minimum_driver_span_days=minimum_driver_span_days,
        minimum_harvest_events_per_season=minimum_harvest_events_per_season,
        minimum_harvest_span_days=minimum_harvest_span_days,
        maximum_driver_interval_hours=maximum_driver_interval_hours,
        minimum_driver_coverage_fraction=minimum_driver_coverage_fraction,
        protocols=protocols,
    )
    require_independent_validation_package(package_assessment)

    test_drivers = driver_frame[
        driver_frame["season_id"].astype(str).isin(test_seasons)
    ].copy()
    canonical_driver_scenarios: pd.DataFrame | None = None
    driver_scenario_audit: dict[str, object] | None = None
    if uncertainty_driver_scenarios is not None:
        canonical_driver_scenarios, driver_scenario_audit = (
            validate_uncertainty_driver_scenarios(
                test_drivers,
                uncertainty_driver_scenarios,
            )
        )

    train_drivers = driver_frame[
        driver_frame["season_id"].astype(str).isin(train_seasons)
    ].copy()
    train_observed = harvest_events[
        harvest_events["season_id"].astype(str).isin(train_seasons)
    ][["season_id", "timestamp", "fresh_kg_m2"]].copy()
    parameter_bootstrap = bootstrap_harvest_parameter_estimates(
        train_drivers,
        train_observed,
        base_temperature_c=base_temperature_c,
        maturity_bounds_deg_day=maturity_bounds_deg_day,
        dry_matter_fraction_bounds=dry_matter_fraction_bounds,
        maturity_grid_size=maturity_grid_size,
        bootstrap_samples=parameter_bootstrap_samples,
        block_days=7,
        bootstrap_seed=bootstrap_seed,
    )
    calibration = parameter_bootstrap["point_calibration"]
    fitted = calibration["parameters"]
    parameters = HarvestCohortParameters(
        base_temperature_c=float(fitted["base_temperature_c"]),
        maturity_thermal_time_deg_day=float(
            fitted["maturity_thermal_time_deg_day"]
        ),
        dry_matter_fraction=float(fitted["dry_matter_fraction"]),
    )

    validation_metrics: dict[str, float] | None = None
    if validation_seasons:
        validation_drivers = driver_frame[
            driver_frame["season_id"].astype(str).isin(validation_seasons)
        ]
        validation_observed = harvest_events[
            harvest_events["season_id"].astype(str).isin(validation_seasons)
        ][["timestamp", "fresh_kg_m2"]]
        validation_prediction = simulate_harvest_series(validation_drivers, parameters)
        validation_metrics, _ = evaluate_harvest_predictions(
            validation_observed,
            validation_prediction[["timestamp", "fresh_kg_m2"]],
        )

    test_observed = harvest_events[
        harvest_events["season_id"].astype(str).isin(test_seasons)
    ][["season_id", "timestamp", "fresh_kg_m2"]]
    test_prediction = simulate_harvest_series(test_drivers, parameters)
    _, test_aligned = evaluate_harvest_predictions(
        test_observed,
        test_prediction[["season_id", "timestamp", "fresh_kg_m2"]],
    )
    intervals = joint_harvest_prediction_intervals(
        calibration["aligned_training_predictions"],
        test_drivers,
        parameter_bootstrap["parameter_draws"],
        forecast_index=test_aligned[["season_id", "date"]],
        base_temperature_c=base_temperature_c,
        forecast_driver_scenarios=canonical_driver_scenarios,
        bootstrap_samples=bootstrap_samples,
        block_days=7,
        bootstrap_seed=bootstrap_seed,
    )
    scored_prediction = pd.DataFrame(
        {
            "season_id": test_aligned["season_id"],
            "timestamp": test_aligned["date"],
            "fresh_kg_m2": test_aligned["predicted_fresh_kg_m2"],
        }
    ).merge(
        intervals,
        left_on=["season_id", "timestamp"],
        right_on=["season_id", "date"],
        how="left",
    ).drop(columns="date")
    test_metrics, final_aligned = evaluate_harvest_predictions(
        test_observed,
        scored_prediction,
    )
    acceptance = evaluate_target_acceptance(
        test_metrics, thresholds=acceptance_thresholds
    )
    uncertainty_summary = _conditional_uncertainty_assessment(
        parameter_bootstrap["audit"],
        driver_scenario_audit,
    )
    return {
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "split": {
            "train_seasons": train_seasons,
            "validation_seasons": validation_seasons,
            "test_seasons": test_seasons,
        },
        "parameters": fitted,
        "training_metrics": calibration["training_metrics"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "metrics_scope": "held_out_target_test",
        "batch_semantics": "greenhouse_calendar_day_total",
        "driver_validation_passed": True,
        "calibration_package_assessment": package_assessment,
        "acceptance": acceptance,
        "acceptance_thresholds": acceptance["thresholds"],
        "candidate_scores": calibration["candidate_scores"],
        "parameter_bootstrap_draws": parameter_bootstrap["parameter_draws"],
        "parameter_bootstrap_audit": parameter_bootstrap["audit"],
        "driver_scenario_audit": driver_scenario_audit,
        "uncertainty_driver_scenarios": canonical_driver_scenarios,
        "test_predictions": final_aligned,
        "uncertainty_method": uncertainty_summary["method"],
        "uncertainty_assessment": uncertainty_summary,
    }


def run_pipeline(
    *,
    sql_path: str | Path,
    dataset_config_path: str | Path,
    output_root: str | Path,
    crop_workbook_path: str | Path | None = None,
    harvest_events_path: str | Path | None = None,
    season_manifest_path: str | Path | None = None,
    harvest_protocol_path: str | Path | None = None,
    harvest_workbook_path: str | Path | None = None,
    drivers_path: str | Path | None = None,
    uncertainty_driver_scenarios_path: str | Path | None = None,
    standing_crop_validation_path: str | Path | None = None,
    harvest_config_path: str | Path = "configs/crops/chengdu_tomato_harvest.yml",
    base_temperature_c: float | None = None,
) -> dict[str, Any]:
    (
        harvest_events_path,
        season_manifest_path,
        harvest_protocol_path,
    ) = validate_harvest_pipeline_inputs(
        harvest_events_path=harvest_events_path,
        season_manifest_path=season_manifest_path,
        harvest_protocol_path=harvest_protocol_path,
        harvest_workbook_path=harvest_workbook_path,
        drivers_path=drivers_path,
        uncertainty_driver_scenarios_path=uncertainty_driver_scenarios_path,
    )
    config = yaml.safe_load(Path(dataset_config_path).read_text(encoding="utf-8"))
    harvest_config = yaml.safe_load(
        Path(harvest_config_path).read_text(encoding="utf-8")
    )
    acceptance_thresholds = parse_acceptance_thresholds(
        harvest_config["validation"]["acceptance"]
    )
    standing_crop_validation: dict[str, Any] | None = None
    if standing_crop_validation_path is not None:
        standing_crop_validation = json.loads(
            Path(standing_crop_validation_path).read_text(encoding="utf-8")
        )
        if not isinstance(standing_crop_validation, dict):
            raise ValueError("standing crop validation artifact must contain a JSON object")
    crop_config = config["crop_observations"]
    target_code = str(crop_config["target_greenhouse_codes"][0])
    target_site = harvest_config["target_site"]
    normalization_areas_m2 = {
        target_code: float(target_site["greenhouse_area_m2"])
    }
    sql_observations = extract_crop_observations(
        sql_path,
        target_greenhouse_ids=crop_config["target_greenhouse_ids"],
        target_greenhouse_codes=crop_config["target_greenhouse_codes"],
        source_sites=crop_config["source_sites"],
        timezone=config["source"]["timezone"],
    )
    target_workbook_observations: pd.DataFrame | None = None
    crop_workbook_audit: dict[str, object] | None = None
    if crop_workbook_path is not None:
        target_density, density_audit = resolve_target_plant_density(target_site)
        target_workbook_observations, crop_workbook_audit = extract_target_crop_workbook(
            crop_workbook_path,
            greenhouse_id=int(target_site["greenhouse_id"]),
            greenhouse_code=str(target_site["greenhouse_code"]),
            planting_id=int(target_site["planting_id"]),
            planting_code=str(target_site["planting_code"]),
            cultivar=str(target_site["cultivar"]),
            plant_density_plants_m2=target_density,
            timezone=config["source"]["timezone"],
        )
        crop_workbook_audit["plant_density_resolution"] = density_audit
    observations = combine_crop_observation_sources(
        sql_observations, target_workbook_observations
    )
    output = Path(output_root)
    processed = output / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    observations.to_csv(
        processed / "standing_crop_samples.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if crop_workbook_audit is not None:
        (output / "crop_workbook_quality.json").write_text(
            json.dumps(crop_workbook_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    dry_matter = estimate_fruit_dry_matter_fraction(
        sql_observations,
        bootstrap_samples=10_000,
        bootstrap_seed=20260727,
    )
    if harvest_events_path is None:
        harvest_events = pd.DataFrame(
            columns=[
                "greenhouse_id",
                "greenhouse_code",
                "target_eligible",
                "season_id",
                "timestamp",
                "fresh_kg_m2",
            ]
        )
        season_manifest_audit: dict[str, Any] = {
            "status": "missing_target_harvest_events_and_season_manifest",
            "season_count": 0,
            "complete_season_count": 0,
            "ongoing_season_count": 0,
            "event_count": 0,
            "complete_season_event_count": 0,
            "ongoing_season_event_count": 0,
            "complete_seasons": [],
            "ongoing_seasons": [],
        }
        measured_harvest_summary: list[dict[str, object]] = []
        measured_batch_semantics = "greenhouse_calendar_day_total"
    else:
        harvest_events = load_harvest_events(
            harvest_events_path,
            target_greenhouse_ids=crop_config["target_greenhouse_ids"],
            target_greenhouse_codes=crop_config["target_greenhouse_codes"],
            normalization_areas_m2=normalization_areas_m2,
            source_sites=crop_config["source_sites"],
            timezone=config["source"]["timezone"],
        )
        harvest_seasons = load_harvest_seasons(
            season_manifest_path,
            target_greenhouse_ids=crop_config["target_greenhouse_ids"],
            target_greenhouse_codes=crop_config["target_greenhouse_codes"],
            timezone=config["source"]["timezone"],
        )
        harvest_events, season_manifest_audit = bind_harvest_events_to_seasons(
            harvest_events, harvest_seasons
        )
        measured_artifacts = build_measured_harvest_artifacts(harvest_events)
        measured_harvest_summary = measured_artifacts["season_summary"]
        measured_batch_semantics = str(measured_artifacts["batch_semantics"])
        harvest_events.to_csv(
            processed / "harvest_events.csv",
            index=False,
            encoding="utf-8-sig",
        )
        harvest_seasons.to_csv(
            processed / "harvest_seasons.csv",
            index=False,
            encoding="utf-8-sig",
        )
        measured_artifacts["event_trajectory"].to_csv(
            processed / "measured_harvest_event_trajectory.csv",
            index=False,
            encoding="utf-8-sig",
        )
        measured_artifacts["daily_trajectory"].to_csv(
            processed / "measured_daily_harvest_trajectory.csv",
            index=False,
            encoding="utf-8-sig",
        )
    experiment: dict[str, Any] | None = None
    if drivers_path is not None:
        if harvest_events_path is None:
            raise ValueError("--drivers requires --harvest-events")
        if base_temperature_c is None:
            raise ValueError("--drivers requires an explicit --base-temperature-c")
        calibration_config = harvest_config["calibration"]
        drivers = pd.read_csv(drivers_path)
        if harvest_protocol_path is None:
            raise ValueError("--drivers requires --harvest-protocol")
        protocols = load_harvest_protocols(
            harvest_protocol_path,
            target_greenhouse_ids=crop_config["target_greenhouse_ids"],
            target_greenhouse_codes=crop_config["target_greenhouse_codes"],
            timezone=config["source"]["timezone"],
        )
        protocols.to_csv(
            processed / "harvest_protocols.csv",
            index=False,
            encoding="utf-8-sig",
        )
        uncertainty_driver_scenarios = (
            pd.read_csv(uncertainty_driver_scenarios_path)
            if uncertainty_driver_scenarios_path is not None
            else None
        )
        target_events = harvest_events[
            harvest_events["target_eligible"]
            & harvest_events["season_complete_evidence"]
        ].copy()
        experiment = calibrate_and_validate_harvest_seasons(
            drivers,
            target_events,
            protocols=protocols,
            uncertainty_driver_scenarios=uncertainty_driver_scenarios,
            target_greenhouse_id=int(crop_config["target_greenhouse_ids"][0]),
            target_greenhouse_code=target_code,
            base_temperature_c=base_temperature_c,
            maturity_bounds_deg_day=tuple(
                calibration_config["maturity_thermal_time_deg_day_bounds"]
            ),
            dry_matter_fraction_bounds=tuple(
                calibration_config["dry_matter_fraction_bounds"]
            ),
            parameter_bootstrap_samples=int(
                harvest_config["uncertainty"]["parameter_estimation"][
                    "bootstrap_samples"
                ]
            ),
            bootstrap_samples=int(
                harvest_config["uncertainty"]["prediction_ensemble_samples"]
            ),
            bootstrap_seed=20260727,
            acceptance_thresholds=acceptance_thresholds,
        )
        experiment["candidate_scores"].to_csv(
            output / "candidate_scores.csv", index=False
        )
        experiment["test_predictions"].to_csv(
            output / "held_out_test_predictions.csv", index=False
        )
        experiment["parameter_bootstrap_draws"].to_csv(
            output / "parameter_bootstrap_draws.csv", index=False
        )
        canonical_driver_scenarios = experiment.get(
            "uncertainty_driver_scenarios"
        )
        if isinstance(canonical_driver_scenarios, pd.DataFrame):
            canonical_driver_scenarios.to_csv(
                processed / "uncertainty_driver_scenarios.csv",
                index=False,
                encoding="utf-8-sig",
            )
        serializable_experiment = {
            key: value
            for key, value in experiment.items()
            if key
            not in {
                "candidate_scores",
                "parameter_bootstrap_draws",
                "test_predictions",
                "uncertainty_driver_scenarios",
            }
        }
        (output / "calibration_and_validation.json").write_text(
            json.dumps(serializable_experiment, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    report = build_harvest_readiness_report(
        observations,
        harvest_events,
        target_greenhouse_id=int(crop_config["target_greenhouse_ids"][0]),
        target_greenhouse_code=target_code,
        validation_result=experiment,
        standing_crop_validation=standing_crop_validation,
    )
    report["crop_observation_audit"] = {
        "rows": int(len(observations)),
        "duplicates_removed": int(observations.attrs.get("duplicate_rows_removed", 0)),
        "placeholder_zero_rows": int(
            observations["quality_flags"].str.contains("placeholder_zero", na=False).sum()
        ),
        "source_row_counts": observations.attrs.get("source_row_counts", {}),
        "target_workbook": crop_workbook_audit,
        "observation_semantics": "standing_crop_sample_not_harvest_event",
    }
    report["source_site_dry_matter_fraction"] = dry_matter
    report["harvest_season_manifest_audit"] = season_manifest_audit
    report["measured_harvest_trajectory_audit"] = {
        "batch_semantics": measured_batch_semantics,
        "physical_event_artifact": (
            "processed/measured_harvest_event_trajectory.csv"
            if harvest_events_path is not None
            else None
        ),
        "daily_batch_artifact": (
            "processed/measured_daily_harvest_trajectory.csv"
            if harvest_events_path is not None
            else None
        ),
        "season_summary": measured_harvest_summary,
    }
    coverage = (
        harvest_events["harvest_coverage_fraction"].to_numpy(dtype=float)
        if "harvest_coverage_fraction" in harvest_events.columns
        and not harvest_events.empty
        else np.asarray([], dtype=float)
    )
    report["harvest_area_normalization"] = {
        "model_target_area_basis": "fixed_configured_production_area",
        "normalization_areas_m2": normalization_areas_m2,
        "actual_harvested_area_role": "coverage_and_local_intensity_audit_only",
        "fresh_kg_m2_formula": "harvested_fresh_kg_divided_by_normalization_area_m2",
        "event_count": int(len(harvest_events)),
        "coverage_fraction_min": float(coverage.min()) if len(coverage) else None,
        "coverage_fraction_mean": float(coverage.mean()) if len(coverage) else None,
        "coverage_fraction_max": float(coverage.max()) if len(coverage) else None,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chengdu harvest calibration inputs")
    parser.add_argument("--sql", required=True)
    parser.add_argument(
        "--dataset-config",
        default="configs/datasets/chengdu_agri_greenhouse_001.yml",
    )
    parser.add_argument(
        "--output-root",
        default="results/chengdu_agri_greenhouse_001/harvest_model",
    )
    parser.add_argument("--harvest-events")
    parser.add_argument("--season-manifest")
    parser.add_argument("--harvest-protocol")
    parser.add_argument("--harvest-workbook")
    parser.add_argument("--crop-workbook")
    parser.add_argument("--drivers")
    parser.add_argument("--uncertainty-driver-scenarios")
    parser.add_argument("--standing-crop-validation")
    parser.add_argument(
        "--harvest-config",
        default="configs/crops/chengdu_tomato_harvest.yml",
    )
    parser.add_argument("--base-temperature-c", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_pipeline(
        sql_path=args.sql,
        dataset_config_path=args.dataset_config,
        output_root=args.output_root,
        crop_workbook_path=args.crop_workbook,
        harvest_events_path=args.harvest_events,
        season_manifest_path=args.season_manifest,
        harvest_protocol_path=args.harvest_protocol,
        harvest_workbook_path=args.harvest_workbook,
        drivers_path=args.drivers,
        uncertainty_driver_scenarios_path=args.uncertainty_driver_scenarios,
        standing_crop_validation_path=args.standing_crop_validation,
        harvest_config_path=args.harvest_config,
        base_temperature_c=args.base_temperature_c,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
