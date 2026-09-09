from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from glassgym.configs.default_params import init_default_params

from experiments.crop.run_chengdu_target_crop_validation import (
    action_toward_control_target,
    calibrate_organ_allocation_multipliers,
    calibrate_organ_allocation_proportional,
    evaluate_organ_dry_predictions,
    summarize_initial_thermal_sum_sensitivity,
    summarize_biomass_flux_period,
    summarize_assimilation_capacity_sensitivity,
    summarize_stage_sink_structure_comparison,
    calibrate_fruit_allocation_multiplier,
    prepare_observed_climate,
    prepare_observed_control_targets,
    simulate_forced_crop_trajectory,
    stage_sink_parameter_multipliers,
)


def test_prepare_observed_control_targets_expands_hourly_states_without_lookahead():
    controls = pd.DataFrame(
        {
            "timestamp": ["2026-04-01 00:00:00", "2026-04-01 01:00:00"],
            "uBoil": [0.0, 0.0],
            "uCO2": [0.0, 0.0],
            "uThScr": [0.0, 1.0],
            "uVent": [0.0, 0.4],
            "uLamp": [0.0, 0.0],
            "uBlScr": [0.0, 0.2],
        }
    )

    expanded = prepare_observed_control_targets(
        controls,
        start="2026-04-01 00:00:00",
        end="2026-04-01 02:00:00",
        dt_seconds=900,
    )

    assert len(expanded) == 8
    assert str(expanded["timestamp"].dt.tz) == "Asia/Shanghai"
    assert expanded.loc[3, "uThScr"] == 0.0
    assert expanded.loc[4, "uThScr"] == 1.0
    assert expanded.loc[4, "uVent"] == 0.4
    assert expanded.loc[4, "uBlScr"] == 0.2


def test_prepare_observed_control_targets_rejects_missing_source_interval():
    controls = pd.DataFrame(
        {
            "timestamp": ["2026-04-01 00:00:00", "2026-04-01 03:00:00"],
            **{name: [0.0, 0.0] for name in ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")},
        }
    )

    with pytest.raises(ValueError, match="gap"):
        prepare_observed_control_targets(
            controls,
            start="2026-04-01 00:00:00",
            end="2026-04-01 04:00:00",
            dt_seconds=900,
        )


def test_prepare_observed_climate_uses_audited_co2_sensor_and_interpolates_invalid_rh():
    source = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=3, freq="1h"),
            "air_temperature": [18.0, 19.0, 20.0],
            "canopy_temperature": [18.5, 19.5, 20.5],
            "relative_humidity": [70.0, 0.0, 80.0],
            "co2_concentration": [np.nan, np.nan, np.nan],
            "co2_concentration__e3036__s1529": [420.0, 430.0, 440.0],
        }
    )

    climate, audit = prepare_observed_climate(
        source,
        start="2026-04-01 00:00:00",
        end="2026-04-01 02:00:00",
    )

    assert climate["relative_humidity"].tolist() == pytest.approx([70.0, 75.0, 80.0])
    assert climate["co2_concentration"].tolist() == [420.0, 430.0, 440.0]
    assert audit["co2_source_column"] == "co2_concentration__e3036__s1529"
    assert audit["invalid_relative_humidity_rows"] == 1
    assert audit["interpolated_relative_humidity_rows"] == 1


def test_action_toward_control_target_respects_rate_limit():
    action = action_toward_control_target(
        current=np.array([0.5, 0.5, 0.5]),
        target=np.array([0.0, 0.55, 1.0]),
        delta_u_max=0.1,
    )

    np.testing.assert_allclose(action, [-1.0, 0.5, 1.0])


@pytest.mark.parametrize(
    ("thermal_sum", "schedule", "expected"),
    [
        (400.0, None, {"fruit": 1.0, "leaf": 1.0, "stem": 1.0}),
        (
            400.0,
            {
                "reference_t_can_sum_deg_day": 500.0,
                "exponent": 1.0,
                "maximum_multiplier": 2.0,
                "organs": ["fruit", "leaf", "stem"],
            },
            {"fruit": 1.0, "leaf": 1.0, "stem": 1.0},
        ),
        (
            2000.0,
            {
                "reference_t_can_sum_deg_day": 500.0,
                "exponent": 0.5,
                "maximum_multiplier": 2.0,
                "organs": ["fruit", "leaf", "stem"],
            },
            {"fruit": 2.0, "leaf": 2.0, "stem": 2.0},
        ),
        (
            750.0,
            {
                "reference_t_can_sum_deg_day": 500.0,
                "exponent": 1.0,
                "maximum_multiplier": 2.0,
                "organs": ["fruit"],
            },
            {"fruit": 1.5, "leaf": 1.0, "stem": 1.0},
        ),
    ],
)
def test_stage_sink_parameter_multipliers_follow_registered_schedule(
    thermal_sum, schedule, expected
):
    assert stage_sink_parameter_multipliers(thermal_sum, schedule) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "schedule",
    [
        {
            "reference_t_can_sum_deg_day": 0.0,
            "exponent": 1.0,
            "maximum_multiplier": 2.0,
            "organs": ["fruit"],
        },
        {
            "reference_t_can_sum_deg_day": 500.0,
            "exponent": -1.0,
            "maximum_multiplier": 2.0,
            "organs": ["fruit"],
        },
        {
            "reference_t_can_sum_deg_day": 500.0,
            "exponent": 1.0,
            "maximum_multiplier": 0.5,
            "organs": ["fruit"],
        },
        {
            "reference_t_can_sum_deg_day": 500.0,
            "exponent": 1.0,
            "maximum_multiplier": 2.0,
            "organs": ["root"],
        },
    ],
)
def test_stage_sink_parameter_multipliers_reject_invalid_schedule(schedule):
    with pytest.raises(ValueError):
        stage_sink_parameter_multipliers(500.0, schedule)


def test_forced_climate_crop_replay_keeps_crop_states_finite_and_nonnegative():
    timestamps = pd.date_range("2026-04-01", periods=3, freq="1h", tz="Asia/Shanghai")
    climate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": [18.0, 19.0, 20.0],
            "canopy_temperature": [18.0, 19.5, 20.5],
            "relative_humidity": [75.0, 70.0, 68.0],
            "co2_concentration": [450.0, 450.0, 450.0],
        }
    )
    controls = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            **{name: [0.0, 0.0] for name in ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")},
        }
    )
    weather = np.zeros((3, 10), dtype=float)
    weather[:, 1] = 15.0
    weather[:, 2] = 1200.0
    weather[:, 3] = 720.0
    weather[:, 6] = 15.0

    trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state={
            "cBuf": 0.0,
            "cLeaf": 6000.0,
            "cStem": 7000.0,
            "cFruit": 0.0,
            "tCanSum": 0.0,
        },
        fruit_dry_matter_fraction=0.08,
        substep_seconds=300,
    )

    assert len(trajectory) == 3
    assert np.isfinite(trajectory.select_dtypes(include=[np.number])).all().all()
    assert (trajectory[["c_buffer_mg_m2", "c_leaf_mg_m2", "c_stem_mg_m2", "standing_dry_kg_m2"]] >= 0.0).all().all()
    assert trajectory.loc[0, "standing_fresh_kg_m2"] == 0.0
    assert trajectory.loc[2, "canopy_thermal_sum_deg_day"] > 0.0
    flux_columns = [
        "gross_photosynthesis_kg_m2",
        "buffer_to_leaf_kg_m2",
        "buffer_to_stem_kg_m2",
        "buffer_to_fruit_kg_m2",
        "growth_respiration_kg_m2",
        "leaf_maintenance_kg_m2",
        "stem_maintenance_kg_m2",
        "fruit_maintenance_kg_m2",
        "leaf_pruning_kg_m2",
        "native_harvested_dry_kg_m2",
        "crop_mass_balance_residual_kg_m2",
    ]
    assert set(flux_columns).issubset(trajectory.columns)
    assert (trajectory[flux_columns[:-1]] >= -1e-12).all().all()
    assert trajectory["crop_mass_balance_residual_kg_m2"].abs().max() < 1e-10

    trajectory_without_flux = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state={
            "cBuf": 0.0,
            "cLeaf": 6000.0,
            "cStem": 7000.0,
            "cFruit": 0.0,
            "tCanSum": 0.0,
        },
        fruit_dry_matter_fraction=0.08,
        substep_seconds=300,
        include_flux_diagnostics=False,
    )
    assert "gross_photosynthesis_kg_m2" not in trajectory_without_flux.columns
    np.testing.assert_allclose(
        trajectory_without_flux["standing_dry_kg_m2"],
        trajectory["standing_dry_kg_m2"],
        rtol=0.0,
        atol=1e-12,
    )

    inactive_schedule = {
        "reference_t_can_sum_deg_day": 10_000.0,
        "exponent": 1.0,
        "maximum_multiplier": 2.0,
        "organs": ["fruit", "leaf", "stem"],
    }
    scheduled_trajectory = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state={
            "cBuf": 0.0,
            "cLeaf": 6000.0,
            "cStem": 7000.0,
            "cFruit": 0.0,
            "tCanSum": 0.0,
        },
        fruit_dry_matter_fraction=0.08,
        substep_seconds=300,
        include_flux_diagnostics=False,
        stage_sink_schedule=inactive_schedule,
    )
    multiplier_columns = [
        "stage_sink_multiplier_fruit",
        "stage_sink_multiplier_leaf",
        "stage_sink_multiplier_stem",
    ]
    assert (scheduled_trajectory[multiplier_columns] == 1.0).all().all()
    np.testing.assert_allclose(
        scheduled_trajectory["standing_dry_kg_m2"],
        trajectory_without_flux["standing_dry_kg_m2"],
        rtol=0.0,
        atol=1e-12,
    )


def test_fruit_allocation_calibration_recovers_known_grid_candidate():
    timestamps = pd.date_range("2026-04-01", periods=4, freq="1h", tz="Asia/Shanghai")
    climate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": 22.0,
            "canopy_temperature": 23.0,
            "relative_humidity": 70.0,
            "co2_concentration": 500.0,
        }
    )
    controls = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            **{name: 0.0 for name in ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")},
        }
    )
    weather = np.zeros((4, 10), dtype=float)
    weather[:, 0] = 300.0
    weather[:, 1] = 20.0
    weather[:, 2] = 1500.0
    weather[:, 3] = 720.0
    weather[:, 6] = 18.0
    initial = {
        "cBuf": 10000.0,
        "cLeaf": 30000.0,
        "cStem": 30000.0,
        "cFruit": 0.0,
        "tCanSum": 800.0,
    }
    known_parameters = np.asarray(init_default_params(216))
    known_parameters[154] *= 0.5
    known = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        parameters=known_parameters,
        substep_seconds=300,
    )

    result = calibrate_fruit_allocation_multiplier(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        calibration_timestamp=timestamps[-1],
        observed_fresh_kg_m2=float(known.iloc[-1]["standing_fresh_kg_m2"]),
        candidate_multipliers=[0.5, 1.0],
        substep_seconds=300,
    )

    assert result["best_multiplier"] == 0.5
    assert result["absolute_error_kg_m2"] == pytest.approx(0.0, abs=1e-10)
    assert len(result["candidate_scores"]) == 2


def test_organ_allocation_calibration_recovers_known_candidate_without_validation_data():
    timestamps = pd.date_range("2026-04-01", periods=4, freq="1h", tz="Asia/Shanghai")
    climate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": 22.0,
            "canopy_temperature": 23.0,
            "relative_humidity": 70.0,
            "co2_concentration": 500.0,
        }
    )
    controls = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            **{name: 0.0 for name in ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")},
        }
    )
    weather = np.zeros((4, 10), dtype=float)
    weather[:, 0] = 300.0
    weather[:, 1] = 20.0
    weather[:, 2] = 1500.0
    weather[:, 3] = 720.0
    weather[:, 6] = 18.0
    initial = {
        "cBuf": 10000.0,
        "cLeaf": 30000.0,
        "cStem": 30000.0,
        "cFruit": 0.0,
        "tCanSum": 800.0,
    }
    known_parameters = np.asarray(init_default_params(216))
    known_parameters[[154, 155, 156]] *= [0.5, 0.75, 1.25]
    known = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        parameters=known_parameters,
        substep_seconds=300,
    )
    target = known.iloc[-1]

    result = calibrate_organ_allocation_multipliers(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        calibration_timestamp=timestamps[-1],
        observed_dry_kg_m2={
            "fruit": float(target["standing_dry_kg_m2"]),
            "leaf": float(target["c_leaf_mg_m2"]) * 1e-6,
            "stem": float(target["c_stem_mg_m2"]) * 1e-6,
        },
        candidate_multipliers={
            "fruit": [0.5, 1.0],
            "leaf": [0.75, 1.0],
            "stem": [1.0, 1.25],
        },
        substep_seconds=300,
    )

    assert result["best_multipliers"] == {
        "fruit": 0.5,
        "leaf": 0.75,
        "stem": 1.25,
    }
    assert result["objective_nrmse"] == pytest.approx(0.0, abs=1e-10)
    assert result["calibration_timestamp"] == timestamps[-1].isoformat()
    assert len(result["candidate_scores"]) == 8


def test_proportional_organ_calibration_improves_fit_without_mutating_base_parameters():
    timestamps = pd.date_range("2026-04-01", periods=5, freq="1h", tz="Asia/Shanghai")
    climate = pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": 22.0,
            "canopy_temperature": 23.0,
            "relative_humidity": 70.0,
            "co2_concentration": 500.0,
        }
    )
    controls = pd.DataFrame(
        {
            "timestamp": timestamps[:-1],
            **{name: 0.0 for name in ("uBoil", "uCO2", "uThScr", "uVent", "uLamp", "uBlScr")},
        }
    )
    weather = np.zeros((5, 10), dtype=float)
    weather[:, 0] = 300.0
    weather[:, 1] = 20.0
    weather[:, 2] = 1500.0
    weather[:, 3] = 720.0
    weather[:, 6] = 18.0
    initial = {
        "cBuf": 10000.0,
        "cLeaf": 30000.0,
        "cStem": 30000.0,
        "cFruit": 5000.0,
        "tCanSum": 800.0,
    }
    base_parameters = np.asarray(init_default_params(216), dtype=float)
    base_parameters[129] *= 1.25
    untouched = base_parameters.copy()
    known_parameters = base_parameters.copy()
    known_parameters[[154, 155, 156]] *= [0.5, 0.75, 1.25]
    known = simulate_forced_crop_trajectory(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        parameters=known_parameters,
        substep_seconds=300,
        include_flux_diagnostics=False,
    )
    target = known.iloc[-1]

    result = calibrate_organ_allocation_proportional(
        climate=climate,
        controls=controls,
        weather=weather,
        initial_crop_state=initial,
        fruit_dry_matter_fraction=0.08,
        calibration_timestamp=timestamps[-1],
        observed_dry_kg_m2={
            "fruit": float(target["standing_dry_kg_m2"]),
            "leaf": float(target["c_leaf_mg_m2"]) * 1e-6,
            "stem": float(target["c_stem_mg_m2"]) * 1e-6,
        },
        base_parameters=base_parameters,
        initial_multipliers={"fruit": 1.0, "leaf": 1.0, "stem": 1.0},
        iterations=3,
        multiplier_bounds=(0.05, 2.0),
        substep_seconds=300,
    )

    np.testing.assert_array_equal(base_parameters, untouched)
    assert len(result["history"]) == 3
    assert result["final_objective_nrmse"] < result["initial_objective_nrmse"]
    assert result["final_objective_nrmse"] < 0.01
    assert result["final_objective_nrmse"] == pytest.approx(
        min(item["objective_nrmse"] for item in result["history"])
    )
    assert result["selected_iteration_from_calibration"] in {1, 2, 3}
    assert result["selection_metric"] == "calibration_organ_nrmse"
    assert set(result["final_multipliers"]) == {"fruit", "leaf", "stem"}
    assert result["calibration_timestamp"] == timestamps[-1].isoformat()


def test_evaluate_organ_dry_predictions_scores_each_organ_from_fresh_samples():
    observations = pd.DataFrame(
        {
            "observation_date": ["2026-04-27", "2026-04-27"],
            "leaf_fresh_kg_m2": [1.0, 1.2],
            "stem_fresh_kg_m2": [2.0, 2.0],
            "ripe_fruit_fresh_kg_m2": [3.0, 5.0],
        }
    )
    trajectory = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-04-27", tz="Asia/Shanghai")],
            "c_leaf_mg_m2": [110000.0],
            "c_stem_mg_m2": [180000.0],
            "standing_dry_kg_m2": [0.8],
        }
    )

    result = evaluate_organ_dry_predictions(
        observations,
        trajectory,
        observation_timestamp="2026-04-27",
        dry_matter_fractions={"leaf": 0.1, "stem": 0.1, "fruit": 0.2},
    )

    assert result["replicate_count"] == 2
    assert result["organs"]["leaf"]["observed_dry_kg_m2"] == pytest.approx(0.11)
    assert result["organs"]["leaf"]["residual_dry_kg_m2"] == pytest.approx(0.0)
    assert result["organs"]["stem"]["relative_error"] == pytest.approx(-0.1)
    assert result["organs"]["fruit"]["relative_error"] == pytest.approx(0.0)


def test_summarize_initial_thermal_sum_sensitivity_reports_envelope_without_selection():
    rows = [
        {
            "scenario": "low",
            "initial_t_can_sum_deg_day": 250.0,
            "rg_fruit_multiplier": 0.4,
            "calibration_observed_fresh_kg_m2": 0.60,
            "calibration_predicted_fresh_kg_m2": 0.59,
            "validation_predicted_fresh_kg_m2": 1.8,
        },
        {
            "scenario": "median",
            "initial_t_can_sum_deg_day": 280.0,
            "rg_fruit_multiplier": 0.4,
            "calibration_observed_fresh_kg_m2": 0.60,
            "calibration_predicted_fresh_kg_m2": 0.61,
            "validation_predicted_fresh_kg_m2": 2.0,
        },
        {
            "scenario": "high",
            "initial_t_can_sum_deg_day": 310.0,
            "rg_fruit_multiplier": 0.3,
            "calibration_observed_fresh_kg_m2": 0.60,
            "calibration_predicted_fresh_kg_m2": 0.58,
            "validation_predicted_fresh_kg_m2": 2.2,
        },
    ]

    result, aligned = summarize_initial_thermal_sum_sensitivity(
        rows,
        validation_observed_fresh_kg_m2=2.1,
    )

    assert result["scenario_count"] == 3
    assert result["validation_envelope_low_fresh_kg_m2"] == pytest.approx(1.8)
    assert result["validation_envelope_high_fresh_kg_m2"] == pytest.approx(2.2)
    assert result["validation_envelope_contains_observed"] is True
    assert result["best_case_absolute_relative_error"] == pytest.approx(0.1 / 2.1)
    assert result["worst_case_absolute_relative_error"] == pytest.approx(0.3 / 2.1)
    assert result["scenario_selected_from_validation"] is False
    assert result["uncertainty_type"] == "structural_sensitivity_not_predictive_interval"
    assert "selected_scenario" not in result
    assert aligned["validation_residual_fresh_kg_m2"].tolist() == pytest.approx(
        [-0.3, -0.1, 0.1]
    )


def test_summarize_stage_sink_structures_reports_posthoc_envelope_without_selection():
    rows = [
        {
            "scenario": "constant",
            "calibration_organ_nrmse": 0.05,
            "development_organ_nrmse": 0.25,
            "development_predicted_fresh_kg_m2": 1.70,
            "development_observed_fresh_kg_m2": 2.40,
        },
        {
            "scenario": "sqrt_all",
            "calibration_organ_nrmse": 0.05,
            "development_organ_nrmse": 0.18,
            "development_predicted_fresh_kg_m2": 2.05,
            "development_observed_fresh_kg_m2": 2.40,
        },
        {
            "scenario": "linear_all",
            "calibration_organ_nrmse": 0.05,
            "development_organ_nrmse": 0.12,
            "development_predicted_fresh_kg_m2": 2.30,
            "development_observed_fresh_kg_m2": 2.40,
        },
        {
            "scenario": "sqrt_fruit",
            "calibration_organ_nrmse": 0.05,
            "development_organ_nrmse": 0.20,
            "development_predicted_fresh_kg_m2": 1.95,
            "development_observed_fresh_kg_m2": 2.40,
        },
        {
            "scenario": "linear_fruit",
            "calibration_organ_nrmse": 0.05,
            "development_organ_nrmse": 0.16,
            "development_predicted_fresh_kg_m2": 2.15,
            "development_observed_fresh_kg_m2": 2.40,
        },
    ]

    result, aligned = summarize_stage_sink_structure_comparison(rows)

    assert result["scenario_count"] == 5
    assert result["development_fresh_envelope_kg_m2"] == pytest.approx([1.7, 2.3])
    assert result["development_fresh_envelope_contains_observed"] is False
    assert result["best_case_development_fresh_absolute_relative_error"] == pytest.approx(
        0.1 / 2.4
    )
    assert result["posthoc_structure_development"] is True
    assert result["independent_validation"] is False
    assert result["scenario_selected_from_development"] is False
    assert result["adoption_status"] == "diagnostic_not_final_target_parameterization"
    assert "selected_scenario" not in result
    assert aligned.loc[
        aligned["scenario"].eq("constant"),
        "development_fresh_residual_kg_m2",
    ].item() == pytest.approx(-0.7)


def test_summarize_biomass_flux_period_compares_model_and_observed_increments():
    timestamps = pd.date_range("2026-04-27", periods=3, freq="1D", tz="Asia/Shanghai")
    trajectory = pd.DataFrame(
        {
            "timestamp": timestamps,
            "c_buffer_mg_m2": [10000.0, 12000.0, 15000.0],
            "c_leaf_mg_m2": [100000.0, 125000.0, 150000.0],
            "c_stem_mg_m2": [200000.0, 225000.0, 250000.0],
            "standing_dry_kg_m2": [0.30, 0.35, 0.40],
            "gross_photosynthesis_kg_m2": [0.0, 0.20, 0.20],
            "buffer_to_leaf_kg_m2": [0.0, 0.03, 0.03],
            "buffer_to_stem_kg_m2": [0.0, 0.03, 0.03],
            "buffer_to_fruit_kg_m2": [0.0, 0.06, 0.06],
            "growth_respiration_kg_m2": [0.0, 0.01, 0.01],
            "leaf_maintenance_kg_m2": [0.0, 0.005, 0.005],
            "stem_maintenance_kg_m2": [0.0, 0.005, 0.005],
            "fruit_maintenance_kg_m2": [0.0, 0.005, 0.005],
            "leaf_pruning_kg_m2": [0.0, 0.0, 0.0],
            "native_harvested_dry_kg_m2": [0.0, 0.0, 0.0],
            "crop_mass_balance_residual_kg_m2": [0.0, 1e-14, -1e-14],
        }
    )

    result = summarize_biomass_flux_period(
        trajectory,
        start_timestamp=timestamps[0],
        end_timestamp=timestamps[-1],
        observed_start_dry_kg_m2={"leaf": 0.10, "stem": 0.20, "fruit": 0.30},
        observed_end_dry_kg_m2={"leaf": 0.18, "stem": 0.27, "fruit": 0.45},
        specific_leaf_area_m2_mg=2.66e-5,
    )

    assert result["flux_totals_kg_m2"]["gross_photosynthesis"] == pytest.approx(0.40)
    assert result["model_organ_change_kg_m2"] == pytest.approx(
        {"leaf": 0.05, "stem": 0.05, "fruit": 0.10}
    )
    assert result["observed_aboveground_change_kg_m2"] == pytest.approx(0.30)
    assert result["model_aboveground_change_kg_m2"] == pytest.approx(0.20)
    assert result["aboveground_change_deficit_kg_m2"] == pytest.approx(0.10)
    assert result["model_fraction_of_observed_change"] == pytest.approx(2.0 / 3.0)
    assert result["model_retained_crop_change_kg_m2"] == pytest.approx(0.205)
    assert result["external_loss_fraction_of_gross_photosynthesis"] == pytest.approx(0.05 / 0.40)
    assert result["required_gross_if_losses_and_buffer_unchanged_kg_m2"] == pytest.approx(0.355)
    assert result["required_to_modeled_gross_ratio"] == pytest.approx(0.355 / 0.40)
    assert result["modeled_lai_start_m2_m2"] == pytest.approx(2.66)
    assert result["modeled_lai_end_m2_m2"] == pytest.approx(3.99)
    assert result["mass_balance_residual_absolute_sum_kg_m2"] == pytest.approx(2e-14)
    assert result["holdout_used_for_parameter_selection"] is False


def test_assimilation_capacity_summary_reports_nonselective_holdout_envelope():
    rows = [
        {
            "capacity_multiplier": 0.75,
            "calibration_organ_nrmse": 0.03,
            "validation_organ_nrmse": 0.30,
            "validation_predicted_fresh_kg_m2": 1.7,
            "validation_observed_fresh_kg_m2": 2.0,
        },
        {
            "capacity_multiplier": 1.0,
            "calibration_organ_nrmse": 0.02,
            "validation_organ_nrmse": 0.20,
            "validation_predicted_fresh_kg_m2": 1.9,
            "validation_observed_fresh_kg_m2": 2.0,
        },
        {
            "capacity_multiplier": 1.5,
            "calibration_organ_nrmse": 0.04,
            "validation_organ_nrmse": 0.25,
            "validation_predicted_fresh_kg_m2": 2.1,
            "validation_observed_fresh_kg_m2": 2.0,
        },
    ]

    result, aligned = summarize_assimilation_capacity_sensitivity(rows)

    assert result["registered_capacity_multipliers"] == [0.75, 1.0, 1.5]
    assert result["validation_fresh_envelope_kg_m2"] == pytest.approx([1.7, 2.1])
    assert result["validation_fresh_envelope_contains_observed"] is True
    assert result["best_case_validation_fresh_absolute_relative_error"] == pytest.approx(0.05)
    assert result["scenario_selected_from_validation"] is False
    assert result["uncertainty_type"] == "structural_sensitivity_not_predictive_interval"
    assert "selected_capacity_multiplier" not in result
    assert aligned["validation_fresh_relative_error"].tolist() == pytest.approx(
        [-0.15, -0.05, 0.05]
    )
