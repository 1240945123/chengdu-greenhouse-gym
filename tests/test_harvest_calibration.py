from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from experiments.crop.calibrate_harvest_model import (
    _draw_season_residual_blocks,
    assess_target_harvest_data,
    bootstrap_harvest_parameter_estimates,
    build_harvest_drivers_from_greenlight,
    calibrate_harvest_parameters,
    estimate_fruit_dry_matter_fraction,
    joint_harvest_prediction_intervals,
    residual_block_prediction_intervals,
    simulate_harvest_series,
    split_harvest_seasons,
    validate_uncertainty_driver_scenarios,
)
from glassgym.models.harvest import HarvestCohortParameters


def _uncertainty_point_drivers() -> pd.DataFrame:
    timestamps = pd.date_range("2026-05-01 08:00", periods=4, freq="D")
    return pd.DataFrame(
        {
            "season_id": "test_1",
            "timestamp": timestamps,
            "net_fruit_dry_matter_change_kg_m2": 0.01,
            "air_temperature_c": 20.0,
            "dt_hours": 24.0,
            "pick": [False, True, False, True],
            "pick_source": "management_protocol",
            "driver_model_status": "accepted_target_crop_model",
            "season_complete": True,
            "fruit_change_source": "greenlight_fruit_state_balance",
            "air_temperature_source": "greenlight_simulated_indoor",
            "greenhouse_id": 63,
            "greenhouse_code": "GH-PIDU",
            "planting_code": "P-test-1",
            "initial_fruit_dry_matter_kg_m2": [0.02, 0.0, 0.0, 0.0],
            "initial_fruit_maturity_fraction": [0.1, 0.0, 0.0, 0.0],
        }
    )


def _uncertainty_scenarios() -> pd.DataFrame:
    point = _uncertainty_point_drivers()
    parts = []
    for scenario_id, growth, source in (
        ("climate_low", 0.006, "greenhouse_climate_model_parameter_uncertainty"),
        ("weather_high", 0.018, "outdoor_weather_scenario_uncertainty"),
    ):
        scenario = point.copy()
        scenario["net_fruit_dry_matter_change_kg_m2"] = growth
        scenario["driver_scenario_id"] = scenario_id
        scenario["driver_uncertainty_source"] = source
        parts.append(scenario)
    return pd.concat(parts, ignore_index=True)


def test_validates_aligned_traceable_uncertainty_driver_scenarios():
    scenarios, audit = validate_uncertainty_driver_scenarios(
        _uncertainty_point_drivers(),
        _uncertainty_scenarios(),
    )

    assert len(scenarios) == 8
    assert audit["scenario_count"] == 2
    assert audit["scenario_ids"] == ["climate_low", "weather_high"]
    assert audit["uncertainty_sources"] == [
        "greenhouse_climate_model_parameter_uncertainty",
        "outdoor_weather_scenario_uncertainty",
    ]
    assert audit["scenario_sampling_policy"] == "uniform"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "duplicate"),
        (lambda frame: frame.drop(index=0), "keys"),
        (
            lambda frame: frame.assign(
                pick=frame["pick"].mask(frame.index == 0, True)
            ),
            "pick",
        ),
        (
            lambda frame: frame.assign(
                greenhouse_code=frame["greenhouse_code"].mask(
                    frame.index == 0, "OTHER"
                )
            ),
            "greenhouse_code",
        ),
        (
            lambda frame: frame.assign(driver_uncertainty_source="unknown_noise"),
            "uncertainty source",
        ),
    ],
)
def test_rejects_invalid_uncertainty_driver_scenarios(mutator, message):
    invalid = mutator(_uncertainty_scenarios().copy()).reset_index(drop=True)

    with pytest.raises(ValueError, match=message):
        validate_uncertainty_driver_scenarios(
            _uncertainty_point_drivers(),
            invalid,
        )


def test_driver_scenarios_widen_joint_cumulative_prediction_intervals():
    training = pd.DataFrame(
        {
            "season_id": "train_1",
            "date": pd.date_range("2025-05-01", periods=4, freq="D"),
            "observed_fresh_kg_m2": [0.0, 0.1, 0.0, 0.1],
            "predicted_fresh_kg_m2": [0.0, 0.1, 0.0, 0.1],
        }
    )
    point = _uncertainty_point_drivers()
    draws = pd.DataFrame(
        {
            "maturity_thermal_time_deg_day": [5.0, 5.0],
            "dry_matter_fraction": [0.08, 0.08],
        }
    )
    forecast_index = pd.DataFrame(
        {
            "season_id": "test_1",
            "date": pd.to_datetime(point["timestamp"]).dt.normalize(),
        }
    )
    kwargs = {
        "forecast_index": forecast_index,
        "base_temperature_c": 10.0,
        "bootstrap_samples": 500,
        "block_days": 2,
        "bootstrap_seed": 19,
    }

    fixed = joint_harvest_prediction_intervals(training, point, draws, **kwargs)
    ensemble = joint_harvest_prediction_intervals(
        training,
        point,
        draws,
        forecast_driver_scenarios=_uncertainty_scenarios(),
        **kwargs,
    )
    repeated = joint_harvest_prediction_intervals(
        training,
        point,
        draws,
        forecast_driver_scenarios=_uncertainty_scenarios(),
        **kwargs,
    )

    fixed_width = (
        fixed["cumulative_p975_kg_m2"].iloc[-1]
        - fixed["cumulative_p025_kg_m2"].iloc[-1]
    )
    ensemble_width = (
        ensemble["cumulative_p975_kg_m2"].iloc[-1]
        - ensemble["cumulative_p025_kg_m2"].iloc[-1]
    )
    assert ensemble_width > fixed_width + 0.05
    pd.testing.assert_frame_equal(ensemble, repeated)


def test_weighted_dry_matter_fraction_uses_fresh_mass_denominator():
    observations = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["2025-04-01", "2025-04-01", "2025-04-18"]
            ),
            "ripe_fruit_fresh_kg_m2": [1.0, 3.0, 6.0],
            "ripe_fruit_dry_kg_m2": [0.10, 0.30, 0.42],
        }
    )

    estimate = estimate_fruit_dry_matter_fraction(
        observations,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )

    assert math.isclose(estimate["estimate"], 0.082)
    assert estimate["observation_count"] == 3
    assert estimate["date_block_count"] == 2
    assert 0.0 < estimate["ci95_low"] <= estimate["estimate"] <= estimate["ci95_high"] < 1.0
    assert estimate == estimate_fruit_dry_matter_fraction(
        observations,
        bootstrap_samples=200,
        bootstrap_seed=7,
    )


def test_dry_matter_estimator_drops_missing_pairs_but_rejects_invalid_pairs():
    observations = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(
                ["2025-04-01", "2025-04-18", "2025-03-01"]
            ),
            "ripe_fruit_fresh_kg_m2": [1.0, None, 0.0],
            "ripe_fruit_dry_kg_m2": [0.08, None, 0.0],
        }
    )
    estimate = estimate_fruit_dry_matter_fraction(observations, bootstrap_samples=20)
    assert estimate["observation_count"] == 1
    assert estimate["estimate"] == 0.08

    invalid = observations.copy()
    invalid.loc[0, "ripe_fruit_dry_kg_m2"] = 1.2
    try:
        estimate_fruit_dry_matter_fraction(invalid, bootstrap_samples=20)
    except ValueError as exc:
        assert "dry matter" in str(exc)
    else:
        raise AssertionError("dry mass greater than fresh mass must be rejected")


def test_target_sufficiency_requires_independent_target_season():
    target_dates = pd.date_range("2026-05-01", periods=8, freq="5D")
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8 + ["XINDU"],
            "greenhouse_id": [63] * 8 + [1],
            "target_eligible": [True] * 8 + [False],
            "season_id": ["2026_spring"] * 8 + ["2025_spring"],
            "timestamp": list(target_dates) + [pd.Timestamp("2025-05-01")],
            "fresh_kg_m2": [1.0] * 8 + [2.0],
            "season_complete_evidence": [True] * 8 + [False],
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["target_harvest_event_count"] == 8
    assert status["target_season_count"] == 1
    assert status["can_calibrate_target"]
    assert not status["can_validate_target_independently"]
    assert status["status"] == "target_calibration_only"


def test_short_or_sparse_target_season_is_not_treated_as_complete():
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8,
            "greenhouse_id": [63] * 8,
            "target_eligible": [True] * 8,
            "season_id": ["short"] * 8,
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="D"),
            "fresh_kg_m2": [1.0] * 8,
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["usable_target_seasons"] == []
    assert status["status"] == "insufficient_target_harvest_events"


def test_target_sufficiency_requires_explicit_season_completion_evidence():
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8,
            "greenhouse_id": [63] * 8,
            "target_eligible": [True] * 8,
            "season_id": ["ongoing"] * 8,
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="5D"),
            "fresh_kg_m2": [1.0] * 8,
            "season_complete_evidence": [False] * 8,
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["usable_target_seasons"] == []
    assert status["season_coverage"][0]["season_complete_evidence"] is False
    assert status["status"] == "insufficient_target_harvest_events"


def test_target_sufficiency_treats_missing_completion_evidence_as_incomplete():
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8,
            "greenhouse_id": [63] * 8,
            "target_eligible": [True] * 8,
            "season_id": ["unknown"] * 8,
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="5D"),
            "fresh_kg_m2": [1.0] * 8,
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["usable_target_seasons"] == []
    assert status["season_coverage"][0]["season_complete_evidence"] is False


def test_missing_target_events_cannot_be_replaced_by_source_site_samples():
    events = pd.DataFrame(
        {
            "greenhouse_code": ["XINDU"],
            "greenhouse_id": [1],
            "target_eligible": [False],
            "season_id": ["2025_spring"],
            "timestamp": pd.to_datetime(["2025-05-01"]),
            "fresh_kg_m2": [2.0],
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["status"] == "missing_target_harvest_events"
    assert not status["can_calibrate_target"]
    assert not status["can_validate_target_independently"]


@pytest.mark.parametrize("evidence_class", ["external_observed", "simulated_prior"])
def test_target_sufficiency_rejects_spoofed_non_target_evidence(evidence_class):
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8,
            "greenhouse_id": [63] * 8,
            "target_eligible": [True] * 8,
            "evidence_class": [evidence_class] * 8,
            "season_id": ["spoofed"] * 8,
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="5D"),
            "fresh_kg_m2": [1.0] * 8,
            "season_complete_evidence": [True] * 8,
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["target_harvest_event_count"] == 0
    assert status["rejected_evidence_class_counts"] == {evidence_class: 8}
    assert status["status"] == "missing_target_harvest_events"


def test_target_sufficiency_rejects_matching_code_with_wrong_id():
    events = pd.DataFrame(
        {
            "greenhouse_code": ["PIDU"] * 8,
            "greenhouse_id": [1] * 8,
            "target_eligible": [False] * 8,
            "season_id": ["s1"] * 8,
            "timestamp": pd.date_range("2026-05-01", periods=8, freq="5D"),
            "fresh_kg_m2": 1.0,
        }
    )

    status = assess_target_harvest_data(
        events, target_greenhouse_id=63, target_greenhouse_code="PIDU"
    )

    assert status["status"] == "missing_target_harvest_events"


def test_season_split_is_disjoint_and_never_uses_test_for_fit():
    events = pd.DataFrame(
        {
            "season_id": ["s1", "s2", "s3", "s1"],
            "timestamp": pd.to_datetime(
                ["2024-05-01", "2025-05-01", "2026-05-01", "2024-05-04"]
            ),
            "fresh_kg_m2": [1.0, 1.1, 1.2, 0.5],
        }
    )

    split = split_harvest_seasons(
        events,
        train_seasons=["s1"],
        validation_seasons=["s2"],
        test_seasons=["s3"],
    )

    assert split["train"]["season_id"].unique().tolist() == ["s1"]
    assert split["validation"]["season_id"].unique().tolist() == ["s2"]
    assert split["test"]["season_id"].unique().tolist() == ["s3"]

    try:
        split_harvest_seasons(
            events,
            train_seasons=["s1", "s3"],
            validation_seasons=["s2"],
            test_seasons=["s3"],
        )
    except ValueError as exc:
        assert "disjoint" in str(exc)
    else:
        raise AssertionError("test seasons must never overlap training seasons")


def test_synthetic_calibration_recovers_harvest_timing_and_fresh_mass_scale():
    timestamps = pd.date_range("2026-03-01 08:00", periods=60, freq="D")
    drivers = pd.DataFrame(
        {
            "timestamp": timestamps,
            "net_fruit_dry_matter_change_kg_m2": [0.01] * len(timestamps),
            "air_temperature_c": [20.0] * len(timestamps),
            "dt_hours": [24.0] * len(timestamps),
            "pick": [(index % 3) == 0 for index in range(len(timestamps))],
        }
    )
    truth = HarvestCohortParameters(
        base_temperature_c=10.0,
        maturity_thermal_time_deg_day=100.0,
        dry_matter_fraction=0.08,
        minimum_pick_fresh_kg_m2=0.0,
    )
    synthetic = simulate_harvest_series(drivers, truth)
    observed = synthetic.loc[synthetic["fresh_kg_m2"] > 0.0, ["timestamp", "fresh_kg_m2"]]

    calibration = calibrate_harvest_parameters(
        drivers,
        observed,
        base_temperature_c=10.0,
        maturity_bounds_deg_day=(70.0, 130.0),
        dry_matter_fraction_bounds=(0.06, 0.10),
        maturity_grid_size=13,
    )

    assert abs(calibration["parameters"]["maturity_thermal_time_deg_day"] - 100.0) <= 5.0
    assert abs(calibration["parameters"]["dry_matter_fraction"] - 0.08) < 0.002
    assert calibration["training_metrics"]["batch_wmape"] < 1e-4
    assert calibration["training_metrics"]["first_harvest_date_error_days"] == 0.0


def test_parameter_bootstrap_is_reproducible_and_bounded():
    timestamps = pd.date_range("2026-03-01 08:00", periods=60, freq="D")
    drivers = pd.DataFrame(
        {
            "season_id": "s1",
            "timestamp": timestamps,
            "net_fruit_dry_matter_change_kg_m2": 0.01,
            "air_temperature_c": 20.0,
            "dt_hours": 24.0,
            "pick": [(index % 3) == 0 for index in range(len(timestamps))],
        }
    )
    truth = HarvestCohortParameters(
        base_temperature_c=10.0,
        maturity_thermal_time_deg_day=100.0,
        dry_matter_fraction=0.08,
    )
    predicted = simulate_harvest_series(drivers, truth)
    observed = predicted.loc[
        predicted["fresh_kg_m2"] > 0.0,
        ["season_id", "timestamp", "fresh_kg_m2"],
    ].copy()
    observed["fresh_kg_m2"] *= np.resize([0.92, 1.08, 1.03], len(observed))
    kwargs = {
        "base_temperature_c": 10.0,
        "maturity_bounds_deg_day": (70.0, 130.0),
        "dry_matter_fraction_bounds": (0.06, 0.10),
        "maturity_grid_size": 7,
        "bootstrap_samples": 8,
        "block_days": 3,
        "bootstrap_seed": 17,
    }

    first = bootstrap_harvest_parameter_estimates(drivers, observed, **kwargs)
    second = bootstrap_harvest_parameter_estimates(drivers, observed, **kwargs)

    pd.testing.assert_frame_equal(first["parameter_draws"], second["parameter_draws"])
    draws = first["parameter_draws"]
    assert len(draws) == 8
    assert draws["maturity_thermal_time_deg_day"].between(70.0, 130.0).all()
    assert draws["dry_matter_fraction"].between(0.06, 0.10).all()
    assert first["audit"]["accepted_draws"] == 8
    assert first["audit"]["unique_parameter_draws"] >= 2


def test_parameter_bootstrap_requires_at_least_two_draws():
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_harvest_parameter_estimates(
            pd.DataFrame(),
            pd.DataFrame(),
            base_temperature_c=10.0,
            maturity_bounds_deg_day=(70.0, 130.0),
            dry_matter_fraction_bounds=(0.06, 0.10),
            bootstrap_samples=1,
        )


def test_joint_parameter_residual_intervals_are_ordered_and_reset_by_season():
    training = pd.DataFrame(
        {
            "season_id": ["train"] * 6,
            "date": pd.date_range("2025-05-01", periods=6, freq="D"),
            "observed_fresh_kg_m2": [0.0, 1.1, 0.0, 0.9, 0.0, 1.0],
            "predicted_fresh_kg_m2": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    driver_parts = []
    index_parts = []
    for season_id, start in (("test_a", "2026-05-01"), ("test_b", "2027-05-01")):
        dates = pd.date_range(start, periods=4, freq="D")
        driver_parts.append(
            pd.DataFrame(
                {
                    "season_id": season_id,
                    "timestamp": dates,
                    "net_fruit_dry_matter_change_kg_m2": 0.01,
                    "air_temperature_c": 20.0,
                    "dt_hours": 24.0,
                    "pick": True,
                    "initial_fruit_dry_matter_kg_m2": [0.08, 0.0, 0.0, 0.0],
                    "initial_fruit_maturity_fraction": [1.0, 0.0, 0.0, 0.0],
                }
            )
        )
        index_parts.append(pd.DataFrame({"season_id": season_id, "date": dates}))
    draws = pd.DataFrame(
        {
            "replicate_id": [0, 1, 2],
            "maturity_thermal_time_deg_day": [5.0, 8.0, 10.0],
            "dry_matter_fraction": [0.075, 0.08, 0.09],
            "objective": [0.1, 0.2, 0.3],
        }
    )

    intervals = joint_harvest_prediction_intervals(
        training,
        pd.concat(driver_parts, ignore_index=True),
        draws,
        forecast_index=pd.concat(index_parts, ignore_index=True),
        base_temperature_c=10.0,
        bootstrap_samples=100,
        block_days=2,
        bootstrap_seed=9,
    )

    for prefix in ("batch", "cumulative"):
        values = intervals[
            [
                f"{prefix}_p025_kg_m2",
                f"{prefix}_p10_kg_m2",
                f"{prefix}_p50_kg_m2",
                f"{prefix}_p90_kg_m2",
                f"{prefix}_p975_kg_m2",
            ]
        ].to_numpy()
        assert np.all(np.diff(values, axis=1) >= -1e-12)
    for _, season in intervals.groupby("season_id", sort=False):
        assert season.iloc[0]["cumulative_p50_kg_m2"] == pytest.approx(
            season.iloc[0]["batch_p50_kg_m2"]
        )
        assert season["cumulative_p50_kg_m2"].is_monotonic_increasing


def test_residual_block_uncertainty_is_reproducible_and_non_decreasing():
    aligned = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=8, freq="D"),
            "observed_fresh_kg_m2": [0.0, 1.0, 0.0, 1.2, 0.0, 0.0, 0.8, 0.0],
            "predicted_fresh_kg_m2": [0.0, 0.9, 0.0, 1.0, 0.0, 0.1, 1.0, 0.0],
        }
    )

    first = residual_block_prediction_intervals(
        aligned,
        bootstrap_samples=200,
        block_days=3,
        bootstrap_seed=11,
    )
    second = residual_block_prediction_intervals(
        aligned,
        bootstrap_samples=200,
        block_days=3,
        bootstrap_seed=11,
    )

    pd.testing.assert_frame_equal(first, second)
    assert {
        "batch_p025_kg_m2",
        "batch_p10_kg_m2",
        "batch_p50_kg_m2",
        "batch_p90_kg_m2",
        "batch_p975_kg_m2",
    }.issubset(first.columns)
    assert (first["batch_p025_kg_m2"] <= first["batch_p975_kg_m2"]).all()
    assert (first["batch_p10_kg_m2"] <= first["batch_p90_kg_m2"]).all()
    assert (first["cumulative_p025_kg_m2"] <= first["cumulative_p975_kg_m2"]).all()
    assert (first["cumulative_p10_kg_m2"] <= first["cumulative_p90_kg_m2"]).all()
    assert (first["cumulative_p50_kg_m2"].diff().fillna(0.0) >= 0.0).all()


def test_residual_blocks_never_cross_training_season_boundaries():
    rng = np.random.default_rng(17)

    sampled = _draw_season_residual_blocks(
        [
            np.array([1.0, 1.0, 1.0]),
            np.array([10.0, 10.0, 10.0]),
        ],
        length=40,
        block_days=4,
        rng=rng,
    )

    assert len(sampled) == 40
    for start in range(0, len(sampled), 4):
        assert len(set(sampled[start : start + 4])) == 1


def test_multiseason_uncertainty_resets_cumulative_forecast_by_season():
    training = pd.DataFrame(
        {
            "season_id": ["train_1"] * 3 + ["train_2"] * 3,
            "date": pd.to_datetime(
                [
                    "2024-05-01",
                    "2024-05-02",
                    "2024-05-03",
                    "2025-05-01",
                    "2025-05-02",
                    "2025-05-03",
                ]
            ),
            "observed_fresh_kg_m2": [1.1, 1.1, 1.1, 2.0, 2.0, 2.0],
            "predicted_fresh_kg_m2": [1.0] * 6,
        }
    )
    forecast = pd.DataFrame(
        {
            "season_id": ["test_1", "test_1", "test_2", "test_2"],
            "date": pd.to_datetime(
                ["2026-05-01", "2026-05-02", "2027-05-01", "2027-05-02"]
            ),
            "predicted_fresh_kg_m2": [1.0] * 4,
        }
    )

    intervals = residual_block_prediction_intervals(
        training,
        forecast_predictions=forecast,
        bootstrap_samples=500,
        block_days=2,
        bootstrap_seed=9,
    )

    assert intervals["season_id"].tolist() == forecast["season_id"].tolist()
    for column in (
        "cumulative_p025_kg_m2",
        "cumulative_p10_kg_m2",
        "cumulative_p50_kg_m2",
        "cumulative_p90_kg_m2",
        "cumulative_p975_kg_m2",
    ):
        assert intervals.loc[2, column] < intervals.loc[1, column]


def test_simulation_resets_crop_state_between_seasons():
    drivers = pd.DataFrame(
        {
            "season_id": ["s1", "s1", "s2", "s2"],
            "timestamp": pd.to_datetime(
                ["2025-05-01", "2025-05-02", "2026-05-01", "2026-05-02"]
            ),
            "net_fruit_dry_matter_change_kg_m2": [0.08, 0.0, 0.08, 0.0],
            "air_temperature_c": [20.0] * 4,
            "dt_hours": [0.0, 48.0, 0.0, 48.0],
            "pick": [False, True, False, True],
        }
    )
    parameters = HarvestCohortParameters(
        base_temperature_c=10.0,
        maturity_thermal_time_deg_day=20.0,
        dry_matter_fraction=0.08,
    )

    prediction = simulate_harvest_series(drivers, parameters)

    assert prediction["season_id"].tolist() == ["s1", "s1", "s2", "s2"]
    assert prediction.loc[prediction["season_id"] == "s1", "fresh_kg_m2"].sum() == 1.0
    assert prediction.loc[prediction["season_id"] == "s2", "fresh_kg_m2"].sum() == 1.0
    assert prediction.loc[prediction["season_id"] == "s2", "cumulative_fresh_kg_m2"].iloc[0] == 0.0


def test_greenlight_driver_uses_net_fruit_state_change_plus_native_harvest():
    trajectory = pd.DataFrame(
        {
            "season_id": ["s1", "s1"],
            "timestamp": pd.to_datetime(["2026-05-01 00:15", "2026-05-01 00:30"]),
            "c_fruit_previous_mg_m2": [100_000.0, 101_000.0],
            "c_fruit_mg_m2": [101_000.0, 100_500.0],
            "harvested_dry_matter_mg_m2": [0.0, 1_000.0],
            "air_temperature": [22.0, 21.0],
            "pick": [False, True],
            "pick_source": ["management_protocol"] * 2,
            "season_complete": [True] * 2,
            "greenhouse_id": [63] * 2,
            "greenhouse_code": ["PIDU"] * 2,
            "planting_code": ["P1"] * 2,
        }
    )

    drivers = build_harvest_drivers_from_greenlight(
        trajectory,
        dt_seconds=900.0,
        driver_model_status="accepted_target_crop_model",
    )

    assert drivers["net_fruit_dry_matter_change_kg_m2"].tolist() == [0.001, 0.0005]
    assert drivers["dt_hours"].tolist() == [0.25, 0.25]
    assert drivers["fruit_change_source"].eq(
        "greenlight_fruit_state_balance"
    ).all()
    assert drivers["air_temperature_source"].eq(
        "greenlight_simulated_indoor"
    ).all()
    assert drivers["driver_model_status"].eq("accepted_target_crop_model").all()
    assert drivers["initial_fruit_dry_matter_kg_m2"].tolist() == pytest.approx([0.1, 0.0])


def test_simulation_includes_initial_greenlight_fruit_inventory():
    drivers = pd.DataFrame(
        {
            "season_id": ["s1", "s1"],
            "timestamp": pd.to_datetime(["2026-05-01", "2026-05-03"]),
            "net_fruit_dry_matter_change_kg_m2": [0.0, 0.0],
            "initial_fruit_dry_matter_kg_m2": [0.08, 0.0],
            "initial_fruit_maturity_fraction": [0.0, 0.0],
            "air_temperature_c": [20.0, 20.0],
            "dt_hours": [0.0, 48.0],
            "pick": [False, True],
        }
    )
    parameters = HarvestCohortParameters(
        base_temperature_c=10.0,
        maturity_thermal_time_deg_day=20.0,
        dry_matter_fraction=0.08,
    )

    prediction = simulate_harvest_series(drivers, parameters)

    assert prediction["fresh_kg_m2"].sum() == 1.0
