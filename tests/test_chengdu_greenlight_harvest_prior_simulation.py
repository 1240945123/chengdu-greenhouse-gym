import numpy as np
import pandas as pd
import pytest
import json

from experiments.crop.simulate_chengdu_greenlight_harvest_priors import (
    assemble_high_fidelity_ensemble,
    canonical_chengdu_seasons,
    collect_greenlight_trajectories,
    load_transfer_priors,
    load_greenlight_trajectory_cache,
    run_greenlight_season,
    sample_harvest_scenarios,
    simulate_cohort_scenario,
    write_ensemble_artifacts,
)


def test_canonical_seasons_cover_six_complete_120_day_windows():
    seasons = canonical_chengdu_seasons()

    assert seasons["season_id"].tolist() == [
        "2023_spring",
        "2023_autumn",
        "2024_spring",
        "2024_autumn",
        "2025_spring",
        "2025_autumn",
    ]
    assert seasons["growth_year"].tolist() == [2023, 2023, 2024, 2024, 2025, 2025]
    assert seasons["start_day"].tolist() == [59, 226, 60, 227, 59, 226]
    assert seasons["season_days"].eq(120).all()


def test_latin_hypercube_sampling_is_deterministic_and_covers_sensitivity_levels():
    season_ids = ["2023_spring", "2023_autumn"]

    first = sample_harvest_scenarios(
        season_ids,
        draws_per_climate=3,
        seed=42,
    )
    second = sample_harvest_scenarios(
        season_ids,
        draws_per_climate=3,
        seed=42,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 18
    assert set(first["climate_scenario"]) == {"calibrated_low", "calibrated", "calibrated_high"}
    assert first.groupby("season_id")["climate_scenario"].nunique().eq(3).all()
    assert first["maturity_thermal_time_deg_day"].between(
        583.7624166768941, 612.7034811152284
    ).all()
    assert first["dry_matter_fraction"].between(0.0758238837, 0.0846179301).all()
    assert first["pick_interval_days"].eq(5).all()
    assert first["pick_hour"].between(7, 10).all()
    assert first["scenario_id"].str.startswith("SIM_").all()
    assert first["transfer_calibration_status"].eq(
        "external_transfer_calibrated_not_target_validated"
    ).all()
    paired_columns = [
        "maturity_thermal_time_deg_day",
        "dry_matter_fraction",
        "minimum_pick_fresh_kg_m2",
        "pick_interval_days",
        "pick_hour",
    ]
    for _, paired in first.groupby(["season_id", "cohort_draw"]):
        assert len(paired) == 3
        assert paired[paired_columns].nunique().eq(1).all()


def test_transfer_priors_load_registered_wur_intervals(tmp_path):
    path = tmp_path / "process_priors.json"
    path.write_text(
        json.dumps(
            {
                "evidence_class": "external_observed",
                "target_eligible": False,
                "transfer_role": "external_process_prior_not_target_calibration",
                "source_dois": ["10.test/wur"],
                "metrics": {
                    "maturity_thermal_time_deg_day": {
                        "ci95_low": 580.0,
                        "ci95_high": 620.0,
                    },
                    "median_interpick_days": {"estimate": 5.0},
                },
            }
        ),
        encoding="utf-8",
    )

    priors = load_transfer_priors(path)

    assert priors["maturity_thermal_time_deg_day"] == (580.0, 620.0)
    assert priors["pick_interval_days"] == (5, 5)
    assert priors["source_dois"] == ["10.test/wur"]


def _toy_greenlight_trajectory() -> pd.DataFrame:
    timestamps = pd.date_range("2023-03-01", periods=24 * 12, freq="h")
    fruit = np.arange(len(timestamps) + 1, dtype=float) * 10_000.0
    return pd.DataFrame(
        {
            "season_id": "2023_spring",
            "timestamp": timestamps,
            "c_fruit_previous_mg_m2": fruit[:-1],
            "c_fruit_mg_m2": fruit[1:],
            "harvested_dry_matter_mg_m2": 0.0,
            "allocated_fruit_dry_matter_mg_m2": 10_000.0,
            "air_temperature": 25.0,
            "uBoil": 0.0,
            "uCO2": 0.0,
        }
    )


def test_cohort_scenario_emits_isolated_events_and_closes_dry_matter_balance():
    sample = pd.Series(
        {
            "scenario_id": "SIM_2023_spring_calibrated_000",
            "season_id": "2023_spring",
            "climate_scenario": "calibrated",
            "climate_calibration_strength": 1.0,
            "base_temperature_c": 10.0,
            "maturity_thermal_time_deg_day": 10.0,
            "dry_matter_fraction": 0.08,
            "minimum_pick_fresh_kg_m2": 0.0,
            "pick_interval_days": 4,
            "pick_hour": 8,
            "initial_fruit_maturity_fraction": 0.0,
        }
    )

    drivers, events, summary = simulate_cohort_scenario(
        _toy_greenlight_trajectory(), sample
    )

    assert not events.empty
    assert events["synthetic_event_id"].str.startswith("SIM_").all()
    assert events["synthetic_greenhouse_code"].str.startswith("SIM_").all()
    assert events["evidence_class"].eq("simulated_prior").all()
    assert not events["target_eligible"].any()
    assert drivers["pick_source"].eq("simulated_external_prior_protocol").all()
    assert drivers["driver_model_status"].eq("diagnostic_simulated_prior_model").all()
    assert summary["dry_matter_balance_error_kg_m2"] == pytest.approx(0.0, abs=1e-10)
    assert summary["partial_yield_kg_m2"] == pytest.approx(events["fresh_kg_m2"].sum())
    assert summary["target_eligible"] is False


def test_cohort_scenario_rejects_disabled_control_activity():
    trajectory = _toy_greenlight_trajectory()
    trajectory.loc[0, "uBoil"] = 0.1
    sample = sample_harvest_scenarios(
        ["2023_spring"], draws_per_climate=1, seed=7
    ).iloc[0]

    with pytest.raises(ValueError, match="heating and CO2 controls must remain disabled"):
        simulate_cohort_scenario(trajectory, sample)


def test_cohort_loss_is_scaled_to_its_own_inventory_after_independent_pick():
    trajectory = _toy_greenlight_trajectory()
    loss_index = 8 * 24 + 9
    increments = np.full(len(trajectory), 10_000.0)
    increments[loss_index] = -50_000.0
    fruit = np.r_[0.0, np.cumsum(increments)]
    trajectory["c_fruit_previous_mg_m2"] = fruit[:-1]
    trajectory["c_fruit_mg_m2"] = fruit[1:]
    trajectory["allocated_fruit_dry_matter_mg_m2"] = np.maximum(increments, 0.0)
    sample = pd.Series(
        {
            "scenario_id": "SIM_2023_spring_calibrated_loss",
            "season_id": "2023_spring",
            "climate_scenario": "calibrated",
            "climate_calibration_strength": 1.0,
            "base_temperature_c": 10.0,
            "maturity_thermal_time_deg_day": 10.0,
            "dry_matter_fraction": 0.08,
            "minimum_pick_fresh_kg_m2": 0.0,
            "pick_interval_days": 4,
            "pick_hour": 8,
            "initial_fruit_maturity_fraction": 0.0,
        }
    )

    _drivers, _events, summary = simulate_cohort_scenario(trajectory, sample)

    assert summary["applied_nonharvest_loss_dry_matter_kg_m2"] > 0.0
    assert summary["dry_matter_balance_error_kg_m2"] == pytest.approx(0.0, abs=1e-10)


def test_greenlight_runner_records_hourly_state_and_disabled_controls():
    season = canonical_chengdu_seasons().iloc[0]

    trajectory = run_greenlight_season(
        season,
        climate_scenario="calibrated",
        climate_calibration_strength=1.0,
        seed=0,
        max_steps=3,
    )

    assert len(trajectory) == 3
    assert trajectory["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert trajectory["season_id"].eq("2023_spring").all()
    assert trajectory["climate_scenario"].eq("calibrated").all()
    assert trajectory["crop_model"].eq("Vanthoor2011_GreenLight").all()
    assert trajectory[["uBoil", "uCO2"]].eq(0.0).all().all()
    assert np.isfinite(
        trajectory[
            [
                "air_temperature",
                "c_fruit_previous_mg_m2",
                "c_fruit_mg_m2",
                "harvested_dry_matter_mg_m2",
            ]
        ].to_numpy(float)
    ).all()


def test_ensemble_reuses_climate_trajectories_and_audits_all_artifacts():
    seasons = pd.DataFrame(
        [{"season_id": "2023_spring", "growth_year": 2023, "start_day": 59, "season_days": 12}]
    )
    samples = sample_harvest_scenarios(
        ["2023_spring"], draws_per_climate=2, seed=11
    )
    samples["maturity_thermal_time_deg_day"] = 10.0
    calls = []

    def fake_runner(season, *, climate_scenario, climate_calibration_strength, seed):
        calls.append((climate_scenario, climate_calibration_strength, seed))
        frame = _toy_greenlight_trajectory()
        frame["climate_scenario"] = climate_scenario
        frame["climate_calibration_strength"] = climate_calibration_strength
        frame["crop_model"] = "Vanthoor2011_GreenLight"
        return frame

    trajectories, drivers, events, summaries, audit = assemble_high_fidelity_ensemble(
        seasons,
        samples,
        trajectory_runner=fake_runner,
        seed=11,
    )

    assert len(calls) == 3
    assert len(trajectories) == 3 * 12 * 24
    assert drivers["scenario_id"].nunique() == 6
    assert events["scenario_id"].nunique() == 6
    assert len(summaries) == 6
    assert audit["trajectory_count"] == 3
    assert audit["scenario_count"] == 6
    assert audit["pipeline"]["climate_backend"] == "ChengduPhysics_calibrated_sensitivity"
    assert audit["pipeline"]["crop_backend"] == "Vanthoor2011_GreenLight"
    assert audit["pipeline"]["harvest_backend"] == "dry_matter_age_cohort"
    assert audit["sampling_design"]["paired_across_climate_scenarios"] is True
    assert audit["quality_gates"]["all_mass_balances_closed"] is True
    assert audit["target_eligible"] is False


def test_trajectory_collector_runs_each_season_climate_pair_once():
    seasons = canonical_chengdu_seasons().iloc[:2]
    calls = []

    def fake_runner(season, *, climate_scenario, climate_calibration_strength, seed):
        calls.append((season["season_id"], climate_scenario, seed))
        return pd.DataFrame({"season_id": [season["season_id"]]})

    collected = collect_greenlight_trajectories(
        seasons,
        seed=19,
        max_workers=1,
        trajectory_runner=fake_runner,
    )

    assert len(calls) == 6
    assert set(collected) == {
        (season_id, climate)
        for season_id in seasons["season_id"]
        for climate in ("calibrated_low", "calibrated", "calibrated_high")
    }


def test_artifact_writer_creates_separate_traceable_outputs(tmp_path):
    frames = {
        "greenlight_state_trajectories.csv": pd.DataFrame({"trajectory_id": ["x"]}),
        "harvest_drivers.csv": pd.DataFrame({"scenario_id": ["SIM_x"]}),
        "sampled_parameters.csv": pd.DataFrame({"scenario_id": ["SIM_x"]}),
        "synthetic_harvest_events.csv": pd.DataFrame({"scenario_id": ["SIM_x"]}),
        "synthetic_season_summaries.csv": pd.DataFrame({"scenario_id": ["SIM_x"]}),
    }
    audit = {"target_eligible": False, "scenario_count": 1}

    written = write_ensemble_artifacts(tmp_path, frames=frames, audit=audit)

    assert set(path.name for path in written) == set(frames) | {"simulation_audit.json"}
    assert pd.read_csv(tmp_path / "sampled_parameters.csv").iloc[0]["scenario_id"] == "SIM_x"
    assert '"target_eligible": false' in (tmp_path / "simulation_audit.json").read_text(
        encoding="utf-8"
    )
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_trajectory_artifact_can_be_reused_without_environment_rerun(tmp_path):
    path = tmp_path / "greenlight_state_trajectories.csv"
    pd.DataFrame(
        {
            "season_id": ["2023_spring", "2023_spring", "2023_spring"],
            "climate_scenario": ["calibrated", "calibrated", "calibrated_low"],
            "timestep": [0, 1, 0],
        }
    ).to_csv(path, index=False)

    cache = load_greenlight_trajectory_cache(path)

    assert set(cache) == {
        ("2023_spring", "calibrated"),
        ("2023_spring", "calibrated_low"),
    }
    assert cache[("2023_spring", "calibrated")]["timestep"].tolist() == [0, 1]
