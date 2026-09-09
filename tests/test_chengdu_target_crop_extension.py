from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.crop.extend_chengdu_target_crop_trajectory import (
    fill_long_canopy_temperature_gaps,
    load_observed_weather_horizon,
    run_driver_horizon_extension,
)


def test_driver_horizon_extension_runs_observed_short_slice(tmp_path: Path):
    report = run_driver_horizon_extension(
        crop_observations_path=(
            "results/chengdu_agri_greenhouse_001/harvest_model/processed/"
            "standing_crop_samples.csv"
        ),
        controls_path=(
            "data/processed/chengdu_agri/greenhouse_001/controls/controls_1h.csv"
        ),
        aligned_climate_path=(
            "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
        ),
        env_config_path="configs/envs/ChengduSingleGreenhouseEnv.yml",
        harvest_config_path="configs/crops/chengdu_tomato_harvest.yml",
        organ_calibration_path=(
            "results/chengdu_agri_greenhouse_001/harvest_model/"
            "target_crop_validation/organ_allocation_calibration.json"
        ),
        output_root=tmp_path,
        simulation_start="2026-04-01 00:00:00",
        simulation_end="2026-04-01 03:00:00",
        substep_seconds=3600,
    )

    assert report["trajectory_rows"] == 4
    assert report["simulation_start"] == "2026-04-01T00:00:00+08:00"
    assert report["simulation_end"] == "2026-04-01T03:00:00+08:00"
    assert report["driver_model_status"] == "diagnostic_not_final"
    assert report["season_complete"] is False
    assert report["target_harvest_validated"] is False
    assert report["use"] == "driver_preparation_only_not_calibration_evidence"
    assert set(report["organ_allocation_multipliers"]) == {"fruit", "leaf", "stem"}
    assert report["mass_balance_residual_absolute_sum_kg_m2"] < 1e-8
    trajectory_path = tmp_path / "target_crop_trajectory_driver_horizon.csv"
    audit_path = tmp_path / "target_crop_trajectory_driver_horizon_audit.json"
    assert trajectory_path.exists() and audit_path.exists()
    trajectory = pd.read_csv(trajectory_path)
    values = trajectory[
        [
            "standing_dry_kg_m2",
            "net_fruit_dry_matter_change_kg_m2",
            "air_temperature_c",
            "canopy_thermal_sum_deg_day",
        ]
    ].to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_long_canopy_gap_uses_local_hourly_air_temperature_delta_proxy():
    timestamps = pd.date_range("2026-06-01", periods=72, freq="h")
    hour = timestamps.hour.to_numpy()
    air = 24.0 + 3.0 * np.sin(2.0 * np.pi * hour / 24.0)
    canopy = air - (1.0 + hour / 24.0)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": air,
            "canopy_temperature": canopy,
        }
    )
    frame.loc[24:47, "canopy_temperature"] = np.nan

    filled, audit = fill_long_canopy_temperature_gaps(
        frame,
        maximum_interpolation_gap_hours=8,
        local_reference_window_days=2,
    )

    assert filled["canopy_temperature"].isna().sum() == 0
    assert np.allclose(filled.loc[24:47, "canopy_temperature"], canopy[24:48])
    assert audit["proxy_filled_rows"] == 24
    assert audit["long_gap_count"] == 1
    assert audit["method"] == "local_hourly_median_canopy_minus_air_delta"
    assert audit["proxy_reference_mae_c"] < 1e-12


def test_observed_weather_horizon_supports_partial_final_day_without_next_year():
    weather, audit = load_observed_weather_horizon(
        "data/processed/chengdu_agri/greenhouse_001/weather/Chengdu/2026.csv",
        simulation_start="2026-04-01 00:00:00",
        simulation_end="2026-07-20 09:00:00",
    )

    assert weather.shape == (2650, 10)
    assert np.isfinite(weather).all()
    assert audit["weather_rows"] == 2650
    assert audit["partial_final_day_supported"] is True
    assert audit["source_start_day_number"] == 91
    assert audit["source_end_day_number"] == 201
