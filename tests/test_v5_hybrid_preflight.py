import numpy as np

from experiments.controllers.run_v5_hybrid_preflight import run_policy_scenario


def test_short_v5_hybrid_policy_scenario_reports_control_and_crop_gates():
    metrics, trajectory = run_policy_scenario(
        scenario="mid",
        target_controls=np.array([0.5, 0.5, 0.5, 0.5]),
        growth_year=2025,
        start_day=226,
        episode_days=1,
        max_steps=4,
    )

    assert len(trajectory) == 4
    assert metrics["scenario"] == "mid"
    assert metrics["steps"] == 4
    assert metrics["all_values_finite"] is True
    assert metrics["numerical_failure"] is False
    assert metrics["crop_carbon_nonnegative"] is True
    assert 0.0 <= metrics["joint_comfort_fraction"] <= 1.0
    assert 0.0 <= metrics["safety_intervention_fraction"] <= 1.0
    assert set(["uRoofVent", "uFan", "air_temperature", "relative_humidity", "reward"]).issubset(
        trajectory.columns
    )
