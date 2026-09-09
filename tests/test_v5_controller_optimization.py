import pandas as pd

from experiments.controllers.v5_controller_optimization import (
    CONFIRMATION_SCENARIO,
    HISTORICAL_SMOKE_SCENARIO,
    TRAINING_SCENARIOS,
    VALIDATION_SCENARIOS,
    aggregate_validation_candidates,
    assess_confirmation,
    select_best_candidate,
)
from experiments.controllers.run_v5_controller_optimization import (
    build_confirmation_plan,
    build_mpc_candidates,
    build_protocol_manifest,
)
from experiments.controllers.run_v5_residual_ppo_optimization import (
    SECOND_CONFIRMATION_SCENARIO,
    _report,
    build_residual_protocol_manifest,
)


def _row(candidate_id, day, reward, *, temperature=0.2, humidity=0.5, variation=1.0):
    return {
        "candidate_id": candidate_id,
        "growth_year": 2025,
        "start_day": day,
        "cumulative_reward": reward,
        "temperature_band_mae_c": temperature,
        "relative_humidity_band_mae_percent": humidity,
        "total_executed_action_variation": variation,
        "mean_controller_inference_ms": 1.0,
        "completed_episode": True,
        "numerical_failure": False,
        "all_values_finite": True,
        "residual_fallback_count": 0,
        "crop_carbon_nonnegative": True,
        "safety_intervention_fraction": 0.0,
    }


def test_protocol_scenarios_are_disjoint():
    roles = [set(TRAINING_SCENARIOS), set(VALIDATION_SCENARIOS), {CONFIRMATION_SCENARIO}]
    assert all(not roles[i] & roles[j] for i in range(3) for j in range(i + 1, 3))
    assert HISTORICAL_SMOKE_SCENARIO not in set().union(*roles)


def test_validation_selection_uses_all_scenarios_and_reward_first():
    rows = []
    for day, reward_a, reward_b in zip((59, 74, 89), (-10, -10, -10), (-8, -8, -20)):
        rows.extend([_row("a", day, reward_a), _row("b", day, reward_b)])
    summary = aggregate_validation_candidates(pd.DataFrame(rows), expected_scenarios=VALIDATION_SCENARIOS)

    selected = select_best_candidate(summary)

    assert selected["candidate_id"] == "a"
    assert selected["mean_cumulative_reward"] == -10.0


def test_validation_selection_rejects_incomplete_candidate():
    frame = pd.DataFrame([_row("complete", day, -10) for _, day in VALIDATION_SCENARIOS] + [_row("partial", 59, -1)])
    summary = aggregate_validation_candidates(frame, expected_scenarios=VALIDATION_SCENARIOS)

    assert select_best_candidate(summary)["candidate_id"] == "complete"


def test_confirmation_requires_reward_and_non_reward_gates():
    pid = _row("pid", 104, -10, temperature=1.0, humidity=2.0)
    passing = _row("candidate", 104, -9, temperature=1.2, humidity=2.8)
    failing = _row("candidate", 104, -9, temperature=1.3, humidity=2.8)

    assert assess_confirmation(passing, pid)["promoted"] is True
    result = assess_confirmation(failing, pid)
    assert result["promoted"] is False
    assert result["gates"]["temperature_band_mae"] is False


def test_confirmation_prefers_active_material_safety_over_legacy_strict_noise():
    pid = _row("pid", 104, -10, temperature=1.0, humidity=2.0)
    candidate = _row("candidate", 104, -9, temperature=1.0, humidity=2.0)
    pid["safety_intervention_fraction"] = 0.0
    candidate["safety_intervention_fraction"] = 0.5
    pid["active_safety_intervention_fraction"] = 0.0
    candidate["active_safety_intervention_fraction"] = 0.01

    result = assess_confirmation(candidate, pid)

    assert result["gates"]["safety_intervention"] is True
    assert result["safety_metric"] == "active_safety_intervention_fraction"
    assert result["promoted"] is True


def test_runner_manifest_preserves_frozen_data_roles():
    manifest = build_protocol_manifest(episode_days=1, seeds=[0, 1], checkpoints=[8192])

    assert manifest["validation_scenarios"] == [
        {"growth_year": 2025, "start_day": 59},
        {"growth_year": 2025, "start_day": 74},
        {"growth_year": 2025, "start_day": 89},
    ]
    assert manifest["confirmation_scenario"]["start_day"] == 104
    assert manifest["historical_smoke_scenario"]["used_for_selection"] is False


def test_mpc_candidate_ids_are_stable_and_unique():
    first = build_mpc_candidates()
    second = build_mpc_candidates()

    assert first == second
    assert len({candidate["candidate_id"] for candidate in first}) == len(first)
    assert all(candidate["algorithm"] == "mpc" for candidate in first)


def test_confirmation_plan_contains_only_frozen_selections_and_pid():
    selections = {
        "mpc": {"candidate_id": "mpc_selected"},
        "ppo": {"candidate_id": "ppo_selected"},
        "sac": {"candidate_id": "sac_selected"},
    }

    plan = build_confirmation_plan(selections)

    assert [item["candidate_id"] for item in plan] == [
        "pid",
        "mpc_selected",
        "ppo_selected",
        "sac_selected",
    ]


def test_residual_protocol_uses_new_confirmation_and_explicit_label():
    manifest = build_residual_protocol_manifest(
        seeds=[0, 1], checkpoints=[4096, 8192], residual_scale=0.25
    )

    assert SECOND_CONFIRMATION_SCENARIO == (2025, 119)
    assert manifest["algorithm_label"] == "residual_ppo"
    assert manifest["pure_ppo_first_stage_result"] == "failed"
    assert manifest["first_confirmation_scenario"]["used_for_tuning"] is False
    assert manifest["second_confirmation_scenario"]["start_day"] == 119


def test_residual_report_names_all_safety_metric_definitions():
    summary = pd.DataFrame(
        [
            {
                "candidate_id": "residual_ppo",
                "mean_cumulative_reward": -1.0,
                "mean_temperature_band_mae_c": 1.0,
                "mean_relative_humidity_band_mae_percent": 2.0,
                "mean_total_executed_action_variation": 3.0,
                "feasible": True,
            }
        ]
    )
    confirmation = pd.DataFrame(
        [
            {
                "algorithm": "pid",
                "candidate_id": "pid",
                "cumulative_reward": -1.0,
                "temperature_band_mae_c": 1.0,
                "relative_humidity_band_mae_percent": 2.0,
                "joint_comfort_fraction": 0.5,
                "safety_intervention_fraction": 0.5,
                "material_safety_intervention_fraction": 0.1,
                "active_safety_intervention_fraction": 0.0,
                "mean_executed_actuator_effort": 0.2,
                "total_executed_action_variation": 1.0,
                "fruit_state_change_mg_m2": 1.0,
            }
        ]
    )
    decisions = {
        "residual_ppo": {
            "promoted": True,
            "reward_improvement_over_pid": 0.1,
            "safety_metric": "active_safety_intervention_fraction",
            "gates": {"safety_intervention": True},
        }
    }

    report = _report(summary, confirmation, decisions)

    assert "legacy strict" in report
    assert "material_safety_intervention_fraction" in report
    assert "active_safety_intervention_fraction" in report
