import pandas as pd
import pytest

from experiments.reports.audit_chengdu_data_quality_v4 import (
    build_quality_audit_bundle,
    attach_rollout_quality,
    compare_regime_candidate,
    summarize_stratified_rollouts,
)


def _trajectory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transition_quality_weight": [1.0, 0.6, 0.0, 1.0],
            "target_heat_regime": ["normal", "high", "extreme", "normal"],
            "extreme_heat_physical_support": [False, False, True, False],
        }
    )


def _rollouts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_index": [0, 0, 1, 1],
            "horizon": [1, 2, 1, 2],
            "pred_air_temperature": [21.0, 33.0, 34.0, 39.0],
            "true_air_temperature": [20.0, 35.0, 35.0, 42.0],
            "pred_relative_humidity": [70.0, 76.0, 78.0, 72.0],
            "true_relative_humidity": [72.0, 80.0, 80.0, 78.0],
        }
    )


def test_rollout_quality_uses_the_actual_target_transition_index():
    attached = attach_rollout_quality(_rollouts(), _trajectory())

    assert attached["target_transition_index"].tolist() == [0, 1, 1, 2]
    assert attached["target_quality_tier"].tolist() == ["high", "medium", "medium", "low"]
    assert attached["target_heat_regime"].tolist() == ["normal", "high", "high", "extreme"]
    assert bool(attached.iloc[-1]["extreme_heat_physical_support"])


def test_stratified_summary_reports_count_mae_rmse_and_bias():
    attached = attach_rollout_quality(_rollouts(), _trajectory())
    summary = summarize_stratified_rollouts(attached, ["target_quality_tier"])

    medium = summary[summary["target_quality_tier"] == "medium"].iloc[0]
    assert medium["count"] == 2
    assert medium["air_temperature_mae"] == pytest.approx(1.5)
    assert medium["air_temperature_rmse"] == pytest.approx((2.5) ** 0.5)
    assert medium["air_temperature_bias"] == pytest.approx(-1.5)
    assert medium["relative_humidity_mae"] == pytest.approx(3.0)
    assert medium["relative_humidity_rmse"] == pytest.approx((10.0) ** 0.5)
    assert medium["relative_humidity_bias"] == pytest.approx(-3.0)


def test_rollout_quality_rejects_indices_outside_the_split():
    rollouts = _rollouts().iloc[[0]].copy()
    rollouts["start_index"] = 10

    with pytest.raises(ValueError, match="outside trajectory"):
        attach_rollout_quality(rollouts, _trajectory())


def test_quality_audit_bundle_writes_reproducible_report_files(tmp_path):
    trajectory_path = tmp_path / "test.csv"
    rollout_path = tmp_path / "rollouts.csv"
    _trajectory().to_csv(trajectory_path, index=False)
    _rollouts().to_csv(rollout_path, index=False)

    audit = build_quality_audit_bundle(trajectory_path, rollout_path, tmp_path / "audit")

    assert audit["trajectory_rows"] == 4
    assert audit["rollout_rows"] == 4
    assert audit["quality_counts"] == {"high": 2, "medium": 1, "low": 1}
    assert (tmp_path / "audit" / "rollouts_with_quality.csv").exists()
    assert (tmp_path / "audit" / "metrics_by_quality.csv").exists()
    assert (tmp_path / "audit" / "metrics_by_heat_regime.csv").exists()
    assert (tmp_path / "audit" / "metrics_by_horizon_quality_heat.csv").exists()
    assert (tmp_path / "audit" / "audit.json").exists()
    report = (tmp_path / "audit" / "audit.md").read_text(encoding="utf-8")
    assert "Quality-aware data audit" in report


def _gate_rollouts(*, high_t_error: float, high_rh_error: float, normal_scale: float):
    return pd.DataFrame(
        {
            "horizon": [24, 72, 24, 72],
            "target_heat_regime": ["normal", "normal", "high", "high"],
            "pred_air_temperature": [20.0 + normal_scale, 20.0 + normal_scale, 35.0 + high_t_error, 35.0 + high_t_error],
            "true_air_temperature": [20.0, 20.0, 35.0, 35.0],
            "pred_relative_humidity": [70.0 + 5.0 * normal_scale, 70.0 + 5.0 * normal_scale, 70.0 + high_rh_error, 70.0 + high_rh_error],
            "true_relative_humidity": [70.0, 70.0, 70.0, 70.0],
        }
    )


def test_dual_regime_gate_promotes_only_aggregate_and_high_heat_improvement():
    reference = _gate_rollouts(high_t_error=4.0, high_rh_error=10.0, normal_scale=1.0)
    candidate = _gate_rollouts(high_t_error=3.0, high_rh_error=8.0, normal_scale=1.01)

    decision = compare_regime_candidate(reference, candidate, normal_tolerance=0.02)

    assert decision["aggregate_improved"]
    assert decision["high_temperature_improved"]
    assert decision["normal_regime_within_tolerance"]
    assert decision["promote"]


def test_dual_regime_gate_rejects_candidate_when_high_heat_humidity_worsens():
    reference = _gate_rollouts(high_t_error=4.0, high_rh_error=10.0, normal_scale=1.0)
    candidate = _gate_rollouts(high_t_error=3.0, high_rh_error=11.0, normal_scale=0.9)

    decision = compare_regime_candidate(reference, candidate, normal_tolerance=0.02)

    assert decision["aggregate_improved"]
    assert not decision["high_temperature_improved"]
    assert not decision["promote"]
