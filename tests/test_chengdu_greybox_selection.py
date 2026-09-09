from __future__ import annotations

import inspect

import pandas as pd

from experiments.reports.select_chengdu_v4_greybox_residual import (
    build_validation_candidates,
    chronological_fit_calibration_split,
    rank_base_candidates,
    select_from_config,
    select_validation_candidate,
    reproducibility_code_hashes,
    resolve_fit_sample_weights,
)


def test_chronological_split_preserves_order_and_nonoverlap():
    frame = pd.DataFrame(
        {"timestamp": pd.date_range("2026-04-01", periods=10, freq="h"), "value": range(10)}
    )

    fit, calibration = chronological_fit_calibration_split(frame, fit_fraction=0.8)

    assert fit["value"].tolist() == list(range(8))
    assert calibration["value"].tolist() == [8, 9]
    assert fit["timestamp"].max() < calibration["timestamp"].min()


def test_calibration_ranking_is_deterministic():
    candidates = [
        {"candidate_id": 2, "calibration_score": 2.0},
        {"candidate_id": 1, "calibration_score": 1.0},
        {"candidate_id": 0, "calibration_score": 1.0},
    ]

    selected = rank_base_candidates(candidates, top_k=2)

    assert [item["candidate_id"] for item in selected] == [0, 1]


def test_validation_candidates_include_one_physical_fallback():
    base = [
        {"candidate_id": 3, "model": {"schema_version": "ridge_residual_v1", "alpha": 1.0}}
    ]

    candidates = build_validation_candidates(base, gains=[0.25, 0.5, 1.0])

    assert [item["gain"] for item in candidates if item["candidate_kind"] == "hybrid"] == [0.25, 0.5, 1.0]
    fallback = [item for item in candidates if item["candidate_kind"] == "physical_fallback"]
    assert len(fallback) == 1
    assert fallback[0]["gain"] == 0.0


def test_validation_selection_rejects_nonphysical_low_score():
    candidates = [
        {"candidate_id": 0, "score": 0.1, "physical_envelope_pass": False},
        {"candidate_id": 1, "score": 1.2, "physical_envelope_pass": True},
        {"candidate_id": 2, "score": 1.5, "physical_envelope_pass": True},
    ]

    assert select_validation_candidate(candidates)["candidate_id"] == 1


def test_selection_api_has_no_test_split_argument():
    assert "test_csv" not in inspect.signature(select_from_config).parameters


def test_reproducibility_hashes_cover_fit_and_state_feedback_modules():
    hashes = reproducibility_code_hashes()

    assert set(hashes) == {"selector", "residual_correction", "multistep_evaluator"}
    assert all(len(value) == 64 for value in hashes.values())


def test_fit_sample_weights_require_configured_v4_column():
    frame = pd.DataFrame({"transition_quality_weight": [1.0, 0.6, 0.0]})

    weights = resolve_fit_sample_weights(frame, "transition_quality_weight")

    assert weights.tolist() == [1.0, 0.6, 0.0]
    assert resolve_fit_sample_weights(frame, None) is None
    try:
        resolve_fit_sample_weights(frame, "missing_weight")
    except ValueError as error:
        assert "missing_weight" in str(error)
    else:
        raise AssertionError("Missing configured sample-weight column was accepted")


def test_fit_sample_weights_support_capped_heat_regime_boosts():
    frame = pd.DataFrame(
        {"target_heat_regime": ["normal", "high", "extreme", "normal"]}
    )

    weights = resolve_fit_sample_weights(
        frame,
        None,
        heat_regime_weights={"normal": 1.0, "high": 5.0, "extreme": 5.0},
    )

    assert weights.tolist() == [1.0, 5.0, 5.0, 1.0]
