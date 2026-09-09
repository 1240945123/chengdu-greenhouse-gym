from __future__ import annotations

import numpy as np


def test_parameter_spec_rejects_nonpositive_or_reversed_bounds():
    from experiments.reports.identify_chengdu_physics_v3 import validate_parameter_specs

    try:
        validate_parameter_specs({"bad": {"index": 208, "lower": 2.0, "upper": 1.0, "initial": 1.5}})
    except ValueError as exc:
        assert "lower" in str(exc)
    else:
        raise AssertionError("reversed bounds must be rejected")


def test_parameter_spec_keeps_sparse_pad_parameters_frozen():
    from experiments.reports.identify_chengdu_physics_v3 import split_parameter_specs

    specs = {
        "air_capacity": {"index": 208, "lower": 2.0, "upper": 20.0, "initial": 8.0},
        "pad_cooling": {"index": 218, "fixed": 0.0, "identifiable": False},
    }
    free, fixed = split_parameter_specs(specs)

    assert list(free) == ["air_capacity"]
    assert fixed == {218: 0.0}


def test_soft_l1_cost_limits_outlier_influence():
    from experiments.reports.identify_chengdu_physics_v3 import soft_l1_cost

    residual = np.array([1.0, 1.0, 20.0])
    assert soft_l1_cost(residual) < float(np.sum(residual**2))


def test_sensitivity_audit_reports_rank_and_weak_parameter():
    from experiments.reports.identify_chengdu_physics_v3 import finite_difference_sensitivity

    def residual(values):
        return np.array([values[0], 2.0 * values[0], 0.0 * values[1]])

    audit = finite_difference_sensitivity(residual, np.array([2.0, 3.0]), ["active", "weak"])

    assert audit["rank"] == 1
    assert audit["weak_parameter_names"] == ["weak"]
    assert len(audit["singular_values"]) == 2


def test_identifier_selection_api_has_no_test_dataset_argument():
    from inspect import signature
    from experiments.reports.identify_chengdu_physics_v3 import identify_and_select

    assert "test" not in signature(identify_and_select).parameters


def test_bounded_fit_uses_a_difference_step_visible_above_integrator_tolerance():
    from experiments.reports.identify_chengdu_physics_v3 import fit_bounded_soft_l1

    def quantized_residual(values):
        return np.array([round(float(values[0]), 5) - 2.0])

    result = fit_bounded_soft_l1(
        quantized_residual,
        start=np.array([1.0]),
        lower=np.array([0.1]),
        upper=np.array([3.0]),
        max_nfev=20,
    )

    assert result.x[0] > 1.9


def test_validation_can_select_unfitted_physical_reference_over_fitted_candidates():
    import pandas as pd
    from experiments.reports.identify_chengdu_physics_v3 import identify_and_select

    data = pd.DataFrame({"x": [1]})

    def fit_fn(_data, start):
        return {"values": start + 10.0}

    def validation_fn(_data, candidate):
        score = float(candidate["values"][0])
        return {"score": score, "physical_envelope_pass": True}

    selected, audit = identify_and_select(
        data,
        data,
        [np.array([2.0])],
        reference_start=np.array([1.0]),
        fit_fn=fit_fn,
        validation_fn=validation_fn,
    )

    assert selected["candidate_kind"] == "unfitted_physical_reference"
    assert audit["selected_candidate_id"] == 0
