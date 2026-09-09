from __future__ import annotations

import pandas as pd


def test_candidate_fitting_and_selection_never_receives_test_data():
    from experiments.reports.calibrate_chengdu_thermal_v2 import fit_and_select_candidates

    train = pd.DataFrame({"role": ["train"]})
    validation = pd.DataFrame({"role": ["validation"]})
    candidates = {"a": 1.0, "b": 2.0, "c": 3.0}
    calls = []

    def evaluator(data, candidate, role):
        calls.append((data.iloc[0]["role"], role, candidate))
        if role == "fit":
            return {"score": {1.0: 0.1, 2.0: 0.2, 3.0: 2.0}[candidate], "physical_envelope_pass": True}
        return {"score": {1.0: 0.8, 2.0: 0.3}[candidate], "physical_envelope_pass": True}

    selected, audit = fit_and_select_candidates(
        train,
        validation,
        candidates,
        evaluator=evaluator,
        shortlist_size=2,
    )

    assert selected == 2.0
    assert audit["selected_candidate_id"] == "b"
    assert {(data_role, evaluation_role) for data_role, evaluation_role, _ in calls} == {
        ("train", "fit"),
        ("validation", "selection"),
    }
    assert all(candidate != 3.0 for data_role, _, candidate in calls if data_role == "validation")


def test_selection_rejects_physically_invalid_candidate_even_with_lower_error():
    from experiments.reports.calibrate_chengdu_thermal_v2 import fit_and_select_candidates

    data = pd.DataFrame({"role": ["data"]})

    def evaluator(_data, candidate, role):
        if role == "fit":
            return {"score": candidate, "physical_envelope_pass": True}
        return {"score": candidate, "physical_envelope_pass": candidate > 0.1}

    selected, audit = fit_and_select_candidates(
        data,
        data,
        {"invalid": 0.1, "valid": 0.2},
        evaluator=evaluator,
        shortlist_size=2,
    )

    assert selected == 0.2
    assert audit["rejected_candidate_ids"] == ["invalid"]
