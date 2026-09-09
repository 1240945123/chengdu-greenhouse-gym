import inspect

from experiments.reports.select_chengdu_physics_v4 import (
    generate_moisture_candidates,
    generate_thermal_candidates,
    rank_train_candidates,
    select_from_config,
    select_validation_candidate,
)


GRID = {
    "thermal_grid": {
        "longwave_scale": [0.0, 1.0],
        "floor_capacity_scale": [1.0, 2.0],
        "latent_heat_scale": [0.0, 1.0],
    },
    "moisture_grid": {
        "moisture_buffer_rate_s": [0.0, 1.0 / 7200.0],
        "moisture_buffer_capacity_ratio": [1.0, 4.0],
    },
}


def test_candidate_generation_is_deterministic_and_contains_v3_reference():
    first = generate_thermal_candidates(GRID)
    second = generate_thermal_candidates(GRID)

    assert first == second
    assert first[0]["candidate_kind"] == "v3_reference"
    assert first[0]["extension"] == [0.0, 1.0, 0.0, 1.0, 0.0]
    assert len({tuple(item["extension"]) for item in first}) == len(first)


def test_train_ranking_retains_reference_even_when_not_in_top_scores():
    candidates = [
        {"candidate_kind": "v3_reference", "extension": [0, 1, 0, 1, 0], "score": 9.0, "physical_envelope_pass": True},
        {"candidate_kind": "thermal", "extension": [1, 1, 0, 1, 1], "score": 1.0, "physical_envelope_pass": True},
        {"candidate_kind": "thermal", "extension": [0.5, 2, 0, 1, 0.5], "score": 2.0, "physical_envelope_pass": True},
    ]

    ranked = rank_train_candidates(candidates, top_k=2)

    assert len(ranked) == 2
    assert candidates[1] in ranked
    assert candidates[0] in ranked


def test_moisture_generation_combines_unique_thermal_finalists():
    thermal = [
        {"candidate_kind": "v3_reference", "extension": [0.0, 1.0, 0.0, 1.0, 0.0]},
        {"candidate_kind": "thermal", "extension": [1.0, 2.0, 0.0, 1.0, 1.0]},
    ]

    candidates = generate_moisture_candidates(thermal, GRID)

    assert any(item["extension"] == [1.0, 2.0, 1.0 / 7200.0, 4.0, 1.0] for item in candidates)
    assert len({tuple(item["extension"]) for item in candidates}) == len(candidates)


def test_validation_selection_rejects_nonphysical_candidate():
    evaluated = [
        {"candidate_id": 0, "score": 0.5, "physical_envelope_pass": False},
        {"candidate_id": 1, "score": 1.5, "physical_envelope_pass": True},
        {"candidate_id": 2, "score": 2.0, "physical_envelope_pass": True},
    ]

    assert select_validation_candidate(evaluated)["candidate_id"] == 1


def test_selection_api_has_no_test_split_argument():
    assert "test_csv" not in inspect.signature(select_from_config).parameters
