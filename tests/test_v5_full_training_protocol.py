from pathlib import Path

import pytest

from experiments.controllers.v5_full_training_protocol import (
    load_v5_full_training_protocol,
    should_continue_training,
)


CONFIG = Path("configs/benchmarks/chengdu_v5_full_training.yml")


def test_protocol_expands_all_valid_training_windows_without_temporal_leakage():
    protocol = load_v5_full_training_protocol(CONFIG)

    assert protocol.episode_days == 14
    assert len(protocol.training_scenarios) == 428
    assert protocol.training_scenarios[0] == (2023, 59)
    assert protocol.training_scenarios[-1] == (2024, 333)
    assert {year for year, _day in protocol.training_scenarios} == {2023, 2024}
    assert all(year < 2025 for year, _day in protocol.training_scenarios)


def test_protocol_uses_non_overlapping_spring_validation_and_autumn_holdout():
    protocol = load_v5_full_training_protocol(CONFIG)

    assert protocol.validation_scenarios == tuple(
        (2025, day) for day in (59, 73, 87, 101, 115, 129, 143, 157)
    )
    assert protocol.temporal_holdout == (2025, 226, 120)
    assert protocol.seeds == (0, 1, 2)
    assert protocol.checkpoints == (20_480, 102_400, 307_200, 614_400)


def test_protocol_checkpoints_align_with_ppo_rollout_length():
    protocol = load_v5_full_training_protocol(CONFIG)
    rollout_steps = int(protocol.algorithms["ppo"]["n_steps"])

    assert all(checkpoint % rollout_steps == 0 for checkpoint in protocol.checkpoints)


def test_protocol_rejects_validation_overlap_with_holdout(tmp_path):
    invalid = tmp_path / "invalid.yml"
    text = CONFIG.read_text(encoding="utf-8").replace(
        "validation_start_days: [59, 73, 87, 101, 115, 129, 143, 157]",
        "validation_start_days: [226]",
    )
    invalid.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="validation.*holdout"):
        load_v5_full_training_protocol(invalid)


def test_training_continues_when_validation_improvement_is_material():
    assert should_continue_training({100_000: -100.0, 300_000: -95.0}) is True
    assert should_continue_training({100_000: -100.0, 300_000: -99.5}) is False
    assert should_continue_training({100_000: -100.0}) is True
