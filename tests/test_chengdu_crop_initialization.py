from __future__ import annotations

import pandas as pd
import pytest

from processing.chengdu_crop_initialization import build_target_crop_initial_state


def _observations() -> pd.DataFrame:
    rows = []
    for sample, scale in (("S1", 1.0), ("S2", 2.0), ("S3", 3.0)):
        rows.append(
            {
                "observation_type": "standing_crop_sample",
                "observation_date": "2025-06-01T00:00:00+08:00",
                "greenhouse_id": 1,
                "greenhouse_code": "GH-XINDU",
                "sample_plant_id": sample,
                "source_site": "chengdu_xindu_experimental_base",
                "target_eligible": False,
                "leaf_fresh_kg_m2": scale,
                "leaf_dry_kg_m2": scale * 0.10,
                "stem_fresh_kg_m2": scale,
                "stem_dry_kg_m2": scale * 0.08,
                "root_fresh_kg_m2": scale,
                "root_dry_kg_m2": scale * 0.12,
                "ripe_fruit_fresh_kg_m2": scale,
                "ripe_fruit_dry_kg_m2": scale * 0.06,
            }
        )
    for sample, leaf, stem, root in (
        ("1", 0.020, 0.030, 0.010),
        ("2", 0.025, 0.032, 0.011),
        ("3", 0.027, 0.034, 0.012),
    ):
        rows.append(
            {
                "observation_type": "standing_crop_sample",
                "observation_date": "2026-03-30T00:00:00+08:00",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "sample_plant_id": sample,
                "source_site": "chengdu_pidu_target_greenhouse",
                "target_eligible": True,
                "leaf_fresh_kg_m2": leaf,
                "leaf_dry_kg_m2": None,
                "stem_fresh_kg_m2": stem,
                "stem_dry_kg_m2": None,
                "root_fresh_kg_m2": root,
                "root_dry_kg_m2": None,
                "ripe_fruit_fresh_kg_m2": 0.0,
                "ripe_fruit_dry_kg_m2": None,
            }
        )
    return pd.DataFrame(rows)


def test_build_target_crop_initial_state_uses_target_replicates_and_source_pairs():
    state, audit = build_target_crop_initial_state(
        _observations(),
        target_greenhouse_id=63,
        target_greenhouse_code="GH-PIDU",
        baseline_date="2026-03-30",
    )

    assert state["cBuf"] == 0.0
    assert state["cLeaf"] == pytest.approx((0.020 + 0.025 + 0.027) / 3 * 0.10 * 1e6)
    assert state["cStem"] == pytest.approx((0.030 + 0.032 + 0.034) / 3 * 0.08 * 1e6)
    assert state["cFruit"] == 0.0
    assert state["tCanSum"] == 0.0
    assert audit["baseline_sample_count"] == 3
    assert audit["baseline_date"] == "2026-03-30T00:00:00+08:00"
    assert audit["dry_matter_fraction_transfer"]["leaf"]["estimate"] == pytest.approx(0.10)
    assert audit["dry_matter_fraction_transfer"]["ripe_fruit"]["estimate"] == pytest.approx(0.06)
    assert audit["root_state_applied_to_greenlight"] is False
    assert audit["initial_fruit_semantics"] == "observed_zero_not_imputed"
    assert audit["canopy_thermal_sum_semantics"] == (
        "unknown_prebaseline_accumulation_initialized_zero_not_validated"
    )


def test_build_target_crop_initial_state_rejects_non_standing_target_rows():
    observations = _observations()
    observations.loc[observations["greenhouse_id"].eq(63), "observation_type"] = "harvest_event"

    with pytest.raises(ValueError, match="standing-crop baseline"):
        build_target_crop_initial_state(
            observations,
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            baseline_date="2026-03-30",
        )


def test_build_target_crop_initial_state_requires_positive_source_pairs():
    observations = _observations()
    observations.loc[observations["greenhouse_id"].eq(1), "leaf_fresh_kg_m2"] = 0.0

    with pytest.raises(ValueError, match="leaf dry-matter fraction"):
        build_target_crop_initial_state(
            observations,
            target_greenhouse_id=63,
            target_greenhouse_code="GH-PIDU",
            baseline_date="2026-03-30",
        )
