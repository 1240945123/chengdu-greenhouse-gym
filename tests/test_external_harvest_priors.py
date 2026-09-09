from __future__ import annotations

import pandas as pd
import pytest

from experiments.crop.estimate_external_harvest_priors import (
    estimate_external_harvest_priors,
)


def _observations() -> pd.DataFrame:
    rows = []
    for team, offset, batches in (
        ("A", 60, [0.4, 0.6, 0.8]),
        ("B", 65, [0.5, 0.7, 0.9]),
        ("C", 70, [0.6, 0.8, 1.0]),
    ):
        for index, batch in enumerate(batches):
            rows.append(
                {
                    "team": team,
                    "harvest_date": pd.Timestamp("2019-12-16")
                    + pd.Timedelta(days=offset + index * 5),
                    "batch_fresh_kg_m2": batch,
                    "evidence_class": "external_observed",
                    "target_eligible": False,
                    "source_doi": "10.4121/example",
                }
            )
    return pd.DataFrame(rows)


def test_estimate_external_harvest_priors_uses_compartment_level_statistics():
    result = estimate_external_harvest_priors(
        _observations(), bootstrap_samples=500, bootstrap_seed=17
    )

    assert result["compartment_count"] == 3
    assert result["event_count"] == 9
    assert result["metrics"]["first_harvest_days_from_crop_start"]["estimate"] == 65
    assert result["metrics"]["median_interpick_days"]["estimate"] == 5
    assert result["metrics"]["seasonal_yield_kg_m2"]["estimate"] == pytest.approx(2.1)
    assert result["bootstrap_unit"] == "greenhouse_compartment"


def test_estimate_external_harvest_priors_is_deterministic_for_fixed_seed():
    first = estimate_external_harvest_priors(
        _observations(), bootstrap_samples=200, bootstrap_seed=8
    )
    second = estimate_external_harvest_priors(
        _observations(), bootstrap_samples=200, bootstrap_seed=8
    )

    assert first == second


def test_estimate_external_harvest_priors_rejects_fewer_than_three_compartments():
    observations = _observations().query("team != 'C'")

    with pytest.raises(ValueError, match="three"):
        estimate_external_harvest_priors(observations)


def test_estimate_external_harvest_priors_rejects_non_external_evidence():
    observations = _observations()
    observations.loc[0, "evidence_class"] = "target_observed"

    with pytest.raises(ValueError, match="external_observed"):
        estimate_external_harvest_priors(observations)
