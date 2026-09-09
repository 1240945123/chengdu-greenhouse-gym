from __future__ import annotations

import pandas as pd

from experiments.crop.simulate_chengdu_harvest_priors import (
    simulate_weather_conditioned_harvest_priors,
)


def _external_observations() -> pd.DataFrame:
    rows = []
    for team, scale in (("A", 1.0), ("B", 1.2), ("C", 0.8)):
        for day, batch in ((2, 0.4), (5, 0.6), (8, 0.8)):
            rows.append(
                {
                    "team": team,
                    "harvest_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
                    "batch_fresh_kg_m2": batch * scale,
                    "evidence_class": "external_observed",
                    "target_eligible": False,
                }
            )
    return pd.DataFrame(rows)


def _weather(season_index: int) -> pd.DataFrame:
    timestamps = pd.date_range(
        f"202{season_index}-03-01", periods=10 * 24, freq="1h"
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "air_temperature": 20.0 + season_index,
            "global_radiation": [300.0 if 7 <= value.hour <= 18 else 0.0 for value in timestamps],
        }
    )


def test_simulation_covers_six_seasons_and_is_target_ineligible():
    weather = {f"season_{index}": _weather(index) for index in range(1, 7)}

    events, summaries, audit = simulate_weather_conditioned_harvest_priors(
        _external_observations(),
        weather,
        external_crop_start="2020-01-01",
        draws_per_season=2,
        seed=42,
    )

    assert events["season_id"].nunique() == 6
    assert summaries["scenario_id"].nunique() == 12
    assert events["scenario_id"].nunique() == 12
    assert events["evidence_class"].eq("simulated_prior").all()
    assert events["target_eligible"].eq(False).all()
    assert events["synthetic_greenhouse_code"].str.startswith("SIM_").all()
    assert audit["season_count"] == 6
    assert audit["method"] == "external_curve_weather_conditioned_bootstrap"


def test_simulation_is_deterministic_for_fixed_seed():
    weather = {f"season_{index}": _weather(index) for index in range(1, 7)}

    first = simulate_weather_conditioned_harvest_priors(
        _external_observations(), weather, external_crop_start="2020-01-01", seed=7
    )
    second = simulate_weather_conditioned_harvest_priors(
        _external_observations(), weather, external_crop_start="2020-01-01", seed=7
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]


def test_simulation_records_fresh_to_dry_conversion_and_cumulative_mass():
    weather = {f"season_{index}": _weather(index) for index in range(1, 7)}

    events, summaries, _ = simulate_weather_conditioned_harvest_priors(
        _external_observations(),
        weather,
        external_crop_start="2020-01-01",
        draws_per_season=1,
        seed=3,
        dry_matter_fraction_bounds=(0.08, 0.08),
    )

    assert (events["batch_dry_kg_m2"] == events["batch_fresh_kg_m2"] * 0.08).all()
    final_events = events.groupby("scenario_id")["cumulative_fresh_kg_m2"].last()
    final_summaries = summaries.set_index("scenario_id")["partial_yield_kg_m2"]
    pd.testing.assert_series_equal(final_events, final_summaries, check_names=False)
