from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _weather_features(frame: pd.DataFrame, *, base_temperature_c: float) -> dict[str, Any]:
    required = {"timestamp", "air_temperature", "global_radiation"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"weather season missing columns: {missing}")
    weather = frame[list(required)].copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="coerce")
    weather["air_temperature"] = pd.to_numeric(
        weather["air_temperature"], errors="coerce"
    )
    weather["global_radiation"] = pd.to_numeric(
        weather["global_radiation"], errors="coerce"
    )
    if weather.isna().any().any() or len(weather) < 48:
        raise ValueError("weather season must contain at least 48 complete hourly rows")
    weather = weather.sort_values("timestamp", kind="stable")
    if weather["timestamp"].duplicated().any():
        raise ValueError("weather season timestamps must be unique")
    intervals = weather["timestamp"].diff().dropna()
    if not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("weather season must form a continuous hourly grid")
    if (weather["global_radiation"] < 0).any():
        raise ValueError("global radiation must be non-negative")
    return {
        "start": weather["timestamp"].iloc[0],
        "end": weather["timestamp"].iloc[-1],
        "duration_days": int(len(weather) // 24),
        "gdd": float(
            np.maximum(
                weather["air_temperature"].to_numpy(float) - base_temperature_c,
                0.0,
            ).sum()
            / 24.0
        ),
        "radiation_mj_m2": float(
            weather["global_radiation"].to_numpy(float).sum() * 3600.0 / 1e6
        ),
    }


def simulate_weather_conditioned_harvest_priors(
    external_observations: pd.DataFrame,
    weather_seasons: Mapping[str, pd.DataFrame],
    *,
    external_crop_start: str | pd.Timestamp = "2019-12-16",
    draws_per_season: int = 20,
    seed: int = 20260729,
    base_temperature_c: float = 10.0,
    yield_noise_sigma: float = 0.08,
    dry_matter_fraction_bounds: tuple[float, float] = (
        0.07582388370007698,
        0.08461793001033059,
    ),
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required_external = {
        "team",
        "harvest_date",
        "batch_fresh_kg_m2",
        "evidence_class",
        "target_eligible",
    }
    missing = sorted(required_external - set(external_observations.columns))
    if missing:
        raise ValueError(f"external observations missing columns: {missing}")
    if len(weather_seasons) < 6:
        raise ValueError("at least six Chengdu weather seasons are required")
    if draws_per_season <= 0:
        raise ValueError("draws_per_season must be positive")
    low_dmf, high_dmf = map(float, dry_matter_fraction_bounds)
    if not 0.0 < low_dmf <= high_dmf < 1.0:
        raise ValueError("dry matter fraction bounds must lie within (0, 1)")
    external = external_observations.copy()
    if not external["evidence_class"].astype(str).eq("external_observed").all():
        raise ValueError("source curves must be external_observed")
    eligible = external["target_eligible"]
    if pd.api.types.is_bool_dtype(eligible):
        any_eligible = bool(eligible.any())
    else:
        normalized = eligible.astype(str).str.strip().str.lower()
        if not normalized.isin({"true", "false"}).all():
            raise ValueError("external target_eligible values must be strict booleans")
        any_eligible = bool(normalized.eq("true").any())
    if any_eligible:
        raise ValueError("external source curves must be target ineligible")
    external["harvest_date"] = pd.to_datetime(
        external["harvest_date"], errors="coerce"
    )
    external["batch_fresh_kg_m2"] = pd.to_numeric(
        external["batch_fresh_kg_m2"], errors="coerce"
    )
    if external[["harvest_date", "batch_fresh_kg_m2"]].isna().any().any():
        raise ValueError("external harvest dates and masses must be complete")
    if (external["batch_fresh_kg_m2"] < 0).any():
        raise ValueError("external batch masses must be non-negative")
    crop_start = pd.Timestamp(external_crop_start)
    external["crop_day"] = (
        external["harvest_date"] - crop_start
    ).dt.total_seconds() / 86400.0
    if (external["crop_day"] < 0).any():
        raise ValueError("external harvest predates declared crop start")
    teams = sorted(external["team"].astype(str).unique().tolist())
    if len(teams) < 3:
        raise ValueError("at least three external compartments are required")

    weather_features = {
        str(season_id): _weather_features(
            frame, base_temperature_c=float(base_temperature_c)
        )
        for season_id, frame in sorted(weather_seasons.items())
    }
    gdd_reference = float(np.median([item["gdd"] for item in weather_features.values()]))
    radiation_reference = float(
        np.median([item["radiation_mj_m2"] for item in weather_features.values()])
    )
    if gdd_reference <= 0.0 or radiation_reference <= 0.0:
        raise ValueError("weather ensemble must have positive GDD and radiation")

    rng = np.random.default_rng(int(seed))
    event_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for season_index, (season_id, features) in enumerate(weather_features.items()):
        timing_scale = float(np.clip(gdd_reference / features["gdd"], 0.70, 1.30))
        radiation_scale = float(
            np.clip((features["radiation_mj_m2"] / radiation_reference) ** 0.35, 0.70, 1.30)
        )
        for draw in range(int(draws_per_season)):
            team = teams[int(rng.integers(0, len(teams)))]
            yield_noise = float(
                rng.lognormal(mean=-0.5 * yield_noise_sigma**2, sigma=yield_noise_sigma)
            )
            dry_fraction = float(rng.uniform(low_dmf, high_dmf))
            scenario_id = f"SIM_{season_id}_{draw:03d}"
            template = external.loc[external["team"].astype(str).eq(team)].copy()
            template["mapped_crop_day"] = np.rint(
                template["crop_day"] * timing_scale
            ).astype(int)
            template = template.loc[
                template["mapped_crop_day"].between(0, features["duration_days"] - 1)
            ].copy()
            template["batch_fresh_kg_m2"] *= radiation_scale * yield_noise
            template = (
                template.groupby("mapped_crop_day", as_index=False)["batch_fresh_kg_m2"]
                .sum()
                .sort_values("mapped_crop_day")
            )
            template["harvest_datetime"] = pd.Timestamp(features["start"]).normalize() + pd.to_timedelta(
                template["mapped_crop_day"], unit="D"
            ) + pd.Timedelta(hours=8)
            template["cumulative_fresh_kg_m2"] = template["batch_fresh_kg_m2"].cumsum()
            template["batch_dry_kg_m2"] = template["batch_fresh_kg_m2"] * dry_fraction
            template["synthetic_event_id"] = [
                f"{scenario_id}_{index:03d}" for index in range(len(template))
            ]
            template["scenario_id"] = scenario_id
            template["season_id"] = season_id
            template["synthetic_greenhouse_code"] = f"SIM_CHENGDU_{season_index:02d}"
            template["synthetic_planting_code"] = scenario_id
            template["source_template_team"] = team
            template["evidence_class"] = "simulated_prior"
            template["target_eligible"] = False
            template["dry_matter_fraction"] = dry_fraction
            template["timing_scale"] = timing_scale
            template["radiation_scale"] = radiation_scale
            template["yield_noise_multiplier"] = yield_noise
            event_frames.append(template)
            summaries.append(
                {
                    "scenario_id": scenario_id,
                    "season_id": season_id,
                    "source_template_team": team,
                    "event_count": int(len(template)),
                    "first_harvest_datetime": (
                        template["harvest_datetime"].iloc[0].isoformat()
                        if len(template) else None
                    ),
                    "partial_yield_kg_m2": float(template["batch_fresh_kg_m2"].sum()),
                    "dry_matter_fraction": dry_fraction,
                    "timing_scale": timing_scale,
                    "radiation_scale": radiation_scale,
                    "yield_noise_multiplier": yield_noise,
                    "weather_gdd": float(features["gdd"]),
                    "weather_radiation_mj_m2": float(features["radiation_mj_m2"]),
                    "evidence_class": "simulated_prior",
                    "target_eligible": False,
                }
            )
    events = pd.concat(event_frames, ignore_index=True)
    events = events[
        [
            "synthetic_event_id",
            "scenario_id",
            "season_id",
            "synthetic_greenhouse_code",
            "synthetic_planting_code",
            "source_template_team",
            "harvest_datetime",
            "batch_fresh_kg_m2",
            "cumulative_fresh_kg_m2",
            "batch_dry_kg_m2",
            "dry_matter_fraction",
            "timing_scale",
            "radiation_scale",
            "yield_noise_multiplier",
            "evidence_class",
            "target_eligible",
        ]
    ].sort_values(["season_id", "scenario_id", "harvest_datetime"], kind="stable").reset_index(drop=True)
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["season_id", "scenario_id"], kind="stable"
    ).reset_index(drop=True)
    audit = {
        "method": "external_curve_weather_conditioned_bootstrap",
        "seed": int(seed),
        "draws_per_season": int(draws_per_season),
        "season_count": len(weather_features),
        "scenario_count": int(len(summary_frame)),
        "event_count": int(len(events)),
        "external_compartment_count": len(teams),
        "gdd_base_temperature_c": float(base_temperature_c),
        "gdd_reference": gdd_reference,
        "radiation_reference_mj_m2": radiation_reference,
        "yield_noise_sigma": float(yield_noise_sigma),
        "dry_matter_fraction_bounds": [low_dmf, high_dmf],
        "evidence_class": "simulated_prior",
        "target_eligible": False,
        "weather_features": {
            name: {
                "start": value["start"].isoformat(),
                "end": value["end"].isoformat(),
                "duration_days": value["duration_days"],
                "gdd": value["gdd"],
                "radiation_mj_m2": value["radiation_mj_m2"],
            }
            for name, value in weather_features.items()
        },
        "limitations": [
            "statistical_prior_not_greenlight_state_simulation",
            "wur_curve_transfer_from_dutch_high_tech_glasshouse",
            "outdoor_weather_scaling_not_indoor_climate_validation",
            "partial_120_day_seasons",
            "not_target_calibration_or_validation_evidence",
        ],
    }
    return events, summary_frame, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate target-ineligible Chengdu tomato harvest prior scenarios."
    )
    parser.add_argument("--external-harvest-csv", required=True)
    parser.add_argument("--weather-dir", required=True)
    parser.add_argument("--output-events", required=True)
    parser.add_argument("--output-summaries", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--draws-per-season", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    weather_paths = sorted(Path(args.weather_dir).glob("20??_*.csv"))
    weather = {path.stem: pd.read_csv(path) for path in weather_paths}
    events, summaries, audit = simulate_weather_conditioned_harvest_priors(
        pd.read_csv(args.external_harvest_csv),
        weather,
        draws_per_season=args.draws_per_season,
        seed=args.seed,
    )
    event_path = Path(args.output_events)
    summary_path = Path(args.output_summaries)
    audit_path = Path(args.output_audit)
    for path in (event_path, summary_path, audit_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(event_path, index=False)
    summaries.to_csv(summary_path, index=False)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
