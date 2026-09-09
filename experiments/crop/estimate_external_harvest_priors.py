from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "team",
    "harvest_date",
    "batch_fresh_kg_m2",
    "evidence_class",
    "target_eligible",
    "source_doi",
}


def _strict_false(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return bool((~series).all())
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("target_eligible must contain strict booleans")
    return bool(normalized.eq("false").all())


def _metric_summary(values: np.ndarray, bootstrap: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(np.median(values)),
        "observed_min": float(np.min(values)),
        "observed_max": float(np.max(values)),
        "ci80_low": float(np.quantile(bootstrap, 0.10)),
        "ci80_high": float(np.quantile(bootstrap, 0.90)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
    }


def estimate_external_harvest_priors(
    observations: pd.DataFrame,
    *,
    crop_start: str | pd.Timestamp = "2019-12-16",
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260729,
) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(observations.columns))
    if missing:
        raise ValueError(f"external harvest observations missing columns: {missing}")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    frame = observations.copy()
    if not frame["evidence_class"].astype(str).eq("external_observed").all():
        raise ValueError("all prior rows must have evidence_class=external_observed")
    if not _strict_false(frame["target_eligible"]):
        raise ValueError("external prior rows must have target_eligible=false")
    frame["harvest_date"] = pd.to_datetime(frame["harvest_date"], errors="coerce")
    frame["batch_fresh_kg_m2"] = pd.to_numeric(
        frame["batch_fresh_kg_m2"], errors="coerce"
    )
    if frame[["harvest_date", "batch_fresh_kg_m2"]].isna().any().any():
        raise ValueError("external harvest dates and batch masses must be valid")
    if (frame["batch_fresh_kg_m2"] < 0).any():
        raise ValueError("external batch masses must be non-negative")
    compartment_count = int(frame["team"].nunique())
    if compartment_count < 3:
        raise ValueError("at least three greenhouse compartments are required")
    start = pd.Timestamp(crop_start)
    summaries: list[dict[str, float | str]] = []
    for team, group in frame.groupby("team", sort=True):
        ordered = group.sort_values("harvest_date")
        intervals = ordered["harvest_date"].diff().dropna().dt.total_seconds() / 86400.0
        if intervals.empty or (intervals <= 0).any():
            raise ValueError(f"team {team} lacks valid chronological harvest intervals")
        summaries.append(
            {
                "team": str(team),
                "first_harvest_days_from_crop_start": float(
                    (ordered["harvest_date"].iloc[0] - start).total_seconds() / 86400.0
                ),
                "median_interpick_days": float(intervals.median()),
                "seasonal_yield_kg_m2": float(ordered["batch_fresh_kg_m2"].sum()),
                "mean_batch_kg_m2": float(ordered["batch_fresh_kg_m2"].mean()),
                "event_count": int(len(ordered)),
            }
        )
    summary_frame = pd.DataFrame(summaries)
    metric_names = (
        "first_harvest_days_from_crop_start",
        "median_interpick_days",
        "seasonal_yield_kg_m2",
        "mean_batch_kg_m2",
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    sampled_indices = rng.integers(
        0,
        compartment_count,
        size=(int(bootstrap_samples), compartment_count),
    )
    metrics: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = summary_frame[metric].to_numpy(dtype=float)
        bootstrap = np.median(values[sampled_indices], axis=1)
        metrics[metric] = _metric_summary(values, bootstrap)
    return {
        "source_dataset": "wur_agc2_v2",
        "source_dois": sorted(frame["source_doi"].astype(str).unique().tolist()),
        "evidence_class": "external_observed",
        "target_eligible": False,
        "transfer_role": "external_prior_and_pipeline_validation_only",
        "crop_start": start.strftime("%Y-%m-%d"),
        "compartment_count": compartment_count,
        "event_count": int(len(frame)),
        "bootstrap_unit": "greenhouse_compartment",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "compartment_summaries": summaries,
        "metrics": metrics,
        "limitations": [
            "different_country_and_outdoor_climate",
            "high_tech_glasshouse_not_pidu_plastic_greenhouse",
            "cherry_tomato_cultivar_axiany_not_target_cultivar",
            "single_external_crop_cycle",
            "not_target_calibration_or_validation_evidence",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate transferable harvest priors from WUR AGC2 observations."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    args = parser.parse_args()
    observations = pd.read_csv(args.input_csv)
    result = estimate_external_harvest_priors(
        observations,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
