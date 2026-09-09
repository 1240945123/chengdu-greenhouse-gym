from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXCEL_ORIGIN = pd.Timestamp("1899-12-30")
DEFAULT_SOURCE_DOI = "10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7"


def excel_serial_to_timestamp(value: float | int | str) -> pd.Timestamp:
    serial = float(value)
    if not np.isfinite(serial):
        raise ValueError("Excel serial timestamp must be finite")
    return EXCEL_ORIGIN + pd.to_timedelta(serial, unit="D")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _time_column(frame: pd.DataFrame) -> str:
    for candidate in ("%time", "%Time", "time", "Time"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("WUR file has no registered Excel serial time column")


def _load_climate(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if "Tair" not in frame.columns:
        raise ValueError(f"WUR climate file has no Tair column: {path}")
    time_column = _time_column(frame)
    result = pd.DataFrame(
        {
            "timestamp": frame[time_column].map(excel_serial_to_timestamp),
            "temperature_c": pd.to_numeric(frame["Tair"], errors="coerce"),
        }
    ).dropna()
    result = result.sort_values("timestamp").drop_duplicates("timestamp")
    if len(result) < 2:
        raise ValueError(f"WUR climate file has insufficient valid rows: {path}")
    return result.reset_index(drop=True)


def _integrate_degree_days(
    climate: pd.DataFrame,
    *,
    harvest_timestamp: pd.Timestamp,
    development_days: float,
    base_temperature_c: float,
) -> float:
    duration = float(development_days)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("Development duration must be positive and finite")
    end = pd.Timestamp(harvest_timestamp)
    start = end - pd.Timedelta(days=duration)
    times = climate["timestamp"]
    temperatures = climate["temperature_c"].to_numpy(dtype=float)
    if start < times.iloc[0] or end > times.iloc[-1]:
        raise ValueError("Climate data do not cover the reported development interval")

    interpolation_times = pd.DatetimeIndex([start, end])
    indexed = pd.Series(temperatures, index=pd.DatetimeIndex(times))
    indexed = indexed.reindex(indexed.index.union(interpolation_times)).sort_index()
    indexed = indexed.interpolate(method="time").loc[start:end]
    values = np.maximum(indexed.to_numpy(dtype=float) - float(base_temperature_c), 0.0)
    elapsed_days = np.diff(indexed.index.to_numpy()) / np.timedelta64(1, "D")
    elapsed_days = np.asarray(elapsed_days, dtype=float)
    if not len(elapsed_days):
        raise ValueError("Development interval contains no climate duration")
    return float(np.sum((values[:-1] + values[1:]) * 0.5 * elapsed_days))


def _load_dry_matter_fractions(path: Path) -> np.ndarray:
    frame = _read_csv(path)
    if "DMC_fruit" not in frame.columns:
        raw = path.read_text(encoding="utf-8-sig")
        header, separator, body = raw.partition("\n")
        if "Weight\tDMC_fruit" not in header:
            raise ValueError(f"WUR quality file has no DMC_fruit column: {path}")
        repaired = header.replace("Weight\tDMC_fruit", "Weight,DMC_fruit")
        frame = pd.read_csv(io.StringIO(repaired + separator + body), skipinitialspace=True)
        frame.columns = [str(column).strip() for column in frame.columns]
    values = pd.to_numeric(frame["DMC_fruit"], errors="coerce").dropna().to_numpy(dtype=float)
    values = values[(values > 0.0) & (values <= 100.0)] / 100.0
    return values


def _metric_summary(
    values_by_compartment: dict[str, float],
    *,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = np.asarray(list(values_by_compartment.values()), dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Process-prior metric has no finite compartment estimates")
    indices = rng.integers(0, len(values), size=(int(bootstrap_samples), len(values)))
    means = values[indices].mean(axis=1)
    q025, q10, q90, q975 = np.quantile(means, [0.025, 0.10, 0.90, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci80_low": float(q10),
        "ci80_high": float(q90),
        "ci95_low": float(q025),
        "ci95_high": float(q975),
        "observed_min": float(values.min()),
        "observed_max": float(values.max()),
    }


def estimate_wur_process_priors(
    *,
    root: str | Path,
    observations_path: str | Path,
    base_temperature_c: float = 10.0,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260817,
    minimum_compartments: int = 5,
    minimum_maturity_events: int = 50,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    observations = pd.read_csv(observations_path, skipinitialspace=True)
    required = {"team", "harvest_date", "Truss development time", "batch_fresh_kg_m2"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"External observations missing columns: {', '.join(missing)}")

    source_root = Path(root)
    summaries: list[dict[str, Any]] = []
    maturity_by_team: dict[str, float] = {}
    dry_matter_by_team: dict[str, float] = {}
    batch_by_team: dict[str, float] = {}
    pick_interval_by_team: dict[str, float] = {}
    maturity_event_count = 0
    dry_matter_observation_count = 0

    for team, group in observations.groupby("team", sort=True):
        team_name = str(team).strip()
        folder = source_root / team_name
        if not folder.is_dir():
            continue
        climate = _load_climate(folder / "GreenhouseClimate.csv")
        # Iterating with iterrows is deliberate here because source column names
        # contain spaces and punctuation that are not stable namedtuple fields.
        maturity_values: list[float] = []
        for _, row in group.iterrows():
            duration = pd.to_numeric(row["Truss development time"], errors="coerce")
            if not np.isfinite(duration):
                continue
            try:
                maturity_values.append(
                    _integrate_degree_days(
                        climate,
                        harvest_timestamp=pd.Timestamp(row["harvest_date"]),
                        development_days=float(duration),
                        base_temperature_c=base_temperature_c,
                    )
                )
            except ValueError:
                continue

        dry_matter = _load_dry_matter_fractions(folder / "TomQuality.csv")
        harvest_dates = pd.to_datetime(group["harvest_date"], errors="coerce").dropna().sort_values()
        intervals = harvest_dates.diff().dt.total_seconds().dropna().to_numpy(dtype=float) / 86_400.0
        batches = pd.to_numeric(group["batch_fresh_kg_m2"], errors="coerce").dropna().to_numpy(dtype=float)
        batches = batches[batches >= 0.0]
        if maturity_values:
            maturity_by_team[team_name] = float(np.median(maturity_values))
        if len(dry_matter):
            dry_matter_by_team[team_name] = float(np.median(dry_matter))
        if len(batches):
            batch_by_team[team_name] = float(np.mean(batches))
        if len(intervals):
            pick_interval_by_team[team_name] = float(np.median(intervals))
        maturity_event_count += len(maturity_values)
        dry_matter_observation_count += len(dry_matter)
        summaries.append(
            {
                "team": team_name,
                "maturity_event_count": len(maturity_values),
                "maturity_thermal_time_deg_day_median": (
                    float(np.median(maturity_values)) if maturity_values else None
                ),
                "fruit_dry_matter_observation_count": int(len(dry_matter)),
                "fruit_dry_matter_fraction_median": (
                    float(np.median(dry_matter)) if len(dry_matter) else None
                ),
                "mean_batch_kg_m2": float(np.mean(batches)) if len(batches) else None,
                "median_interpick_days": float(np.median(intervals)) if len(intervals) else None,
            }
        )

    compartment_count = len(maturity_by_team)
    if compartment_count < int(minimum_compartments):
        raise ValueError(
            f"Only {compartment_count} compartments have maturity evidence; "
            f"at least {minimum_compartments} are required"
        )
    if maturity_event_count < int(minimum_maturity_events):
        raise ValueError(
            f"Only {maturity_event_count} maturity events are valid; "
            f"at least {minimum_maturity_events} are required"
        )

    rng = np.random.default_rng(bootstrap_seed)
    metrics = {
        "maturity_thermal_time_deg_day": _metric_summary(
            maturity_by_team, bootstrap_samples=bootstrap_samples, rng=rng
        ),
        "fruit_dry_matter_fraction": _metric_summary(
            dry_matter_by_team, bootstrap_samples=bootstrap_samples, rng=rng
        ),
        "mean_batch_kg_m2": _metric_summary(
            batch_by_team, bootstrap_samples=bootstrap_samples, rng=rng
        ),
        "median_interpick_days": _metric_summary(
            pick_interval_by_team, bootstrap_samples=bootstrap_samples, rng=rng
        ),
    }
    source_dois = sorted(
        set(observations.get("source_doi", pd.Series([DEFAULT_SOURCE_DOI])).dropna().astype(str))
    )
    return {
        "method": "reported_truss_duration_temperature_integral",
        "evidence_class": "external_observed",
        "target_eligible": False,
        "transfer_role": "external_process_prior_not_target_calibration",
        "source_dataset": "wur_agc2_v2",
        "source_dois": source_dois or [DEFAULT_SOURCE_DOI],
        "cultivar": "Axiany",
        "facility": "WUR high-tech glasshouse compartments, Bleiswijk",
        "base_temperature_c": float(base_temperature_c),
        "bootstrap_unit": "greenhouse_compartment",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "compartment_count": int(compartment_count),
        "maturity_event_count": int(maturity_event_count),
        "dry_matter_observation_count": int(dry_matter_observation_count),
        "compartment_summaries": summaries,
        "metrics": metrics,
        "limitations": [
            "external_country_climate_and_greenhouse",
            "cherry_tomato_cultivar_not_pidu_target_cultivar",
            "single_external_crop_cycle",
            "reported_truss_duration_not_direct_cohort_age_observation",
            "not_target_calibration_or_validation_evidence",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate WUR tomato harvest-process priors")
    parser.add_argument("--root", required=True)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-temperature-c", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = estimate_wur_process_priors(
        root=args.root,
        observations_path=args.observations,
        base_temperature_c=args.base_temperature_c,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
