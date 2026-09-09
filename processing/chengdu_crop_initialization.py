from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


_ORGANS = ("leaf", "stem", "root", "ripe_fruit")
_STATE_ORGANS = {"cLeaf": "leaf", "cStem": "stem", "cFruit": "ripe_fruit"}


def build_target_crop_initial_state(
    observations: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    baseline_date: str | pd.Timestamp,
    timezone: str = "Asia/Shanghai",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Estimate GreenLight crop mass states from a dated target plant sample."""
    required = {
        "observation_type",
        "observation_date",
        "greenhouse_id",
        "greenhouse_code",
        "sample_plant_id",
        "source_site",
        "target_eligible",
    }
    for organ in _ORGANS:
        required.update({f"{organ}_fresh_kg_m2", f"{organ}_dry_kg_m2"})
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"crop observations missing columns: {', '.join(missing)}")

    frame = observations.copy()
    frame["observation_date"] = frame["observation_date"].map(
        lambda value: _target_timestamp(value, timezone)
    )
    if frame["observation_date"].isna().any():
        raise ValueError("crop observation dates must be valid")

    target_id = pd.to_numeric(frame["greenhouse_id"], errors="coerce")
    identity = target_id.eq(int(target_greenhouse_id)) & frame[
        "greenhouse_code"
    ].astype(str).eq(str(target_greenhouse_code))
    target_rows = frame.loc[identity & frame["target_eligible"].eq(True)].copy()
    baseline = _target_timestamp(baseline_date, timezone).normalize()
    baseline_rows = target_rows.loc[
        target_rows["observation_date"].dt.normalize().eq(baseline)
        & target_rows["observation_type"].eq("standing_crop_sample")
    ].copy()
    if baseline_rows.empty:
        raise ValueError("target standing-crop baseline is missing for the requested date")
    if baseline_rows["sample_plant_id"].astype(str).duplicated().any():
        raise ValueError("target standing-crop baseline sample IDs must be unique")

    source_rows = frame.loc[~frame["target_eligible"].eq(True)].copy()
    fractions = {
        organ: _estimate_dry_matter_fraction(source_rows, organ) for organ in _ORGANS
    }

    baseline_statistics: dict[str, dict[str, float]] = {}
    organ_dry_kg_m2: dict[str, float] = {}
    for organ in _ORGANS:
        column = f"{organ}_fresh_kg_m2"
        values = pd.to_numeric(baseline_rows[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"target baseline {organ} fresh mass must be finite and non-negative")
        baseline_statistics[organ] = _sample_statistics(values)
        organ_dry_kg_m2[organ] = float(values.mean() * fractions[organ]["estimate"])

    state = {"cBuf": 0.0, "tCanSum": 0.0}
    for state_name, organ in _STATE_ORGANS.items():
        state[state_name] = organ_dry_kg_m2[organ] * 1e6

    fruit_values = pd.to_numeric(
        baseline_rows["ripe_fruit_fresh_kg_m2"], errors="coerce"
    ).to_numpy(dtype=float)
    audit: dict[str, Any] = {
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "baseline_date": baseline.isoformat(),
        "baseline_sample_count": int(len(baseline_rows)),
        "baseline_fresh_mass_statistics_kg_m2": baseline_statistics,
        "dry_matter_fraction_transfer": fractions,
        "greenlight_initial_crop_state_mg_m2": state,
        "root_estimated_dry_matter_kg_m2": organ_dry_kg_m2["root"],
        "root_state_applied_to_greenlight": False,
        "buffer_state_source": "unobserved_initialized_zero",
        "canopy_thermal_sum_semantics": (
            "unknown_prebaseline_accumulation_initialized_zero_not_validated"
        ),
        "initial_fruit_semantics": (
            "observed_zero_not_imputed"
            if np.allclose(fruit_values, 0.0)
            else "fresh_mass_converted_with_source_site_fraction"
        ),
        "status": "target_state_initialized_with_source_site_dry_fraction_transfer",
    }
    return state, audit


def _estimate_dry_matter_fraction(
    source_rows: pd.DataFrame, organ: str
) -> dict[str, float | int | str]:
    fresh = pd.to_numeric(source_rows[f"{organ}_fresh_kg_m2"], errors="coerce")
    dry = pd.to_numeric(source_rows[f"{organ}_dry_kg_m2"], errors="coerce")
    valid = fresh.gt(0.0) & dry.ge(0.0) & np.isfinite(fresh) & np.isfinite(dry)
    ratios = (dry.loc[valid] / fresh.loc[valid]).to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0) & (ratios < 1.0)]
    if len(ratios) < 2:
        raise ValueError(f"at least two positive paired samples are required for {organ} dry-matter fraction")
    return {
        "estimate": float(np.median(ratios)),
        "pair_count": int(len(ratios)),
        "minimum": float(np.min(ratios)),
        "maximum": float(np.max(ratios)),
        "method": "source_site_paired_sample_median",
        "transfer_scope": "xindu_to_pidu_not_target_calibration",
    }


def _sample_statistics(values: np.ndarray) -> dict[str, float]:
    count = len(values)
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if count > 1 else float("nan")
    standard_error = standard_deviation / np.sqrt(count) if count > 1 else float("nan")
    half_width = (
        float(student_t.ppf(0.975, df=count - 1) * standard_error)
        if count > 1
        else float("nan")
    )
    return {
        "count": int(count),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def _target_timestamp(value: object, timezone: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(timezone)
    return timestamp.tz_convert(timezone)
