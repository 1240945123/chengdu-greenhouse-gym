from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common.harvest_evaluation import (
    batch_prediction_intervals,
    evaluate_harvest_predictions,
    percentile_prediction_intervals,
)
from glassgym.models.harvest import HarvestCohortModel, HarvestCohortParameters


DRIVER_UNCERTAINTY_SOURCES = {
    "greenhouse_climate_model_parameter_uncertainty",
    "outdoor_weather_scenario_uncertainty",
}


def build_harvest_drivers_from_greenlight(
    trajectory: pd.DataFrame,
    *,
    dt_seconds: float,
    driver_model_status: str,
    initial_fruit_maturity_fraction: float = 0.0,
) -> pd.DataFrame:
    """Convert GreenLight fruit-state accounting into cohort-model drivers."""
    required = {
        "season_id",
        "timestamp",
        "c_fruit_previous_mg_m2",
        "c_fruit_mg_m2",
        "harvested_dry_matter_mg_m2",
        "air_temperature",
        "pick",
        "pick_source",
        "season_complete",
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
    }
    missing = sorted(required - set(trajectory.columns))
    if missing:
        raise ValueError(f"GreenLight trajectory missing columns: {', '.join(missing)}")
    if not np.isfinite(float(dt_seconds)) or float(dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    if not 0.0 <= float(initial_fruit_maturity_fraction) <= 1.0:
        raise ValueError("initial_fruit_maturity_fraction must be in [0, 1]")
    if not str(driver_model_status):
        raise ValueError("driver_model_status must be non-empty")
    frame = trajectory.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame.empty or frame["timestamp"].isna().any():
        raise ValueError("GreenLight trajectory timestamps must be valid and non-empty")
    numeric_columns = (
        "c_fruit_previous_mg_m2",
        "c_fruit_mg_m2",
        "harvested_dry_matter_mg_m2",
        "air_temperature",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not np.isfinite(frame[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise ValueError("GreenLight fruit states, harvest, and temperature must be finite")
    if (frame["harvested_dry_matter_mg_m2"] < 0.0).any():
        raise ValueError("GreenLight native harvested dry matter must be non-negative")
    for _, season in frame.groupby("season_id", sort=False):
        previous = season["c_fruit_previous_mg_m2"].to_numpy(dtype=float)[1:]
        prior_current = season["c_fruit_mg_m2"].to_numpy(dtype=float)[:-1]
        if len(previous) and not np.allclose(
            previous,
            prior_current,
            rtol=1e-8,
            atol=1e-6,
        ):
            raise ValueError("GreenLight fruit state is discontinuous within a season")
    net_change_mg = (
        frame["c_fruit_mg_m2"]
        - frame["c_fruit_previous_mg_m2"]
        + frame["harvested_dry_matter_mg_m2"]
    )
    first_in_season = ~frame["season_id"].astype(str).duplicated()
    initial_fruit_dm = np.where(
        first_in_season,
        frame["c_fruit_previous_mg_m2"].to_numpy(dtype=float) * 1e-6,
        0.0,
    )
    return pd.DataFrame(
        {
            "season_id": frame["season_id"].astype(str),
            "timestamp": frame["timestamp"],
            "net_fruit_dry_matter_change_kg_m2": net_change_mg * 1e-6,
            "initial_fruit_dry_matter_kg_m2": initial_fruit_dm,
            "initial_fruit_maturity_fraction": np.where(
                first_in_season, float(initial_fruit_maturity_fraction), 0.0
            ),
            "air_temperature_c": frame["air_temperature"],
            "dt_hours": float(dt_seconds) / 3600.0,
            "pick": frame["pick"],
            "pick_source": frame["pick_source"],
            "season_complete": frame["season_complete"],
            "fruit_change_source": "greenlight_fruit_state_balance",
            "air_temperature_source": "greenlight_simulated_indoor",
            "driver_model_status": str(driver_model_status),
            "greenhouse_id": frame["greenhouse_id"],
            "greenhouse_code": frame["greenhouse_code"].astype(str),
            "planting_code": frame["planting_code"].astype(str),
        }
    )


def estimate_fruit_dry_matter_fraction(
    observations: pd.DataFrame,
    *,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 0,
) -> dict[str, float | int]:
    """Estimate fresh-fruit dry-matter fraction with date-block uncertainty."""
    fresh_column = "ripe_fruit_fresh_kg_m2"
    dry_column = "ripe_fruit_dry_kg_m2"
    required = {"observation_date", fresh_column, dry_column}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"crop observations missing columns: {', '.join(missing)}")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    valid = observations[list(required)].copy()
    valid["observation_date"] = pd.to_datetime(valid["observation_date"], errors="coerce")
    valid[fresh_column] = pd.to_numeric(valid[fresh_column], errors="coerce")
    valid[dry_column] = pd.to_numeric(valid[dry_column], errors="coerce")
    valid = valid.dropna()
    valid = valid[~((valid[fresh_column] == 0.0) & (valid[dry_column] == 0.0))]
    if valid.empty:
        raise ValueError("no complete fruit fresh/dry observation pairs")
    if (valid[fresh_column] <= 0.0).any():
        raise ValueError("fruit fresh mass must be positive")
    if (valid[dry_column] <= 0.0).any():
        raise ValueError("fruit dry matter must be positive")
    if (valid[dry_column] > valid[fresh_column]).any():
        raise ValueError("fruit dry matter cannot exceed fruit fresh mass")

    estimate = float(
        valid[dry_column].sum() / valid[fresh_column].sum()
    )
    blocks = [group for _, group in valid.groupby("observation_date", sort=True)]
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(int(bootstrap_samples), dtype=float)
    for sample_index in range(int(bootstrap_samples)):
        selected = rng.integers(0, len(blocks), size=len(blocks))
        fresh = sum(float(blocks[index][fresh_column].sum()) for index in selected)
        dry = sum(float(blocks[index][dry_column].sum()) for index in selected)
        bootstrap[sample_index] = dry / fresh
    ci80_low, ci80_high = np.quantile(bootstrap, [0.10, 0.90])
    ci95_low, ci95_high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "estimate": estimate,
        "ci80_low": float(ci80_low),
        "ci80_high": float(ci80_high),
        "ci95_low": float(ci95_low),
        "ci95_high": float(ci95_high),
        "observation_count": int(len(valid)),
        "date_block_count": int(len(blocks)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }


def assess_target_harvest_data(
    harvest_events: pd.DataFrame,
    *,
    target_greenhouse_id: int,
    target_greenhouse_code: str,
    minimum_events_per_season: int = 8,
    minimum_event_span_days: int = 30,
) -> dict[str, object]:
    required = {
        "greenhouse_id",
        "greenhouse_code",
        "target_eligible",
        "season_id",
        "timestamp",
        "fresh_kg_m2",
    }
    missing = sorted(required - set(harvest_events.columns))
    if missing:
        raise ValueError(f"harvest events missing columns: {', '.join(missing)}")
    if minimum_events_per_season <= 0:
        raise ValueError("minimum_events_per_season must be positive")
    if minimum_event_span_days <= 0:
        raise ValueError("minimum_event_span_days must be positive")
    greenhouse_ids = pd.to_numeric(harvest_events["greenhouse_id"], errors="coerce")
    eligible = _strict_boolean_series(
        harvest_events["target_eligible"], "target_eligible"
    )
    if "evidence_class" in harvest_events.columns:
        evidence_class = harvest_events["evidence_class"].astype(str).str.strip()
    else:
        evidence_class = pd.Series(
            "target_observed", index=harvest_events.index, dtype="object"
        )
    accepted_evidence = evidence_class.eq("target_observed")
    matching_identity = (
        greenhouse_ids.eq(int(target_greenhouse_id))
        & harvest_events["greenhouse_code"].astype(str).eq(str(target_greenhouse_code))
        & eligible
    )
    rejected_evidence_class_counts = {
        str(name): int(count)
        for name, count in evidence_class[matching_identity & ~accepted_evidence]
        .value_counts()
        .sort_index()
        .items()
    }
    target = harvest_events[
        matching_identity & accepted_evidence
    ].copy()
    if target.empty:
        return {
            "target_greenhouse_id": int(target_greenhouse_id),
            "target_greenhouse_code": str(target_greenhouse_code),
            "target_harvest_event_count": 0,
            "target_season_count": 0,
            "usable_target_seasons": [],
            "season_coverage": [],
            "can_calibrate_target": False,
            "can_validate_target_independently": False,
            "rejected_evidence_class_counts": rejected_evidence_class_counts,
            "status": "missing_target_harvest_events",
        }
    target["timestamp"] = pd.to_datetime(target["timestamp"], errors="coerce")
    target["fresh_kg_m2"] = pd.to_numeric(target["fresh_kg_m2"], errors="coerce")
    if target["timestamp"].isna().any() or target["fresh_kg_m2"].isna().any():
        raise ValueError("target harvest events contain invalid timestamps or masses")
    if (target["fresh_kg_m2"] < 0.0).any():
        raise ValueError("target harvest masses must be non-negative")
    coverage: list[dict[str, object]] = []
    usable: list[str] = []
    for season_id, group in target.groupby("season_id", sort=True):
        span_days = float(
            (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
            / 86400.0
        )
        event_count = int(len(group))
        completion_evidence = (
            bool(_strict_boolean_series(
                group["season_complete_evidence"], "season_complete_evidence"
            ).all())
            if "season_complete_evidence" in group.columns
            else False
        )
        is_usable = (
            event_count >= int(minimum_events_per_season)
            and span_days >= float(minimum_event_span_days)
            and completion_evidence
        )
        if is_usable:
            usable.append(str(season_id))
        coverage.append(
            {
                "season_id": str(season_id),
                "event_count": event_count,
                "event_span_days": span_days,
                "season_complete_evidence": completion_evidence,
                "meets_minimum_coverage": is_usable,
            }
        )
    can_calibrate = len(usable) >= 1
    can_validate = len(usable) >= 2
    status = (
        "target_validation_ready"
        if can_validate else "target_calibration_only"
        if can_calibrate else "insufficient_target_harvest_events"
    )
    return {
        "target_greenhouse_id": int(target_greenhouse_id),
        "target_greenhouse_code": str(target_greenhouse_code),
        "target_harvest_event_count": int(len(target)),
        "target_season_count": int(target["season_id"].nunique()),
        "usable_target_seasons": usable,
        "season_coverage": coverage,
        "can_calibrate_target": can_calibrate,
        "can_validate_target_independently": can_validate,
        "rejected_evidence_class_counts": rejected_evidence_class_counts,
        "status": status,
    }


def split_harvest_seasons(
    harvest_events: pd.DataFrame,
    *,
    train_seasons: Iterable[str],
    validation_seasons: Iterable[str],
    test_seasons: Iterable[str],
) -> dict[str, pd.DataFrame]:
    required = {"season_id", "timestamp"}
    missing = sorted(required - set(harvest_events.columns))
    if missing:
        raise ValueError(f"harvest events missing columns: {', '.join(missing)}")
    requested = {
        "train": {str(value) for value in train_seasons},
        "validation": {str(value) for value in validation_seasons},
        "test": {str(value) for value in test_seasons},
    }
    if not all(requested.values()):
        raise ValueError("train, validation, and test season sets must be non-empty")
    if (
        requested["train"] & requested["validation"]
        or requested["train"] & requested["test"]
        or requested["validation"] & requested["test"]
    ):
        raise ValueError("train, validation, and test seasons must be disjoint")
    available = set(harvest_events["season_id"].astype(str))
    unknown = sorted(set.union(*requested.values()) - available)
    if unknown:
        raise ValueError(f"unknown harvest seasons: {', '.join(unknown)}")

    result: dict[str, pd.DataFrame] = {}
    for name, season_ids in requested.items():
        frame = harvest_events[
            harvest_events["season_id"].astype(str).isin(season_ids)
        ].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["timestamp"].isna().any():
            raise ValueError(f"{name} split contains invalid timestamps")
        result[name] = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if not (
        result["train"]["timestamp"].max() < result["validation"]["timestamp"].min()
        < result["test"]["timestamp"].min()
    ):
        raise ValueError("season splits must be chronological: train, validation, test")
    return result


def simulate_harvest_series(
    drivers: pd.DataFrame,
    parameters: HarvestCohortParameters,
) -> pd.DataFrame:
    """Run the cohort model from net fruit-growth and climate driver rows."""
    required = {
        "timestamp",
        "net_fruit_dry_matter_change_kg_m2",
        "air_temperature_c",
        "dt_hours",
        "pick",
    }
    missing = sorted(required - set(drivers.columns))
    if missing:
        raise ValueError(f"harvest drivers missing columns: {', '.join(missing)}")
    if drivers.empty:
        raise ValueError("harvest drivers must not be empty")
    frame = drivers.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("harvest driver timestamps must be valid")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("harvest driver timestamps must be chronological")
    frame["pick"] = _strict_boolean_series(frame["pick"], "pick")
    for column in (
        "initial_fruit_dry_matter_kg_m2",
        "initial_fruit_maturity_fraction",
    ):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite")
    model = HarvestCohortModel(parameters)
    has_seasons = "season_id" in frame.columns
    current_season: str | None = None
    rows: list[dict[str, float | str | pd.Timestamp]] = []
    for row in frame.itertuples(index=False):
        season_id = str(getattr(row, "season_id")) if has_seasons else None
        if has_seasons and season_id != current_season:
            model = HarvestCohortModel(
                parameters,
                initial_fruit_dry_matter_kg_m2=float(
                    getattr(row, "initial_fruit_dry_matter_kg_m2")
                ),
                initial_fruit_maturity_fraction=float(
                    getattr(row, "initial_fruit_maturity_fraction")
                ),
            )
            current_season = season_id
        result = model.step(
            net_fruit_dry_matter_change_kg_m2=float(
                getattr(row, "net_fruit_dry_matter_change_kg_m2")
            ),
            air_temperature_c=float(getattr(row, "air_temperature_c")),
            dt_hours=float(getattr(row, "dt_hours")),
            pick=bool(getattr(row, "pick")),
        )
        output_row: dict[str, float | str | pd.Timestamp] = {
                "timestamp": pd.Timestamp(getattr(row, "timestamp")),
                "harvested_dry_matter_kg_m2": result.harvested_dry_matter_kg_m2,
                "fresh_kg_m2": result.harvested_fresh_kg_m2,
                "cumulative_fresh_kg_m2": result.cumulative_harvested_fresh_kg_m2,
                "mature_dry_matter_kg_m2": result.mature_dry_matter_kg_m2,
                "standing_dry_matter_kg_m2": result.standing_dry_matter_kg_m2,
        }
        if season_id is not None:
            output_row["season_id"] = season_id
        rows.append(output_row)
    return pd.DataFrame(rows)


def _strict_boolean_series(values: pd.Series, name: str) -> pd.Series:
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = values.map(
        lambda value: mapping.get(value.lower().strip() if isinstance(value, str) else value)
    )
    if parsed.isna().any():
        raise ValueError(f"{name} must contain strict boolean values")
    return parsed.astype(bool)


def calibrate_harvest_parameters(
    drivers: pd.DataFrame,
    observed_harvest: pd.DataFrame,
    *,
    base_temperature_c: float,
    maturity_bounds_deg_day: tuple[float, float],
    dry_matter_fraction_bounds: tuple[float, float],
    maturity_grid_size: int = 61,
) -> dict[str, object]:
    """Fit identifiable maturity timing and dry/fresh scale on training events."""
    maturity_low, maturity_high = map(float, maturity_bounds_deg_day)
    dm_low, dm_high = map(float, dry_matter_fraction_bounds)
    if not 0.0 < maturity_low < maturity_high:
        raise ValueError("maturity bounds must be positive and increasing")
    if not 0.0 < dm_low < dm_high < 1.0:
        raise ValueError("dry-matter fraction bounds must lie within (0, 1)")
    if maturity_grid_size < 3:
        raise ValueError("maturity_grid_size must be at least 3")

    candidates = np.linspace(maturity_low, maturity_high, int(maturity_grid_size))
    best: tuple[float, float, float, pd.DataFrame, dict[str, float]] | None = None
    candidate_records: list[dict[str, float]] = []
    for maturity in candidates:
        dry_parameters = HarvestCohortParameters(
            base_temperature_c=float(base_temperature_c),
            maturity_thermal_time_deg_day=float(maturity),
            dry_matter_fraction=1.0,
            minimum_pick_fresh_kg_m2=0.0,
        )
        dry_prediction = simulate_harvest_series(drivers, dry_parameters)
        harvested_dm = dry_prediction["harvested_dry_matter_kg_m2"].to_numpy(dtype=float)
        if harvested_dm.sum() <= 0.0:
            continue

        def objective(dm_fraction: float) -> float:
            prediction_data: dict[str, object] = {
                "timestamp": dry_prediction["timestamp"],
                "fresh_kg_m2": harvested_dm / float(dm_fraction),
            }
            if "season_id" in observed_harvest.columns:
                prediction_data["season_id"] = dry_prediction["season_id"]
            predicted = pd.DataFrame(prediction_data)
            metrics, _ = evaluate_harvest_predictions(observed_harvest, predicted)
            onset = metrics["first_harvest_date_error_days"]
            onset_penalty = abs(onset) / 30.0 if np.isfinite(onset) else 10.0
            return float(
                metrics["batch_wmape"]
                + metrics["cumulative_wmape"]
                + onset_penalty
            )

        fit = minimize_scalar(
            objective,
            bounds=(dm_low, dm_high),
            method="bounded",
            options={"xatol": 1e-10},
        )
        dm_fraction = float(fit.x)
        prediction_data = {
            "timestamp": dry_prediction["timestamp"],
            "fresh_kg_m2": harvested_dm / dm_fraction,
        }
        if "season_id" in observed_harvest.columns:
            prediction_data["season_id"] = dry_prediction["season_id"]
        predicted = pd.DataFrame(prediction_data)
        metrics, aligned = evaluate_harvest_predictions(observed_harvest, predicted)
        score = float(fit.fun)
        candidate_records.append(
            {
                "maturity_thermal_time_deg_day": float(maturity),
                "dry_matter_fraction": dm_fraction,
                "objective": score,
            }
        )
        if best is None or score < best[0]:
            best = (score, float(maturity), dm_fraction, aligned, metrics)
    if best is None:
        raise ValueError("no candidate produced a positive harvest")
    score, maturity, dm_fraction, aligned, metrics = best
    return {
        "parameters": {
            "base_temperature_c": float(base_temperature_c),
            "maturity_thermal_time_deg_day": maturity,
            "dry_matter_fraction": dm_fraction,
        },
        "objective": score,
        "training_metrics": metrics,
        "aligned_training_predictions": aligned,
        "candidate_scores": pd.DataFrame(candidate_records),
        "fit_scope": "training_only",
    }


def _draw_season_residual_blocks(
    residual_groups: list[np.ndarray],
    *,
    length: int,
    block_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(length) <= 0:
        raise ValueError("residual forecast length must be positive")
    if int(block_days) <= 0:
        raise ValueError("block_days must be positive")
    groups = [np.asarray(group, dtype=float) for group in residual_groups]
    if not groups or any(group.ndim != 1 or len(group) == 0 for group in groups):
        raise ValueError("residual groups must contain non-empty one-dimensional arrays")
    if any(not np.isfinite(group).all() for group in groups):
        raise ValueError("residual groups must be finite")
    blocks: list[np.ndarray] = []
    blocks_needed = int(np.ceil(int(length) / int(block_days)))
    offsets = np.arange(int(block_days))
    for _ in range(blocks_needed):
        group = groups[int(rng.integers(0, len(groups)))]
        start = int(rng.integers(0, len(group)))
        blocks.append(group[(start + offsets) % len(group)])
    return np.concatenate(blocks)[: int(length)]


def bootstrap_harvest_parameter_estimates(
    drivers: pd.DataFrame,
    observed_harvest: pd.DataFrame,
    *,
    base_temperature_c: float,
    maturity_bounds_deg_day: tuple[float, float],
    dry_matter_fraction_bounds: tuple[float, float],
    maturity_grid_size: int = 61,
    bootstrap_samples: int = 200,
    block_days: int = 7,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Refit harvest parameters on season-aware residual bootstrap samples."""
    if int(bootstrap_samples) < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    if int(block_days) <= 0:
        raise ValueError("block_days must be positive")
    point_calibration = calibrate_harvest_parameters(
        drivers,
        observed_harvest,
        base_temperature_c=base_temperature_c,
        maturity_bounds_deg_day=maturity_bounds_deg_day,
        dry_matter_fraction_bounds=dry_matter_fraction_bounds,
        maturity_grid_size=maturity_grid_size,
    )
    aligned = point_calibration["aligned_training_predictions"].copy()
    observed = aligned["observed_fresh_kg_m2"].to_numpy(dtype=float)
    fitted = aligned["predicted_fresh_kg_m2"].to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(fitted).all():
        raise ValueError("aligned training batches must be finite")
    if "season_id" in aligned.columns:
        season_values = aligned["season_id"].astype(str).reset_index(drop=True)
        season_ids = pd.unique(season_values).tolist()
        positions = [
            np.flatnonzero(season_values.eq(season_id).to_numpy())
            for season_id in season_ids
        ]
    else:
        season_values = None
        season_ids = [None]
        positions = [np.arange(len(aligned))]
    centered_residual_groups = []
    for season_positions in positions:
        residuals = observed[season_positions] - fitted[season_positions]
        centered_residual_groups.append(residuals - float(np.mean(residuals)))
    numerical_zero_threshold = 1e-8 * max(1.0, float(np.max(fitted)))

    rng = np.random.default_rng(int(bootstrap_seed))
    requested = int(bootstrap_samples)
    maximum_attempts = max(requested * 5, requested + 10)
    records: list[dict[str, float | int]] = []
    rejected = 0
    attempts = 0
    while len(records) < requested and attempts < maximum_attempts:
        attempts += 1
        pseudo_batches = np.empty(len(aligned), dtype=float)
        for season_positions, residual_group in zip(
            positions, centered_residual_groups
        ):
            sampled = _draw_season_residual_blocks(
                [residual_group],
                length=len(season_positions),
                block_days=int(block_days),
                rng=rng,
            )
            pseudo_batches[season_positions] = np.clip(
                fitted[season_positions] + sampled, 0.0, None
            )
        pseudo_batches[pseudo_batches < numerical_zero_threshold] = 0.0
        pseudo_observed = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(aligned["date"], errors="coerce"),
                "fresh_kg_m2": pseudo_batches,
            }
        )
        if season_values is not None:
            pseudo_observed.insert(0, "season_id", season_values)
        try:
            fitted_draw = calibrate_harvest_parameters(
                drivers,
                pseudo_observed,
                base_temperature_c=base_temperature_c,
                maturity_bounds_deg_day=maturity_bounds_deg_day,
                dry_matter_fraction_bounds=dry_matter_fraction_bounds,
                maturity_grid_size=maturity_grid_size,
            )
        except ValueError:
            rejected += 1
            continue
        parameters = fitted_draw["parameters"]
        records.append(
            {
                "replicate_id": len(records),
                "maturity_thermal_time_deg_day": float(
                    parameters["maturity_thermal_time_deg_day"]
                ),
                "dry_matter_fraction": float(parameters["dry_matter_fraction"]),
                "objective": float(fitted_draw["objective"]),
            }
        )
    if len(records) < 2:
        raise ValueError("fewer than 2 valid parameter bootstrap draws were accepted")
    draws = pd.DataFrame(records)
    maturity = draws["maturity_thermal_time_deg_day"].to_numpy(dtype=float)
    dry_fraction = draws["dry_matter_fraction"].to_numpy(dtype=float)
    audit = {
        "method": "season_residual_moving_block_parameter_refit_bootstrap",
        "requested_draws": requested,
        "attempted_draws": attempts,
        "accepted_draws": int(len(draws)),
        "rejected_draws": int(rejected),
        "unique_parameter_draws": int(
            len(
                draws.drop_duplicates(
                    [
                        "maturity_thermal_time_deg_day",
                        "dry_matter_fraction",
                    ]
                )
            )
        ),
        "block_days": int(block_days),
        "bootstrap_seed": int(bootstrap_seed),
        "maturity_thermal_time_deg_day_quantiles": {
            "p025": float(np.quantile(maturity, 0.025)),
            "p50": float(np.quantile(maturity, 0.50)),
            "p975": float(np.quantile(maturity, 0.975)),
        },
        "dry_matter_fraction_quantiles": {
            "p025": float(np.quantile(dry_fraction, 0.025)),
            "p50": float(np.quantile(dry_fraction, 0.50)),
            "p975": float(np.quantile(dry_fraction, 0.975)),
        },
    }
    return {
        "point_calibration": point_calibration,
        "parameter_draws": draws,
        "audit": audit,
    }


def joint_harvest_prediction_intervals(
    aligned_training_predictions: pd.DataFrame,
    forecast_drivers: pd.DataFrame,
    parameter_draws: pd.DataFrame,
    *,
    forecast_index: pd.DataFrame,
    base_temperature_c: float,
    forecast_driver_scenarios: pd.DataFrame | None = None,
    bootstrap_samples: int = 2_000,
    block_days: int = 7,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Propagate driver scenarios, parameter draws, and seasonal residuals."""
    training_required = {
        "date",
        "observed_fresh_kg_m2",
        "predicted_fresh_kg_m2",
    }
    if missing := sorted(training_required - set(aligned_training_predictions.columns)):
        raise ValueError(f"aligned predictions missing columns: {', '.join(missing)}")
    draw_required = {
        "maturity_thermal_time_deg_day",
        "dry_matter_fraction",
    }
    if missing := sorted(draw_required - set(parameter_draws.columns)):
        raise ValueError(f"parameter draws missing columns: {', '.join(missing)}")
    if len(parameter_draws) < 2:
        raise ValueError("parameter draws must contain at least 2 rows")
    if int(bootstrap_samples) < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    if int(block_days) <= 0:
        raise ValueError("block_days must be positive")
    if forecast_index.empty or "date" not in forecast_index.columns:
        raise ValueError("forecast_index must contain non-empty date values")

    training = aligned_training_predictions.copy()
    observed = pd.to_numeric(
        training["observed_fresh_kg_m2"], errors="coerce"
    ).to_numpy(dtype=float)
    fitted = pd.to_numeric(
        training["predicted_fresh_kg_m2"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(fitted).all():
        raise ValueError("aligned training batches must be finite")
    residuals = observed - fitted
    if "season_id" in training.columns:
        training_seasons = training["season_id"].astype(str)
        residual_groups = [
            residuals[training_seasons.eq(season_id).to_numpy()]
            for season_id in pd.unique(training_seasons)
        ]
    else:
        residual_groups = [residuals]

    index_frame = forecast_index.copy().reset_index(drop=True)
    index_frame["date"] = pd.to_datetime(index_frame["date"], errors="coerce").dt.normalize()
    if index_frame["date"].isna().any():
        raise ValueError("forecast_index dates must be valid")
    has_forecast_seasons = "season_id" in index_frame.columns
    if has_forecast_seasons:
        if "season_id" not in forecast_drivers.columns:
            raise ValueError("forecast drivers must include season_id")
        index_frame["season_id"] = index_frame["season_id"].astype(str)
        if index_frame.duplicated(["season_id", "date"]).any():
            raise ValueError("forecast_index season dates must be unique")
        driver_seasons = set(forecast_drivers["season_id"].astype(str))
        index_seasons = set(index_frame["season_id"])
        if driver_seasons != index_seasons:
            raise ValueError("forecast driver and index seasons must match")
        forecast_positions = [
            np.flatnonzero(index_frame["season_id"].eq(season_id).to_numpy())
            for season_id in pd.unique(index_frame["season_id"])
        ]
    else:
        if "season_id" in forecast_drivers.columns:
            raise ValueError("forecast_index must include season_id for multiseason drivers")
        if index_frame["date"].duplicated().any():
            raise ValueError("forecast_index dates must be unique")
        forecast_positions = [np.arange(len(index_frame))]

    draw_values = parameter_draws[
        ["maturity_thermal_time_deg_day", "dry_matter_fraction"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(draw_values).all()
        or (draw_values[:, 0] <= 0.0).any()
        or (draw_values[:, 1] <= 0.0).any()
        or (draw_values[:, 1] >= 1.0).any()
    ):
        raise ValueError("parameter draws must contain finite physical values")

    driver_sets: list[pd.DataFrame]
    if forecast_driver_scenarios is None:
        driver_sets = [forecast_drivers]
    else:
        canonical_scenarios, _ = validate_uncertainty_driver_scenarios(
            forecast_drivers,
            forecast_driver_scenarios,
        )
        driver_sets = [
            scenario.copy()
            for _, scenario in canonical_scenarios.groupby(
                "driver_scenario_id", sort=False
            )
        ]

    def aligned_parameter_forecast(
        drivers: pd.DataFrame,
        draw_index: int,
    ) -> np.ndarray:
        maturity, dry_fraction = draw_values[int(draw_index)]
        parameters = HarvestCohortParameters(
            base_temperature_c=float(base_temperature_c),
            maturity_thermal_time_deg_day=float(maturity),
            dry_matter_fraction=float(dry_fraction),
        )
        simulated = simulate_harvest_series(drivers, parameters)
        simulated_dates = pd.to_datetime(
            simulated["timestamp"], errors="coerce"
        ).dt.normalize()
        masses = simulated["fresh_kg_m2"].to_numpy(dtype=float)
        if has_forecast_seasons:
            keys = pd.MultiIndex.from_arrays(
                [simulated["season_id"].astype(str), simulated_dates]
            )
            daily = pd.Series(masses, index=keys).groupby(level=[0, 1]).sum()
            target = pd.MultiIndex.from_frame(index_frame[["season_id", "date"]])
        else:
            daily = pd.Series(masses, index=simulated_dates).groupby(level=0).sum()
            target = pd.DatetimeIndex(index_frame["date"])
        return daily.reindex(target, fill_value=0.0).to_numpy(dtype=float)

    joint_forecasts = np.vstack(
        [
            aligned_parameter_forecast(drivers, draw_index)
            for drivers in driver_sets
            for draw_index in range(len(parameter_draws))
        ]
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    sample_count = int(bootstrap_samples)
    forecast_length = len(index_frame)
    batch_ensemble = np.empty((sample_count, forecast_length), dtype=float)
    cumulative_ensemble = np.empty((sample_count, forecast_length), dtype=float)
    for sample_index in range(sample_count):
        selected = int(rng.integers(0, len(joint_forecasts)))
        parameter_forecast = joint_forecasts[selected]
        for positions in forecast_positions:
            sampled_residuals = _draw_season_residual_blocks(
                residual_groups,
                length=len(positions),
                block_days=int(block_days),
                rng=rng,
            )
            batches = np.clip(
                parameter_forecast[positions] + sampled_residuals, 0.0, None
            )
            batch_ensemble[sample_index, positions] = batches
            cumulative_ensemble[sample_index, positions] = np.cumsum(batches)

    intervals: dict[str, np.ndarray] = dict(
        batch_prediction_intervals(batch_ensemble)
    )
    cumulative_columns = (
        "cumulative_p025_kg_m2",
        "cumulative_p10_kg_m2",
        "cumulative_p50_kg_m2",
        "cumulative_p90_kg_m2",
        "cumulative_p975_kg_m2",
    )
    intervals.update(
        {column: np.empty(forecast_length, dtype=float) for column in cumulative_columns}
    )
    for positions in forecast_positions:
        season_intervals = percentile_prediction_intervals(
            cumulative_ensemble[:, positions]
        )
        for column, values in season_intervals.items():
            intervals[column][positions] = values
    scale = max(1.0, float(np.max(cumulative_ensemble)))
    numerical_padding = 1e-8 * scale
    for prefix in ("batch", "cumulative"):
        for quantile in ("p025", "p10"):
            column = f"{prefix}_{quantile}_kg_m2"
            intervals[column] = np.clip(
                intervals[column] - numerical_padding, 0.0, None
            )
        for quantile in ("p90", "p975"):
            intervals[f"{prefix}_{quantile}_kg_m2"] += numerical_padding
    result = index_frame.copy()
    for column, values in intervals.items():
        result[column] = values
    return result


def validate_uncertainty_driver_scenarios(
    point_drivers: pd.DataFrame,
    scenario_drivers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate traceable scenario drivers against the point forecast keys."""
    driver_columns = {
        "season_id",
        "timestamp",
        "net_fruit_dry_matter_change_kg_m2",
        "air_temperature_c",
        "dt_hours",
        "pick",
        "pick_source",
        "driver_model_status",
        "season_complete",
        "fruit_change_source",
        "air_temperature_source",
        "greenhouse_id",
        "greenhouse_code",
        "planting_code",
        "initial_fruit_dry_matter_kg_m2",
        "initial_fruit_maturity_fraction",
    }
    scenario_columns = driver_columns | {
        "driver_scenario_id",
        "driver_uncertainty_source",
    }
    if missing := sorted(driver_columns - set(point_drivers.columns)):
        raise ValueError(f"point forecast driver columns missing: {', '.join(missing)}")
    if missing := sorted(scenario_columns - set(scenario_drivers.columns)):
        raise ValueError(f"scenario driver columns missing: {', '.join(missing)}")
    if point_drivers.empty or scenario_drivers.empty:
        raise ValueError("point and scenario drivers must not be empty")

    point = point_drivers.copy()
    scenarios = scenario_drivers.copy()
    for frame in (point, scenarios):
        frame["season_id"] = frame["season_id"].astype("string").str.strip()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        if frame["season_id"].isna().any() or frame["season_id"].eq("").any():
            raise ValueError("driver season_id values must be present")
        if frame["timestamp"].isna().any():
            raise ValueError("driver timestamps must be valid")
        frame["season_id"] = frame["season_id"].astype(str)
    scenario_ids = scenarios["driver_scenario_id"].astype("string").str.strip()
    if scenario_ids.isna().any() or scenario_ids.eq("").any():
        raise ValueError("driver scenario IDs must be present")
    scenarios["driver_scenario_id"] = scenario_ids.astype(str)
    scenario_order = pd.unique(scenarios["driver_scenario_id"]).tolist()
    if len(scenario_order) < 2:
        raise ValueError("uncertainty drivers require at least two scenarios")
    if point.duplicated(["season_id", "timestamp"]).any():
        raise ValueError("point forecast driver keys must be unique")
    if scenarios.duplicated(
        ["driver_scenario_id", "season_id", "timestamp"]
    ).any():
        raise ValueError("duplicate uncertainty driver scenario keys")

    source = scenarios["driver_uncertainty_source"].astype("string").str.strip()
    if source.isna().any() or not source.isin(DRIVER_UNCERTAINTY_SOURCES).all():
        raise ValueError("driver uncertainty source is missing or unsupported")
    scenarios["driver_uncertainty_source"] = source.astype(str)
    if (
        scenarios.groupby("driver_scenario_id")["driver_uncertainty_source"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise ValueError("each driver scenario must declare one uncertainty source")

    numeric_columns = [
        "net_fruit_dry_matter_change_kg_m2",
        "air_temperature_c",
        "dt_hours",
        "greenhouse_id",
        "initial_fruit_dry_matter_kg_m2",
        "initial_fruit_maturity_fraction",
    ]
    for frame in (point, scenarios):
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[numeric_columns].to_numpy(dtype=float)).all():
            raise ValueError("uncertainty driver numeric values must be finite")
        frame["pick"] = _strict_boolean_series(frame["pick"], "pick")
        frame["season_complete"] = _strict_boolean_series(
            frame["season_complete"], "season_complete"
        )
    if (scenarios["dt_hours"] <= 0.0).any():
        raise ValueError("uncertainty driver dt_hours must be positive")
    if not scenarios["season_complete"].all():
        raise ValueError("uncertainty driver scenarios must be complete")
    if not scenarios["driver_model_status"].astype(str).eq(
        "accepted_target_crop_model"
    ).all():
        raise ValueError("uncertainty driver model status must be accepted")
    if not scenarios["pick_source"].astype(str).eq("management_protocol").all():
        raise ValueError("uncertainty driver pick source must be management_protocol")

    point_indexed = point.set_index(["season_id", "timestamp"])
    point_keys = point_indexed.index
    aligned_parts: list[pd.DataFrame] = []
    invariant_text = [
        "pick_source",
        "driver_model_status",
        "fruit_change_source",
        "air_temperature_source",
        "greenhouse_code",
        "planting_code",
    ]
    for scenario_id in scenario_order:
        scenario = scenarios[
            scenarios["driver_scenario_id"].eq(scenario_id)
        ].set_index(["season_id", "timestamp"])
        if len(scenario) != len(point_indexed) or set(scenario.index) != set(point_keys):
            raise ValueError(f"driver scenario {scenario_id} keys do not match point forecast")
        scenario = scenario.loc[point_keys].copy()
        for column in invariant_text:
            if not scenario[column].astype(str).eq(
                point_indexed[column].astype(str)
            ).all():
                raise ValueError(f"driver scenario {scenario_id} {column} mismatch")
        if not np.allclose(
            scenario["dt_hours"].to_numpy(dtype=float),
            point_indexed["dt_hours"].to_numpy(dtype=float),
        ):
            raise ValueError(f"driver scenario {scenario_id} dt_hours mismatch")
        if not scenario["greenhouse_id"].eq(point_indexed["greenhouse_id"]).all():
            raise ValueError(f"driver scenario {scenario_id} greenhouse_id mismatch")
        if not scenario["pick"].eq(point_indexed["pick"]).all():
            raise ValueError(f"driver scenario {scenario_id} pick mismatch")
        aligned_parts.append(scenario.reset_index())
    canonical = pd.concat(aligned_parts, ignore_index=True)
    audit = {
        "scenario_count": int(len(scenario_order)),
        "scenario_ids": scenario_order,
        "uncertainty_sources": sorted(
            set(canonical["driver_uncertainty_source"].astype(str))
        ),
        "row_count": int(len(canonical)),
        "rows_per_scenario": int(len(point)),
        "scenario_sampling_policy": "uniform",
    }
    return canonical, audit


def residual_block_prediction_intervals(
    aligned_training_predictions: pd.DataFrame,
    *,
    forecast_predictions: pd.DataFrame | None = None,
    bootstrap_samples: int = 2_000,
    block_days: int = 7,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Build predictive intervals using circular moving-block residual bootstrap."""
    required = {"date", "observed_fresh_kg_m2", "predicted_fresh_kg_m2"}
    missing = sorted(required - set(aligned_training_predictions.columns))
    if missing:
        raise ValueError(f"aligned predictions missing columns: {', '.join(missing)}")
    if aligned_training_predictions.empty:
        raise ValueError("aligned predictions must not be empty")
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least 2")
    if block_days <= 0:
        raise ValueError("block_days must be positive")
    training = aligned_training_predictions.copy()
    observed = training["observed_fresh_kg_m2"].to_numpy(dtype=float)
    fitted = training["predicted_fresh_kg_m2"].to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(fitted).all():
        raise ValueError("aligned observed and predicted batches must be finite")
    residuals = observed - fitted
    if "season_id" in training.columns:
        training_seasons = training["season_id"].astype(str)
        residual_groups = [
            residuals[training_seasons.eq(season_id).to_numpy()]
            for season_id in pd.unique(training_seasons)
        ]
    else:
        residual_groups = [residuals]
    if forecast_predictions is None:
        forecast = fitted
        dates = pd.to_datetime(training["date"], errors="coerce")
        forecast_seasons = (
            training["season_id"].astype(str).reset_index(drop=True)
            if "season_id" in training.columns
            else None
        )
    else:
        forecast_required = {"date", "predicted_fresh_kg_m2"}
        forecast_missing = sorted(forecast_required - set(forecast_predictions.columns))
        if forecast_missing:
            raise ValueError(
                f"forecast predictions missing columns: {', '.join(forecast_missing)}"
            )
        forecast = forecast_predictions["predicted_fresh_kg_m2"].to_numpy(dtype=float)
        dates = pd.to_datetime(forecast_predictions["date"], errors="coerce")
        forecast_seasons = (
            forecast_predictions["season_id"].astype(str).reset_index(drop=True)
            if "season_id" in forecast_predictions.columns
            else None
        )
        if dates.isna().any() or not np.isfinite(forecast).all():
            raise ValueError("forecast dates and predicted batches must be valid")
    if dates.isna().any():
        raise ValueError("forecast dates and predicted batches must be valid")
    forecast_length = len(forecast)
    if forecast_length == 0:
        raise ValueError("forecast predictions must not be empty")
    if forecast_seasons is None:
        forecast_positions = [np.arange(forecast_length)]
    else:
        forecast_positions = [
            np.flatnonzero(forecast_seasons.eq(season_id).to_numpy())
            for season_id in pd.unique(forecast_seasons)
        ]
    rng = np.random.default_rng(bootstrap_seed)
    cumulative_ensemble = np.empty(
        (int(bootstrap_samples), forecast_length), dtype=float
    )
    batch_ensemble = np.empty(
        (int(bootstrap_samples), forecast_length), dtype=float
    )
    for sample_index in range(int(bootstrap_samples)):
        for positions in forecast_positions:
            sampled_residuals = _draw_season_residual_blocks(
                residual_groups,
                length=len(positions),
                block_days=int(block_days),
                rng=rng,
            )
            simulated_batches = np.clip(
                forecast[positions] + sampled_residuals, 0.0, None
            )
            batch_ensemble[sample_index, positions] = simulated_batches
            cumulative_ensemble[sample_index, positions] = np.cumsum(
                simulated_batches
            )
    intervals = {
        column: np.empty(forecast_length, dtype=float)
        for column in (
            "cumulative_p025_kg_m2",
            "cumulative_p10_kg_m2",
            "cumulative_p50_kg_m2",
            "cumulative_p90_kg_m2",
            "cumulative_p975_kg_m2",
        )
    }
    for positions in forecast_positions:
        season_intervals = percentile_prediction_intervals(
            cumulative_ensemble[:, positions]
        )
        for column, values in season_intervals.items():
            intervals[column][positions] = values
    intervals.update(batch_prediction_intervals(batch_ensemble))
    scale = max(1.0, float(np.max(np.cumsum(np.clip(forecast, 0.0, None)))))
    numerical_padding = 1e-8 * scale
    intervals["cumulative_p025_kg_m2"] = np.clip(
        intervals["cumulative_p025_kg_m2"] - numerical_padding,
        0.0,
        None,
    )
    intervals["cumulative_p10_kg_m2"] = np.clip(
        intervals["cumulative_p10_kg_m2"] - numerical_padding,
        0.0,
        None,
    )
    intervals["cumulative_p90_kg_m2"] += numerical_padding
    intervals["cumulative_p975_kg_m2"] += numerical_padding
    intervals["batch_p025_kg_m2"] = np.clip(
        intervals["batch_p025_kg_m2"] - numerical_padding,
        0.0,
        None,
    )
    intervals["batch_p10_kg_m2"] = np.clip(
        intervals["batch_p10_kg_m2"] - numerical_padding,
        0.0,
        None,
    )
    intervals["batch_p90_kg_m2"] += numerical_padding
    intervals["batch_p975_kg_m2"] += numerical_padding
    result = pd.DataFrame({"date": dates})
    if forecast_seasons is not None:
        result.insert(0, "season_id", forecast_seasons)
    for column, values in intervals.items():
        result[column] = values
    return result
