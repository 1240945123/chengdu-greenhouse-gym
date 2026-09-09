from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from processing.chengdu_crop_observations import (
    CROP_OBSERVATION_COLUMNS,
    _crop_frame,
    validate_crop_observations,
)


def extract_target_crop_workbook(
    path: str | Path,
    *,
    greenhouse_id: int,
    greenhouse_code: str,
    planting_id: int,
    planting_code: str,
    cultivar: str,
    plant_density_plants_m2: float,
    biomass_sheet: str = "郫都1",
    morphology_sheet: str = "郫都2",
    timezone: str = "Asia/Shanghai",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize dated Pidu workbook rows as standing-crop observations."""
    source = Path(path)
    density = float(plant_density_plants_m2)
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("plant_density_plants_m2 must be finite and positive")
    try:
        biomass_raw = pd.read_excel(
            source, sheet_name=biomass_sheet, header=None, engine="openpyxl"
        )
        morphology_raw = pd.read_excel(
            source, sheet_name=morphology_sheet, header=None, engine="openpyxl"
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to read target crop workbook {source}: {exc}") from exc

    biomass = _dated_rows(
        biomass_raw, minimum_columns=15, label="biomass", timezone=timezone
    )
    morphology = _dated_rows(
        morphology_raw, minimum_columns=8, label="morphology", timezone=timezone
    )
    undated = biomass[biomass["observation_date"].isna()].copy()
    dated_biomass = biomass.dropna(subset=["observation_date"]).copy()
    dated_morphology = morphology.dropna(subset=["observation_date"]).copy()

    dated_biomass = dated_biomass.rename(
        columns={
            3: "root_fresh_g_per_plant",
            4: "stem_fresh_g_per_plant",
            5: "leaf_fresh_g_per_plant",
            6: "ripe_fruit_fresh_g_per_plant",
        }
    )
    dated_morphology = dated_morphology.rename(
        columns={
            3: "plant_height_cm",
            4: "stem_diameter_mm",
            5: "stem_node_count",
            6: "flower_truss_count",
            7: "fruit_truss_count",
        }
    )
    fresh_columns = [
        "root_fresh_g_per_plant",
        "stem_fresh_g_per_plant",
        "leaf_fresh_g_per_plant",
        "ripe_fruit_fresh_g_per_plant",
    ]
    morphology_columns = [
        "plant_height_cm",
        "stem_diameter_mm",
        "stem_node_count",
        "flower_truss_count",
        "fruit_truss_count",
    ]
    _numeric_nonnegative(dated_biomass, fresh_columns, "fresh mass")
    _numeric_nonnegative(dated_morphology, morphology_columns, "morphology")

    key = ["observation_date", "sample_plant_id"]
    if dated_biomass.duplicated(key).any() or dated_morphology.duplicated(key).any():
        raise ValueError("duplicate workbook observation date and sample plant ID")
    merged = dated_biomass[
        key + ["source_line"] + fresh_columns
    ].merge(
        dated_morphology[key + morphology_columns],
        on=key,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("biomass and morphology workbook rows must have matching dated samples")

    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        date = pd.Timestamp(record["observation_date"])
        sample_id = int(record["sample_plant_id"])
        row: dict[str, object] = {column: None for column in CROP_OBSERVATION_COLUMNS}
        row.update(
            {
                "observation_id": int(date.strftime("%Y%m%d")) * 10 + sample_id,
                "observation_type": "standing_crop_sample",
                "observation_date": date,
                "greenhouse_id": int(greenhouse_id),
                "greenhouse_code": str(greenhouse_code),
                "planting_id": int(planting_id),
                "planting_code": str(planting_code),
                "crop_name": "tomato",
                "cultivar": str(cultivar),
                "sample_plant_id": str(sample_id),
                "source_site": "chengdu_pidu_target_greenhouse",
                "target_eligible": True,
                "quality_flags": (
                    "manual_workbook;secondary_mass_header_ambiguous_not_used;"
                    "standing_crop_sample_not_harvest_event"
                ),
                "plant_density_plants_m2": density,
                "source_table": f"xlsx:{biomass_sheet}+{morphology_sheet}",
                "source_line": int(record["source_line"]),
            }
        )
        for column in morphology_columns + fresh_columns:
            row[column] = float(record[column])
        row["plant_fresh_g_per_plant"] = float(
            sum(row[column] for column in fresh_columns)
        )
        for organ in ("plant", "root", "stem", "leaf", "ripe_fruit"):
            grams = float(row[f"{organ}_fresh_g_per_plant"])
            row[f"{organ}_fresh_kg_m2"] = grams * density / 1000.0
        rows.append(row)

    frame = _crop_frame(rows)
    validate_crop_observations(frame)
    yield_columns = [column for column in (12, 13, 14) if column in undated.columns]
    undated_yield_rows = (
        int(undated[yield_columns].notna().any(axis=1).sum()) if yield_columns else 0
    )
    audit = {
        "source_file": str(source),
        "biomass_sheet": biomass_sheet,
        "morphology_sheet": morphology_sheet,
        "dated_observation_rows": int(len(frame)),
        "observation_dates": [
            value.isoformat() for value in frame["observation_date"].drop_duplicates()
        ],
        "undated_sample_rows_excluded": int(len(undated)),
        "undated_rows_with_reported_truss_yield": undated_yield_rows,
        "secondary_mass_columns_used": False,
        "observation_semantics": "standing_crop_sample_not_harvest_event",
    }
    return frame, audit


def _dated_rows(
    raw: pd.DataFrame, *, minimum_columns: int, label: str, timezone: str
) -> pd.DataFrame:
    if raw.shape[0] < 3 or raw.shape[1] < minimum_columns:
        raise ValueError(f"{label} sheet has an unexpected layout")
    rows = raw.iloc[2:].copy()
    rows["source_line"] = rows.index + 1
    rows["sample_plant_id"] = pd.to_numeric(rows.iloc[:, 2], errors="coerce")
    rows = rows.dropna(subset=["sample_plant_id"])
    if not np.equal(rows["sample_plant_id"], np.floor(rows["sample_plant_id"])).all():
        raise ValueError(f"{label} sample plant IDs must be integers")
    rows["sample_plant_id"] = rows["sample_plant_id"].astype(int)
    explicit_dates = pd.to_datetime(rows.iloc[:, 0], errors="coerce")
    explicit_dates = explicit_dates.dt.tz_localize(timezone)
    rows["observation_date"] = explicit_dates.ffill()
    return rows


def _numeric_nonnegative(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[columns].isna().any().any():
        raise ValueError(f"dated workbook {label} values must be numeric and complete")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(f"dated workbook {label} values must be finite and non-negative")
