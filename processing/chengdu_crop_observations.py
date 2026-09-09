from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from processing.chengdu_sql_controls import split_mysql_values


SQL_TABLE_WHITELIST = frozenset({"planting_info", "plant_message_info"})

PLANTING_FIELDS = (
    "id",
    "code",
    "type",
    "createdDateTime",
    "updatedDateTime",
    "version",
    "deleted",
    "disable",
    "crop_name",
    "plant_nums",
    "seedingDateTime",
    "plantingDateTime",
    "firflowerDateTime",
    "earlyFruitDateTime",
    "matureDateTime",
    "endDateTime",
    "plant_den",
    "stem_den",
    "greenhouse_code",
    "rowSpace",
    "plantSpace",
    "booleanCurrent",
    "plantStage",
    "booleanMain",
    "image",
    "plant_area_name",
    "plant_area_id",
    "experimentName",
    "cropIntroduction",
    "matureSpecies",
    "greenHouseId",
)

MESSAGE_FIELDS = (
    "id",
    "greenhouseCode",
    "plantingCode",
    "plantingInfoCropName",
    "createdDateTime",
    "updatedDateTime",
    "version",
    "deleted",
    "disable",
    "plantStage",
    "plantHeight",
    "stemDia",
    "plantFrewei",
    "plantDryWei",
    "leaveFrewei",
    "layerFlowerNumber",
    "layerFruitNumber",
    "leaveDryWei",
    "petioleFrewei",
    "petioleDryWei",
    "stemFrewei",
    "stemDryWei",
    "ripeFruitFrewei",
    "ripeFruitDryWei",
    "rootFrewei",
    "rootDryWei",
    "lai",
    "oneFruitWei",
    "twoFruitWei",
    "threeFruitWei",
    "fourFruitWei",
    "collectDateTime",
    "fixedPlantHeight",
    "fixedStemDia",
    "fixedLayerFlowerNumber",
    "fixedLayerFruitNumber",
    "stipesNumber",
    "fixedStipesNumber",
    "sweetness",
    "acidity",
    "unitPlantNumber",
    "images",
    "oneNumberFruit",
    "twoNumberFruit",
    "threeNumberFruit",
    "fourNumberFruit",
    "pesticideResidue",
    "fixedOneNumberFruit",
    "fixedTowNumberFruit",
    "fixedThreeNumberFruit",
    "fixedFourNumberFruit",
)

_MASS_SOURCE_FIELDS = {
    "plant_fresh": "plantFrewei",
    "plant_dry": "plantDryWei",
    "leaf_fresh": "leaveFrewei",
    "leaf_dry": "leaveDryWei",
    "petiole_fresh": "petioleFrewei",
    "petiole_dry": "petioleDryWei",
    "stem_fresh": "stemFrewei",
    "stem_dry": "stemDryWei",
    "ripe_fruit_fresh": "ripeFruitFrewei",
    "ripe_fruit_dry": "ripeFruitDryWei",
    "root_fresh": "rootFrewei",
    "root_dry": "rootDryWei",
}

CROP_OBSERVATION_COLUMNS = [
    "observation_id",
    "observation_type",
    "observation_date",
    "greenhouse_id",
    "greenhouse_code",
    "planting_id",
    "planting_code",
    "crop_name",
    "cultivar",
    "plant_stage",
    "sample_plant_id",
    "source_site",
    "target_eligible",
    "quality_flags",
    "plant_density_plants_m2",
    "lai_m2_m2",
    "plant_height_cm",
    "stem_diameter_mm",
    "stem_node_count",
    "flower_truss_count",
    "fruit_truss_count",
    *[
        column
        for name in _MASS_SOURCE_FIELDS
        for column in (f"{name}_g_per_plant", f"{name}_kg_m2")
    ],
    "source_table",
    "source_line",
]

_TABLE_RE = re.compile(r"^\s*INSERT\s+INTO\s+`?(?P<table>[A-Za-z0-9_]+)`?", re.IGNORECASE)
_INSERT_RE = re.compile(
    r"^\s*INSERT\s+INTO\s+`?(?P<table>[A-Za-z0-9_]+)`?\s+VALUES\s+(?P<values>.+?)\s*;\s*$",
    re.IGNORECASE,
)


def extract_crop_observations(
    sql_path: str | Path,
    *,
    target_greenhouse_ids: Collection[int] = (),
    target_greenhouse_codes: Collection[str] = (),
    source_sites: Mapping[str, str] | None = None,
    timezone: str = "Asia/Shanghai",
) -> pd.DataFrame:
    """Stream standing crop samples from the two whitelisted SQL tables."""
    if bool(target_greenhouse_ids) != bool(target_greenhouse_codes):
        raise ValueError(
            "target_greenhouse_ids and target_greenhouse_codes must be configured together"
        )
    planting_versions: dict[str, list[tuple[dict[str, str | None], int]]] = {}
    messages: list[tuple[dict[str, str | None], int]] = []

    with Path(sql_path).open("r", encoding="utf-8-sig") as sql_file:
        for line_number, line in enumerate(sql_file, start=1):
            table_match = _TABLE_RE.match(line)
            if table_match is None:
                continue
            table = table_match.group("table").lower()
            if table not in SQL_TABLE_WHITELIST:
                continue

            insert_match = _INSERT_RE.match(line)
            if insert_match is None:
                raise ValueError(f"line {line_number}: malformed INSERT for {table}")
            try:
                values = split_mysql_values(insert_match.group("values"))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc

            fields = PLANTING_FIELDS if table == "planting_info" else MESSAGE_FIELDS
            if len(values) != len(fields):
                raise ValueError(
                    f"line {line_number}: {table} row has {len(values)} values; "
                    f"expected {len(fields)}"
                )
            record = dict(zip(fields, values, strict=True))
            if table == "planting_info":
                code = record["code"]
                if not code:
                    raise ValueError(f"line {line_number}: planting code is required")
                planting_versions.setdefault(code, []).append((record, line_number))
            else:
                messages.append((record, line_number))

    plantings = {
        code: max(versions, key=_record_rank)
        for code, versions in planting_versions.items()
    }
    active_messages = [item for item in messages if _is_active(item[0])]
    active_messages, duplicate_rows_removed = _deduplicate_messages(active_messages)
    rows = []
    for message, line_number in active_messages:
        planting_code = message["plantingCode"]
        if not planting_code or planting_code not in plantings:
            raise ValueError(
                f"line {line_number}: no planting_info row for planting code {planting_code!r}"
            )
        planting, planting_line = plantings[planting_code]
        if not _is_active(planting):
            continue
        try:
            rows.append(
                _canonical_row(
                    message,
                    planting,
                    line_number=line_number,
                    target_greenhouse_ids=set(target_greenhouse_ids),
                    target_greenhouse_codes=set(target_greenhouse_codes),
                    source_sites=source_sites or {},
                    timezone=timezone,
                )
            )
        except ValueError as exc:
            message_text = str(exc)
            if message_text.startswith("line "):
                raise
            raise ValueError(f"line {line_number}: {message_text}") from exc

    frame = _crop_frame(rows)
    frame.attrs["duplicate_rows_removed"] = duplicate_rows_removed
    validate_crop_observations(frame)
    return frame


def validate_crop_observations(frame: pd.DataFrame) -> None:
    """Validate canonical observation dates and nonnegative mass measurements."""
    missing_columns = [column for column in CROP_OBSERVATION_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing canonical crop observation columns: {missing_columns}")
    dates = pd.to_datetime(frame["observation_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("observation_date must contain valid, non-null dates")
    if not frame["observation_type"].eq("standing_crop_sample").all():
        raise ValueError("observation_type must be standing_crop_sample")
    for column in ("observation_id", "greenhouse_id", "greenhouse_code", "planting_code"):
        if frame[column].isna().any():
            raise ValueError(f"{column} must not be null")
    density = pd.to_numeric(frame["plant_density_plants_m2"], errors="coerce")
    if not np.isfinite(density.to_numpy(dtype=float)).all() or (density <= 0.0).any():
        raise ValueError("plant_density_plants_m2 must be finite and positive")

    mass_columns = [
        column
        for column in CROP_OBSERVATION_COLUMNS
        if column.endswith("_g_per_plant") or column.endswith("_kg_m2")
    ]
    for column in mass_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"{column} must be numeric when present")
        present = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all():
            raise ValueError(f"{column} must be finite when present")
        if (numeric < 0).any():
            raise ValueError(f"{column} must be nonnegative")
    for organ in ("plant", "leaf", "petiole", "stem", "ripe_fruit", "root"):
        for unit in ("g_per_plant", "kg_m2"):
            fresh = pd.to_numeric(frame[f"{organ}_fresh_{unit}"], errors="coerce")
            dry = pd.to_numeric(frame[f"{organ}_dry_{unit}"], errors="coerce")
            if (dry.notna() & fresh.notna() & (dry > fresh)).any():
                raise ValueError(f"{organ} dry mass cannot exceed fresh mass")
        for mass_type in ("fresh", "dry"):
            grams = pd.to_numeric(
                frame[f"{organ}_{mass_type}_g_per_plant"], errors="coerce"
            )
            area = pd.to_numeric(frame[f"{organ}_{mass_type}_kg_m2"], errors="coerce")
            paired = grams.notna() & area.notna()
            expected = grams * density / 1000.0
            if paired.any() and not np.allclose(
                area[paired], expected[paired], rtol=1e-9, atol=1e-12
            ):
                raise ValueError(
                    f"{organ} {mass_type} mass area conversion is inconsistent"
                )


def _canonical_row(
    message: dict[str, str | None],
    planting: dict[str, str | None],
    *,
    line_number: int,
    target_greenhouse_ids: set[int],
    target_greenhouse_codes: set[str],
    source_sites: Mapping[str, str],
    timezone: str,
) -> dict[str, object]:
    observation_date = _timestamp(message["collectDateTime"], "observation date", timezone)
    density = _required_float(planting["plant_den"], "plant density")
    if density <= 0:
        raise ValueError("plant density must be greater than zero")

    greenhouse_id = _optional_int(planting["greenHouseId"], "greenhouse ID")
    greenhouse_code = planting["greenhouse_code"] or message["greenhouseCode"]
    if not greenhouse_code:
        raise ValueError("greenhouse code is required")
    if message["greenhouseCode"] and message["greenhouseCode"] != greenhouse_code:
        raise ValueError("planting and observation greenhouse codes do not match")

    identifiers_configured = bool(target_greenhouse_ids and target_greenhouse_codes)
    id_match = greenhouse_id in target_greenhouse_ids if identifiers_configured else False
    code_match = greenhouse_code in target_greenhouse_codes if identifiers_configured else False
    target_eligible = bool(id_match and code_match)
    quality_flags: list[str] = []
    if identifiers_configured and id_match != code_match:
        quality_flags.append("target_identifier_mismatch")
    if message["plantHeight"] is not None:
        quality_flags.append("plant_height_unit_inferred_cm")

    parsed_masses = {
        canonical_name: _optional_nonnegative_float(
            message[source_field], f"{canonical_name}_g_per_plant"
        )
        for canonical_name, source_field in _MASS_SOURCE_FIELDS.items()
    }
    other_mass_values = [
        value for name, value in parsed_masses.items() if name != "plant_fresh"
    ]
    if parsed_masses["plant_fresh"] == 0.0 and all(
        value is None for value in other_mass_values
    ):
        parsed_masses["plant_fresh"] = None
        quality_flags.append("placeholder_zero_plant_fresh")

    row: dict[str, object] = {
        "observation_id": _required_int(message["id"], "observation ID"),
        "observation_type": "standing_crop_sample",
        "observation_date": observation_date,
        "greenhouse_id": greenhouse_id,
        "greenhouse_code": greenhouse_code,
        "planting_id": _required_int(planting["id"], "planting ID"),
        "planting_code": planting["code"],
        "crop_name": planting["type"] or message["plantingInfoCropName"],
        "cultivar": planting["crop_name"],
        "plant_stage": message["plantStage"],
        "sample_plant_id": message["unitPlantNumber"],
        "source_site": source_sites.get(greenhouse_code, greenhouse_code),
        "target_eligible": target_eligible,
        "quality_flags": ";".join(quality_flags),
        "plant_density_plants_m2": density,
        "lai_m2_m2": _optional_nonnegative_float(message["lai"], "lai_m2_m2"),
        "plant_height_cm": _optional_nonnegative_float(
            message["plantHeight"], "plant_height_cm"
        ),
        "stem_diameter_mm": _optional_nonnegative_float(
            message["stemDia"], "stem_diameter_mm"
        ),
        "source_table": "plant_message_info",
        "source_line": line_number,
    }
    for canonical_name in _MASS_SOURCE_FIELDS:
        grams = parsed_masses[canonical_name]
        row[f"{canonical_name}_g_per_plant"] = grams
        row[f"{canonical_name}_kg_m2"] = (
            None if grams is None else grams * density / 1000.0
        )
    return row


def _is_active(record: Mapping[str, str | None]) -> bool:
    return record.get("deleted") not in {"1", 1} and record.get("disable") not in {"1", 1}


def _required_int(value: str | None, label: str) -> int:
    parsed = _optional_int(value, label)
    if parsed is None:
        raise ValueError(f"{label} is required")
    return parsed


def _optional_int(value: str | None, label: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc


def _required_float(value: str | None, label: str) -> float:
    parsed = _optional_float(value, label)
    if parsed is None:
        raise ValueError(f"{label} is required")
    return parsed


def _optional_float(value: str | None, label: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if not pd.notna(parsed):
        raise ValueError(f"invalid {label}: {value!r}")
    return parsed


def _optional_nonnegative_float(value: str | None, label: str) -> float | None:
    parsed = _optional_float(value, label)
    if parsed is not None and parsed < 0:
        raise ValueError(f"{label} must be nonnegative")
    return parsed


def _timestamp(value: str | None, label: str, timezone: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{label} is required")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"invalid {label}: {value!r}")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone)
    else:
        timestamp = timestamp.tz_convert(timezone)
    return timestamp


def _deduplicate_messages(
    messages: list[tuple[dict[str, str | None], int]],
) -> tuple[list[tuple[dict[str, str | None], int]], int]:
    grouped: dict[tuple[str, str, str], list[tuple[dict[str, str | None], int]]] = {}
    passthrough: list[tuple[dict[str, str | None], int]] = []
    for item in messages:
        record, _ = item
        if not record.get("plantingCode") or not record.get("unitPlantNumber") or not record.get("collectDateTime"):
            passthrough.append(item)
            continue
        key = (
            str(record["plantingCode"]),
            str(record["unitPlantNumber"]),
            str(record["collectDateTime"]),
        )
        grouped.setdefault(key, []).append(item)

    kept = list(passthrough)
    removed = 0
    for items in grouped.values():
        kept.append(max(items, key=_record_rank))
        removed += len(items) - 1
    return kept, removed


def _record_rank(
    item: tuple[dict[str, str | None], int],
) -> tuple[int, pd.Timestamp, int, int]:
    record, line_number = item
    updated = pd.to_datetime(record.get("updatedDateTime"), errors="coerce")
    if pd.isna(updated):
        updated = pd.Timestamp.min
    return (
        int(record.get("version") or 0),
        updated,
        int(record.get("id") or 0),
        int(line_number),
    )


def _crop_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=CROP_OBSERVATION_COLUMNS)
    if frame.empty:
        frame = frame.astype(
            {
                "observation_id": "int64",
                "observation_date": "datetime64[ns]",
                "greenhouse_id": "Int64",
                "planting_id": "int64",
                "target_eligible": "bool",
                "source_line": "int64",
            }
        )
        return frame
    frame["observation_id"] = frame["observation_id"].astype("int64")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    frame["greenhouse_id"] = pd.array(frame["greenhouse_id"], dtype="Int64")
    frame["planting_id"] = frame["planting_id"].astype("int64")
    frame["target_eligible"] = frame["target_eligible"].astype("bool")
    frame["source_line"] = frame["source_line"].astype("int64")
    return frame.sort_values(["observation_date", "observation_id"], kind="stable").reset_index(
        drop=True
    )
