from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable, Mapping

import py7zr
import numpy as np
import pandas as pd
import requests


EXPECTED_ARTICLE_ID = 12764777
EXPECTED_FILE_ID = 24757220
EXPECTED_DOI = "10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7"
ARTICLE_API_URL = f"https://data.4tu.nl/v2/articles/{EXPECTED_ARTICLE_ID}"
SUPPORTED_ARCHIVE_EXTENSIONS = {".csv", ".pdf"}
WUR_TEAMS = (
    "AICU",
    "Automatoes",
    "Digilog",
    "IUACAAS",
    "Reference",
    "TheAutomators",
)
PRODUCTION_AREA_M2 = 62.5
CROP_START = pd.Timestamp("2019-12-16")
CROP_END = pd.Timestamp("2020-05-29")
PRODUCTION_COLUMNS = (
    "%time",
    "ProdA",
    "ProdB",
    "avg_nr_harvested_trusses",
    "Truss development time",
    "Nr_fruits_ClassA",
    "Weight_fruits_ClassA",
    "Nr_fruits_ClassB",
    "Weight_fruits_ClassB",
)


def fetch_article_metadata(
    *,
    http_get: Callable[..., Any] = requests.get,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    response = http_get(ARTICLE_API_URL, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("4TU article metadata must be a JSON object")
    return payload


def validate_article_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if metadata.get("id") != EXPECTED_ARTICLE_ID:
        raise ValueError("Unexpected 4TU article ID")
    if metadata.get("doi") != EXPECTED_DOI:
        raise ValueError("Unexpected 4TU article DOI")
    license_data = metadata.get("license")
    if not isinstance(license_data, Mapping) or license_data.get("name") != "CC0":
        raise ValueError("WUR AGC2 source must retain its CC0 license")
    files = metadata.get("files")
    if not isinstance(files, list):
        raise ValueError("4TU article metadata is missing files")
    matches = [item for item in files if item.get("id") == EXPECTED_FILE_ID]
    if len(matches) != 1:
        raise ValueError("Expected exactly one WUR AGC2 archive file")
    selected = dict(matches[0])
    required = {"name", "size", "computed_md5", "download_url"}
    missing = sorted(required - set(selected))
    if missing:
        raise ValueError(f"WUR AGC2 file metadata missing fields: {missing}")
    if not str(selected["name"]).lower().endswith(".7z"):
        raise ValueError("WUR AGC2 source file must be a 7z archive")
    if int(selected["size"]) <= 0:
        raise ValueError("WUR AGC2 archive size must be positive")
    return selected


def download_verified_archive(
    metadata: Mapping[str, Any],
    destination: str | Path,
    *,
    http_get: Callable[..., Any] = requests.get,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    selected = validate_article_metadata(metadata)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    response = http_get(
        str(selected["download_url"]),
        stream=True,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                size += len(chunk)
                md5.update(chunk)
                sha256.update(chunk)
        if size != int(selected["size"]):
            raise ValueError(
                f"SIZE mismatch: expected {selected['size']}, downloaded {size}"
            )
        if md5.hexdigest().lower() != str(selected["computed_md5"]).lower():
            raise ValueError("MD5 mismatch for downloaded WUR AGC2 archive")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "article_id": EXPECTED_ARTICLE_ID,
        "article_uuid": str(metadata.get("uuid", "")),
        "article_api_url": ARTICLE_API_URL,
        "doi": EXPECTED_DOI,
        "version": int(metadata.get("version", 0)),
        "license": "CC0",
        "file_id": EXPECTED_FILE_ID,
        "file_name": str(selected["name"]),
        "download_url": str(selected["download_url"]),
        "bytes": size,
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(path.resolve()),
        "evidence_class": "external_observed",
        "target_eligible": False,
    }


def validate_archive_member_names(member_names: list[str]) -> None:
    seen: set[str] = set()
    for raw_name in member_names:
        name = str(raw_name)
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not name
            or name != normalized
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or (path.parts and ":" in path.parts[0])
        ):
            raise ValueError(f"unsafe archive member path: {name!r}")
        folded = normalized.casefold()
        if folded in seen:
            raise ValueError(f"duplicate archive member path: {name!r}")
        seen.add(folded)
        if path.suffix.lower() not in SUPPORTED_ARCHIVE_EXTENSIONS:
            raise ValueError(f"unsupported archive member type: {name!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive_inventory(
    archive_path: str | Path,
    staging_directory: str | Path,
) -> dict[str, Any]:
    archive = Path(archive_path)
    staging = Path(staging_directory)
    if staging.exists() and any(staging.iterdir()):
        raise ValueError("archive staging directory must be absent or empty")
    staging.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as source:
        if source.needs_password():
            raise ValueError("encrypted WUR archive is not accepted")
        entries = source.list()
        file_entries = [entry for entry in entries if not entry.is_directory]
        names = [str(entry.filename) for entry in file_entries]
        validate_archive_member_names(names)
        source.extractall(path=staging)
    members: list[dict[str, Any]] = []
    staging_resolved = staging.resolve()
    for entry in file_entries:
        member_path = staging.joinpath(*PurePosixPath(str(entry.filename)).parts)
        resolved = member_path.resolve()
        if staging_resolved not in resolved.parents:
            raise ValueError(f"unsafe extracted member location: {entry.filename!r}")
        if not resolved.is_file():
            raise ValueError(f"archive member was not extracted: {entry.filename!r}")
        members.append(
            {
                "path": str(entry.filename),
                "bytes": int(resolved.stat().st_size),
                "sha256": _sha256_file(resolved),
                "extension": resolved.suffix.lower(),
            }
        )
    return {
        "archive_path": str(archive.resolve()),
        "archive_sha256": _sha256_file(archive),
        "staging_directory": str(staging.resolve()),
        "member_count": len(members),
        "members": members,
    }


def normalize_production_frame(
    frame: pd.DataFrame,
    *,
    team: str,
    source_file: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if team not in WUR_TEAMS:
        raise ValueError(f"unknown WUR AGC2 team: {team!r}")
    raw = frame.copy()
    raw.columns = [str(column).strip() for column in raw.columns]
    missing = sorted(set(PRODUCTION_COLUMNS) - set(raw.columns))
    unexpected = sorted(set(raw.columns) - set(PRODUCTION_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"WUR AGC2 Production.csv schema mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    raw["source_row"] = np.arange(len(raw), dtype=int) + 2
    numeric_columns = list(PRODUCTION_COLUMNS)
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    required_numeric = ["%time", "ProdA", "ProdB"]
    if raw[required_numeric].isna().any().any():
        raise ValueError("WUR production date and ProdA/ProdB must be numeric")
    if (raw[["ProdA", "ProdB"]] < 0).any().any():
        raise ValueError("WUR production batch masses must be non-negative")
    raw["harvest_date"] = pd.Timestamp("1899-12-30") + pd.to_timedelta(
        raw["%time"], unit="D"
    )
    in_window = raw["harvest_date"].between(CROP_START, CROP_END)
    excluded_rows = [
        {
            "source_row": int(row.source_row),
            "source_excel_time": int(row._asdict()["_0"]),
            "parsed_date": row.harvest_date.strftime("%Y-%m-%d"),
            "reason": "outside_declared_crop_window",
        }
        for row in raw.loc[~in_window, ["%time", "source_row", "harvest_date"]].itertuples(
            index=False
        )
    ]
    valid = raw.loc[in_window].copy().sort_values(["harvest_date", "source_row"])
    if valid["harvest_date"].duplicated().any():
        raise ValueError(f"WUR team {team} has duplicate harvest dates")
    valid["batch_class_a_fresh_kg_m2"] = valid["ProdA"]
    valid["batch_class_b_fresh_kg_m2"] = valid["ProdB"]
    valid["batch_fresh_kg_m2"] = valid["ProdA"] + valid["ProdB"]
    valid["cumulative_fresh_kg_m2"] = valid["batch_fresh_kg_m2"].cumsum()
    valid["harvested_area_m2"] = PRODUCTION_AREA_M2
    valid["harvested_fresh_kg"] = valid["batch_fresh_kg_m2"] * PRODUCTION_AREA_M2
    valid["marketable_fresh_kg"] = valid["ProdA"] * PRODUCTION_AREA_M2
    valid["rejected_fresh_kg"] = valid["ProdB"] * PRODUCTION_AREA_M2
    valid["external_harvest_event_id"] = [
        f"WUR_AGC2_{team}_{date:%Y%m%d}" for date in valid["harvest_date"]
    ]
    valid["evidence_class"] = "external_observed"
    valid["target_eligible"] = False
    valid["source_dataset"] = "wur_agc2_v2"
    valid["source_doi"] = EXPECTED_DOI
    valid["team"] = team
    valid["compartment"] = team
    valid["cultivar"] = "Axiany"
    valid["source_file"] = source_file or f"{team}/Production.csv"
    output_columns = [
        "external_harvest_event_id",
        "evidence_class",
        "target_eligible",
        "source_dataset",
        "source_doi",
        "team",
        "compartment",
        "cultivar",
        "harvest_date",
        "batch_class_a_fresh_kg_m2",
        "batch_class_b_fresh_kg_m2",
        "batch_fresh_kg_m2",
        "cumulative_fresh_kg_m2",
        "harvested_area_m2",
        "harvested_fresh_kg",
        "marketable_fresh_kg",
        "rejected_fresh_kg",
        "avg_nr_harvested_trusses",
        "Truss development time",
        "Nr_fruits_ClassA",
        "Weight_fruits_ClassA",
        "Nr_fruits_ClassB",
        "Weight_fruits_ClassB",
        "source_file",
        "source_row",
    ]
    normalized = valid[output_columns].reset_index(drop=True)
    audit = {
        "team": team,
        "source_file": source_file or f"{team}/Production.csv",
        "source_rows": int(len(raw)),
        "normalized_rows": int(len(normalized)),
        "excluded_row_count": len(excluded_rows),
        "excluded_rows": excluded_rows,
        "production_area_m2": PRODUCTION_AREA_M2,
        "mass_semantics": "batch_at_harvest_date",
        "mass_unit": "kg_fresh_m2_growing_area",
        "crop_window": [CROP_START.strftime("%Y-%m-%d"), CROP_END.strftime("%Y-%m-%d")],
        "evidence_class": "external_observed",
        "target_eligible": False,
    }
    return normalized, audit


def build_external_harvest_observations(
    extracted_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(extracted_root)
    observations: list[pd.DataFrame] = []
    team_audits: list[dict[str, Any]] = []
    for team in WUR_TEAMS:
        source = root / team / "Production.csv"
        if not source.is_file():
            raise ValueError(f"missing WUR production file: {source}")
        normalized, audit = normalize_production_frame(
            pd.read_csv(source, skipinitialspace=True),
            team=team,
            source_file=f"{team}/Production.csv",
        )
        observations.append(normalized)
        team_audits.append(audit)
    combined = pd.concat(observations, ignore_index=True)
    if combined["external_harvest_event_id"].duplicated().any():
        raise ValueError("duplicate WUR external harvest event IDs")
    audit = {
        "dataset": "wur_agc2_v2",
        "doi": EXPECTED_DOI,
        "license": "CC0",
        "team_count": len(team_audits),
        "source_rows": sum(item["source_rows"] for item in team_audits),
        "normalized_rows": int(len(combined)),
        "excluded_row_count": sum(item["excluded_row_count"] for item in team_audits),
        "teams": team_audits,
        "mass_semantics": "batch_at_harvest_date",
        "production_area_m2": PRODUCTION_AREA_M2,
        "evidence_class": "external_observed",
        "target_eligible": False,
        "transfer_role": "external_prior_and_pipeline_validation_only",
    }
    return combined, audit


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and verify the public WUR AGC2 tomato dataset."
    )
    parser.add_argument("--output-archive", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--extract-dir")
    parser.add_argument("--output-inventory")
    parser.add_argument("--normalize-root")
    parser.add_argument("--output-normalized")
    parser.add_argument("--output-normalization-audit")
    args = parser.parse_args()
    if bool(args.extract_dir) != bool(args.output_inventory):
        parser.error("--extract-dir and --output-inventory must be supplied together")
    normalization_arguments = (
        args.normalize_root,
        args.output_normalized,
        args.output_normalization_audit,
    )
    if any(normalization_arguments) and not all(normalization_arguments):
        parser.error(
            "--normalize-root, --output-normalized, and "
            "--output-normalization-audit must be supplied together"
        )
    metadata = fetch_article_metadata()
    audit = download_verified_archive(metadata, args.output_archive)
    audit["article_metadata"] = metadata
    _write_json_atomic(Path(args.output_manifest), audit)
    if args.extract_dir:
        inventory = extract_archive_inventory(args.output_archive, args.extract_dir)
        _write_json_atomic(Path(args.output_inventory), inventory)
        audit["archive_inventory"] = inventory
    if args.normalize_root:
        observations, normalization_audit = build_external_harvest_observations(
            args.normalize_root
        )
        normalized_path = Path(args.output_normalized)
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        observations.to_csv(normalized_path, index=False)
        _write_json_atomic(
            Path(args.output_normalization_audit), normalization_audit
        )
        audit["normalization_audit"] = normalization_audit
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
