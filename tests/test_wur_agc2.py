from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import py7zr
import pytest

from processing.wur_agc2 import (
    EXPECTED_ARTICLE_ID,
    EXPECTED_FILE_ID,
    build_external_harvest_observations,
    download_verified_archive,
    extract_archive_inventory,
    normalize_production_frame,
    validate_archive_member_names,
    validate_article_metadata,
)


def _metadata(*, content: bytes = b"verified archive", license_name: str = "CC0"):
    return {
        "id": EXPECTED_ARTICLE_ID,
        "uuid": "8b35675d-7549-49cc-a8c3-dedec8d3f23e",
        "title": "Autonomous Greenhouse Challenge, Second Edition (2019)",
        "doi": "10.4121/uuid:88d22c60-21b3-4ea8-90db-20249a5be2a7",
        "version": 2,
        "license": {"name": license_name},
        "files": [
            {
                "id": EXPECTED_FILE_ID,
                "name": "AutonomousGreenhouseChallenge_edition2.7z",
                "size": len(content),
                "computed_md5": hashlib.md5(content).hexdigest(),
                "download_url": "https://example.test/agc2.7z",
            }
        ],
    }


class _Response:
    def __init__(self, content: bytes):
        self.content = content
        self.url = "https://example.test/agc2.7z"

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]


def test_validate_article_metadata_selects_expected_cc0_archive():
    selected = validate_article_metadata(_metadata())

    assert selected["id"] == EXPECTED_FILE_ID
    assert selected["name"].endswith(".7z")


def test_validate_article_metadata_rejects_non_cc0_license():
    with pytest.raises(ValueError, match="CC0"):
        validate_article_metadata(_metadata(license_name="CC-BY-4.0"))


@pytest.mark.parametrize("corruption", ["size", "md5"])
def test_download_verified_archive_rejects_corrupt_content(
    tmp_path: Path,
    corruption: str,
):
    expected = b"verified archive"
    metadata = _metadata(content=expected)
    if corruption == "size":
        metadata["files"][0]["size"] += 1
        response_content = expected
    else:
        response_content = b"xerified archive"

    with pytest.raises(ValueError, match=corruption.upper()):
        download_verified_archive(
            metadata,
            tmp_path / "archive.7z",
            http_get=lambda *args, **kwargs: _Response(response_content),
        )

    assert not (tmp_path / "archive.7z").exists()
    assert not (tmp_path / "archive.7z.tmp").exists()


def test_download_verified_archive_writes_file_and_provenance(tmp_path: Path):
    content = b"verified archive"
    destination = tmp_path / "archive.7z"

    audit = download_verified_archive(
        _metadata(content=content),
        destination,
        http_get=lambda *args, **kwargs: _Response(content),
    )

    assert destination.read_bytes() == content
    assert audit["article_id"] == EXPECTED_ARTICLE_ID
    assert audit["file_id"] == EXPECTED_FILE_ID
    assert audit["license"] == "CC0"
    assert audit["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    "members, message",
    [
        (["../escape.csv"], "unsafe"),
        (["C:/escape.csv"], "unsafe"),
        (["Team/Production.csv", "team/production.csv"], "duplicate"),
        (["Team/script.exe"], "unsupported"),
    ],
)
def test_validate_archive_member_names_rejects_unsafe_members(members, message):
    with pytest.raises(ValueError, match=message):
        validate_archive_member_names(members)


def test_validate_archive_member_names_accepts_top_level_documentation():
    validate_archive_member_names(["ReadMe.pdf", "Team/Production.csv"])


def test_extract_archive_inventory_rejects_encrypted_archive(tmp_path: Path):
    archive = tmp_path / "encrypted.7z"
    with py7zr.SevenZipFile(archive, mode="w", password="secret") as output:
        output.writestr(b"date,yield\n", "Team/Production.csv")

    with pytest.raises(ValueError, match="encrypted"):
        extract_archive_inventory(archive, tmp_path / "staging")


def test_extract_archive_inventory_hashes_extracted_members(tmp_path: Path):
    archive = tmp_path / "source.7z"
    content = b"date,yield\n2020-01-01,1.5\n"
    with py7zr.SevenZipFile(archive, mode="w") as output:
        output.writestr(content, "Team/Production.csv")

    audit = extract_archive_inventory(archive, tmp_path / "staging")

    assert audit["member_count"] == 1
    assert audit["members"][0]["path"] == "Team/Production.csv"
    assert audit["members"][0]["bytes"] == len(content)
    assert audit["members"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "staging" / "Team" / "Production.csv").read_bytes() == content


def test_normalize_production_frame_emits_batch_yield_with_external_evidence():
    raw = pd.DataFrame(
        {
            "%time": [43880, 43885],
            "ProdA": [0.5, 0.7],
            "ProdB": [0.0, 0.1],
            "avg_nr_harvested_trusses": [1.0, 1.2],
            "Truss development time ": [50.0, 48.0],
            "Nr_fruits_ClassA": [100, 120],
            "Weight_fruits_ClassA": [1000, 1200],
            "Nr_fruits_ClassB": [0, 10],
            "Weight_fruits_ClassB": [0, 100],
        }
    )

    normalized, audit = normalize_production_frame(raw, team="Reference")

    assert normalized["harvest_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-02-19",
        "2020-02-24",
    ]
    assert normalized["batch_fresh_kg_m2"].tolist() == pytest.approx([0.5, 0.8])
    assert normalized["harvested_fresh_kg"].tolist() == pytest.approx([31.25, 50.0])
    assert normalized["evidence_class"].eq("external_observed").all()
    assert normalized["target_eligible"].eq(False).all()
    assert audit["mass_semantics"] == "batch_at_harvest_date"


def test_normalize_production_frame_excludes_out_of_crop_window_without_repair():
    raw = pd.DataFrame(
        {
            "%time": [43510, 43880],
            "ProdA": [0.32, 0.32],
            "ProdB": [0.0, 0.0],
            "avg_nr_harvested_trusses": [None, 1.0],
            "Truss development time ": [45.0, 45.0],
            "Nr_fruits_ClassA": [None, None],
            "Weight_fruits_ClassA": [717, 964],
            "Nr_fruits_ClassB": [0, 0],
            "Weight_fruits_ClassB": [0, 0],
        }
    )

    normalized, audit = normalize_production_frame(raw, team="Reference")

    assert len(normalized) == 1
    assert audit["excluded_rows"][0]["source_excel_time"] == 43510
    assert audit["excluded_rows"][0]["reason"] == "outside_declared_crop_window"


def test_build_external_harvest_observations_normalizes_all_real_teams():
    root = Path("data/external/crops/wur_agc2/extracted")

    observations, audit = build_external_harvest_observations(root)

    assert audit["team_count"] == 6
    assert audit["source_rows"] == 140
    assert audit["normalized_rows"] == 139
    assert observations["team"].nunique() == 6
    assert observations.groupby("team")["cumulative_fresh_kg_m2"].last().between(8, 20).all()
