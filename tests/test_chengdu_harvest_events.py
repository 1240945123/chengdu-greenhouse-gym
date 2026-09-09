from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from processing.chengdu_harvest_events import load_harvest_events as _load_harvest_events


NORMALIZATION_AREAS_M2 = {"GH-PIDU": 192.0, "GH-XINDU": 192.0}


def load_harvest_events(*args, **kwargs):
    kwargs.setdefault("normalization_areas_m2", NORMALIZATION_AREAS_M2)
    return _load_harvest_events(*args, **kwargs)


def _write_csv(path: Path, **overrides: object) -> Path:
    row: dict[str, object] = {
        "harvest_event_id": "H-001",
        "harvest_datetime": "2026-05-01 08:00:00",
        "greenhouse_id": 63,
        "greenhouse_code": "GH-PIDU",
        "planting_code": "P-PIDU-2026",
        "season_id": "2026_spring",
        "cultivar": "Sainirui",
        "harvested_fresh_kg": 96.0,
        "harvested_area_m2": 192.0,
        "marketable_fresh_kg": 90.0,
        "rejected_fresh_kg": 6.0,
        "grade": "mixed",
        "batch_id": "B-001",
    }
    row.update(overrides)
    pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _event_row(**overrides: object) -> dict[str, object]:
    row = {
        "harvest_event_id": "H-001",
        "harvest_datetime": "2026-05-01 08:00:00",
        "greenhouse_id": 63,
        "greenhouse_code": "GH-PIDU",
        "planting_code": "P-PIDU-2026",
        "season_id": "2026_spring",
        "cultivar": "Sainirui",
        "harvested_fresh_kg": 96.0,
        "harvested_area_m2": 192.0,
        "marketable_fresh_kg": 90.0,
        "rejected_fresh_kg": 6.0,
        "grade": "mixed",
        "batch_id": "B-001",
    }
    row.update(overrides)
    return row


def test_loader_reads_xlsx_harvest_events_and_ignores_fully_blank_rows(tmp_path: Path):
    path = tmp_path / "harvest.xlsx"
    rows = pd.DataFrame([_event_row(), {column: None for column in _event_row()}])
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="HarvestEvents", index=False)

    events = load_harvest_events(
        path,
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
    )

    assert events["harvest_event_id"].tolist() == ["H-001"]
    assert events.loc[0, "fresh_kg_m2"] == 0.5


def test_xlsx_loader_rejects_partial_rows_and_missing_sheet(tmp_path: Path):
    partial_path = tmp_path / "partial.xlsx"
    partial = pd.DataFrame([_event_row(), _event_row(harvest_event_id="H-002", harvested_fresh_kg=None)])
    with pd.ExcelWriter(partial_path, engine="openpyxl") as writer:
        partial.to_excel(writer, sheet_name="HarvestEvents", index=False)
    with pytest.raises(ValueError, match="harvested_fresh_kg"):
        load_harvest_events(
            partial_path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )

    missing_path = tmp_path / "missing.xlsx"
    with pd.ExcelWriter(missing_path, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["wrong sheet"]}).to_excel(
            writer, sheet_name="Other", index=False
        )
    with pytest.raises(ValueError, match="HarvestEvents"):
        load_harvest_events(
            missing_path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )


def test_loader_normalizes_target_event_to_fixed_area_and_retains_local_yield(tmp_path: Path):
    path = _write_csv(tmp_path / "harvest.csv", harvested_area_m2=96.0)

    events = load_harvest_events(
        path,
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
        source_sites={"GH-PIDU": "chengdu_pidu_target_greenhouse"},
        timezone="Asia/Shanghai",
    )

    assert events.loc[0, "event_type"] == "harvest_event"
    assert events.loc[0, "target_eligible"]
    assert events.loc[0, "source_site"] == "chengdu_pidu_target_greenhouse"
    assert events.loc[0, "harvested_fresh_kg_m2"] == 0.5
    assert events.loc[0, "fresh_kg_m2"] == 0.5
    assert events.loc[0, "normalization_area_m2"] == 192.0
    assert events.loc[0, "local_harvested_fresh_kg_m2"] == 1.0
    assert events.loc[0, "harvest_coverage_fraction"] == 0.5
    assert str(events.loc[0, "timestamp"].tz) == "Asia/Shanghai"


@pytest.mark.parametrize("evidence_class", ["external_observed", "simulated_prior"])
def test_loader_never_marks_non_target_evidence_as_target_eligible(
    tmp_path: Path,
    evidence_class: str,
):
    path = _write_csv(tmp_path / "harvest.csv", evidence_class=evidence_class)

    events = load_harvest_events(
        path,
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
    )

    assert events.loc[0, "evidence_class"] == evidence_class
    assert not events.loc[0, "target_eligible"]


def test_loader_requires_trusted_area_mapping_and_rejects_overcoverage(tmp_path: Path):
    path = _write_csv(tmp_path / "harvest.csv")
    try:
        _load_harvest_events(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
            normalization_areas_m2={},
        )
    except ValueError as exc:
        assert "normalization" in str(exc) and "GH-PIDU" in str(exc)
    else:
        raise AssertionError("normalization area must come from trusted configuration")

    overcoverage = _write_csv(
        tmp_path / "overcoverage.csv", harvested_area_m2=193.0
    )
    try:
        load_harvest_events(
            overcoverage,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "harvested_area_m2" in str(exc) and "normalization" in str(exc)
    else:
        raise AssertionError("harvested area cannot exceed modeled production area")


def test_loader_rejects_missing_area_instead_of_assuming_whole_greenhouse(tmp_path: Path):
    path = _write_csv(tmp_path / "harvest.csv", harvested_area_m2=None)

    try:
        load_harvest_events(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "harvested_area_m2" in str(exc)
    else:
        raise AssertionError("harvest area is required for kg/m2 conversion")


def test_loader_rejects_component_mass_inconsistency(tmp_path: Path):
    path = _write_csv(
        tmp_path / "harvest.csv",
        marketable_fresh_kg=95.0,
        rejected_fresh_kg=10.0,
    )

    try:
        load_harvest_events(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "marketable" in str(exc) and "rejected" in str(exc)
    else:
        raise AssertionError("components cannot exceed total harvest")


def test_loader_marks_source_site_event_ineligible_for_target_validation(tmp_path: Path):
    path = _write_csv(
        tmp_path / "harvest.csv",
        greenhouse_id=1,
        greenhouse_code="GH-XINDU",
    )

    events = load_harvest_events(
        path,
        target_greenhouse_ids={63},
        target_greenhouse_codes={"GH-PIDU"},
        source_sites={"GH-XINDU": "chengdu_xindu_experimental_base"},
    )

    assert not events.loc[0, "target_eligible"]


def test_loader_rejects_duplicate_ids_negative_mass_and_invalid_dates(tmp_path: Path):
    base = pd.DataFrame(
        [
            {
                "harvest_event_id": "H-001",
                "harvest_datetime": "bad-date",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": "P1",
                "season_id": "s1",
                "cultivar": "c1",
                "harvested_fresh_kg": -1.0,
                "harvested_area_m2": 192.0,
            },
            {
                "harvest_event_id": "H-001",
                "harvest_datetime": "2026-05-02",
                "greenhouse_id": 63,
                "greenhouse_code": "GH-PIDU",
                "planting_code": "P1",
                "season_id": "s1",
                "cultivar": "c1",
                "harvested_fresh_kg": 1.0,
                "harvested_area_m2": 192.0,
            },
        ]
    )
    path = tmp_path / "harvest.csv"
    base.to_csv(path, index=False)

    try:
        load_harvest_events(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        message = str(exc)
        assert "duplicate" in message or "date" in message or "non-negative" in message
    else:
        raise AssertionError("invalid harvest rows must be rejected")


def test_loader_rejects_target_id_code_mismatch_and_infinite_mass(tmp_path: Path):
    mismatch = _write_csv(tmp_path / "mismatch.csv", greenhouse_id=1)
    try:
        load_harvest_events(
            mismatch,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "greenhouse ID and code" in str(exc)
    else:
        raise AssertionError("target ID/code mismatch must be rejected")

    infinite = _write_csv(tmp_path / "infinite.csv", harvested_fresh_kg=float("inf"))
    try:
        load_harvest_events(
            infinite,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("infinite harvest mass must be rejected")


def test_loader_rejects_business_duplicates_and_mixed_season_identity(tmp_path: Path):
    first = pd.read_csv(_write_csv(tmp_path / "one.csv"))
    duplicate = first.copy()
    duplicate.loc[0, "harvest_event_id"] = "H-002"
    duplicate_path = tmp_path / "duplicate.csv"
    pd.concat([first, duplicate], ignore_index=True).to_csv(duplicate_path, index=False)
    try:
        load_harvest_events(
            duplicate_path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "business duplicate" in str(exc)
    else:
        raise AssertionError("identical harvest content with a new ID must be rejected")

    mixed = first.copy()
    mixed.loc[0, "harvest_event_id"] = "H-003"
    mixed.loc[0, "harvest_datetime"] = "2026-05-02 08:00:00"
    mixed.loc[0, "planting_code"] = "OTHER-PLANTING"
    mixed_path = tmp_path / "mixed.csv"
    pd.concat([first, mixed], ignore_index=True).to_csv(mixed_path, index=False)
    try:
        load_harvest_events(
            mixed_path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
        )
    except ValueError as exc:
        assert "season_id" in str(exc) and "identity" in str(exc)
    else:
        raise AssertionError("one season must map to one crop identity")
