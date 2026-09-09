from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.crop.estimate_wur_process_priors import (
    _load_dry_matter_fractions,
    estimate_wur_process_priors,
    excel_serial_to_timestamp,
)


def _excel_serial(timestamp: str) -> float:
    origin = pd.Timestamp("1899-12-30")
    return (pd.Timestamp(timestamp) - origin).total_seconds() / 86_400.0


def _write_compartment(root: Path, team: str, temperature: float, dmc: float) -> None:
    folder = root / team
    folder.mkdir(parents=True)
    times = pd.date_range("2019-12-31", "2020-01-11", freq="1D")
    pd.DataFrame(
        {"%time": [_excel_serial(str(value)) for value in times], "Tair": temperature}
    ).to_csv(folder / "GreenhouseClimate.csv", index=False)
    pd.DataFrame(
        {
            "%time": [_excel_serial("2020-01-11")],
            "DMC_fruit": [dmc],
        }
    ).to_csv(folder / "TomQuality.csv", index=False)


def test_excel_serial_conversion_uses_excel_1900_origin():
    assert excel_serial_to_timestamp(43831.0) == pd.Timestamp("2020-01-01")


def test_reference_quality_header_repairs_registered_missing_comma(tmp_path: Path):
    path = tmp_path / "TomQuality.csv"
    path.write_text(
        "%time,\tFlavour,\tWeight\tDMC_fruit\n43880,\t74,\t7.77,\t9.59\n",
        encoding="utf-8",
    )

    assert _load_dry_matter_fractions(path).tolist() == pytest.approx([0.0959])


def test_process_prior_integrates_thermal_time_and_bootstraps_by_compartment(
    tmp_path: Path,
):
    root = tmp_path / "wur"
    _write_compartment(root, "A", temperature=20.0, dmc=8.0)
    _write_compartment(root, "B", temperature=22.0, dmc=10.0)
    observations = pd.DataFrame(
        {
            "team": ["A", "A", "B", "B"],
            "harvest_date": ["2020-01-10", "2020-01-11"] * 2,
            "Truss development time": [10.0, 10.0] * 2,
            "batch_fresh_kg_m2": [0.5, 0.5, 0.7, 0.7],
            "source_doi": ["10.test/source"] * 4,
        }
    )
    observations_path = tmp_path / "observations.csv"
    observations.to_csv(observations_path, index=False)

    result = estimate_wur_process_priors(
        root=root,
        observations_path=observations_path,
        base_temperature_c=10.0,
        bootstrap_samples=200,
        bootstrap_seed=17,
        minimum_compartments=2,
        minimum_maturity_events=2,
    )

    assert result["compartment_count"] == 2
    assert result["maturity_event_count"] == 4
    summaries = {row["team"]: row for row in result["compartment_summaries"]}
    assert summaries["A"]["maturity_thermal_time_deg_day_median"] == pytest.approx(100.0)
    assert summaries["B"]["maturity_thermal_time_deg_day_median"] == pytest.approx(120.0)
    assert result["metrics"]["maturity_thermal_time_deg_day"]["estimate"] == pytest.approx(110.0)
    assert result["metrics"]["fruit_dry_matter_fraction"]["estimate"] == pytest.approx(0.09)
    assert result["bootstrap_unit"] == "greenhouse_compartment"
    assert result["transfer_role"] == "external_process_prior_not_target_calibration"

    output = tmp_path / "priors.json"
    output.write_text(json.dumps(result), encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8"))["base_temperature_c"] == 10.0


def test_process_prior_rejects_insufficient_compartment_evidence(tmp_path: Path):
    root = tmp_path / "wur"
    _write_compartment(root, "A", temperature=20.0, dmc=8.0)
    observations = pd.DataFrame(
        {
            "team": ["A"],
            "harvest_date": ["2020-01-11"],
            "Truss development time": [10.0],
            "batch_fresh_kg_m2": [0.5],
            "source_doi": ["10.test/source"],
        }
    )
    path = tmp_path / "observations.csv"
    observations.to_csv(path, index=False)

    with pytest.raises(ValueError, match="compartments"):
        estimate_wur_process_priors(
            root=root,
            observations_path=path,
            minimum_compartments=2,
            minimum_maturity_events=1,
            bootstrap_samples=20,
        )
