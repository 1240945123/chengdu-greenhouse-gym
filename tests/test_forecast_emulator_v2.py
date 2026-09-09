from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from experiments.weather.fit_forecast_emulator import (
    fit_forecast_emulator_v2,
    load_forecast_emulator_artifact,
    save_forecast_emulator_artifact,
)
from glassgym.components.forecast import HistoricalForecastEmulatorV2
from experiments.controllers.benchmark_protocol import build_environment, load_benchmark_config

SOURCE_HASHES = {"2023": "a" * 64, "2024": "b" * 64}


def _pattern(days: int = 5, steps_per_day: int = 4) -> np.ndarray:
    rows = days * steps_per_day
    values = np.zeros((rows, 11), dtype=float)
    phase = np.arange(rows) % steps_per_day
    values[:, 0] = np.maximum(0.0, np.sin(2.0 * np.pi * phase / steps_per_day)) * 500.0
    values[:, 1] = 18.0 + 8.0 * np.sin(2.0 * np.pi * phase / steps_per_day)
    values[:, 2] = 1500.0 + 300.0 * np.cos(2.0 * np.pi * phase / steps_per_day)
    values[:, 4] = 1.0 + phase
    return values


def test_forecast_fit_enforces_2023_fit_and_2024_select_roles():
    values = _pattern()
    with pytest.raises(ValueError, match="2023 fit"):
        fit_forecast_emulator_v2(
            values, values, fit_year=2025, select_year=2024, steps_per_day=4,
            horizon_steps=2,
        )
    with pytest.raises(ValueError, match="2024 select"):
        fit_forecast_emulator_v2(
            values, values, fit_year=2023, select_year=2025, steps_per_day=4,
            horizon_steps=2,
        )


def test_selected_forecast_is_no_worse_than_persistence_and_is_deterministic():
    fit = _pattern()
    select = _pattern()
    first = fit_forecast_emulator_v2(
        fit, select, fit_year=2023, select_year=2024, steps_per_day=4,
        horizon_steps=3, candidate_blend_weights=(0.0, 0.5, 1.0),
        source_sha256=SOURCE_HASHES,
    )
    second = fit_forecast_emulator_v2(
        fit, select, fit_year=2023, select_year=2024, steps_per_day=4,
        horizon_steps=3, candidate_blend_weights=(0.0, 0.5, 1.0),
        source_sha256=SOURCE_HASHES,
    )

    assert first.manifest["selection_weighted_mae"] <= (
        first.manifest["persistence_weighted_mae"] + 1e-12
    )
    assert first.manifest["selected_blend_weight"] == pytest.approx(1.0)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.climatology, second.climatology)


def test_runtime_emulator_cannot_observe_poisoned_future():
    climatology = _pattern(days=2)
    provider = HistoricalForecastEmulatorV2(
        climatology=climatology,
        blend_weight=0.75,
        steps_per_day=4,
        source_model_version="historical-forecast-emulator-v2",
        error_model_id="sha256:test",
    )
    realized = _pattern(days=3)
    poisoned = realized.copy()
    poisoned[3:] = 1e12

    first = provider.issue(
        realized[:3], issue_timestep=2, horizon_steps=3,
        issue_day_index=0, issue_step_of_day=2,
    )
    second = provider.issue(
        poisoned[:3], issue_timestep=2, horizon_steps=3,
        issue_day_index=0, issue_step_of_day=2,
    )

    np.testing.assert_array_equal(first.values, second.values)
    assert first.source_model_version == "historical-forecast-emulator-v2"
    assert first.error_model_id == "sha256:test"
    assert np.all(first.values[:, 10] == 0.0)


def test_forecast_artifact_manifest_is_checksum_addressable():
    artifact = fit_forecast_emulator_v2(
        _pattern(), _pattern(), fit_year=2023, select_year=2024,
        steps_per_day=4, horizon_steps=2,
        source_sha256=SOURCE_HASHES,
    )
    encoded = json.dumps(
        artifact.manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    assert artifact.manifest_sha256 == hashlib.sha256(encoded).hexdigest()


def test_forecast_artifact_round_trip_verifies_payload_checksum(tmp_path):
    artifact = fit_forecast_emulator_v2(
        _pattern(), _pattern(), fit_year=2023, select_year=2024,
        steps_per_day=4, horizon_steps=2,
        source_sha256=SOURCE_HASHES,
    )
    manifest_path = save_forecast_emulator_artifact(artifact, tmp_path)
    loaded = load_forecast_emulator_artifact(manifest_path)

    assert loaded.fingerprint == artifact.fingerprint
    np.testing.assert_array_equal(loaded.climatology, artifact.climatology)

    payload_path = tmp_path / "climatology.npz"
    payload_path.write_bytes(payload_path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_forecast_emulator_artifact(manifest_path)


def test_runtime_rejects_tampered_manifest_identity(tmp_path):
    artifact = fit_forecast_emulator_v2(
        _pattern(), _pattern(), fit_year=2023, select_year=2024,
        steps_per_day=4, horizon_steps=2,
        source_sha256={"2023": "a" * 64, "2024": "b" * 64},
    )
    manifest_path = save_forecast_emulator_artifact(artifact, tmp_path)
    document = json.loads(manifest_path.read_text())
    document["artifact_manifest"]["selected_blend_weight"] = 0.123
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="fingerprint"):
        HistoricalForecastEmulatorV2.from_artifact(manifest_path)


def test_runtime_compares_sources_against_trusted_configuration(tmp_path):
    artifact = fit_forecast_emulator_v2(
        _pattern(), _pattern(), fit_year=2023, select_year=2024,
        steps_per_day=4, horizon_steps=2, source_sha256=SOURCE_HASHES,
    )
    manifest_path = save_forecast_emulator_artifact(artifact, tmp_path)

    with pytest.raises(ValueError, match="trusted source"):
        HistoricalForecastEmulatorV2.from_artifact(
            manifest_path,
            expected_artifact_fingerprint=artifact.fingerprint,
            expected_source_sha256={"2023": "c" * 64, "2024": "d" * 64},
        )


def test_leap_day_is_removed_before_calendar_aligned_selection():
    fit = np.zeros((365, 11), dtype=float)
    fit[:, 1] = np.arange(365)
    select = np.zeros((366, 11), dtype=float)
    select[:59] = fit[:59]
    select[59] = -999.0
    select[60:] = fit[59:]

    artifact = fit_forecast_emulator_v2(
        fit, select, fit_year=2023, select_year=2024,
        steps_per_day=1, horizon_steps=2,
        candidate_blend_weights=(0.0, 1.0), select_is_leap_year=True,
        source_sha256={"2023": "a" * 64, "2024": "b" * 64},
    )

    assert artifact.manifest["leap_day_policy"] == "drop_february_29_from_select"
    assert artifact.manifest["selected_blend_weight"] == pytest.approx(1.0)
    assert artifact.manifest["selection_weighted_mae"] == pytest.approx(0.0)


def test_chengdu_benchmark_uses_frozen_historical_emulator():
    config = load_benchmark_config("smoke")
    env = build_environment(config, config.test_start_day, episode_days=1)
    try:
        env.reset(seed=0)
        forecast = env.issue_forecast()
    finally:
        env.close()

    assert isinstance(env.forecast_provider, HistoricalForecastEmulatorV2)
    assert forecast.source_model_version == "historical-forecast-emulator-v2"
    assert forecast.error_model_id.startswith("sha256:")
    assert env.forecast_provider.blend_weight == pytest.approx(0.75)
