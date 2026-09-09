from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class IssuedForecastV2:
    schema_version: str
    issue_timestep: int
    valid_timesteps: np.ndarray
    lead_steps: np.ndarray
    values: np.ndarray
    source_model_version: str
    error_model_id: str | None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64).copy()
        valid = np.asarray(self.valid_timesteps, dtype=np.int64).copy()
        leads = np.asarray(self.lead_steps, dtype=np.int64).copy()
        if self.schema_version != "forecast-provider-v2":
            raise ValueError("Unsupported forecast schema version")
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Forecast values must be a finite two-dimensional array")
        if valid.shape != (len(values),) or leads.shape != (len(values),):
            raise ValueError("Forecast valid times and leads must match its horizon")
        if not np.array_equal(leads, np.arange(1, len(values) + 1)):
            raise ValueError("Forecast lead steps must start at one")
        if not np.array_equal(valid, int(self.issue_timestep) + leads):
            raise ValueError("Forecast valid times must follow its issue time")
        values.setflags(write=False)
        valid.setflags(write=False)
        leads.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_timesteps", valid)
        object.__setattr__(self, "lead_steps", leads)


class PersistenceForecastProviderV2:
    schema_version = "forecast-provider-v2"

    def __init__(
        self,
        source_model_version: str = "persistence-v1",
        error_model_id: str | None = None,
    ) -> None:
        self.source_model_version = str(source_model_version)
        self.error_model_id = error_model_id

    def issue(
        self,
        history: np.ndarray,
        *,
        issue_timestep: int,
        horizon_steps: int,
        issue_day_index: int | None = None,
        issue_step_of_day: int | None = None,
    ) -> IssuedForecastV2:
        history_values = np.asarray(history, dtype=np.float64)
        issue = int(issue_timestep)
        horizon = int(horizon_steps)
        if issue < 0 or history_values.ndim != 2 or len(history_values) != issue + 1:
            raise ValueError("Forecast provider requires exact issue-time history")
        if horizon <= 0:
            raise ValueError("horizon_steps must be positive")
        if not np.all(np.isfinite(history_values)):
            raise ValueError("Forecast history must be finite")
        leads = np.arange(1, horizon + 1, dtype=np.int64)
        return IssuedForecastV2(
            schema_version=self.schema_version,
            issue_timestep=issue,
            valid_timesteps=issue + leads,
            lead_steps=leads,
            values=np.repeat(history_values[-1:], horizon, axis=0),
            source_model_version=self.source_model_version,
            error_model_id=self.error_model_id,
        )


class HistoricalForecastEmulatorV2(PersistenceForecastProviderV2):
    def __init__(
        self,
        *,
        climatology: np.ndarray,
        blend_weight: float,
        steps_per_day: int,
        source_model_version: str,
        error_model_id: str,
    ) -> None:
        super().__init__(source_model_version, error_model_id)
        values = np.asarray(climatology, dtype=np.float64).copy()
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("Historical forecast climatology must be a finite matrix")
        if not 0.0 <= float(blend_weight) <= 1.0:
            raise ValueError("Forecast blend_weight must be in [0, 1]")
        if int(steps_per_day) <= 0:
            raise ValueError("Forecast steps_per_day must be positive")
        values.setflags(write=False)
        self.climatology = values
        self.blend_weight = float(blend_weight)
        self.steps_per_day = int(steps_per_day)
        self.artifact_manifest: dict[str, object] | None = None
        self.artifact_fingerprint: str | None = None

    @classmethod
    def from_artifact(
        cls,
        manifest_path: str | Path,
        *,
        expected_artifact_fingerprint: str | None = None,
        expected_source_sha256: dict[str, str] | None = None,
    ) -> "HistoricalForecastEmulatorV2":
        path = Path(manifest_path)
        document = json.loads(path.read_text(encoding="utf-8"))
        payload_path = path.parent / str(document["payload_file"])
        digest = hashlib.sha256(payload_path.read_bytes()).hexdigest()
        if digest != document["payload_sha256"]:
            raise ValueError("Forecast climatology payload checksum mismatch")
        manifest = document["artifact_manifest"]
        if (
            manifest.get("schema_version") != "historical-forecast-emulator-v2"
            or manifest.get("fit_year") != 2023
            or manifest.get("select_year") != 2024
            or manifest.get("retrospective_year") != 2025
            or manifest.get("fit_role") != "fit"
            or manifest.get("select_role") != "select"
        ):
            raise ValueError("Forecast artifact roles or schema are invalid")
        source_hashes = manifest.get("source_sha256", {})
        if set(source_hashes) != {"2023", "2024"} or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value.lower())
            for value in source_hashes.values()
        ):
            raise ValueError("Forecast artifact source provenance is invalid")
        with np.load(payload_path, allow_pickle=False) as payload:
            climatology = np.asarray(payload["climatology"], dtype=np.float64)
        manifest_digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        artifact_digest = hashlib.sha256()
        artifact_digest.update(manifest_digest.encode())
        artifact_digest.update(np.ascontiguousarray(climatology).tobytes())
        recomputed_fingerprint = artifact_digest.hexdigest()
        if recomputed_fingerprint != document.get("artifact_fingerprint"):
            raise ValueError("Forecast artifact fingerprint mismatch")
        if (
            expected_artifact_fingerprint is not None
            and recomputed_fingerprint != expected_artifact_fingerprint
        ):
            raise ValueError("Forecast artifact does not match trusted fingerprint")
        if (
            expected_source_sha256 is not None
            and source_hashes != expected_source_sha256
        ):
            raise ValueError("Forecast artifact does not match trusted source hashes")
        provider = cls(
            climatology=climatology,
            blend_weight=float(manifest["selected_blend_weight"]),
            steps_per_day=int(manifest["steps_per_day"]),
            source_model_version="historical-forecast-emulator-v2",
            error_model_id=f"sha256:{recomputed_fingerprint}",
        )
        provider.artifact_manifest = dict(manifest)
        provider.artifact_fingerprint = recomputed_fingerprint
        return provider

    def issue(
        self,
        history: np.ndarray,
        *,
        issue_timestep: int,
        horizon_steps: int,
        issue_day_index: int | None = None,
        issue_step_of_day: int | None = None,
    ) -> IssuedForecastV2:
        history_values = np.asarray(history, dtype=np.float64)
        issue = int(issue_timestep)
        horizon = int(horizon_steps)
        if history_values.ndim != 2 or len(history_values) != issue + 1:
            raise ValueError("Forecast provider requires exact issue-time history")
        if not np.all(np.isfinite(history_values)) or horizon <= 0:
            raise ValueError("Forecast history and horizon are invalid")
        if issue_day_index is None or issue_step_of_day is None:
            raise ValueError("Historical emulator requires issue calendar position")
        base = int(issue_day_index) * self.steps_per_day + int(issue_step_of_day)
        leads = np.arange(1, horizon + 1, dtype=np.int64)
        analog = self.climatology[(base + leads) % len(self.climatology)]
        persistence = np.repeat(history_values[-1:], horizon, axis=0)
        values = (1.0 - self.blend_weight) * persistence + self.blend_weight * analog
        if values.shape[1] > 10:
            values[:, 10] = max(float(history_values[-1, 10]), 0.0)
        return IssuedForecastV2(
            schema_version=self.schema_version,
            issue_timestep=issue,
            valid_timesteps=issue + leads,
            lead_steps=leads,
            values=values,
            source_model_version=self.source_model_version,
            error_model_id=self.error_model_id,
        )
