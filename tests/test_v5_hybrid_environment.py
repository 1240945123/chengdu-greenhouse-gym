import json

import numpy as np
import pandas as pd
import pytest

from experiments.controllers.benchmark_protocol import _load_environment_kwargs
from experiments.controllers.v5_hybrid_environment import (
    DEFAULT_RESIDUAL_PATH,
    V5ResidualClimatePostprocessor,
    build_v5_hybrid_environment,
)
from experiments.reports.chengdu_residual_correction import (
    apply_ridge_residual_corrector,
)
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import rh2vaporDens, vaporDens2pres, vaporPres2rh


class FixedClimatePostprocessor:
    def correct(self, *, raw_state, **_kwargs):
        corrected = np.asarray(raw_state, dtype=float).copy()
        corrected[2] = 21.25
        corrected[15] = 1500.0
        return corrected, {"climate_postprocessed": True, "corrected_air_temperature": 21.25}


def test_state_postprocessor_runs_before_observation_and_transition_info():
    kwargs = _load_environment_kwargs()
    kwargs.update(
        {
            "season_length": 1,
            "weather_scenario_sampler_kwargs": {
                "location": "Chengdu",
                "growth_year": 2025,
                "start_day": 226,
            },
            "state_postprocessor": FixedClimatePostprocessor(),
            "nu": 8,
            "u_min": [0.0] * 8,
            "u_max": [1.0] * 8,
            "controlled_inputs": ["uVent", "uBlScr", "uPadFan", "uPadPump"],
        }
    )
    env = GreenLightEnv(**kwargs)
    env.reset(seed=0)

    observation, reward, _terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )

    assert not truncated
    assert np.isfinite(reward)
    assert env.x[2] == 21.25
    assert observation["IndoorClimateObservations"][1] == 21.25
    assert info["climate_postprocessed"] is True
    assert info["corrected_air_temperature"] == 21.25


def test_v5_residual_correction_scales_hourly_gain_and_preserves_rh_state():
    model = {
        "schema_version": "ridge_residual_v1",
        "feature_columns": ["pred_air_temperature", "uFan"],
        "feature_means": [0.0, 0.0],
        "feature_scales": [1.0, 1.0],
        "target_names": ["air_temperature", "relative_humidity"],
        "coefficients": {
            "air_temperature": [4.0, 0.0, 0.0],
            "relative_humidity": [8.0, 0.0, 0.0],
        },
        "gain": 1.0,
    }
    processor = V5ResidualClimatePostprocessor(model)
    raw = np.zeros(28, dtype=float)
    raw[2] = 20.0
    raw[15] = vaporDens2pres(20.0, rh2vaporDens(20.0, 50.0))

    corrected, info = processor.correct(
        previous_state=raw.copy(),
        raw_state=raw,
        controls=np.zeros(8, dtype=float),
        disturbance=np.array([100.0, 15.0, 1000.0, 700.0, 2.0, 10.0, 18.0, 0, 1, 1, 0]),
        dt_seconds=900.0,
        hour_of_day=12.0,
        day_of_year=150.0,
    )

    assert info["residual_gain_scale"] == pytest.approx(0.25)
    assert corrected[2] == pytest.approx(21.0)
    assert info["corrected_relative_humidity"] == pytest.approx(52.0)
    assert vaporPres2rh(corrected[2], corrected[15]) == pytest.approx(52.0)


@pytest.mark.parametrize("hour", [0.0, 6.25, 12.0, 19.75, 23.5])
def test_v5_fast_residual_prediction_matches_authoritative_dataframe_path(hour):
    model = json.loads(DEFAULT_RESIDUAL_PATH.read_text(encoding="utf-8"))
    processor = V5ResidualClimatePostprocessor(model)
    raw_temperature = 26.4
    raw_rh = 78.2
    outdoor_temperature = 22.1
    outdoor_rh = 84.3
    radiation = 417.5
    wind = 1.8
    controls = np.array([0.0, 0.0, 0.35, 0.42, 0.0, 0.18, 0.27, 0.31])
    row = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2001-01-01")
                + pd.to_timedelta(hour, unit="h"),
                "pred_air_temperature": raw_temperature,
                "pred_relative_humidity": raw_rh,
                "d_air_temperature": outdoor_temperature,
                "d_relative_humidity": outdoor_rh,
                "d_global_radiation": radiation,
                "d_wind_speed": wind,
                "uRoofVent": controls[3],
                "uFan": controls[6],
                "uPad": controls[7],
                "uBlScr": controls[5],
                "uThScr": controls[2],
            }
        ]
    )
    gain = 0.1875
    expected = apply_ridge_residual_corrector(row, model, gain=gain).iloc[0]

    actual_temperature, actual_rh, fallback = processor.predict_corrected_climate(
        raw_temperature=raw_temperature,
        raw_relative_humidity=raw_rh,
        outdoor_temperature=outdoor_temperature,
        outdoor_relative_humidity=outdoor_rh,
        radiation=radiation,
        wind_speed=wind,
        controls=controls,
        hour_of_day=hour,
        applied_gain=gain,
    )

    assert actual_temperature == pytest.approx(expected["pred_air_temperature"], abs=1e-12)
    assert actual_rh == pytest.approx(expected["pred_relative_humidity"], abs=1e-12)
    assert fallback is bool(expected["residual_fallback"])


def test_real_v5_hybrid_environment_has_eight_controls_and_two_safe_actions():
    env = build_v5_hybrid_environment(
        growth_year=2025,
        start_day=226,
        episode_days=1,
        dt_seconds=900,
    )
    observation, _reset_info = env.reset(seed=0)

    assert env.nu == 8
    assert len(env.p) == 225
    assert env.action_space.shape == (4,)
    assert observation["ControlObservations"].shape == (8,)

    _observation, reward, _terminated, truncated, info = env.step(
        np.zeros(4, dtype=np.float32)
    )

    assert not truncated
    assert np.isfinite(reward)
    assert info["climate_postprocessed"] is True
    assert np.isfinite(env.x).all()
    assert np.allclose(env.u[[0, 1, 2, 4]], 0.0)
    assert env.u[3] > 0.0
    assert env.u[6] > 0.0
