import casadi as ca
import numpy as np
import pytest
import yaml
from pathlib import Path

from glassgym.configs.default_params import init_default_params
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import init_state
from glassgym.models.ChengduPhysics.ode import ODE as chengdu_ode
from glassgym.models.ChengduPhysicsLegacy.ode import ODE as chengdu_legacy_ode
from glassgym.models.GreenLight.ode import ODE as greenlight_ode
from glassgym.models.GreenLight.crop import crop_fluxes
from experiments.controllers.benchmark_protocol import load_benchmark_config
from experiments.controllers.tune_benchmark_classical import build_environment
from RL.utils import build_env_kwargs, load_env_params


def _evaluate(expr: ca.SX | ca.DM) -> np.ndarray:
    return ca.Function("evaluate_crop_derivatives", [], [expr])()["o0"].full().ravel()


def test_chengdu_uses_greenlight_vanthoor_crop_derivatives():
    disturbance = np.array(
        [350.0, 22.0, 1800.0, 720.0, 1.2, 15.0, 18.0, 12.0, 1.0, 1.0],
        dtype=np.float64,
    )
    state = init_state(disturbance)
    control = np.array([0.0, 0.0, 0.2, 0.3, 0.0, 0.0], dtype=np.float64)
    params = init_default_params(208)

    chengdu = _evaluate(chengdu_ode(ca.DM(state), ca.DM(control), ca.DM(disturbance), ca.DM(params)))
    greenlight = _evaluate(greenlight_ode(ca.DM(state), ca.DM(control), ca.DM(disturbance), ca.DM(params)))

    np.testing.assert_allclose(chengdu[22:28], greenlight[22:28], rtol=1e-10, atol=1e-10)


def test_crop_organ_allocation_is_zero_when_carbohydrate_buffer_is_empty():
    disturbance = np.array(
        [0.0, 18.0, 1400.0, 720.0, 0.0, 12.0, 16.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    state = init_state(disturbance)
    state[22] = 0.0
    fluxes = crop_fluxes(
        ca.DM(state), ca.DM(np.zeros(6)), ca.DM(disturbance), ca.DM(init_default_params(208))
    )

    assert float(fluxes["buffer_to_leaf"]) == pytest.approx(0.0, abs=1e-12)
    assert float(fluxes["buffer_to_stem"]) == pytest.approx(0.0, abs=1e-12)
    assert float(fluxes["buffer_to_fruit"]) == pytest.approx(0.0, abs=1e-12)


def test_chengdu_canopy_temperature_has_negative_feedback_above_solar_target():
    disturbance = np.array(
        [600.0, 30.0, 1800.0, 720.0, 1.0, 20.0, 20.0, 15.0, 1.0, 1.0],
        dtype=np.float64,
    )
    state = init_state(disturbance)
    state[2] = 30.0
    state[4] = 60.0
    derivatives = _evaluate(chengdu_ode(
        ca.DM(state), ca.DM(np.zeros(6)), ca.DM(disturbance), ca.DM(init_default_params(208))
    ))

    assert derivatives[4] < 0.0


def test_environment_reports_explicit_vanthoor_harvest_diagnostics():
    env_kwargs = load_env_params("ChengduControllerBenchmark", "configs/envs/")
    env_kwargs, _ = build_env_kwargs(env_kwargs)
    env_kwargs.update(
        {
            "nu": 8,
            "u_min": [0.0] * 8,
            "u_max": [1.0] * 8,
            "controlled_inputs": ["uVent", "uBlScr", "uPadFan", "uPadPump"],
        }
    )
    env = GreenLightEnv(**env_kwargs)
    env.reset(seed=1)

    try:
        _obs, _reward, _terminated, truncated, info = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
    finally:
        env.close()

    assert not truncated
    assert info["crop_model"] == "Vanthoor2011_GreenLight"
    assert info["fruit_harvest_rate_mg_m2_s"] >= 0.0
    assert info["fruit_allocation_rate_mg_m2_s"] >= 0.0
    assert info["harvested_dry_matter_mg_m2"] >= 0.0
    assert info["allocated_fruit_dry_matter_mg_m2"] >= 0.0
    assert info["dry_matter_fraction"] == pytest.approx(0.0627)
    assert info["c_fruit_previous_mg_m2"] >= 0.0
    for name in ("c_buffer_mg_m2", "c_leaf_mg_m2", "c_stem_mg_m2", "c_fruit_mg_m2"):
        assert np.isfinite(info[name])


def test_crop_config_records_literature_provenance_and_external_validation_only():
    path = Path("configs/crops/vanthoor_tomato.yml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["model"]["source_doi"] == "10.1016/j.biosystemseng.2011.08.005"
    assert config["implementation"]["greenlight_doi"] == "10.1016/j.biosystemseng.2020.03.010"
    assert config["parameterization"]["target_site_harvest_calibrated"] is False
    assert config["parameterization"]["dry_matter_fraction"] == pytest.approx(0.0627)
    assert config["external_validation"]["fresh_yield_kg_m2"] == pytest.approx(
        [10.1625, 10.314]
    )
    assert config["external_validation"]["use"] == "plausibility_only"


def test_chengdu_calibration_does_not_overwrite_greenlight_crop_parameters():
    benchmark = load_benchmark_config("smoke")
    env = build_environment(benchmark, benchmark.test_start_day, episode_days=1)
    try:
        assert len(env.p) == 216
        assert env.p[206] == pytest.approx(0.0627)
        assert env.p[208] == pytest.approx(4.0)
        assert env.p[215] == pytest.approx(1.6)
    finally:
        env.close()


def test_primary_chengdu_backend_removes_legacy_fan_pad_coupling_from_roof_vent():
    disturbance = np.array(
        [0.0, 20.0, 1800.0, 720.0, 1.0, 15.0, 18.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )
    state = init_state(disturbance)
    state[2] = 30.0
    controls = np.zeros(6, dtype=np.float64)
    controls[3] = 1.0
    params = init_default_params(216)

    primary = _evaluate(
        chengdu_ode(ca.DM(state), ca.DM(controls), ca.DM(disturbance), ca.DM(params))
    )
    legacy = _evaluate(
        chengdu_legacy_ode(ca.DM(state), ca.DM(controls), ca.DM(disturbance), ca.DM(params))
    )

    assert primary[2] > legacy[2]
    expected_removed_cooling = 18_000.0 * (30.0 - 20.0) / (
        1.2 * 1005.0 * (192.0 * 4.7)
    )
    assert primary[2] - legacy[2] == pytest.approx(expected_removed_cooling)
