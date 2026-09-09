import unittest
from dataclasses import replace

import numpy as np

from glassgym.components.mpc import CandidateActionGenerator, LightweightMPCController
from experiments.controllers.tune_benchmark_classical import _mpc_step_context
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import satVp
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


class TestCandidateActionGenerator(unittest.TestCase):
    def test_includes_current_control(self):
        generator = CandidateActionGenerator(
            levels={
                "uBoil": [0.0, 0.5, 1.0],
                "uCO2": [0.0, 0.5],
                "uVent": [0.0, 0.5],
                "uLamp": [0.0, 1.0],
            },
            include_current=True,
            include_conservative=True,
        )
        current = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
        candidates = generator.generate(current)
        self.assertTrue(any(np.allclose(c, current) for c in candidates))

    def test_candidates_are_clipped_to_unit_interval(self):
        generator = CandidateActionGenerator(
            levels={"uBoil": [-1.0, 2.0], "uCO2": [2.0], "uVent": [-1.0], "uLamp": [2.0]},
            include_current=True,
            include_conservative=True,
        )
        current = np.array([-5, 2, 0, 3, -1, 8], dtype=np.float32)
        candidates = generator.generate(current)
        self.assertTrue(np.all(candidates >= 0.0), candidates)
        self.assertTrue(np.all(candidates <= 1.0), candidates)


class TestLightweightMPCController(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params("GreenLightEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env_kwargs["normalize_actions"] = False
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        params["horizon_steps"] = 1
        params["candidate_levels"] = {
            "uBoil": [0.0, 0.5],
            "uCO2": [0.0, 0.5],
            "uVent": [0.0, 0.5],
            "uLamp": [0.0],
        }
        self.controller = LightweightMPCController(**params)

    def _ctx(self):
        return _mpc_step_context(self.env)

    def test_predict_returns_six_controls(self):
        u = self.controller.predict(self._ctx())
        self.assertEqual(u.shape, (6,))

    def test_predict_returns_controls_in_unit_interval(self):
        u = self.controller.predict(self._ctx())
        self.assertTrue(np.all(u >= 0.0), u)
        self.assertTrue(np.all(u <= 1.0), u)

    def test_predict_requires_restricted_controller_context(self):
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        controller = LightweightMPCController(**params)
        with self.assertRaises(TypeError):
            controller.predict(object())

    def test_controller_owns_frozen_surrogate_without_environment_binding(self):
        self.assertFalse(hasattr(self.controller, "env"))
        self.assertFalse(hasattr(self.controller, "transition_model"))
        self.assertEqual(
            self.controller.controller_model.config.schema_version,
            "controller-model-v2",
        )

    def test_short_rollout_does_not_crash(self):
        for _ in range(3):
            ctx = self._ctx()
            u = self.controller.predict(ctx)
            _obs, _reward, terminated, truncated, _info = self.env.step(u.astype(np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)

    def test_low_temperature_selects_some_heating(self):
        ctx = self._ctx()
        cold_ctx = replace(
            ctx,
            indoor_temperature=12.0,
            u=np.zeros(6, dtype=np.float32),
        )

        u = self.controller.predict(cold_ctx)

        self.assertGreater(u[0], 0.0)

    def test_benchmark_slew_rate_is_applied_inside_rollout(self):
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        params["max_action_change"] = 0.1
        controller = LightweightMPCController(**params)

        applied = controller._apply_slew_rate(
            previous=np.zeros(6),
            target=np.ones(6),
        )

        np.testing.assert_allclose(applied, np.full(6, 0.1))

    def test_benchmark_stage_score_matches_climate_reward_terms(self):
        params = load_model_hyperparams("mpc", "GreenLightEnv")
        params["comfort_objective"] = {
            "temp_day_low": 20.0,
            "temp_day_high": 28.0,
            "temp_night_low": 16.0,
            "temp_night_high": 24.0,
            "rh_low": 60.0,
            "rh_high": 85.0,
            "day_start": 6.0,
            "day_end": 20.0,
            "temperature_weight": 1.0,
            "humidity_weight": 1.0,
            "lamp_weight": 0.1,
            "effort_weight": 0.2,
            "action_change_weight": 0.3,
        }
        controller = LightweightMPCController(**params)
        x = np.zeros(28)
        x[2] = 30.0
        x[15] = 0.9 * satVp(30.0)
        d = np.zeros(10)
        d[8] = 1.0

        idle_score = controller._stage_score(
            x=x,
            prev_x=x,
            u=np.zeros(6),
            prev_u=np.zeros(6),
            d=d,
        )
        disabled_controls = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        disabled_score = controller._stage_score(
            x=x,
            prev_x=x,
            u=disabled_controls,
            prev_u=disabled_controls,
            d=d,
        )

        self.assertAlmostEqual(idle_score, -0.325)
        self.assertAlmostEqual(disabled_score, idle_score)

    def test_rollout_scores_the_post_transition_hour(self):
        ctx = self._ctx()
        self.assertAlmostEqual(
            self.controller._prediction_hour(ctx, horizon_index=0),
            (ctx.hour_of_day + ctx.dt / 3600.0) % 24.0,
        )


if __name__ == "__main__":
    unittest.main()


