import unittest

import casadi as ca
import numpy as np

from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import init_state
from RL.utils import build_env_kwargs, load_env_params


class ChengduPhysicsModelTest(unittest.TestCase):
    def test_backend_integrator_advances_state_without_nan(self):
        env_kwargs = load_env_params("GreenLightEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env_kwargs["model_backend"] = "ChengduPhysics"
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=1)

        _obs, _reward, _terminated, truncated, _info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        self.assertFalse(truncated)
        self.assertFalse(np.isnan(env.x).any())
        self.assertEqual(env.x.shape, (28,))

    def test_chengdu_env_config_resets_and_steps(self):
        env_kwargs = load_env_params("ChengduSingleGreenhouseEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env = GreenLightEnv(**env_kwargs)

        obs, info = env.reset(seed=1)
        _obs, reward, _terminated, truncated, _info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))

        self.assertEqual(info["scenario"]["location"], "Chengdu")
        self.assertEqual(info["scenario"]["growth_year"], 2026)
        self.assertFalse(truncated)
        self.assertIsInstance(float(reward), float)

    def test_chengdu_env_survives_300_zero_control_steps(self):
        env_kwargs = load_env_params("ChengduSingleGreenhouseEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=1)

        for _ in range(300):
            _obs, _reward, _terminated, truncated, _info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            self.assertFalse(truncated)
            self.assertFalse(np.isnan(env.x).any())

    def test_heating_increases_air_temperature_tendency(self):
        from glassgym.models.ChengduPhysics.ode import ODE

        d = np.array([300.0, 10.0, 900.0, 720.0, 0.5, 8.0, 10.0, 0.0, 1.0, 1.0])
        x = init_state(d)
        p = np.zeros(208, dtype=np.float64)
        u_off = np.zeros(6, dtype=np.float64)
        u_heat = np.zeros(6, dtype=np.float64)
        u_heat[0] = 1.0

        dx_off = ca.Function("dx_off", [], [ODE(ca.DM(x), ca.DM(u_off), ca.DM(d), ca.DM(p))])()["o0"].full().flatten()
        dx_heat = ca.Function("dx_heat", [], [ODE(ca.DM(x), ca.DM(u_heat), ca.DM(d), ca.DM(p))])()["o0"].full().flatten()

        self.assertGreater(dx_heat[2], dx_off[2])


if __name__ == "__main__":
    unittest.main()
