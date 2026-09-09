import unittest
import numpy as np
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import load_env_params, build_env_kwargs

class TestGreenLightEnv(unittest.TestCase):
    def setUp(self):
        # Set up environment parameters
        self.env_id = "GreenLightEnv"
        self.env_config_path = "configs/envs/"
        self.env_kwargs = load_env_params(self.env_id, self.env_config_path)
        self.env_kwargs, _ = build_env_kwargs(self.env_kwargs)

        # Initialize environment
        self.env = GreenLightEnv(**self.env_kwargs)
        self.env.reset(seed=42)

    def test_reward_normalisation(self):
        """Test the configured reward function's normalization bounds."""
        self.env.reset(seed=42)
        max_reward = 0.328 * 900 * 1e-6 / 0.065 * 1.6
        self.assertAlmostEqual(self.env.reward_fn.max_profit, max_reward)
        self.assertEqual(self.env.reward_fn.variable_costs, 0)

    def test_reset(self):
        """Test environment reset functionality"""
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, dict)
        self.assertTrue(self.env.observation_space.contains(obs))

        # Check initial state
        self.assertEqual(self.env.timestep, 0)
        self.assertFalse(self.env.terminated)

    def test_step(self):
        """Test environment step functionality"""
        self.env.reset()
        
        # Take a random action
        action = self.env.action_space.sample()
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertIsInstance(obs, dict)
        self.assertTrue(self.env.observation_space.contains(obs))

        # Check reward is float
        self.assertIsInstance(reward, (int, float))
        
        # Check timestep increment
        self.assertEqual(self.env.timestep, 1)

    def test_reward(self):
        """Test reward functionality"""
        self.env.reset()
        action = np.ones(self.env.nu)*-1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsInstance(reward, (int, float))
        self.assertTrue(np.isfinite(reward))
        self.assertIn("variable_costs", info)
        

    def test_action_scaling(self):
        """Test action scaling functionality"""
        action = self.env.action_space.sample()
        scaled_action = self.env.action_scheme.to_full_control_input(action)

        # Check scaled action bounds
        self.assertTrue(np.all(scaled_action >= self.env.u_min))
        self.assertTrue(np.all(scaled_action <= self.env.u_max))

    def test_episode_termination(self):
        """Test if episode terminates correctly"""
        self.env.reset()
        
        self.env.timestep = self.env.N - 1
        action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, _ = self.env.step(action)

        self.assertTrue(terminated)

    def test_reset_accepts_local_initial_crop_state_override(self):
        self.env.reset(
            seed=42,
            options={
                "initial_crop_state_mg_m2": {
                    "cBuf": 10.0,
                    "cLeaf": 20.0,
                    "cStem": 30.0,
                    "cFruit": 0.0,
                    "tCanSum": 123.0,
                }
            },
        )

        np.testing.assert_allclose(self.env.x[22:26], [10.0, 20.0, 30.0, 0.0])
        self.assertEqual(self.env.x[26], 123.0)
        self.env.reset(seed=42)
        np.testing.assert_allclose(
            self.env.x[22:26], [0.0, 9.5283e4, 2.5107e5, 5.5338e4]
        )

    def test_reset_rejects_invalid_initial_crop_state_override(self):
        for override, message in (
            ({"cFruit": -1.0}, "finite and non-negative"),
            ({"unknown": 1.0}, "unknown crop state"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    self.env.reset(
                        seed=42,
                        options={"initial_crop_state_mg_m2": override},
                    )

if __name__ == '__main__':
    unittest.main()
    


