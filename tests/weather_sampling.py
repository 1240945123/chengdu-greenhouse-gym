import unittest

import numpy as np

from glassgym.components.weather import WeatherRepository
from glassgym.environments.greenlight_env import GreenLightEnv
from glassgym.environments.utils import load_weather_data
from RL.utils import build_env_kwargs, load_env_params, make_eval_vec_env, make_training_vec_env


def load_test_env_kwargs():
    env_kwargs = load_env_params("GreenLightEnv", "configs/envs/")
    env_kwargs, _eval_scenarios = build_env_kwargs(env_kwargs)
    return env_kwargs


def reference_weather_repository():
    return WeatherRepository(
        weather_data_dir="data/processed/amsterdam_reference/weather/",
        load_weather_data_fn=load_weather_data,
    )


class TestWeatherSampling(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_test_env_kwargs()
        env_kwargs["weather_scenario_sampler"] = "fixed"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "location": "Amsterdam",
            "growth_year": 2010,
            "start_day": 1,
        }
        env_kwargs["weather_repository"] = reference_weather_repository()

        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)

    def test_weather_fixed_scenario_sampler(self):
        scenario = self.env.weather_scenario_sampler.sample(np.random.default_rng(0))
        self.assertEqual(scenario.location, "Amsterdam")
        self.assertEqual(scenario.growth_year, 2010)
        self.assertEqual(scenario.start_day, 1)
        self.env.reset(seed=0)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2010)
        self.assertEqual(self.env.start_day, 1)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        weather_data_1 = self.env.weather_data.copy()
        self.env.reset(seed=0)
        self.assertTrue(np.allclose(weather_data_1, self.env.weather_data))

    def test_weather_random_scenario_sampler(self):
        env_kwargs = load_test_env_kwargs()
        env_kwargs["weather_scenario_sampler"] = "random"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "locations": ["Amsterdam"],
            "growth_years": [2008, 2009, 2010],
            "start_days": range(1, 3),
        }
        env_kwargs["weather_repository"] = reference_weather_repository()
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=123)
        scenario = env.weather_scenario_sampler.sample(np.random.default_rng(0))
        self.assertEqual(scenario.location, "Amsterdam")
        self.assertIn(scenario.growth_year, [2008, 2009, 2010])
        self.assertIn(scenario.start_day, [1, 2])

    def test_reset_with_scenario(self):
        options = {
            "scenario": {
                "location": "Amsterdam",
                "growth_year": 2009,
                "start_day": 2,
            }
        }
        self.env.reset(seed=0, options=options)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2009)
        self.assertEqual(self.env.start_day, 2)
        self.assertEqual(self.env.weather_data.shape[-1], 10)
        weather_data_1 = self.env.weather_data.copy()
        self.env.reset(seed=0, options=options)
        self.assertTrue(np.allclose(weather_data_1, self.env.weather_data))
        self.env.reset(seed=0)
        self.assertEqual(self.env.location, "Amsterdam")
        self.assertEqual(self.env.growth_year, 2010)
        self.assertEqual(self.env.start_day, 1)
        self.assertFalse(np.allclose(weather_data_1, self.env.weather_data))

    def test_cycling_scenario_sampler(self):
        env_kwargs = load_test_env_kwargs()
        env_kwargs["weather_scenario_sampler"] = "cycling"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "scenarios": [
                dict(location="Amsterdam", growth_year=2010, start_day=1),
                dict(location="Amsterdam", growth_year=2009, start_day=2),
                dict(location="Amsterdam", growth_year=2008, start_day=3),
            ]
        }
        env_kwargs["weather_repository"] = reference_weather_repository()
        env = GreenLightEnv(**env_kwargs)
        env.reset(seed=123)
        self.assertEqual(env.growth_year, 2010)
        env.reset(seed=0)
        self.assertEqual(env.growth_year, 2009)
        weather_data_1 = env.weather_data.copy()

        env.reset(seed=0, options={"scenario_index": 1})
        self.assertEqual(env.growth_year, 2009)
        self.assertTrue(np.allclose(weather_data_1, env.weather_data))
        env.reset(seed=0, options={"scenario_index": 2})
        self.assertEqual(env.growth_year, 2008)

    def test_vectorized_weather_sampling(self):
        env_kwargs = load_test_env_kwargs()
        env_kwargs["weather_scenario_sampler"] = "random"
        env_kwargs["weather_scenario_sampler_kwargs"] = {
            "locations": ["Amsterdam"],
            "growth_years": [2008, 2009, 2010],
            "start_days": range(0, 3),
        }
        env_kwargs["weather_repository"] = reference_weather_repository()
        env = make_training_vec_env(
            env_id="GreenLightEnv",
            env_kwargs=env_kwargs,
            seed=123,
            n_envs=2,
        )

        env.reset()
        self.assertEqual(
            env.env_method("get_wrapper_attr", "location"),
            ["Amsterdam"] * 2,
        )
        for year in env.env_method("get_wrapper_attr", "growth_year"):
            self.assertIn(year, [2008, 2009, 2010])
        env.close()

    def test_eval_env_vectorized_weather_sampling(self):
        env_kwargs = load_test_env_kwargs()
        env_kwargs["weather_repository"] = reference_weather_repository()
        eval_scenarios = [
            dict(location="Amsterdam", growth_year=2008, start_day=1),
            dict(location="Amsterdam", growth_year=2009, start_day=2),
        ]

        eval_env = make_eval_vec_env(
            env_id="GreenLightEnv",
            env_kwargs=env_kwargs,
            seed=123,
            n_envs=2,
            eval_weather_scenarios=eval_scenarios,
        )
        eval_env.reset()

        for i, env_sampler in enumerate(eval_env.env_method("get_wrapper_attr", "weather_scenario_sampler")):
            scenario = env_sampler.sample(np.random.default_rng(0))
            self.assertEqual(scenario.location, eval_scenarios[i]["location"])
            self.assertEqual(scenario.growth_year, eval_scenarios[i]["growth_year"])
            self.assertEqual(scenario.start_day, eval_scenarios[i]["start_day"])
        eval_env.close()


if __name__ == "__main__":
    unittest.main()
