import unittest
from dataclasses import replace

import numpy as np

from glassgym.components.pid import PIDController, PIDLoop
from glassgym.core.types import StepContext
from glassgym.environments.greenlight_env import GreenLightEnv
from RL.utils import build_env_kwargs, load_env_params, load_model_hyperparams


class TestPIDLoop(unittest.TestCase):
    def test_output_is_clipped_to_bounds(self):
        loop = PIDLoop(kp=10.0, ki=0.0, kd=0.0, output_min=0.0, output_max=1.0)
        self.assertEqual(loop.compute(setpoint=10.0, measurement=0.0, dt=1.0), 1.0)

    def test_reset_clears_state(self):
        loop = PIDLoop(kp=1.0, ki=1.0, kd=1.0, output_min=0.0, output_max=10.0)
        loop.compute(setpoint=2.0, measurement=1.0, dt=1.0)
        loop.reset()
        self.assertEqual(loop.integral, 0.0)
        self.assertIsNone(loop.prev_error)


class TestPIDController(unittest.TestCase):
    def setUp(self):
        env_kwargs = load_env_params("GreenLightEnv", "configs/envs/")
        env_kwargs, _ = build_env_kwargs(env_kwargs)
        env_kwargs["normalize_actions"] = False
        self.env = GreenLightEnv(**env_kwargs)
        self.env.reset(seed=42)
        params = load_model_hyperparams("pid", "GreenLightEnv")
        self.controller = PIDController(**params)

    def _ctx(self):
        return StepContext(
            t=self.env.timestep,
            dt=self.env.dt,
            Np=self.env.Np,
            x_prev=self.env.x_prev,
            x=self.env.x,
            u=self.env.u,
            p=self.env.p,
            d=self.env.weather_data,
            hour_of_day=self.env.hour_of_day,
            day_of_year=self.env.day_of_year,
        )

    def test_predict_returns_six_controls(self):
        u = self.controller.predict(self._ctx())
        self.assertEqual(u.shape, (6,))

    def test_predict_returns_controls_in_unit_interval(self):
        u = self.controller.predict(self._ctx())
        self.assertTrue(np.all(u >= 0.0), u)
        self.assertTrue(np.all(u <= 1.0), u)

    def test_low_temperature_increases_heating(self):
        cold_ctx = self._ctx()
        warm_ctx = self._ctx()
        cold_x = cold_ctx.x.copy()
        warm_x = warm_ctx.x.copy()
        cold_x[2] = 14.0
        warm_x[2] = 22.0
        cold_ctx = replace(cold_ctx, x=cold_x)
        warm_ctx = replace(warm_ctx, x=warm_x)

        self.controller.reset()
        cold_u = self.controller.predict(cold_ctx)
        self.controller.reset()
        warm_u = self.controller.predict(warm_ctx)

        self.assertGreater(cold_u[0], warm_u[0])

    def test_low_co2_increases_dosing_during_day(self):
        low_ctx = self._ctx()
        high_ctx = self._ctx()
        low_x = low_ctx.x.copy()
        high_x = high_ctx.x.copy()
        low_x[0] = 0.15
        high_x[0] = 1.2
        low_ctx = replace(low_ctx, x=low_x, hour_of_day=12.0)
        high_ctx = replace(high_ctx, x=high_x, hour_of_day=12.0)

        self.controller.reset()
        low_u = self.controller.predict(low_ctx)
        self.controller.reset()
        high_u = self.controller.predict(high_ctx)

        self.assertGreaterEqual(low_u[1], high_u[1])

    def test_high_rh_increases_ventilation(self):
        humid_ctx = self._ctx()
        dry_ctx = self._ctx()
        humid_x = humid_ctx.x.copy()
        dry_x = dry_ctx.x.copy()
        humid_x[15] = 2500.0
        dry_x[15] = 500.0
        humid_ctx = replace(humid_ctx, x=humid_x)
        dry_ctx = replace(dry_ctx, x=dry_x)

        self.controller.reset()
        humid_u = self.controller.predict(humid_ctx)
        self.controller.reset()
        dry_u = self.controller.predict(dry_ctx)

        self.assertGreater(humid_u[3], dry_u[3])

    def test_short_rollout_does_not_crash(self):
        for _ in range(10):
            ctx = self._ctx()
            u = self.controller.predict(ctx)
            _obs, _reward, terminated, truncated, _info = self.env.step(u.astype(np.float32))
            self.assertFalse(terminated)
            self.assertFalse(truncated)


if __name__ == "__main__":
    unittest.main()


