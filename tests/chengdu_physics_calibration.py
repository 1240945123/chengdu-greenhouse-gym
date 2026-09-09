import unittest

import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from glassgym.environments.utils import rh2vaporDens, vaporDens2pres


class ChengduPhysicsCalibrationTest(unittest.TestCase):
    def test_evaluate_one_step_predictions_reports_target_metrics(self):
        from experiments.reports.evaluate_chengdu_physics import evaluate_one_step_predictions

        trajectories = pd.DataFrame(
            {
                "x_air_temperature": [20.0, 21.0],
                "x_relative_humidity": [70.0, 72.0],
                "x_co2_concentration": [650.0, 660.0],
                "uBoil": [0.0, 0.0],
                "uCO2": [0.0, 0.0],
                "uThScr": [0.0, 0.0],
                "uVent": [0.0, 0.0],
                "uLamp": [0.0, 0.0],
                "uBlScr": [0.0, 0.0],
                "d_global_radiation": [0.0, 100.0],
                "d_air_temperature": [19.5, 20.5],
                "d_relative_humidity": [75.0, 74.0],
                "d_wind_speed": [0.5, 0.8],
                "next_x_air_temperature": [20.2, 21.1],
                "next_x_relative_humidity": [70.5, 71.8],
                "next_x_co2_concentration": [648.0, 657.0],
            }
        )

        metrics, predictions = evaluate_one_step_predictions(trajectories, sample_limit=2)

        self.assertEqual(len(predictions), 2)
        for key in [
            "air_temperature_mae",
            "relative_humidity_mae",
            "mean_mae",
            "num_samples",
        ]:
            self.assertIn(key, metrics)
        self.assertNotIn("co2_concentration_mae", metrics)
        self.assertEqual(metrics["num_samples"], 2)
        self.assertTrue(np.isfinite(predictions["pred_air_temperature"]).all())
        self.assertTrue(np.isfinite(predictions["pred_co2_concentration"]).all())

    def test_chengdu_parameter_vector_changes_air_temperature_tendency(self):
        from glassgym.models.ChengduPhysics.ode import ODE
        import casadi as ca

        x = np.zeros(28, dtype=np.float64)
        x[0] = 650.0
        x[2] = 20.0
        x[4] = 20.0
        x[9] = 20.0
        x[15] = vaporDens2pres(20.0, rh2vaporDens(20.0, 70.0))

        u = np.zeros(6, dtype=np.float64)
        d = np.array([300.0, 10.0, 900.0, 720.0, 0.5, 4.0, 10.0, 0.0, 1.0, 1.0])
        base_p = np.ones(216, dtype=np.float64)
        high_solar_p = base_p.copy()
        high_solar_p[209] = 2.0

        dx_base = ca.Function("dx_base", [], [ODE(ca.DM(x), ca.DM(u), ca.DM(d), ca.DM(base_p))])()["o0"].full().flatten()
        dx_high = ca.Function("dx_high", [], [ODE(ca.DM(x), ca.DM(u), ca.DM(d), ca.DM(high_solar_p))])()["o0"].full().flatten()

        self.assertGreater(dx_high[2], dx_base[2])

    def test_calibrate_parameter_multipliers_selects_lowest_score_and_saves_json(self):
        from experiments.reports.calibrate_chengdu_physics import calibrate_parameter_multipliers

        trajectories = pd.DataFrame({"dummy": [1.0]})
        search_space = {209: [0.5, 1.0, 1.5], 210: [0.8, 1.0]}

        def evaluator(_trajectories, params):
            return {"mean_mae": abs(params[209] - 1.5) + abs(params[210] - 0.8)}

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "best.json"
            best_params, best_metrics = calibrate_parameter_multipliers(
                trajectories,
                search_space=search_space,
                evaluator=evaluator,
                output_path=output_path,
            )

            self.assertEqual(best_params[209], 1.5)
            self.assertEqual(best_params[210], 0.8)
            self.assertEqual(best_metrics["mean_mae"], 0.0)
            self.assertTrue(output_path.exists())

    def test_linear_residual_corrector_learns_affine_target_mapping(self):
        from experiments.reports.chengdu_residual_correction import (
            DEFAULT_RESIDUAL_FEATURES,
            evaluate_corrected_predictions,
            fit_linear_residual_corrector,
        )

        train = pd.DataFrame(
            {
                "pred_air_temperature": [10.0, 12.0, 14.0, 16.0],
                "pred_relative_humidity": [50.0, 55.0, 60.0, 65.0],
                "x_air_temperature": [9.0, 11.0, 13.0, 15.0],
                "x_relative_humidity": [49.0, 54.0, 59.0, 64.0],
                "uVent": [0.0, 0.5, 0.0, 1.0],
                "uBlScr": [0.0, 0.0, 1.0, 1.0],
                "uThScr": [1.0, 0.0, 1.0, 0.0],
                "d_global_radiation": [0.0, 100.0, 200.0, 300.0],
                "d_wind_speed": [0.2, 0.3, 0.4, 0.5],
                "next_x_air_temperature": [11.0, 13.0, 15.0, 17.0],
                "next_x_relative_humidity": [48.0, 53.0, 58.0, 63.0],
            }
        )

        model = fit_linear_residual_corrector(train, feature_columns=DEFAULT_RESIDUAL_FEATURES)
        metrics, corrected = evaluate_corrected_predictions(train, model)

        self.assertLess(metrics["air_temperature_mae"], 1e-8)
        self.assertLess(metrics["relative_humidity_mae"], 1e-8)
        self.assertIn("corrected_air_temperature", corrected.columns)

    def test_multistep_rollout_reports_horizon_metrics(self):
        from experiments.reports.evaluate_chengdu_multistep import evaluate_multistep_rollouts

        trajectories = pd.DataFrame(
            {
                "x_air_temperature": [20.0, 21.0, 22.0, 23.0],
                "x_relative_humidity": [70.0, 69.0, 68.0, 67.0],
                "next_x_air_temperature": [21.0, 22.0, 23.0, 24.0],
                "next_x_relative_humidity": [69.0, 68.0, 67.0, 66.0],
            }
        )

        def predictor(row):
            return {
                "pred_air_temperature": float(row["x_air_temperature"]) + 1.0,
                "pred_relative_humidity": float(row["x_relative_humidity"]) - 1.0,
            }

        metrics, rollouts = evaluate_multistep_rollouts(
            trajectories,
            predictor=predictor,
            horizons=[1, 2, 3],
            start_indices=[0],
        )

        self.assertEqual(len(rollouts), 3)
        self.assertEqual(metrics["horizon_1_air_temperature_mae"], 0.0)
        self.assertEqual(metrics["horizon_3_relative_humidity_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
