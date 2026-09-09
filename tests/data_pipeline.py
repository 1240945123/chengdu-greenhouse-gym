import tempfile
import unittest
from pathlib import Path

import pandas as pd

from processing.dataset_config import load_dataset_config
from processing.chengdu_weather import (
    convert_aligned_weather_dataframe,
    convert_weather_dataframe,
    write_greenlight_weather_csv,
)
from processing.chengdu_trajectory import build_trajectory_dataframe, split_trajectory_dataframe
from processing.chengdu_controls import (
    build_hourly_controls,
    build_hourly_controls_from_states,
    merge_controls_with_hourly_wide,
)
from processing.chengdu_control_states import reconstruct_minute_controls


class DatasetPipelineTest(unittest.TestCase):
    def test_load_dataset_config_returns_known_chengdu_paths(self):
        config = load_dataset_config("chengdu_agri_greenhouse_001")

        self.assertEqual(config["dataset_id"], "chengdu_agri_greenhouse_001")
        self.assertEqual(
            config["processed_data"]["weather_dir"],
            "data/processed/chengdu_agri/greenhouse_001/weather/",
        )

    def test_convert_weather_dataframe_to_greenlight_columns(self):
        raw = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
                "global_radiation": [0.0, 10.0],
                "wind_speed": [1.2, 1.3],
                "air_temperature": [8.0, 8.2],
                "sky_temperature": [5.0, 5.1],
                "co2_concentration": [420.0, 421.0],
                "relative_humidity": [80.0, 79.0],
            }
        )

        converted = convert_weather_dataframe(raw)

        self.assertEqual(
            list(converted.columns),
            [
                "time",
                "global radiation",
                "wind speed",
                "air temperature",
                "sky temperature",
                "CO2 concentration",
                "day number",
                "RH",
            ],
        )
        self.assertEqual(converted["time"].tolist(), [0.0, 300.0])
        self.assertEqual(converted["day number"].tolist(), [1, 1])
        self.assertEqual(converted["RH"].tolist(), [80.0, 79.0])

    def test_convert_hourly_wide_weather_columns(self):
        raw = pd.DataFrame(
            {
                "time": ["2026-04-05 00:00:00", "2026-04-05 01:00:00"],
                "solar_radiation_mean": [0.0, 50.0],
                "wind_speed_mean": [0.1, 0.2],
                "air_temperature_mean": [15.0, 16.0],
                "co2_mean": [530.0, 540.0],
                "air_humidity_mean": [80.0, 78.0],
            }
        )

        converted = convert_weather_dataframe(raw)

        april_5_seconds = 94 * 86400.0
        self.assertEqual(
            converted["time"].tolist(),
            [april_5_seconds, april_5_seconds + 3600.0],
        )
        self.assertEqual(converted["global radiation"].tolist(), [0.0, 50.0])
        self.assertEqual(converted["sky temperature"].tolist(), [9.0, 10.0])

    def test_convert_aligned_weather_uses_outdoor_columns_and_constant_co2(self):
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-01", periods=3, freq="h"),
                "air_temperature": [25.0, 26.0, 27.0],
                "relative_humidity": [70.0, 71.0, 72.0],
                "outdoor_air_temperature": [12.0, 13.0, 14.0],
                "outdoor_relative_humidity": [80.0, 81.0, 82.0],
                "global_radiation": [0.0, 20.0, 40.0],
                "wind_speed": [0.1, 0.2, 0.3],
            }
        )

        converted = convert_aligned_weather_dataframe(raw)

        april_1_seconds = 90 * 86400.0
        self.assertEqual(
            converted["time"].tolist(),
            [april_1_seconds, april_1_seconds + 3600.0, april_1_seconds + 7200.0],
        )
        self.assertEqual(converted["air temperature"].tolist(), [12.0, 13.0, 14.0])
        self.assertEqual(converted["RH"].tolist(), [80.0, 81.0, 82.0])
        self.assertEqual(converted["sky temperature"].tolist(), [6.0, 7.0, 8.0])
        self.assertEqual(converted["CO2 concentration"].tolist(), [400.0] * 3)
        self.assertFalse(converted.isna().any().any())

    def test_write_greenlight_weather_csv_uses_dataset_location_year(self):
        raw = pd.DataFrame(
            {
                "timestamp": ["2024-01-01 00:00:00", "2024-01-01 00:05:00"],
                "global_radiation": [0.0, 10.0],
                "wind_speed": [1.2, 1.3],
                "air_temperature": [8.0, 8.2],
                "sky_temperature": [5.0, 5.1],
                "co2_concentration": [420.0, 421.0],
                "relative_humidity": [80.0, 79.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = write_greenlight_weather_csv(
                raw,
                output_root=Path(tmp),
                location="Chengdu",
                year=2024,
            )

            self.assertTrue(output.exists())
            self.assertEqual(output.name, "2024.csv")
            self.assertEqual(output.parent.name, "Chengdu")

    def test_build_trajectory_dataframe_from_hourly_wide(self):
        raw = pd.DataFrame(
            {
                "time": ["2026-04-05 00:00:00", "2026-04-05 01:00:00", "2026-04-05 02:00:00"],
                "air_temperature_mean": [15.0, 16.0, 17.0],
                "air_humidity_mean": [80.0, 78.0, 76.0],
                "co2_mean": [530.0, 540.0, 550.0],
                "solar_radiation_mean": [0.0, 50.0, 100.0],
                "wind_speed_mean": [0.1, 0.2, 0.3],
                "illumination_mean": [0.0, 10.0, 20.0],
            }
        )

        traj = build_trajectory_dataframe(raw, require_measured_outdoor=False)

        self.assertEqual(len(traj), 2)
        for column in [
            "timestamp",
            "x_air_temperature",
            "x_relative_humidity",
            "x_co2_concentration",
            "uBoil",
            "uCO2",
            "uThScr",
            "uVent",
            "uLamp",
            "uBlScr",
            "d_global_radiation",
            "d_wind_speed",
            "next_x_air_temperature",
            "next_x_relative_humidity",
            "next_x_co2_concentration",
        ]:
            self.assertIn(column, traj.columns)
        self.assertEqual(traj["next_x_air_temperature"].tolist(), [16.0, 17.0])
        self.assertEqual(traj["uBoil"].tolist(), [0.0, 0.0])

    def test_trajectory_uses_outdoor_temperature_and_humidity_as_disturbances(self):
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-05", periods=3, freq="h"),
                "air_temperature": [20.0, 21.0, 22.0],
                "relative_humidity": [80.0, 79.0, 78.0],
                "outdoor_air_temperature": [10.0, 11.0, 12.0],
                "outdoor_relative_humidity": [90.0, 89.0, 88.0],
                "co2_concentration": [500.0, 501.0, 502.0],
                "illumination": [0.0, 10.0, 20.0],
            }
        )

        trajectory = build_trajectory_dataframe(raw)

        self.assertEqual(trajectory["x_air_temperature"].tolist(), [20.0, 21.0])
        self.assertEqual(trajectory["d_air_temperature"].tolist(), [10.0, 11.0])
        self.assertEqual(trajectory["d_relative_humidity"].tolist(), [90.0, 89.0])

    def test_trajectory_requires_measured_outdoor_disturbances_by_default(self):
        raw = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-04-01", periods=2, freq="h"),
                "air_temperature": [20.0, 21.0],
                "relative_humidity": [70.0, 69.0],
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "Missing measured outdoor disturbance columns: outdoor_air_temperature, outdoor_relative_humidity",
        ):
            build_trajectory_dataframe(raw)

    def test_split_trajectory_dataframe_preserves_time_order(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="h"), "value": range(10)})

        splits = split_trajectory_dataframe(df, train_fraction=0.6, val_fraction=0.2)

        self.assertEqual(splits["train"]["value"].tolist(), list(range(6)))
        self.assertEqual(splits["val"]["value"].tolist(), [6, 7])
        self.assertEqual(splits["test"]["value"].tolist(), [8, 9])

    def test_build_hourly_controls_from_device_events(self):
        events = pd.DataFrame(
            {
                "操作时间": [
                    "4/5/2026 12:15:00 AM",
                    "4/5/2026 1:05:00 AM",
                    "4/5/2026 12:35:00 AM",
                ],
                "设备类型": ["外遮阳", "外遮阳", "CO2发生器"],
                "操作类型": ["TURN_ON", "TURN_OFF", "TURN_ON"],
            }
        )
        hourly_index = pd.date_range("2026-04-05 00:00:00", periods=3, freq="h")

        controls = build_hourly_controls(events, hourly_index)

        self.assertAlmostEqual(controls["uBlScr"].iloc[0], 45.0 / 60.0)
        self.assertAlmostEqual(controls["uBlScr"].iloc[1], 5.0 / 60.0)
        self.assertAlmostEqual(controls["uCO2"].iloc[0], 25.0 / 60.0)
        self.assertIn("uVent", controls.columns)

    def test_merge_controls_with_hourly_wide_adds_control_columns(self):
        hourly = pd.DataFrame(
            {
                "time": pd.date_range("2026-04-05 00:00:00", periods=2, freq="h"),
                "air_temperature_mean": [20.0, 21.0],
            }
        )
        events = pd.DataFrame(
            {
                "操作时间": ["4/5/2026 1:05:00 AM"],
                "设备类型": ["补光灯"],
                "操作类型": ["TURN_ON"],
            }
        )

        merged = merge_controls_with_hourly_wide(hourly, events)

        self.assertAlmostEqual(merged["uLamp"].iloc[0], 0.0)
        self.assertAlmostEqual(merged["uLamp"].iloc[1], 55.0 / 60.0)
        self.assertEqual(merged["air_temperature_mean"].tolist(), [20.0, 21.0])

    def test_build_hourly_controls_from_canonical_device_states(self):
        events = pd.DataFrame(
            {
                "log_id": [1, 2, 3],
                "timestamp": pd.to_datetime(
                    ["2026-04-05 00:00:00", "2026-04-05 00:30:00", "2026-04-05 00:00:00"]
                ),
                "controller_id": [46, 46, 47],
                "action": ["TURN_ON_2", "TURN_OFF", "TURN_ON_4"],
            }
        )
        states = reconstruct_minute_controls(
            events,
            pd.date_range("2026-04-05 00:00:00", periods=60, freq="min"),
        )

        hourly = build_hourly_controls_from_states(states)

        self.assertAlmostEqual(hourly.loc[0, "roof_window"], 0.25)
        self.assertAlmostEqual(hourly.loc[0, "fan"], 1.0)
        self.assertAlmostEqual(hourly.loc[0, "uVent"], 1.0)
        self.assertIn("uPad", hourly.columns)
        self.assertIn("roof_window_known_fraction", hourly.columns)


if __name__ == "__main__":
    unittest.main()
