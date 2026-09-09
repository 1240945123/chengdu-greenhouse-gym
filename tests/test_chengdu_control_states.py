import unittest

import numpy as np
import pandas as pd

from processing.chengdu_control_states import (
    add_aggregate_controls,
    derive_hourly_controls,
    reconstruct_minute_controls,
)


def event_frame(rows):
    return pd.DataFrame(
        rows,
        columns=["log_id", "timestamp", "controller_id", "action"],
    ).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))


class ChengduControlStatesTest(unittest.TestCase):
    def test_roof_window_half_open_and_stop_marks_uncertain(self):
        events = event_frame(
            [
                (1, "2026-04-01 00:01:00", 46, "TURN_ON_2"),
                (2, "2026-04-01 00:03:00", 46, "STOP"),
                (3, "2026-04-01 00:05:00", 46, "TURN_OFF"),
            ]
        )
        index = pd.date_range("2026-04-01 00:00", periods=7, freq="min")

        states = reconstruct_minute_controls(events, index)

        np.testing.assert_allclose(states["roof_window"], [0, 0.5, 0.5, 0.5, 0.5, 0, 0])
        self.assertEqual(
            states["roof_window_known"].tolist(),
            [False, True, True, True, True, True, True],
        )
        self.assertEqual(
            states["roof_window_uncertain"].tolist(),
            [False, False, False, True, True, False, False],
        )

    def test_fan_stages_are_preserved(self):
        events = event_frame(
            [
                (1, "2026-04-01 00:00:00", 47, "TURN_ON"),
                (2, "2026-04-01 00:01:00", 47, "TURN_ON_2"),
                (3, "2026-04-01 00:02:00", 47, "TURN_ON_4"),
                (4, "2026-04-01 00:03:00", 47, "TURN_OFF"),
            ]
        )

        states = reconstruct_minute_controls(
            events,
            pd.date_range("2026-04-01", periods=4, freq="min"),
        )

        np.testing.assert_allclose(states["fan"], [1 / 3, 2 / 3, 1, 0])
        self.assertTrue(states["fan_known"].all())
        self.assertFalse(states["fan_uncertain"].any())

    def test_first_command_prehistory_is_zero_but_unknown(self):
        events = event_frame([(1, "2026-04-01 00:02:00", 43, "TURN_ON")])

        states = reconstruct_minute_controls(
            events,
            pd.date_range("2026-04-01", periods=4, freq="min"),
        )

        np.testing.assert_allclose(states["external_shade"], [0, 0, 1, 1])
        self.assertEqual(states["external_shade_known"].tolist(), [False, False, True, True])

    def test_same_minute_commands_use_timestamp_then_log_id(self):
        events = event_frame(
            [
                (5, "2026-04-01 00:01:40", 50, "TURN_OFF"),
                (3, "2026-04-01 00:01:10", 50, "TURN_ON"),
                (6, "2026-04-01 00:01:40", 50, "TURN_ON"),
            ]
        )

        states = reconstruct_minute_controls(
            events,
            pd.date_range("2026-04-01", periods=3, freq="min"),
        )

        np.testing.assert_allclose(states["wet_pad_pump"], [0, 1, 1])
        self.assertEqual(states["wet_pad_pump_known"].tolist(), [False, True, True])

    def test_aggregate_controls_keep_real_devices_and_ignore_heating_co2(self):
        states = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-04-01 00:00:00"]),
                "external_shade": [0.7],
                "roof_thermal_screen": [1.0],
                "side_thermal_screen": [0.0],
                "roof_window": [0.5],
                "fan": [2 / 3],
                "supplemental_lamp": [1.0],
                "co2_generator": [1.0],
                "wet_pad_pump": [1.0],
                "wet_pad_roll_film": [0.8],
            }
        )

        result = add_aggregate_controls(states)

        self.assertAlmostEqual(result.loc[0, "uVent"], 2 / 3)
        self.assertAlmostEqual(result.loc[0, "uPad"], 2 / 3)
        self.assertAlmostEqual(result.loc[0, "uThScr"], 192 / 388)
        self.assertAlmostEqual(result.loc[0, "uLamp"], 1.0)
        self.assertAlmostEqual(result.loc[0, "uBlScr"], 0.7)
        self.assertAlmostEqual(result.loc[0, "uBoil"], 0.0)
        self.assertAlmostEqual(result.loc[0, "uCO2"], 0.0)
        self.assertIn("roof_window", result.columns)

    def test_unsupported_device_action_is_rejected(self):
        events = event_frame([(1, "2026-04-01 00:00:00", 47, "STOP")])

        with self.assertRaisesRegex(ValueError, "controller 47.*STOP"):
            reconstruct_minute_controls(
                events,
                pd.date_range("2026-04-01", periods=1, freq="min"),
            )

    def test_hourly_controls_use_minute_means_and_quality_fractions(self):
        timestamps = pd.date_range("2026-04-01 00:00", periods=60, freq="min")
        minute = pd.DataFrame({"timestamp": timestamps})
        for column in [
            "external_shade",
            "roof_thermal_screen",
            "side_thermal_screen",
            "roof_window",
            "fan",
            "supplemental_lamp",
            "co2_generator",
            "wet_pad_pump",
            "wet_pad_roll_film",
        ]:
            minute[column] = 0.0
            minute[f"{column}_known"] = True
            minute[f"{column}_uncertain"] = False
        minute["fan"] = [0.0] * 30 + [1.0] * 30
        minute["fan_known"] = [False] * 10 + [True] * 50
        minute["fan_uncertain"] = [False] * 40 + [True] * 20
        minute["wet_pad_pump"] = 1.0
        minute["wet_pad_roll_film"] = 1.0

        hourly = derive_hourly_controls(minute)

        self.assertEqual(len(hourly), 1)
        self.assertEqual(hourly.loc[0, "timestamp"], pd.Timestamp("2026-04-01 00:00"))
        self.assertAlmostEqual(hourly.loc[0, "fan"], 0.5)
        self.assertAlmostEqual(hourly.loc[0, "fan_known_fraction"], 50 / 60)
        self.assertAlmostEqual(hourly.loc[0, "fan_uncertain_fraction"], 20 / 60)
        self.assertAlmostEqual(hourly.loc[0, "uVent"], 0.5)
        self.assertAlmostEqual(hourly.loc[0, "uPad"], 0.5)

    def test_hourly_controls_reject_duplicate_minute_timestamps(self):
        minute = reconstruct_minute_controls(
            event_frame([]),
            pd.date_range("2026-04-01", periods=2, freq="min"),
        )
        minute = pd.concat([minute, minute.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "unique"):
            derive_hourly_controls(minute)


if __name__ == "__main__":
    unittest.main()
