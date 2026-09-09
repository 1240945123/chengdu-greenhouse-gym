import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from processing.build_chengdu_dataset import build_dataset
from tests.test_chengdu_sensor_stream import sensor_rows


def controller_row(controller_id, device_type):
    values = [
        str(controller_id),
        "'2026-03-11 16:32:25'",
        "'2026-03-11 16:32:25'",
        "0",
        "0",
        "0",
        "3037",
        "'dev'",
        "'code'",
        f"'{device_type}'",
        f"'{device_type}'",
        "NULL",
        "NULL",
        "NULL",
        "NULL",
        "NULL",
        "0",
        "0",
        "NULL",
        "NULL",
        "NULL",
        "'pidu1'",
        "'open'",
        "'close'",
        "'stop'",
        "'half'",
        "'stage3'",
        "'10'",
    ]
    return f"INSERT INTO `equipment_controller_dev` VALUES ({', '.join(values)});"


def log_row(log_id, timestamp, controller_id, action):
    return (
        "INSERT INTO `equipment_controller_operator_log` VALUES "
        f"({log_id}, '{timestamp}', NULL, 0, 0, 0, {controller_id}, "
        f"'{action}', NULL, '{{\"id\":{controller_id}}}', 2);"
    )


class ChengduDatasetIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sensor_root = self.root / "sensors"
        self.sensor_root.mkdir()
        self.output_root = self.root / "processed"
        self.sql_path = self.root / "backup.sql"
        self.sql_path.write_text(
            "\n".join(
                [
                    "INSERT INTO `account` VALUES (1, NULL, 'private-marker');",
                    controller_row(46, "顶窗"),
                    controller_row(47, "湿帘风机"),
                    log_row(1, "2026-04-01 00:01:00", 46, "TURN_ON_2"),
                    log_row(2, "2026-04-01 00:03:00", 46, "STOP"),
                    log_row(3, "2026-04-01 00:02:00", 47, "TURN_ON_4"),
                    log_row(4, "2026-04-01 01:00:00", 47, "TURN_OFF"),
                ]
            ),
            encoding="utf-8",
        )
        sensor_rows(
            [
                (63, 3044, 1557, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 20.0, True),
                (63, 3044, 1557, "AIR_TEMPERATURE", "2026-04-01 00:01:00:000", 21.0, True),
                (63, 3044, 1558, "AIR_HUMIDITY", "2026-04-01 00:00:00:000", 80.0, True),
                (63, 3044, 1558, "AIR_HUMIDITY", "2026-04-01 01:00:00:000", 70.0, True),
            ]
        ).to_csv(self.sensor_root / "2026-04-01.csv", index=False)
        self.config = {
            "source": {"greenhouse_id": 63},
            "processing": {"chunksize": 2, "interpolation_limit_minutes": 5},
            "sensor_ranges": {
                "AIR_TEMPERATURE": [-30.0, 60.0],
                "AIR_HUMIDITY": [0.0, 100.0],
            },
        }

    def test_build_dataset_writes_aligned_outputs_and_report(self):
        outputs = build_dataset(
            sql_path=self.sql_path,
            sensor_root=self.sensor_root,
            output_root=self.output_root,
            config=self.config,
        )

        self.assertEqual(
            set(outputs),
            {
                "control_events",
                "controls_1min",
                "controls_1h",
                "aligned_1min",
                "aligned_1h",
                "quality_report",
            },
        )
        minute = pd.read_csv(outputs["aligned_1min"])
        hourly = pd.read_csv(outputs["aligned_1h"])
        report = json.loads(Path(outputs["quality_report"]).read_text(encoding="utf-8"))

        self.assertEqual(len(minute), 61)
        self.assertEqual(len(hourly), 2)
        self.assertIn("roof_window", minute.columns)
        self.assertIn("uVent", minute.columns)
        self.assertIn("air_temperature_observed", minute.columns)
        self.assertEqual(report["controls"]["target_events"], 4)
        self.assertEqual(report["controls"]["by_device"]["46"]["TURN_ON_2"], 1)
        self.assertEqual(report["outputs"]["aligned_1min_rows"], 61)
        self.assertEqual(report["outputs"]["aligned_1h_rows"], 2)
        self.assertAlmostEqual(report["outputs"]["complete_air_temp_rh_fraction"], 1 / 61)
        self.assertGreaterEqual(report["sensors"]["overall_interpolated_fraction"], 0.0)
        self.assertLessEqual(report["sensors"]["overall_interpolated_fraction"], 1.0)

        for output in outputs.values():
            self.assertNotIn("private-marker", Path(output).read_text(encoding="utf-8"))

    def test_data_docs_describe_external_sources_and_command_limit(self):
        readme = Path("data/README.md").read_text(encoding="utf-8")
        metadata = Path("data/raw/chengdu_agri/greenhouse_001/metadata.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("build_chengdu_dataset", readme)
        self.assertIn("tomato_full_backup_20260724.sql", readme)
        self.assertIn("command states are not actuator feedback", readme)
        self.assertIn("greenhouse_database_id: 63", metadata)
        self.assertIn("2026-04-01", metadata)
        self.assertIn("2026-07-20", metadata)


if __name__ == "__main__":
    unittest.main()
