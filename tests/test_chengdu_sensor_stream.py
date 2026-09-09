import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from processing.build_chengdu_dataset import derive_hourly_dataset
from processing.chengdu_sensor_stream import (
    SENSOR_INPUT_COLUMNS,
    build_minute_sensor_table,
    clean_sensor_chunk,
    read_sensor_files,
)


SENSOR_RANGES = {
    "AIR_TEMPERATURE": (-30.0, 60.0),
    "AIR_HUMIDITY": (0.0, 100.0),
}


def sensor_rows(rows):
    records = []
    for index, row in enumerate(rows):
        greenhouse_id, equipment_id, sensor_id, param_type, timestamp, value, indoor = row
        records.append(
            {
                "_id": f"row-{index}",
                "devCode": f"dev-{equipment_id}",
                "paramId": f"param-{sensor_id}",
                "timeString": timestamp,
                "paramName": param_type,
                "paramType": param_type,
                "code": "0101",
                "paramCode": "01",
                "paramNo": "01",
                "paramVal": value,
                "booleanIndoor": indoor,
                "paramUnit": "",
                "ctime": "2026-03-31T16:00:00.000Z",
                "ip": "",
                "topic": "sensor",
                "plantingAreaId": 2,
                "equipmentSensorId": sensor_id,
                "equipmentId": equipment_id,
                "greenhouseId": greenhouse_id,
                "extraFieldsJson": "",
            }
        )
    return pd.DataFrame(records, columns=SENSOR_INPUT_COLUMNS)


class ChengduSensorStreamTest(unittest.TestCase):
    def test_clean_sensor_chunk_filters_greenhouse_and_ranges(self):
        raw = sensor_rows(
            [
                (63, 3036, 1500, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 18.0, False),
                (63, 3044, 1558, "AIR_HUMIDITY", "2026-04-01 00:00:00:000", 125.0, True),
                (1, 3013, 1322, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 20.0, False),
            ]
        )

        clean, quality = clean_sensor_chunk(raw, greenhouse_id=63, ranges=SENSOR_RANGES)

        self.assertEqual(len(clean), 2)
        self.assertTrue(np.isnan(clean.loc[clean["param_type"].eq("AIR_HUMIDITY"), "value"]).all())
        self.assertEqual(quality["input_rows"], 3)
        self.assertEqual(quality["target_rows"], 2)
        self.assertEqual(quality["invalid_range"], 1)

    def test_minute_table_keeps_sensor_columns_and_builds_indoor_median(self):
        clean, _ = clean_sensor_chunk(
            sensor_rows(
                [
                    (63, 3044, 1557, "AIR_TEMPERATURE", "2026-04-01 00:00:10:000", 20.0, True),
                    (63, 3044, 1561, "AIR_TEMPERATURE", "2026-04-01 00:00:40:000", 24.0, True),
                    (63, 3036, 1500, "AIR_TEMPERATURE", "2026-04-01 00:00:20:000", 15.0, False),
                    (63, 3044, 1558, "AIR_HUMIDITY", "2026-04-01 00:00:30:000", 80.0, True),
                ]
            ),
            greenhouse_id=63,
            ranges=SENSOR_RANGES,
        )

        minute, quality = build_minute_sensor_table(clean, interpolation_limit=5)

        self.assertAlmostEqual(minute.loc[0, "air_temperature"], 22.0)
        self.assertAlmostEqual(minute.loc[0, "outdoor_air_temperature"], 15.0)
        self.assertAlmostEqual(minute.loc[0, "relative_humidity"], 80.0)
        self.assertIn("air_temperature__e3044__s1557", minute.columns)
        self.assertTrue(minute.loc[0, "air_temperature_observed"])
        self.assertFalse(minute.loc[0, "air_temperature_interpolated"])
        self.assertEqual(quality["rows"], 1)

    def test_interpolation_fills_five_minute_gap_but_not_six_minute_gap(self):
        rows = []
        for minute, value in [(0, 10.0), (6, 16.0), (10, 20.0), (17, 27.0)]:
            rows.append(
                (
                    63,
                    3044,
                    1557,
                    "AIR_TEMPERATURE",
                    f"2026-04-01 00:{minute:02d}:00:000",
                    value,
                    True,
                )
            )
        clean, _ = clean_sensor_chunk(
            sensor_rows(rows),
            greenhouse_id=63,
            ranges=SENSOR_RANGES,
        )

        minute, _ = build_minute_sensor_table(clean, interpolation_limit=5)

        first_gap = minute.loc[minute["timestamp"].between("2026-04-01 00:01", "2026-04-01 00:05")]
        second_gap = minute.loc[minute["timestamp"].between("2026-04-01 00:11", "2026-04-01 00:16")]
        self.assertTrue(first_gap["air_temperature"].notna().all())
        self.assertTrue(first_gap["air_temperature_interpolated"].all())
        self.assertTrue(second_gap["air_temperature"].isna().all())
        self.assertFalse(second_gap["air_temperature_interpolated"].any())

    def test_read_sensor_files_uses_recursive_csv_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2026-04").mkdir()
            sensor_rows(
                [(63, 3036, 1500, "AIR_TEMPERATURE", "2026-04-01 00:00:00:000", 18.0, False)]
            ).to_csv(root / "2026-04" / "2026-04-01.csv", index=False)

            clean, quality = read_sensor_files(
                root,
                greenhouse_id=63,
                ranges=SENSOR_RANGES,
                chunksize=1,
            )

        self.assertEqual(len(clean), 1)
        self.assertEqual(quality["files"], 1)
        self.assertEqual(quality["input_rows"], 1)

    def test_wide_minute_table_does_not_emit_fragmentation_warning(self):
        rows = [
            (
                63,
                3044,
                1600 + sensor,
                "AIR_TEMPERATURE",
                "2026-04-01 00:00:00:000",
                20.0 + sensor / 100,
                True,
            )
            for sensor in range(40)
        ]
        clean, _ = clean_sensor_chunk(
            sensor_rows(rows),
            greenhouse_id=63,
            ranges=SENSOR_RANGES,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", PerformanceWarning)
            minute, _ = build_minute_sensor_table(clean, interpolation_limit=5)
            derive_hourly_dataset(minute)

        fragmentation = [item for item in caught if issubclass(item.category, PerformanceWarning)]
        self.assertEqual(fragmentation, [])


if __name__ == "__main__":
    unittest.main()
