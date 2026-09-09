import tempfile
import unittest
from pathlib import Path

import pandas as pd

from processing.chengdu_sql_controls import (
    SUPPORTED_ACTIONS,
    TARGET_CONTROLLER_IDS,
    extract_control_events,
    split_mysql_values,
)


class ChengduSqlControlsTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)

    def write_fixture(self, sql: str) -> Path:
        path = Path(self._temp_dir.name) / "fixture.sql"
        path.write_text(sql, encoding="utf-8-sig")
        return path

    @staticmethod
    def controller_row(controller_id: int, device_type: str, dev_name: str) -> str:
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
            f"'{dev_name}'",
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
            "NULL",
            "NULL",
            "'10'",
        ]
        return f"INSERT INTO `equipment_controller_dev` VALUES ({', '.join(values)});"

    @staticmethod
    def log_row(
        log_id: int,
        timestamp: str,
        controller_id: int,
        action: str = "TURN_ON",
        params: str = "NULL",
        source_content: str = r"{\"id\":43,\"note\":\"a,b\"}",
        operator_id: str = "2",
    ) -> str:
        return (
            "INSERT INTO `equipment_controller_operator_log` VALUES "
            f"({log_id}, '{timestamp}', NULL, 0, 0, 0, {controller_id}, "
            f"'{action}', {params}, '{source_content}', {operator_id});"
        )

    def test_public_constants_match_supported_controller_scope(self):
        self.assertEqual(TARGET_CONTROLLER_IDS, frozenset(range(43, 52)))
        self.assertEqual(
            SUPPORTED_ACTIONS,
            {"TURN_ON", "TURN_OFF", "STOP", "TURN_ON_2", "TURN_ON_4"},
        )

    def test_split_mysql_values_handles_null_quotes_commas_and_backslashes(self):
        values = split_mysql_values(
            r"(43, NULL, '遮阳网,东侧', '{\"id\":43,\"path\":\"a\\\\b\"}', 'it\'s ready');"
        )

        self.assertEqual(
            values,
            ["43", None, "遮阳网,东侧", '{"id":43,"path":"a\\\\b"}', "it's ready"],
        )

    def test_split_mysql_values_preserves_whitespace_inside_quotes(self):
        values = split_mysql_values("('  padded  ',   NULL   ,   43   );")

        self.assertEqual(values, ["  padded  ", None, "43"])

    def test_extract_target_events_decodes_values_and_excludes_sensitive_rows(self):
        sql = "\n".join(
            [
                "INSERT INTO `account` VALUES (1, NULL, 'private-account-marker');",
                self.controller_row(43, "外遮阳", "遮阳网,东侧"),
                self.controller_row(42, "非目标设备", "private-device-marker"),
                self.log_row(100, "2026-04-02 15:12:05", 43),
                self.log_row(101, "2026-04-02 15:13:05", 42, "TURN_ON_9", source_content="private-log-marker"),
                "INSERT INTO `account` THIS IS MALFORMED AND PRIVATE;",
            ]
        )

        events = extract_control_events(self.write_fixture(sql))

        self.assertEqual(
            list(events.columns),
            [
                "log_id",
                "timestamp",
                "controller_id",
                "device",
                "action",
                "params",
                "source_content",
                "operator_id",
            ],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events.loc[0, "controller_id"], 43)
        self.assertEqual(events.loc[0, "device"], "外遮阳")
        self.assertIsNone(events.loc[0, "params"])
        self.assertEqual(events.loc[0, "source_content"], '{"id":43,"note":"a,b"}')
        self.assertNotIn("private", events.to_csv(index=False))

    def test_extract_target_events_sorts_deterministically_and_uses_typed_ids(self):
        sql = "\n".join(
            [
                self.controller_row(46, "顶窗", "顶窗"),
                self.controller_row(47, "湿帘风机", "湿帘风机"),
                self.log_row(12, "2026-04-02 15:12:06", 47, "TURN_ON_4", operator_id="NULL"),
                self.log_row(11, "2026-04-02 15:12:05", 46, "TURN_ON_2"),
                self.log_row(10, "2026-04-02 15:12:05", 47, "TURN_ON_2"),
            ]
        )

        events = extract_control_events(self.write_fixture(sql))

        self.assertEqual(events["log_id"].tolist(), [10, 11, 12])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(events["timestamp"]))
        self.assertTrue(pd.api.types.is_integer_dtype(events["log_id"]))
        self.assertTrue(pd.api.types.is_integer_dtype(events["controller_id"]))
        self.assertTrue(pd.api.types.is_integer_dtype(events["operator_id"]))
        self.assertIs(pd.NA, events.loc[2, "operator_id"])

    def test_extract_target_events_rejects_unsupported_action_with_line_number(self):
        sql = "\n".join(
            [
                self.controller_row(46, "顶窗", "顶窗"),
                self.log_row(200, "2026-04-02 15:12:05", 46, "TURN_ON_9"),
            ]
        )

        with self.assertRaisesRegex(ValueError, r"line 2.*TURN_ON_9"):
            extract_control_events(self.write_fixture(sql))

    def test_extract_target_events_reports_malformed_target_row_line_number(self):
        sql = "\n".join(
            [
                "-- harmless header",
                self.controller_row(43, "外遮阳", "遮阳网"),
                "INSERT INTO `unrelated` VALUES ('unterminated);",
                "INSERT INTO `equipment_controller_operator_log` VALUES "
                "(300, 'not-a-timestamp', NULL, 0, 0, 0, 43, 'TURN_ON', NULL, '{}', 2);",
            ]
        )

        with self.assertRaisesRegex(ValueError, r"line 4"):
            extract_control_events(self.write_fixture(sql))

    def test_extract_target_events_ignores_malformed_non_target_controller_rows(self):
        sql = "\n".join(
            [
                "INSERT INTO `equipment_controller_dev` VALUES (42, 'unterminated);",
                "INSERT INTO `equipment_controller_operator_log` VALUES "
                "(400, '2026-04-02 15:12:05', NULL, 0, 0, 0, 42, "
                "'TURN_ON', NULL, 'unterminated);",
            ]
        )

        events = extract_control_events(self.write_fixture(sql))

        self.assertTrue(events.empty)
        self.assertEqual(list(events.columns), [
            "log_id",
            "timestamp",
            "controller_id",
            "device",
            "action",
            "params",
            "source_content",
            "operator_id",
        ])

    def test_extract_target_events_ignores_truncated_non_target_controller_definition(self):
        path = self.write_fixture(
            "INSERT INTO `equipment_controller_dev` VALUES (42, 'truncated'"
        )

        events = extract_control_events(path)

        self.assertTrue(events.empty)

    def test_extract_target_events_ignores_truncated_non_target_operator_log(self):
        path = self.write_fixture(
            "INSERT INTO `equipment_controller_operator_log` VALUES "
            "(400, '2026-04-02 15:12:05', NULL, 0, 0, 0, 42, 'truncated'"
        )

        events = extract_control_events(path)

        self.assertTrue(events.empty)

    def test_extract_target_events_reports_truncated_target_row_line_number(self):
        path = self.write_fixture(
            "-- header\n"
            "INSERT INTO `equipment_controller_operator_log` VALUES "
            "(401, '2026-04-02 15:12:05', NULL, 0, 0, 0, 43, 'truncated'"
        )

        with self.assertRaisesRegex(ValueError, r"line 2"):
            extract_control_events(path)

    def test_extract_target_events_rejects_log_truncated_before_controller_id(self):
        path = self.write_fixture(
            "-- header\n"
            "INSERT INTO `equipment_controller_operator_log` VALUES "
            "(402, '2026-04-02 15:12:05', NULL, 0, 0, 42"
        )

        with self.assertRaisesRegex(ValueError, r"line 2"):
            extract_control_events(path)


if __name__ == "__main__":
    unittest.main()
