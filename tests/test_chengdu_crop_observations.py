import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from processing.chengdu_crop_observations import (
    CROP_OBSERVATION_COLUMNS,
    extract_crop_observations,
    validate_crop_observations,
)
from processing.chengdu_sql_controls import split_mysql_values


PLANTING_FIELDS = [
    "id",
    "code",
    "type",
    "createdDateTime",
    "updatedDateTime",
    "version",
    "deleted",
    "disable",
    "crop_name",
    "plant_nums",
    "seedingDateTime",
    "plantingDateTime",
    "firflowerDateTime",
    "earlyFruitDateTime",
    "matureDateTime",
    "endDateTime",
    "plant_den",
    "stem_den",
    "greenhouse_code",
    "rowSpace",
    "plantSpace",
    "booleanCurrent",
    "plantStage",
    "booleanMain",
    "image",
    "plant_area_name",
    "plant_area_id",
    "experimentName",
    "cropIntroduction",
    "matureSpecies",
    "greenHouseId",
]

MESSAGE_FIELDS = [
    "id",
    "greenhouseCode",
    "plantingCode",
    "plantingInfoCropName",
    "createdDateTime",
    "updatedDateTime",
    "version",
    "deleted",
    "disable",
    "plantStage",
    "plantHeight",
    "stemDia",
    "plantFrewei",
    "plantDryWei",
    "leaveFrewei",
    "layerFlowerNumber",
    "layerFruitNumber",
    "leaveDryWei",
    "petioleFrewei",
    "petioleDryWei",
    "stemFrewei",
    "stemDryWei",
    "ripeFruitFrewei",
    "ripeFruitDryWei",
    "rootFrewei",
    "rootDryWei",
    "lai",
    "oneFruitWei",
    "twoFruitWei",
    "threeFruitWei",
    "fourFruitWei",
    "collectDateTime",
    "fixedPlantHeight",
    "fixedStemDia",
    "fixedLayerFlowerNumber",
    "fixedLayerFruitNumber",
    "stipesNumber",
    "fixedStipesNumber",
    "sweetness",
    "acidity",
    "unitPlantNumber",
    "images",
    "oneNumberFruit",
    "twoNumberFruit",
    "threeNumberFruit",
    "fourNumberFruit",
    "pesticideResidue",
    "fixedOneNumberFruit",
    "fixedTowNumberFruit",
    "fixedThreeNumberFruit",
    "fixedFourNumberFruit",
]


def _sql_value(value):
    if value is None:
        return "NULL"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return str(value)


def _insert(table, fields, **overrides):
    values = {field: None for field in fields}
    values.update(overrides)
    rendered = ", ".join(_sql_value(values[field]) for field in fields)
    return f"INSERT INTO `{table}` VALUES ({rendered});"


def planting_row(**overrides):
    values = {
        "id": 56,
        "code": "P-XINDU-2025",
        "type": "tomato",
        "version": 0,
        "deleted": 0,
        "disable": 0,
        "crop_name": "Zhainirui",
        "plant_nums": 2640,
        "plant_den": 2.5,
        "stem_den": 2.5,
        "greenhouse_code": "GH2024",
        "greenHouseId": 1,
    }
    values.update(overrides)
    return _insert("planting_info", PLANTING_FIELDS, **values)


def message_row(**overrides):
    values = {
        "id": 901,
        "greenhouseCode": "GH2024",
        "plantingCode": "P-XINDU-2025",
        "plantingInfoCropName": "tomato unit",
        "version": 0,
        "deleted": 0,
        "disable": 0,
        "plantStage": "fruiting",
        "plantHeight": 31.0,
        "plantFrewei": 1200.0,
        "plantDryWei": 150.0,
        "leaveFrewei": 350.0,
        "leaveDryWei": 50.0,
        "petioleFrewei": 80.0,
        "petioleDryWei": 12.0,
        "stemFrewei": 220.0,
        "stemDryWei": 40.0,
        "ripeFruitFrewei": 500.0,
        "ripeFruitDryWei": 40.0,
        "rootFrewei": 50.0,
        "rootDryWei": 8.0,
        "lai": 3.2,
        "collectDateTime": "2025-04-28 09:30:00",
        "unitPlantNumber": "sample-7",
    }
    values.update(overrides)
    return _insert("plant_message_info", MESSAGE_FIELDS, **values)


class ChengduCropObservationsTest(unittest.TestCase):
    def test_target_identity_requires_id_and_code_together(self):
        with self.assertRaisesRegex(ValueError, "configured together"):
            extract_crop_observations(
                "unused.sql",
                target_greenhouse_ids={63},
            )

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def write_sql(self, *lines):
        path = Path(self.temp.name) / "crop.sql"
        path.write_text("\n".join(lines), encoding="utf-8-sig")
        return path

    def extract(self, path):
        return extract_crop_observations(
            path,
            target_greenhouse_ids={63},
            target_greenhouse_codes={"GH-PIDU"},
            source_sites={
                "GH2024": "Chengdu Xindu experimental base",
                "GH-PIDU": "Chengdu Pidu target greenhouse",
            },
        )

    def test_extracts_canonical_standing_sample_and_converts_g_per_plant_to_kg_m2(self):
        path = self.write_sql(
            planting_row(),
            message_row(),
            planting_row(
                id=59,
                code="P-PIDU-2026",
                crop_name="Target cultivar",
                plant_den=1.0,
                greenhouse_code="GH-PIDU",
                greenHouseId=63,
            ),
            message_row(
                id=902,
                greenhouseCode="GH-PIDU",
                plantingCode="P-PIDU-2026",
                ripeFruitFrewei=300.0,
                ripeFruitDryWei=24.0,
            ),
        )

        observations = self.extract(path)

        self.assertEqual(list(observations.columns), CROP_OBSERVATION_COLUMNS)
        self.assertEqual(observations["observation_type"].unique().tolist(), ["standing_crop_sample"])
        xindu = observations.loc[observations["observation_id"] == 901].iloc[0]
        self.assertEqual(xindu["greenhouse_id"], 1)
        self.assertEqual(xindu["greenhouse_code"], "GH2024")
        self.assertEqual(xindu["planting_code"], "P-XINDU-2025")
        self.assertEqual(xindu["crop_name"], "tomato")
        self.assertEqual(xindu["cultivar"], "Zhainirui")
        self.assertEqual(xindu["source_site"], "Chengdu Xindu experimental base")
        self.assertFalse(xindu["target_eligible"])
        self.assertEqual(xindu["sample_plant_id"], "sample-7")
        self.assertIn("plant_height_unit_inferred_cm", xindu["quality_flags"])
        self.assertAlmostEqual(xindu["ripe_fruit_fresh_kg_m2"], 1.25)
        self.assertAlmostEqual(xindu["ripe_fruit_dry_kg_m2"], 0.1)
        self.assertAlmostEqual(xindu["leaf_dry_kg_m2"], 0.125)
        self.assertAlmostEqual(xindu["stem_dry_kg_m2"], 0.1)
        self.assertAlmostEqual(xindu["root_dry_kg_m2"], 0.02)
        pidu = observations.loc[observations["observation_id"] == 902].iloc[0]
        self.assertTrue(pidu["target_eligible"])
        self.assertEqual(pidu["source_site"], "Chengdu Pidu target greenhouse")
        self.assertAlmostEqual(pidu["ripe_fruit_fresh_kg_m2"], 0.3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(observations["observation_date"]))
        self.assertEqual(str(observations["observation_date"].dt.tz), "Asia/Shanghai")

    def test_placeholder_zero_plant_mass_is_missing_and_quality_flagged(self):
        path = self.write_sql(
            planting_row(),
            message_row(
                plantFrewei=0.0,
                plantDryWei=None,
                leaveFrewei=None,
                leaveDryWei=None,
                petioleFrewei=None,
                petioleDryWei=None,
                stemFrewei=None,
                stemDryWei=None,
                rootFrewei=None,
                rootDryWei=None,
                ripeFruitFrewei=None,
                ripeFruitDryWei=None,
            ),
        )

        observation = self.extract(path).iloc[0]

        self.assertTrue(pd.isna(observation["plant_fresh_g_per_plant"]))
        self.assertTrue(pd.isna(observation["plant_fresh_kg_m2"]))
        self.assertIn("placeholder_zero_plant_fresh", observation["quality_flags"])

    def test_target_eligibility_requires_configured_id_and_code_to_agree(self):
        path = self.write_sql(
            planting_row(greenhouse_code="GH-PIDU", greenHouseId=1),
            message_row(greenhouseCode="GH-PIDU"),
        )

        observations = self.extract(path)

        self.assertFalse(observations.loc[0, "target_eligible"])
        self.assertIn("target_identifier_mismatch", observations.loc[0, "quality_flags"])

    def test_logical_duplicates_keep_latest_record_and_report_count(self):
        path = self.write_sql(
            planting_row(),
            message_row(id=901, updatedDateTime="2025-04-28 10:00:00"),
            message_row(id=902, updatedDateTime="2025-04-28 11:00:00"),
        )

        observations = self.extract(path)

        self.assertEqual(observations["observation_id"].tolist(), [902])
        self.assertEqual(observations.attrs["duplicate_rows_removed"], 1)

    def test_duplicate_planting_code_uses_latest_version_not_file_order(self):
        path = self.write_sql(
            planting_row(
                id=57,
                version=2,
                updatedDateTime="2025-05-01 10:00:00",
                plant_den=3.0,
            ),
            planting_row(
                id=56,
                version=1,
                updatedDateTime="2025-04-01 10:00:00",
                plant_den=2.0,
            ),
            message_row(),
        )

        observations = self.extract(path)

        self.assertEqual(observations.loc[0, "planting_id"], 57)
        self.assertEqual(observations.loc[0, "plant_density_plants_m2"], 3.0)

    def test_only_whitelisted_tables_are_parsed_and_inactive_rows_are_excluded(self):
        path = self.write_sql(
            "INSERT INTO `account` VALUES ('private-marker', 'unterminated);",
            planting_row(),
            message_row(),
            message_row(id=903, deleted=1),
            message_row(id=904, disable=1),
            "INSERT INTO `account` THIS IS MALFORMED AND PRIVATE;",
        )

        with patch(
            "processing.chengdu_crop_observations.split_mysql_values",
            wraps=split_mysql_values,
        ) as parser:
            observations = self.extract(path)

        self.assertEqual(observations["observation_id"].tolist(), [901])
        self.assertEqual(parser.call_count, 4)
        self.assertNotIn("private-marker", observations.to_csv(index=False))

    def test_standing_crop_sample_is_not_labeled_as_an_actual_harvest_event(self):
        observations = self.extract(self.write_sql(planting_row(), message_row()))

        self.assertNotIn("harvest_event", observations["observation_type"].tolist())
        harvest_schema = yaml.safe_load(
            Path("data/schemas/harvest_event_schema.yml").read_text(encoding="utf-8")
        )
        crop_schema = yaml.safe_load(
            Path("data/schemas/crop_observation_schema.yml").read_text(encoding="utf-8")
        )
        self.assertEqual(harvest_schema["event_type"], "harvest_event")
        self.assertEqual(crop_schema["observation_type"], "standing_crop_sample")
        self.assertIn("harvested_fresh_kg", harvest_schema["required_columns"])
        self.assertIn("target_eligible", harvest_schema["required_columns"])
        self.assertNotIn("harvested_fresh_kg", crop_schema["required_columns"])

    def test_rejects_negative_mass_with_source_line_number(self):
        path = self.write_sql(planting_row(), message_row(ripeFruitFrewei=-0.1))

        with self.assertRaisesRegex(ValueError, r"line 2.*ripe_fruit_fresh_g_per_plant.*nonnegative"):
            self.extract(path)

    def test_rejects_missing_or_invalid_observation_date_with_source_line_number(self):
        for invalid_date in (None, "not-a-date"):
            with self.subTest(invalid_date=invalid_date):
                path = self.write_sql(planting_row(), message_row(collectDateTime=invalid_date))
                with self.assertRaisesRegex(ValueError, r"line 2.*observation date"):
                    self.extract(path)

    def test_public_validator_rejects_negative_canonical_mass_and_invalid_date(self):
        valid = self.extract(self.write_sql(planting_row(), message_row()))

        negative = valid.copy()
        negative.loc[0, "ripe_fruit_fresh_kg_m2"] = -1.0
        with self.assertRaisesRegex(ValueError, r"ripe_fruit_fresh_kg_m2.*nonnegative"):
            validate_crop_observations(negative)

        invalid_date = valid.copy()
        invalid_date.loc[0, "observation_date"] = pd.NaT
        with self.assertRaisesRegex(ValueError, "observation_date"):
            validate_crop_observations(invalid_date)

        wrong_type = valid.copy()
        wrong_type.loc[0, "observation_type"] = "harvest_event"
        with self.assertRaisesRegex(ValueError, "observation_type"):
            validate_crop_observations(wrong_type)

        infinite = valid.copy()
        infinite.loc[0, "ripe_fruit_fresh_kg_m2"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_crop_observations(infinite)

        dry_above_fresh = valid.copy()
        dry_above_fresh.loc[0, "ripe_fruit_fresh_kg_m2"] = 0.1
        dry_above_fresh.loc[0, "ripe_fruit_dry_kg_m2"] = 0.2
        with self.assertRaisesRegex(ValueError, "dry.*fresh"):
            validate_crop_observations(dry_above_fresh)

        inconsistent_dry_conversion = valid.copy()
        inconsistent_dry_conversion.loc[0, "ripe_fruit_dry_kg_m2"] = 0.05
        with self.assertRaisesRegex(ValueError, "dry mass area conversion"):
            validate_crop_observations(inconsistent_dry_conversion)

    def test_dataset_config_registers_crop_and_harvest_schemas_and_safe_tables(self):
        config = yaml.safe_load(
            Path("configs/datasets/chengdu_agri_greenhouse_001.yml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            config["expected_streams"]["crop_observation"]["schema"],
            "data/schemas/crop_observation_schema.yml",
        )
        self.assertEqual(
            config["expected_streams"]["harvest_event"]["schema"],
            "data/schemas/harvest_event_schema.yml",
        )
        self.assertEqual(
            config["crop_observations"]["sql_table_whitelist"],
            ["planting_info", "plant_message_info"],
        )
        self.assertEqual(config["crop_observations"]["target_greenhouse_ids"], [63])
        self.assertIn("GH2024", config["crop_observations"]["source_sites"])
        self.assertEqual(config["greenhouse_type"], "multi_span_plastic")

        harvest_schema = yaml.safe_load(
            Path("data/schemas/harvest_event_schema.yml").read_text(encoding="utf-8")
        )
        self.assertIn("season_id", harvest_schema["required_columns"])
        self.assertIn("harvested_area_m2", harvest_schema["required_columns"])


if __name__ == "__main__":
    unittest.main()
