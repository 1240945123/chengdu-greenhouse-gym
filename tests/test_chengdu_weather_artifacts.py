import json
from pathlib import Path

import pandas as pd

from processing.chengdu_weather import convert_aligned_weather_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "processed" / "chengdu_agri" / "greenhouse_001"


def test_live_real_weather_matches_aligned_source():
    aligned = pd.read_csv(DATASET_ROOT / "aligned" / "greenhouse_1h.csv")
    actual = pd.read_csv(DATASET_ROOT / "weather" / "Chengdu" / "2026.csv")
    expected = convert_aligned_weather_dataframe(aligned)

    assert len(actual) == 2650
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert len(actual) // 24 == 110


def test_live_synthetic_manifest_is_training_only_and_hashes_match():
    weather_root = DATASET_ROOT / "weather"
    manifest = json.loads(
        (weather_root / "synthetic_weather_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["source_start_day"] == 0
    assert manifest["source_days"] == 70
    assert len(manifest["outputs"]) == 10
    for output in manifest["outputs"]:
        assert max(output["source_block_absolute_start_days"]) + manifest["block_days"] <= 70
