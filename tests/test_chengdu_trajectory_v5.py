import json

import pandas as pd

from processing.build_chengdu_trajectory_v5 import build_trajectory_v5_bundle


def _write_v4(source_dir):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-07-11 18:00", periods=8, freq="h"),
            "next_timestamp": pd.date_range("2026-07-11 19:00", periods=8, freq="h"),
            "x_air_temperature": range(20, 28),
            "next_x_air_temperature": range(21, 29),
            "x_relative_humidity": range(60, 68),
            "next_x_relative_humidity": range(61, 69),
            "transition_quality_weight": [1.0] * 8,
            "target_heat_regime": ["normal"] * 8,
        }
    )
    source_dir.mkdir()
    frame.iloc[:3].to_csv(source_dir / "train.csv", index=False)
    frame.iloc[3:5].to_csv(source_dir / "val.csv", index=False)
    frame.iloc[5:].to_csv(source_dir / "test.csv", index=False)
    (source_dir / "manifest.json").write_text(
        json.dumps({"schema_version": "chengdu_trajectory_v4_quality_aware"}),
        encoding="utf-8",
    )
    return frame


def test_v5_excludes_targets_at_or_after_cultivation_end_and_preserves_values(tmp_path):
    source = _write_v4(tmp_path / "v4")

    manifest = build_trajectory_v5_bundle(
        tmp_path / "v4",
        tmp_path / "v5",
        cultivation_end_exclusive="2026-07-12 00:00",
        train_fraction=0.5,
        val_fraction=0.25,
    )
    retained = pd.concat(
        [pd.read_csv(tmp_path / "v5" / f"{split}.csv") for split in ("train", "val", "test")],
        ignore_index=True,
    )
    expected = source[pd.to_datetime(source["next_timestamp"]) < pd.Timestamp("2026-07-12")]
    expected = expected.copy()
    for column in ("timestamp", "next_timestamp"):
        retained[column] = pd.to_datetime(retained[column])
        expected[column] = pd.to_datetime(expected[column])

    assert len(retained) == len(expected) == 5
    assert pd.to_datetime(retained["next_timestamp"]).max() < pd.Timestamp("2026-07-12")
    pd.testing.assert_frame_equal(
        retained[expected.columns].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert manifest["source_rows"] == 8
    assert manifest["retained_rows"] == 5
    assert manifest["post_cultivation_excluded_rows"] == 3
    assert sum(manifest["split_rows"].values()) == 5
