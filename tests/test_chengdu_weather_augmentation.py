import hashlib
import json

import numpy as np
import pandas as pd

from processing.chengdu_weather_augmentation import (
    generate_synthetic_weather,
    write_synthetic_weather_ensemble,
)


def source_weather(days: int = 12) -> pd.DataFrame:
    hours = days * 24
    hour = np.arange(hours)
    return pd.DataFrame(
        {
            "time": 90 * 86400.0 + hour * 3600.0,
            "global radiation": np.maximum(0.0, 500.0 * np.sin((hour % 24 - 6) * np.pi / 12)),
            "wind speed": 1.0 + 0.01 * hour,
            "air temperature": 15.0 + 0.02 * hour,
            "sky temperature": 9.0 + 0.02 * hour,
            "CO2 concentration": 400.0,
            "day number": 91 + hour // 24,
            "RH": 80.0 - 0.01 * hour,
        }
    )


def test_synthetic_weather_is_seeded_bounded_and_hourly():
    first, first_meta = generate_synthetic_weather(
        source_weather(), seed=7, days=10, block_days=3
    )
    second, second_meta = generate_synthetic_weather(
        source_weather(), seed=7, days=10, block_days=3
    )
    different, _ = generate_synthetic_weather(
        source_weather(), seed=8, days=10, block_days=3
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_meta == second_meta
    assert not first.equals(different)
    assert len(first) == 240
    assert np.allclose(np.diff(first["time"]), 3600.0)
    assert first["time"].iloc[0] == source_weather()["time"].iloc[0]
    assert first["day number"].iloc[0] == 91
    assert first["global radiation"].ge(0.0).all()
    assert first["wind speed"].ge(0.0).all()
    assert first["RH"].between(0.0, 100.0).all()
    assert first["air temperature"].between(-30.0, 60.0).all()
    assert first_meta["block_days"] == 3
    assert len(first_meta["source_block_start_days"]) >= 4
    source = source_weather()
    assert first["air temperature"].diff().abs().max() <= source["air temperature"].diff().abs().max() + 1.0
    assert first["RH"].diff().abs().max() <= source["RH"].diff().abs().max() + 2.0


def test_unperturbed_generation_preserves_whole_source_days():
    generated, metadata = generate_synthetic_weather(
        source_weather(), seed=3, days=5, block_days=2, perturb=False
    )

    source = source_weather()
    first_source_day = metadata["source_block_start_days"][0]
    expected = source.iloc[first_source_day * 24:(first_source_day + 1) * 24]
    assert generated.iloc[:24]["air temperature"].tolist() == expected["air temperature"].tolist()
    assert generated.iloc[:24]["global radiation"].tolist() == expected["global radiation"].tolist()


def test_ensemble_writer_records_provenance_and_hashes(tmp_path):
    source_path = tmp_path / "2026.csv"
    source_weather().to_csv(source_path, index=False)

    manifest_path = write_synthetic_weather_ensemble(
        source_path,
        output_root=tmp_path,
        location="Chengdu",
        years=(3000, 3001),
        seeds=(0, 1),
        days=10,
        block_days=3,
        source_start_day=1,
        source_days=8,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method"] == "multivariate_moving_block_bootstrap"
    assert manifest["source_sha256"]
    assert manifest["source_start_day"] == 1
    assert manifest["source_days"] == 8
    assert len(manifest["outputs"]) == 2
    for item in manifest["outputs"]:
        output = tmp_path / "Chengdu" / f"{item['year']}.csv"
        assert output.exists()
        assert hashlib.sha256(output.read_bytes()).hexdigest() == item["sha256"]
        assert max(item["source_block_start_days"]) <= 5
