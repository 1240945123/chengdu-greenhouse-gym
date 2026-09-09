"""A2: GH2024 温度驱动果穗率（De Koning 1994）物候验证。

用 accumulate_temperature.normal（日平均温度）驱动 De Koning 穗率模型，
与 plant_message_info.layerFruitNumber（果穗数）的 4 个观测日期对比。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd

from processing.chengdu_sql_controls import split_mysql_values
from processing.chengdu_crop_observations import MESSAGE_FIELDS

SQL = r"E:/school/NKY/玻璃温室数据库信息/tomato_full_backup_20260724.sql"

# 物候关键日期（GH2024 planting_info id=56）
PLANTING_DATE = pd.Timestamp("2025-01-19")
FIRST_FLOWER_DATE = pd.Timestamp("2025-03-11")  # firflowerDateTime
FIRST_FRUIT_DATE = pd.Timestamp("2025-04-01")   # earlyFruitDateTime
MATURE_DATE = pd.Timestamp("2025-04-23")        # matureDateTime


def load_daily_temperature(sql_text: str) -> pd.DataFrame:
    acc = []
    for m in re.finditer(r"INSERT INTO `accumulate_temperature` VALUES \((.*?)\);", sql_text, re.S):
        v = split_mysql_values(m.group(1))
        acc.append((v[6], v[7], v[8], v[9]))  # greenhouseId, date, normal, active
    frame = pd.DataFrame(acc, columns=["greenhouse_id", "date", "normal", "active"])
    for column in ("normal", "active"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["greenhouse_id"] == "1"].sort_values("date").set_index("date")


def load_fruit_truss_observations(sql_text: str) -> pd.DataFrame:
    rows = []
    for m in re.finditer(r"INSERT INTO `plant_message_info` VALUES \((.*?)\);", sql_text, re.S):
        rec = dict(zip(MESSAGE_FIELDS, split_mysql_values(m.group(1))))
        trusses = rec.get("layerFruitNumber")
        if trusses is None or trusses in ("NULL", ""):
            continue
        value = float(trusses)
        if value <= 0:
            continue
        rows.append((pd.Timestamp(rec["collectDateTime"]), value))
    frame = pd.DataFrame(rows, columns=["observation_date", "fruit_truss_count"])
    frame["observation_date"] = frame["observation_date"].dt.normalize()
    return (
        frame.groupby("observation_date")["fruit_truss_count"]
        .agg(mean="mean", count="count")
        .reset_index()
    )


def de_koning_cumulative_trusses(
    temperature: pd.DataFrame, baseline: pd.Timestamp, timezone: str = "Asia/Shanghai"
) -> pd.Series:
    """从 baseline 起累计 De Koning 1994 穗率（日平均温度版）。

    trusses_per_day = max(0, -0.2903 + 0.1454 * ln(T_24h))
    """
    temperature = temperature.copy()
    temperature.index = pd.DatetimeIndex(
        temperature.index.map(lambda t: t.tz_localize(timezone) if t.tzinfo is None else t)
    )
    baseline = pd.Timestamp(baseline).tz_localize(timezone) if pd.Timestamp(baseline).tzinfo is None else pd.Timestamp(baseline)
    temperature = temperature.loc[temperature.index >= baseline]
    temps = temperature["normal"].to_numpy(dtype=float)
    daily_rate = np.maximum(0.0, -0.2903 + 0.1454 * np.log(np.maximum(temps, 1e-6)))
    return pd.Series(np.cumsum(daily_rate), index=temperature.index, name="cumulative_trusses")


def main() -> None:
    sql = open(SQL, encoding="utf-8-sig").read()
    temperature = load_daily_temperature(sql)
    observations = load_fruit_truss_observations(sql)

    print("=== 果穗数观测 ===")
    print(observations.to_string(index=False))
    print()

    results = {}
    for label, baseline in (
        ("定植 2025-01-19", PLANTING_DATE),
        ("始花 2025-03-11", FIRST_FLOWER_DATE),
        ("初果 2025-04-01", FIRST_FRUIT_DATE),
    ):
        cumulative = de_koning_cumulative_trusses(temperature, baseline)
        predicted = []
        for obs_date in observations["observation_date"]:
            # 找到 ≤ obs_date 的最后一个累计值
            past = cumulative[cumulative.index <= pd.Timestamp(obs_date).tz_localize("Asia/Shanghai")]
            predicted.append(float(past.iloc[-1]) if not past.empty else 0.0)
        predicted = np.array(predicted)
        observed = observations["mean"].to_numpy(dtype=float)
        residual = predicted - observed
        wmape = float(np.abs(residual).sum() / observed.sum()) if observed.sum() > 0 else None
        results[label] = {
            "predicted": predicted,
            "mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "bias": float(np.mean(residual)),
            "wmape": wmape,
        }
        print(f"=== baseline = {label} ===")
        for d, p, o in zip(observations["observation_date"].dt.date, predicted, observed):
            print(f"  {d}: 预测 {p:.2f} 穗 vs 观测 {o:.1f} 穗")
        print(f"  MAE={results[label]['mae']:.2f} 穗, RMSE={results[label]['rmse']:.2f}, bias={results[label]['bias']:.2f}, WMAPE={wmape if wmape is None else round(wmape,3)}")
        print()

    # 温度摘要
    season_temp = temperature.loc[PLANTING_DATE:MATURE_DATE]
    print(f"=== 温度摘要（定植~成熟，{len(season_temp)} 天）===")
    print(f"  日平均温度 mean={season_temp['normal'].mean():.1f}°C, min={season_temp['normal'].min():.1f}, max={season_temp['normal'].max():.1f}")
    print(f"  活动积温(>10°C)累计 = {season_temp['active'].sum():.0f} °C·day")
    print(f"  正常积温(>0°C)累计 = {season_temp['normal'].sum():.0f} °C·day")


if __name__ == "__main__":
    main()
