"""执行器操作日志 → 逐时档位状态序列。

读 9 个执行器 xls 的操作日志（事件型），对齐 greenhouse_1h.csv 的时间戳，
前向填充得到每小时每个执行器的档位状态。输出用于模型标定/行为分析。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import xlrd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

XLS_DIR = Path(r"E:\school\final paper\玻璃温室数据库信息\控制器数据")
SENSOR_1H = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
OUT = Path("results/chengdu_agri_greenhouse_001/real_greenhouse/aligned_actuator_states.csv")

# 执行器: (文件名前缀, 标准列名)
ACTUATORS = [
    ("43外遮阳", "shade_cloth"),
    ("44顶保温", "top_insulation"),
    ("45四周保温", "side_insulation"),
    ("46顶窗", "roof_vent"),
    ("47湿帘风机", "pad_fan"),
    ("48补光灯", "grow_lamp"),
    ("49co2发生器", "co2_gen"),
    ("50湿帘水泵", "pad_pump"),
    ("51湿帘卷膜机", "pad_curtain"),
]


def excel_to_datetime(v: float) -> datetime:
    return datetime(1899, 12, 30) + timedelta(days=float(v))


def level_of(op: str) -> int:
    """操作类型 → 档位。TURN_OFF=0, TURN_ON=1, TURN_ON_N=N"""
    if op == "TURN_OFF":
        return 0
    if op == "TURN_ON":
        return 1
    if op.startswith("TURN_ON_"):
        return int(op.split("_")[-1])
    return 0


def load_actuator_events(path: Path) -> pd.DataFrame:
    wb = xlrd.open_workbook(str(path))
    sh = wb.sheet_by_index(0)
    header = [str(sh.cell_value(0, c)) for c in range(sh.ncols)]
    time_col = header.index("操作时间")
    op_col = header.index("操作类型")
    events = []
    for r in range(1, sh.nrows):
        t = sh.cell_value(r, time_col)
        op = str(sh.cell_value(r, op_col))
        if isinstance(t, float) and t > 40000:
            events.append((excel_to_datetime(t), level_of(op)))
    df = pd.DataFrame(events, columns=["timestamp", "level"]).sort_values("timestamp")
    return df


def main():
    sensor = pd.read_csv(SENSOR_1H)
    sensor["timestamp"] = pd.to_datetime(sensor["timestamp"])

    # 每个执行器: 事件 → 前向填充到 sensor 时间戳
    result = sensor[["timestamp"]].copy()
    coverage = {}
    for prefix, colname in ACTUATORS:
        path = XLS_DIR / f"{prefix}.xls"
        events = load_actuator_events(path)
        if events.empty:
            result[colname] = 0
            coverage[colname] = 0
            continue
        # 在 sensor 时间戳上前向填充: 取每个时间戳之前最后一次事件后的档位
        # merge_asof: 每个 sensor 时间戳匹配 <= 它的最后一个事件时间
        levels = pd.merge_asof(
            result[["timestamp"]],
            events.rename(columns={"level": colname}),
            on="timestamp",
            direction="backward",
        )[colname].fillna(0).astype(int)
        result[colname] = levels
        # 覆盖率: 有操作记录的时间段占比
        first_op = events.timestamp.min()
        coverage[colname] = float((sensor.timestamp >= first_op).mean())
        print(f"  {prefix} ({colname}): {len(events)} 事件, 档位 {sorted(events.level.unique())}, "
              f"首操作 {first_op}, 覆盖 {coverage[colname]*100:.0f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT} ({len(result)} 行, {result.columns.tolist()})")

    # 摘要: 各执行器档位分布
    print("\n=== 各执行器档位分布（逐时）===")
    for _, colname in ACTUATORS:
        dist = result[colname].value_counts().sort_index()
        dist_str = ", ".join(f"{k}:{v}" for k, v in dist.items())
        active = (result[colname] > 0).mean()
        print(f"  {colname:16s}: {dist_str} | 开启占比 {active*100:.1f}%")


if __name__ == "__main__":
    main()
