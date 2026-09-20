"""把块自举合成天气（GreenLight 原生格式）转换为 glass_env 期望的窄表格式。

源：`data/.../weather/Chengdu/3000–3009.csv`（10 条 multivariate moving-block-bootstrap 序列）
    列 = time(秒), global radiation, wind speed, air temperature, sky temperature,
         CO2 concentration, day number, RH
目标列 = timestamp, global_radiation, outdoor_air_temperature, outdoor_relative_humidity, wind_speed
    （与 `aligned/greenhouse_1h.csv` 中 env 实际读取的 5 列同名）

时间轴：源 `time` 以秒计，起点 7 776 000 s = 90 天 → 若纪元取当年 1/1，则对应 **4/1**；
       源 `day number` 起点 91（非闰年第 91 天 = 4/1），两者一致，故纪元取 2026-01-01。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("data/processed/chengdu_agri/greenhouse_001")
SRC = ROOT / "weather" / "Chengdu"
DST = ROOT / "weather" / "Chengdu_aligned"

EPOCH = pd.Timestamp("2026-01-01")

COLMAP = {
    "time": None,                       # 单独处理
    "global radiation": "global_radiation",
    "air temperature": "outdoor_air_temperature",
    "RH": "outdoor_relative_humidity",
    "wind speed": "wind_speed",
}
ORDER = ["timestamp", "global_radiation", "outdoor_air_temperature",
         "outdoor_relative_humidity", "wind_speed"]


def convert_one(src: Path, dst: Path) -> dict:
    d = pd.read_csv(src)
    missing = [c for c in COLMAP if c not in d.columns]
    if missing:
        raise ValueError(f"{src.name} 缺列: {missing}")
    ts = EPOCH + pd.to_timedelta(d["time"], unit="s")
    out = pd.DataFrame({"timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S")})
    for s, t in COLMAP.items():
        if t:
            out[t] = d[s].astype(float)
    out = out[ORDER]
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    return {
        "file": dst.name, "rows": len(out),
        "t_start": str(ts.iloc[0]), "t_end": str(ts.iloc[-1]),
        "days": round(len(out) / 24, 1),
        "rad_max": round(float(d["global radiation"].max()), 1),
        "t_out_range": [round(float(d["air temperature"].min()), 1),
                        round(float(d["air temperature"].max()), 1)],
        "rh_range": [round(float(d["RH"].min()), 1), round(float(d["RH"].max()), 1)],
    }


def main() -> None:
    srcs = sorted(SRC.glob("30*.csv"))
    if not srcs:
        print(f"未找到合成天气：{SRC}", file=sys.stderr)
        sys.exit(1)
    print(f"发现 {len(srcs)} 条合成天气 → {DST}")
    rows = []
    for s in srcs:
        info = convert_one(s, DST / f"{s.stem}.csv")
        rows.append(info)
        print(f"  {info['file']:>10}  {info['rows']:>5} 行 ({info['days']:>4} 天)  "
              f"{info['t_start']} → {info['t_end']}  "
              f"辐射≤{info['rad_max']:>6.1f}  室外温{info['t_out_range']}  RH{info['rh_range']}")
    pd.DataFrame(rows).to_csv(DST / "_convert_manifest.csv", index=False)
    print(f"\n已写入 {len(rows)} 个转换文件 + _convert_manifest.csv")


if __name__ == "__main__":
    main()
